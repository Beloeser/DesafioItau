"""Backtest endurecido: lag t-1, stress de custo, holding, multi-seed, walk-forward.

Metricas reportadas: PnL, DD, % posicionado, Calmar (SEM enfatizar Sharpe alto).

Uso:
    .venv-finrl/bin/python tests/run_backtest_endurecido.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

mod = importlib.import_module("03_finrl_trading")
PairsTradingFinRLEnv = mod.PairsTradingFinRLEnv
prever_posicoes = mod.prever_posicoes
treinar_ppo = mod.treinar_ppo
split_formacao_negociacao = mod.split_formacao_negociacao
from avaliar_ganhos import carregar_pipeline, simular_ganhos  # noqa: E402

CSV = RAIZ / "data/processed/pipeline_com_quebras_TAEE_causal.csv"
if not CSV.exists():
    CSV = RAIZ / "data/processed/pipeline_com_quebras_TAEE.csv"
SAIDA = RAIZ / "data/processed/backtest_endurecido_TAEE.csv"
CAPITAL = 100_000.0
TIMESTEPS = 30_000
NET = [64, 64]


def calmar(ret_pct: float, dd_pct: float) -> float:
    dd = abs(dd_pct)
    return float(ret_pct / dd) if dd > 1e-9 else float("inf")


def metricas(df, y, x, h, col, taxa) -> dict:
    _, m = simular_ganhos(df, y, x, h, coluna_sinal=col, capital=CAPITAL, taxa=taxa)
    return {
        "pnl": round(m["pnl_liquido"], 2),
        "ret_pct": round(m["retorno_pct"], 2),
        "max_dd": round(m["max_drawdown_pct"], 3),
        "pos_pct": round(m["dias_posicionado_pct"], 1),
        "calmar": round(calmar(m["retorno_pct"], m["max_drawdown_pct"]), 2),
        "trades": m["trades_fechados"],
        # Sharpe so como referencia secundaria (nao e o foco)
        "sharpe_ref": round(m["sharpe_anualizado"], 2),
    }


def treinar_e_prever(treino, teste, y, x, h, *, lag, taxa, hold, seed) -> pd.DataFrame:
    kw = dict(capital=CAPITAL, taxa=taxa, features_lag=lag, holding_cost=hold)
    env_tr = PairsTradingFinRLEnv(treino, y, x, h, **kw)
    modelo = treinar_ppo(env_tr, TIMESTEPS, seed=seed, net_arch=NET)
    env_te = PairsTradingFinRLEnv(teste.copy(), y, x, h, **kw)
    out = teste.copy()
    out["sinal_finrl"] = prever_posicoes(modelo, env_te)
    return out


def main() -> None:
    df, y, x, h = carregar_pipeline(CSV)
    if "prob_quebra" not in df.columns:
        df["prob_quebra"] = 0.5

    treino, teste = split_formacao_negociacao(
        df, "2022-01-01", "2024-12-31", "2025-01-01", "2026-01-01"
    )
    print(f"Par {y}/{x} | treino {len(treino)}d | teste {len(teste)}d | CSV={CSV.name}")

    linhas = []

    # --- A) Luiz vs FinRL lag0 vs FinRL lag1 (custo 8 bps) ---
    print("\n=== A) Mesmo OOS 2025: Luiz | FinRL same-close | FinRL info-de-ontem ===")
    m_luiz = metricas(teste, y, x, h, "sinal", 0.0008)
    print(f"  Luiz:              PnL {m_luiz['pnl']:,.0f} | DD {m_luiz['max_dd']}% | pos {m_luiz['pos_pct']}% | Calmar {m_luiz['calmar']}")
    linhas.append({"bloco": "A_principal", "nome": "luiz_8bps", "lag": None, "taxa": 0.0008, "hold": 0.0, "seed": None, **m_luiz})

    d0 = treinar_e_prever(treino, teste, y, x, h, lag=0, taxa=0.0008, hold=0.0, seed=42)
    m0 = metricas(d0, y, x, h, "sinal_finrl", 0.0008)
    print(f"  FinRL lag=0 (hoje): PnL {m0['pnl']:,.0f} | DD {m0['max_dd']}% | pos {m0['pos_pct']}% | Calmar {m0['calmar']}")
    linhas.append({"bloco": "A_principal", "nome": "finrl_lag0_same_close", "lag": 0, "taxa": 0.0008, "hold": 0.0, "seed": 42, **m0})

    d1 = treinar_e_prever(treino, teste, y, x, h, lag=1, taxa=0.0008, hold=0.0, seed=42)
    m1 = metricas(d1, y, x, h, "sinal_finrl", 0.0008)
    print(f"  FinRL lag=1 (ontem):PnL {m1['pnl']:,.0f} | DD {m1['max_dd']}% | pos {m1['pos_pct']}% | Calmar {m1['calmar']}")
    linhas.append({"bloco": "A_principal", "nome": "finrl_lag1_realista", "lag": 1, "taxa": 0.0008, "hold": 0.0, "seed": 42, **m1})

    # --- B) Stress de custo no modo lag=1 (o realista) ---
    print("\n=== B) Stress de custo — FinRL lag=1 (info de ontem) ===")
    for taxa in (0.0, 0.0008, 0.0020, 0.0050):
        d = treinar_e_prever(treino, teste, y, x, h, lag=1, taxa=taxa, hold=0.0, seed=42)
        # avalia com a MESMA taxa
        m = metricas(d, y, x, h, "sinal_finrl", taxa)
        ml = metricas(teste, y, x, h, "sinal", taxa)
        print(
            f"  taxa={taxa*1e4:.0f}bps | FinRL PnL {m['pnl']:,.0f} pos {m['pos_pct']}% | "
            f"Luiz PnL {ml['pnl']:,.0f} pos {ml['pos_pct']}%"
        )
        linhas.append({"bloco": "B_stress_custo_lag1", "nome": f"finrl_lag1_taxa_{taxa}", "lag": 1, "taxa": taxa, "hold": 0.0, "seed": 42, **m})
        linhas.append({"bloco": "B_stress_custo_lag1", "nome": f"luiz_taxa_{taxa}", "lag": None, "taxa": taxa, "hold": 0.0, "seed": None, **ml})

    # --- C) Holding cost no lag=0 (para ver se reduz agressividade) ---
    print("\n=== C) Holding cost — FinRL lag=0 com cobranca por dia posicionado ===")
    for hold in (0.0, 0.00005, 0.0001):
        d = treinar_e_prever(treino, teste, y, x, h, lag=0, taxa=0.0008, hold=hold, seed=42)
        m = metricas(d, y, x, h, "sinal_finrl", 0.0008)
        print(f"  hold={hold} | PnL {m['pnl']:,.0f} | pos {m['pos_pct']}% | DD {m['max_dd']}% | Calmar {m['calmar']}")
        linhas.append({"bloco": "C_holding", "nome": f"finrl_lag0_hold_{hold}", "lag": 0, "taxa": 0.0008, "hold": hold, "seed": 42, **m})

    # --- D) Multi-seed no lag=1 ---
    print("\n=== D) Multi-seed — FinRL lag=1, 8 bps, seeds 1..5 ===")
    pnls = []
    for seed in (1, 2, 3, 4, 5):
        d = treinar_e_prever(treino, teste, y, x, h, lag=1, taxa=0.0008, hold=0.0, seed=seed)
        m = metricas(d, y, x, h, "sinal_finrl", 0.0008)
        pnls.append(m["pnl"])
        print(f"  seed={seed} | PnL {m['pnl']:,.0f} | pos {m['pos_pct']}% | DD {m['max_dd']}%")
        linhas.append({"bloco": "D_multiseed_lag1", "nome": f"finrl_lag1_seed_{seed}", "lag": 1, "taxa": 0.0008, "hold": 0.0, "seed": seed, **m})
    print(f"  resumo seeds: mediana R$ {np.median(pnls):,.0f} | min {np.min(pnls):,.0f} | max {np.max(pnls):,.0f}")

    # --- E) Walk-forward simples (lag=1) ---
    print("\n=== E) Walk-forward — FinRL lag=1 vs Luiz ===")
    splits = [
        ("WF_2024", "2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ("WF_2025", "2022-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
        ("WF_2S25", "2022-01-01", "2025-06-30", "2025-07-01", "2025-12-31"),
    ]
    for nome, a, b, c, dte in splits:
        tr, te = split_formacao_negociacao(df, a, b, c, dte)
        if len(tr) < 100 or len(te) < 40:
            print(f"  {nome}: skip (poucos dias)")
            continue
        d = treinar_e_prever(tr, te, y, x, h, lag=1, taxa=0.0008, hold=0.0, seed=42)
        mf = metricas(d, y, x, h, "sinal_finrl", 0.0008)
        ml = metricas(te, y, x, h, "sinal", 0.0008)
        print(
            f"  {nome}: FinRL PnL {mf['pnl']:,.0f} pos {mf['pos_pct']}% | "
            f"Luiz PnL {ml['pnl']:,.0f} pos {ml['pos_pct']}% | Δ {mf['pnl']-ml['pnl']:,.0f}"
        )
        linhas.append({"bloco": "E_walkforward", "nome": f"{nome}_finrl_lag1", "lag": 1, "taxa": 0.0008, "hold": 0.0, "seed": 42, **mf})
        linhas.append({"bloco": "E_walkforward", "nome": f"{nome}_luiz", "lag": None, "taxa": 0.0008, "hold": 0.0, "seed": None, **ml})

    tab = pd.DataFrame(linhas)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(SAIDA, index=False)
    print(f"\nTabela salva: {SAIDA}")
    print("\nColunas principais: bloco, nome, pnl, max_dd, pos_pct, calmar (sharpe_ref so referencia)")


if __name__ == "__main__":
    main()
