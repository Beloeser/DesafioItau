"""Backtests por DIFERENTES janelas de tempo (nao so arquiteturas).

Usa CSV causal da SWANet (labels +5d sem cruzar formacao no treino).
Compara Luiz (sinal) vs FinRL PPO [64,64] 30k em varios splits.
Tambem reporta Sharpe DIARIO*sqrt(252) e Sharpe MENSAL*sqrt(12) + Calmar.

Uso:
    .venv-finrl/bin/python tests/run_backtests_por_periodo.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

modulo = importlib.import_module("03_finrl_trading")
PairsTradingFinRLEnv = modulo.PairsTradingFinRLEnv
prever_posicoes = modulo.prever_posicoes
treinar_ppo = modulo.treinar_ppo
from avaliar_ganhos import carregar_pipeline, simular_ganhos  # noqa: E402

CSV = RAIZ / "data/processed/pipeline_com_quebras_TAEE_causal.csv"
# fallback se ainda nao rodou o script causal
if not CSV.exists():
    CSV = RAIZ / "data/processed/pipeline_com_quebras_TAEE.csv"

SAIDA = RAIZ / "data/processed/backtests_por_periodo_TAEE.csv"
CAPITAL = 100_000.0
TAXA = 0.0008

# Selecoes de tempo: (nome, form_ini, form_fim, neg_ini, neg_fim)
SPLITS = [
    ("atual_3y_1y", "2022-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("treino_curto_1y", "2023-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("treino_longo_5y", "2020-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("mais_backtest_2y", "2022-01-01", "2023-12-31", "2024-01-01", "2025-12-31"),
    ("meio_2p5y_1p5y", "2022-01-01", "2024-06-30", "2024-07-01", "2025-12-31"),
    ("mais_treino_oos_curto", "2022-01-01", "2025-06-30", "2025-07-01", "2025-12-31"),
]


def sharpe_diario(ret: pd.Series) -> float:
    if ret.std(ddof=1) <= 0:
        return 0.0
    return float(ret.mean() / ret.std(ddof=1) * np.sqrt(252))


def sharpe_mensal(pnl: pd.Series, datas: pd.Series, capital: float) -> float:
    s = pd.Series(pnl.values, index=pd.to_datetime(datas))
    mensal = s.resample("ME").sum() / capital
    if len(mensal) < 2 or mensal.std(ddof=1) <= 0:
        return 0.0
    return float(mensal.mean() / mensal.std(ddof=1) * np.sqrt(12))


def calmar(retorno_pct: float, max_dd_pct: float) -> float:
    dd = abs(max_dd_pct)
    if dd < 1e-9:
        return float("inf")
    return float(retorno_pct / dd)


def avaliar(df, y, x, hedge, col: str) -> dict:
    dados, m = simular_ganhos(df, y, x, hedge, coluna_sinal=col, capital=CAPITAL, taxa=TAXA)
    ret = dados["pnl_liquido"] / CAPITAL
    return {
        "pnl": round(m["pnl_liquido"], 2),
        "ret_pct": round(m["retorno_pct"], 2),
        "sharpe_d252": round(sharpe_diario(ret), 3),
        "sharpe_m12": round(sharpe_mensal(dados["pnl_liquido"], dados["data"], CAPITAL), 3),
        "max_dd": round(m["max_drawdown_pct"], 3),
        "calmar": round(calmar(m["retorno_pct"], m["max_drawdown_pct"]), 2),
        "pos_pct": round(m["dias_posicionado_pct"], 1),
        "n_dias": len(df),
    }


def recorte(df, a, b):
    t0 = pd.Timestamp(a, tz="UTC")
    t1 = pd.Timestamp(b, tz="UTC")
    out = df[(df["data"] >= t0) & (df["data"] <= t1)].copy()
    if "prob_quebra" not in out.columns:
        out["prob_quebra"] = 0.5
    else:
        out["prob_quebra"] = out["prob_quebra"].astype(float).fillna(0.5)
    out = out.dropna(subset=["zscore_mr", "spread_observado"]).reset_index(drop=True)
    # Remove qualquer linha nao finita (evita NaN no PPO)
    mask = (
        np.isfinite(out["zscore_mr"].astype(float))
        & np.isfinite(out["spread_observado"].astype(float))
        & np.isfinite(out["prob_quebra"].astype(float))
    )
    out = out.loc[mask].reset_index(drop=True)
    return out


def main() -> None:
    # Para treino_longo precisa de dados desde 2020 — tentar CSV longo se existir
    csv_longo = RAIZ / "data/processed/pipeline_parcial_TAEE_desde_2019.csv"
    df_base, y, x, hedge = carregar_pipeline(CSV)
    if "prob_quebra" not in df_base.columns:
        df_base["prob_quebra"] = 0.5

    # Merge prob_quebra causal no CSV longo se necessario
    if csv_longo.exists():
        df_l, _, _, _ = carregar_pipeline(csv_longo)
        probs = df_base[["data", "prob_quebra"]].drop_duplicates("data")
        df_l = df_l.drop(columns=["prob_quebra"], errors="ignore")
        df_l = df_l.merge(probs, on="data", how="left")
        df_l["prob_quebra"] = df_l["prob_quebra"].astype(float).fillna(0.5)
        if "sinal" not in df_l.columns and "sinal" in df_base.columns:
            df_l = df_l.merge(df_base[["data", "sinal"]], on="data", how="left")
        df_full = df_l
    else:
        df_full = df_base
        if "prob_quebra" not in df_full.columns:
            df_full["prob_quebra"] = 0.5
        else:
            df_full["prob_quebra"] = df_full["prob_quebra"].astype(float).fillna(0.5)

    print(f"Fonte principal: {CSV.name} | full min/max: {df_full['data'].min().date()} .. {df_full['data'].max().date()}")

    linhas = []
    for nome, fi, ff, ni, nf in SPLITS:
        print(f"\n=== SPLIT {nome}: treino {fi}..{ff} | teste {ni}..{nf} ===")
        treino = recorte(df_full, fi, ff)
        teste = recorte(df_full, ni, nf)
        # precisa zscore e preferencialmente sinal no teste
        if "sinal" not in teste.columns:
            print("  SKIP (sem coluna sinal no recorte)")
            continue
        if len(treino) < 80 or len(teste) < 40:
            print(f"  SKIP curto treino={len(treino)} teste={len(teste)}")
            continue
        if treino["data"].max() >= teste["data"].min():
            print("  ERRO overlap — pulando")
            continue

        # Baseline Luiz
        m_luiz = avaliar(teste, y, x, hedge, "sinal")
        print(f"  Luiz: PnL {m_luiz['pnl']} | Sh_d {m_luiz['sharpe_d252']} | Sh_m {m_luiz['sharpe_m12']} | pos {m_luiz['pos_pct']}%")

        # FinRL
        env_tr = PairsTradingFinRLEnv(treino, y, x, hedge, capital=CAPITAL, taxa=TAXA)
        modelo = treinar_ppo(env_tr, timesteps=30_000, seed=42, net_arch=[64, 64])
        env_te = PairsTradingFinRLEnv(teste.copy(), y, x, hedge, capital=CAPITAL, taxa=TAXA)
        pos = prever_posicoes(modelo, env_te)
        dte = teste.copy()
        dte["sinal_finrl"] = pos
        m_fin = avaliar(dte, y, x, hedge, "sinal_finrl")
        print(f"  FinRL: PnL {m_fin['pnl']} | Sh_d {m_fin['sharpe_d252']} | Sh_m {m_fin['sharpe_m12']} | pos {m_fin['pos_pct']}%")

        linhas.append({
            "split": nome,
            "formacao": f"{fi}..{ff}",
            "negociacao": f"{ni}..{nf}",
            "dias_treino": len(treino),
            "dias_teste": len(teste),
            "estrategia": "luiz",
            **{f"luiz_{k}": v for k, v in m_luiz.items()},
        })
        # flatten side by side in one row for table readability
        linhas[-1].update({f"finrl_{k}": v for k, v in m_fin.items()})
        linhas[-1]["delta_pnl"] = round(m_fin["pnl"] - m_luiz["pnl"], 2)

    # Reformat: one row per split with both strategies columns
    tab = pd.DataFrame(linhas)
    # drop nested estrategia key confusion
    if "estrategia" in tab.columns:
        tab = tab.drop(columns=["estrategia"])
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(SAIDA, index=False)

    print("\n" + "=" * 100)
    print("TABELA POR PERIODO — Luiz vs FinRL (PPO [64,64] 30k, 8 bps)")
    print("sharpe_d252 = media/std diaria * sqrt(252)   |  sharpe_m12 = mensal * sqrt(12)")
    print("=" * 100)
    cols = [
        "split", "dias_treino", "dias_teste",
        "luiz_pnl", "luiz_sharpe_d252", "luiz_sharpe_m12", "luiz_max_dd", "luiz_pos_pct",
        "finrl_pnl", "finrl_sharpe_d252", "finrl_sharpe_m12", "finrl_max_dd", "finrl_pos_pct",
        "delta_pnl",
    ]
    cols = [c for c in cols if c in tab.columns]
    print(tab[cols].to_string(index=False))
    print(f"\nSalvo: {SAIDA}")
    print(
        "\nNOTA Sharpe: a formula diaria*sqrt(252) esta correta matematicamente.\n"
        "Valores >5 aparecem porque a curva OOS e muito lisa (DD <1%).\n"
        "Isso e suspeito economicamente (sorte de janela / overfit), nao bug de *sqrt(252).\n"
        "Prefira comparar sharpe_m12 + calmar + varios splits."
    )


if __name__ == "__main__":
    main()
