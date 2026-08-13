"""Backtests FinRL sem vazamento SWANet — varias arquiteturas vs Luiz.

Fluxo:
  1) Retreina SWANet com mask causal (label +5d nao cruza negociacao)
  2) Treina varias politicas PPO (redes/timesteps diferentes)
  3) Avalia no MESMO OOS 2025 e compara com baseline Luiz (coluna sinal)

Uso:
    .venv-finrl/bin/python tests/run_backtests_sem_vazamento.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

modulo = importlib.import_module("03_finrl_trading")
PairsTradingFinRLEnv = modulo.PairsTradingFinRLEnv
prever_posicoes = modulo.prever_posicoes
split_formacao_negociacao = modulo.split_formacao_negociacao
treinar_ppo = modulo.treinar_ppo

from avaliar_ganhos import carregar_pipeline, simular_ganhos  # noqa: E402

PARCIAL = RAIZ / "data/processed/pipeline_parcial_TAEE3_TAEE11.csv"
QUEBRAS = RAIZ / "data/processed/pipeline_com_quebras_TAEE_causal.csv"
SAIDA_TABELA = RAIZ / "data/processed/backtests_sem_vazamento_TAEE.csv"
MODELS = RAIZ / "models" / "backtests_sem_vazamento"
CAPITAL = 100_000.0
TAXA = 0.0008

FORM_INI, FORM_FIM = "2022-01-01", "2024-12-31"
NEG_INI, NEG_FIM = "2025-01-01", "2026-01-01"

# Variantes de treino PPO (todas sobre o mesmo CSV causal)
VARIANTES = [
    {"nome": "ppo_default_50k", "timesteps": 50_000, "net_arch": None},
    {"nome": "ppo_small_32_30k", "timesteps": 30_000, "net_arch": [32]},
    {"nome": "ppo_64x64_30k", "timesteps": 30_000, "net_arch": [64, 64]},
    {"nome": "ppo_64x64_80k", "timesteps": 80_000, "net_arch": [64, 64]},
]


def retreinar_swanet_causal() -> Path:
    import subprocess

    if not PARCIAL.exists():
        raise FileNotFoundError(f"Falta {PARCIAL}")
    QUEBRAS.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(RAIZ / ".venv-finrl/bin/python"),
        str(RAIZ / "src/01_swanet_quebras.py"),
        "--entrada", str(PARCIAL),
        "--saida", str(QUEBRAS),
        "--inicio-formacao", FORM_INI,
        "--fim-formacao", FORM_FIM,
    ]
    print(">>> Retreinando SWANet (mask causal, sem label cruzando 2025)...")
    subprocess.run(cmd, cwd=RAIZ, check=True)
    return QUEBRAS


def metricas(df, y, x, hedge, col: str) -> dict:
    _, m = simular_ganhos(df, y, x, hedge, coluna_sinal=col, capital=CAPITAL, taxa=TAXA)
    return {
        "pnl_liquido": round(m["pnl_liquido"], 2),
        "retorno_pct": round(m["retorno_pct"], 2),
        "sharpe": round(m["sharpe_anualizado"], 3),
        "max_dd_pct": round(m["max_drawdown_pct"], 3),
        "pos_pct": round(m["dias_posicionado_pct"], 1),
        "trades": m["trades_fechados"],
    }


def verifica_zero_janelas_vazando(df: pd.DataFrame) -> int:
    """Conta janelas de formacao cujo label i:i+5 tocaria a negociacao.

    Apos o fix, o TREINO nao usa essas janelas; esta funcao so diagnostica
    quantas ainda EXISTIRIAM no CSV (previsao continua sendo gerada).
    """
    fim = pd.Timestamp(FORM_FIM, tz="UTC")
    seq = 24
    cruzam = 0
    for i in range(seq, len(df) - 5):
        if df["data"].iloc[i] <= fim and df["data"].iloc[i + 4] > fim:
            cruzam += 1
    return cruzam


def main() -> None:
    retreinar_swanet_causal()

    df, y, x, hedge = carregar_pipeline(QUEBRAS)
    cruzam = verifica_zero_janelas_vazando(df)
    print(f"  Janelas na fronteira (ainda existem no CSV, mas excluidas do treino SWANet): {cruzam}")

    treino, teste = split_formacao_negociacao(df, FORM_INI, FORM_FIM, NEG_INI, NEG_FIM)
    print(f"  Split: treino={len(treino)}d | teste={len(teste)}d | par={y}/{x}")

    # Baseline Luiz no mesmo OOS
    base = metricas(teste, y, x, hedge, "sinal")
    linhas = [{
        "estrategia": "luiz_baseline_sinal",
        "timesteps": None,
        "net_arch": None,
        "swanet_causal": True,
        **base,
        "delta_pnl_vs_luiz": 0.0,
    }]
    print(f"\n  Luiz baseline: PnL R$ {base['pnl_liquido']:,.2f} | Sharpe {base['sharpe']} | pos {base['pos_pct']}%")

    MODELS.mkdir(parents=True, exist_ok=True)
    for var in VARIANTES:
        nome = var["nome"]
        print(f"\n>>> Treinando {nome} (ts={var['timesteps']}, arch={var['net_arch']})...")
        env_tr = PairsTradingFinRLEnv(treino, y, x, hedge, capital=CAPITAL, taxa=TAXA)
        modelo = treinar_ppo(
            env_tr,
            timesteps=var["timesteps"],
            seed=42,
            net_arch=var["net_arch"],
        )
        modelo_path = MODELS / f"{nome}.zip"
        modelo.save(str(modelo_path))

        env_te = PairsTradingFinRLEnv(teste.copy(), y, x, hedge, capital=CAPITAL, taxa=TAXA)
        pos = prever_posicoes(modelo, env_te)
        dte = teste.copy()
        dte["sinal_finrl"] = pos
        # Sanidade: ultimo dia nao precisa mais ser forçado a 0
        print(f"  ultimo sinal_finrl = {int(pos[-1])} (antes do fix era sempre 0)")

        m = metricas(dte, y, x, hedge, "sinal_finrl")
        linhas.append({
            "estrategia": nome,
            "timesteps": var["timesteps"],
            "net_arch": str(var["net_arch"]),
            "swanet_causal": True,
            **m,
            "delta_pnl_vs_luiz": round(m["pnl_liquido"] - base["pnl_liquido"], 2),
        })
        # always-long referencia
        dte["always_long"] = 1
        m_long = metricas(dte, y, x, hedge, "always_long")
        print(
            f"  FinRL: PnL R$ {m['pnl_liquido']:,.2f} | Sharpe {m['sharpe']} | "
            f"pos {m['pos_pct']}% | Δ vs Luiz R$ {m['pnl_liquido'] - base['pnl_liquido']:,.2f}"
        )
        print(f"  Always-long (ref): PnL R$ {m_long['pnl_liquido']:,.2f}")

        out_csv = MODELS / f"{nome}_oos.csv"
        dte.to_csv(out_csv, index=False)

    # Always long/short linhas extras na tabela
    dref = teste.copy()
    dref["always_long"] = 1
    dref["always_short"] = -1
    for col, nome in (("always_long", "ref_always_long"), ("always_short", "ref_always_short")):
        m = metricas(dref, y, x, hedge, col)
        linhas.append({
            "estrategia": nome,
            "timesteps": None,
            "net_arch": None,
            "swanet_causal": True,
            **m,
            "delta_pnl_vs_luiz": round(m["pnl_liquido"] - base["pnl_liquido"], 2),
        })

    tab = pd.DataFrame(linhas)
    SAIDA_TABELA.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(SAIDA_TABELA, index=False)

    print("\n" + "=" * 78)
    print("TABELA — mesmo OOS 2025, SWANet sem vazamento de label no treino, 8 bps")
    print("=" * 78)
    cols = ["estrategia", "timesteps", "net_arch", "pnl_liquido", "sharpe", "max_dd_pct", "pos_pct", "delta_pnl_vs_luiz"]
    print(tab[cols].to_string(index=False))
    print(f"\nSalvo em: {SAIDA_TABELA}")


if __name__ == "__main__":
    main()
