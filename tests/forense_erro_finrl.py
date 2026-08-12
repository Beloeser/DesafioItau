#!/usr/bin/env python3
"""Forense: quanto cada 'erro' inflava o FinRL vs Luiz (mesmos sinais, mesma base).

Roda com python3 na raiz do repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from avaliar_ganhos import carregar_pipeline, simular_ganhos  # noqa: E402
from periodos import DATA_FIM_NEGOCIACAO, DATA_INICIO_NEGOCIACAO  # noqa: E402

CAPITAL = 100_000.0
TAXA = 0.0008


def pnl_bug_mesmo_dia(df, y, x, h, col, taxa=TAXA):
    """BUG classico: posicao[t]=sinal[t], lucro no mesmo dia (SEM shift)."""
    d = df.reset_index(drop=True).copy()
    pos = d[col].fillna(0).astype(int)
    spread = d["spread_observado"].astype(float)
    n_y = (CAPITAL / 2.0) / max(float(d[y].iloc[0]), 1e-9)
    n_x = abs(h) * n_y
    pnl = pos * n_y * spread.diff().fillna(0.0)
    notional = n_y * d[y].astype(float) + n_x * d[x].astype(float)
    custos = taxa * notional * pos.diff().abs().fillna(abs(float(pos.iloc[0])))
    return float((pnl - custos).sum())


def main() -> None:
    candidatos = [
        RAIZ / "data/processed/pipeline_com_quebras_TAEE_causal_finrl.csv",
        RAIZ / "data/processed/pipeline_finrl_TAEE.csv",
    ]
    entrada = next((p for p in candidatos if p.exists()), None)
    if entrada is None:
        raise SystemExit("Sem CSV com sinal_finrl.")

    df, y, x, h = carregar_pipeline(entrada)
    t0 = pd.Timestamp(DATA_INICIO_NEGOCIACAO, tz="UTC")
    t1 = pd.Timestamp(DATA_FIM_NEGOCIACAO, tz="UTC")
    oos = df[(df["data"] >= t0) & (df["data"] <= t1)].copy()

    if "sinal_finrl" not in oos.columns:
        raise SystemExit(f"{entrada} sem sinal_finrl")

    pos_fin = float((oos["sinal_finrl"].fillna(0) != 0).mean() * 100)
    pos_luiz = float((oos["sinal"].fillna(0) != 0).mean() * 100)

    linhas = []

    def add(nome: str, col: str, execucao: str, bug=None):
        if bug:
            pnl = pnl_bug_mesmo_dia(oos, y, x, h, col, TAXA)
            sharpe = float("nan")
        else:
            _, m = simular_ganhos(
                oos, y, x, h, col, CAPITAL, TAXA, execucao=execucao
            )
            pnl = m["pnl_liquido"]
            sharpe = m["sharpe_anualizado"]
        linhas.append({"modo": nome, "coluna": col, "pnl": pnl, "sharpe": sharpe})

    # Luiz baseline
    add("Luiz CORRETO (shift + abertura)", "sinal", "abertura")
    add("Luiz GITHUB (shift + fechamento)", "sinal", "fechamento")

    # FinRL mesmos sinais, avaliadores diferentes
    add("FinRL CORRETO (shift + abertura)", "sinal_finrl", "abertura")
    add("FinRL GITHUB (shift + fechamento)", "sinal_finrl", "fechamento")
    add("FinRL BUG mesmo-dia (sem shift)", "sinal_finrl", "fechamento", bug=True)

    tab = pd.DataFrame(linhas)
    pnl_luiz_ok = tab.loc[tab["modo"] == "Luiz CORRETO (shift + abertura)", "pnl"].iloc[0]
    pnl_fin_github = tab.loc[tab["modo"] == "FinRL GITHUB (shift + fechamento)", "pnl"].iloc[0]
    pnl_fin_ok = tab.loc[tab["modo"] == "FinRL CORRETO (shift + abertura)", "pnl"].iloc[0]
    pnl_fin_bug = tab.loc[tab["modo"] == "FinRL BUG mesmo-dia (sem shift)", "pnl"].iloc[0]
    pnl_luiz_github = tab.loc[tab["modo"] == "Luiz GITHUB (shift + fechamento)", "pnl"].iloc[0]

    print("=" * 72)
    print(f"FORENSE TAEE OOS | {entrada.name}")
    print(f"Posicionado: Luiz {pos_luiz:.1f}% | FinRL {pos_fin:.1f}%")
    print("=" * 72)
    for _, r in tab.iterrows():
        sh = f"{r['sharpe']:.2f}" if pd.notna(r["sharpe"]) else "n/a"
        print(f"  {r['modo']:<42} PnL R$ {r['pnl']:10,.0f}  Sharpe {sh}")

    print("\n--- DECOMPOSICAO DO 'RESULTADO BOM DEMAIS' ---")
    print(f"  FinRL GitHub vs Luiz GitHub     : R$ {pnl_fin_github - pnl_luiz_github:+,.0f}  (razao {pnl_fin_github/max(pnl_luiz_github,1):.2f}x)")
    print(f"  FinRL GitHub vs FinRL abertura  : R$ {pnl_fin_github - pnl_fin_ok:+,.0f}  <- erro PRECO (100 vs 103)")
    print(f"  FinRL abertura vs Luiz abertura : R$ {pnl_fin_ok - pnl_luiz_ok:+,.0f}  <- rede + mais dias posicionado")
    print(f"  FinRL BUG mesmo-dia vs GitHub   : R$ {pnl_fin_bug - pnl_fin_github:+,.0f}  <- NAO e o bug do GitHub (shift existe)")
    print("\n  CONCLUSAO: GitHub Luiz-finrl NAO usa bug mesmo-dia.")
    print("  O inflador #1 medido: spread.diff() no CLOSE (nao abertura).")
    print("  O inflador #2: sinal_finrl diferente (rede), nao so o avaliador.")

    out = RAIZ / "data/processed/forense_erro_finrl_TAEE.csv"
    tab.to_csv(out, index=False)
    print(f"\nSalvo: {out}")


if __name__ == "__main__":
    main()
