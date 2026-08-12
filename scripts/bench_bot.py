#!/usr/bin/env python3
"""Avaliador COMUM para todas as branches (Luiz / Luiz-finrl).

Este arquivo e AUTONOMO: nao importa src/. Copia ele para qualquer branch
e rode. A matematica de PnL e a mesma em todos os sufixos.

Saida (colunas identicas, so muda o sufixo no nome do arquivo):
    data/processed/bench_<SUFIXO>.csv

Uso:
    python3 scripts/bench_bot.py --sufixo luiz_antigo \\
        --entrada data/processed/pipeline_TAEE3_TAEE11.csv \\
        --entrada data/processed/pipeline_CGRA3_CGRA4.csv \\
        --entrada data/processed/pipeline_PLAS3_INEP4.csv

    # se o CSV tiver coluna sinal_finrl, ela tambem e avaliada
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CAPITAL = 100_000.0
TAXA = 0.0008  # 8 bps

# Recortes inteligentes: teste SEMPRE depois do treino (sem overlap).
# Luiz (regra) nao "treina"; o recorte de treino so importa para o FinRL.
# O PnL e medido SOMENTE no periodo de teste.
SPLITS = [
    ("oficial_3y_1y", "2022-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("walk_2024", "2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("treino_curto", "2023-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("oos_longo", "2022-01-01", "2024-06-30", "2024-07-01", "2025-12-31"),
]


def carregar(entrada: Path) -> tuple[pd.DataFrame, str, str, float]:
    df = pd.read_csv(entrada)
    df["data"] = pd.to_datetime(df["data"], utc=True)
    y, x = df.columns[1], df.columns[2]
    linha = df.iloc[0]
    hedge = float((linha[y] - linha["spread_observado"]) / linha[x])
    return df, y, x, hedge


def recorte(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    t0 = pd.Timestamp(a, tz="UTC")
    t1 = pd.Timestamp(b, tz="UTC")
    out = df[(df["data"] >= t0) & (df["data"] <= t1)].copy()
    return out.dropna(subset=["spread_observado"]).reset_index(drop=True)


def simular(df: pd.DataFrame, y: str, x: str, hedge: float, coluna: str, taxa: float) -> dict:
    dados = df.reset_index(drop=True).copy()
    if coluna not in dados.columns:
        raise KeyError(coluna)
    preco_y = dados[y].astype(float)
    preco_x = dados[x].astype(float)
    spread = dados["spread_observado"].astype(float)
    # Sinal de ontem vira posicao de hoje (lucro = movimento de ontem->hoje).
    posicao = dados[coluna].fillna(0).astype(int).shift(1).fillna(0).astype(int)
    n_y = (CAPITAL / 2.0) / max(float(preco_y.iloc[0]), 1e-9)
    n_x = abs(hedge) * n_y
    pnl_bruto = posicao * n_y * spread.diff().fillna(0.0)
    notional = n_y * preco_y + n_x * preco_x
    mudanca = posicao.diff().abs().fillna(abs(float(posicao.iloc[0])))
    custos = taxa * notional * mudanca
    pnl_liq = pnl_bruto - custos
    equity = CAPITAL + pnl_liq.cumsum()
    ret = pnl_liq / CAPITAL
    sharpe = 0.0
    if ret.std(ddof=1) > 0:
        sharpe = float(ret.mean() / ret.std(ddof=1) * np.sqrt(252))
    topo = equity.cummax()
    max_dd = float(((equity - topo) / topo).min() * 100)
    return {
        "pnl": float(pnl_liq.sum()),
        "ret_pct": float(pnl_liq.sum() / CAPITAL * 100),
        "max_dd_pct": max_dd,
        "pos_pct": float((posicao != 0).mean() * 100),
        "sharpe_ref": sharpe,
        "n_dias": int(len(dados)),
        "n_pos_dias": int((posicao != 0).sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sufixo", required=True, help="ex.: luiz_antigo | luiz_finrl_antigo | luiz_novo | luiz_finrl_novo")
    p.add_argument("--entrada", type=Path, action="append", required=True, help="CSV do pipeline (pode repetir)")
    p.add_argument("--saida-dir", type=Path, default=Path("data/processed"))
    args = p.parse_args()

    linhas = []
    for path in args.entrada:
        if not path.exists():
            raise FileNotFoundError(path)
        df, y, x, hedge = carregar(path)
        par = f"{y}/{x}"
        colunas = ["sinal"]
        if "sinal_calibrado" in df.columns:
            colunas.append("sinal_calibrado")
        if "sinal_finrl" in df.columns:
            colunas.append("sinal_finrl")
        if "sinal_hibrido" in df.columns:
            colunas.append("sinal_hibrido")

        for nome_split, fi, ff, ni, nf in SPLITS:
            teste = recorte(df, ni, nf)
            if len(teste) < 40:
                print(f"  SKIP {par} {nome_split}: teste curto ({len(teste)}d)")
                continue
            for col in colunas:
                m = simular(teste, y, x, hedge, col, TAXA)
                linhas.append({
                    "sufixo": args.sufixo,
                    "par": par,
                    "arquivo": str(path),
                    "split": nome_split,
                    "formacao": f"{fi}..{ff}",
                    "teste": f"{ni}..{nf}",
                    "coluna_sinal": col,
                    "taxa": TAXA,
                    "capital": CAPITAL,
                    **m,
                })
                print(
                    f"{args.sufixo:22s} {par:14s} {nome_split:16s} {col:14s} "
                    f"PnL {m['pnl']:10.2f}  pos {m['pos_pct']:5.1f}%  DD {m['max_dd_pct']:7.3f}%"
                )

    out = args.saida_dir / f"bench_{args.sufixo}.csv"
    args.saida_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(out, index=False)
    print(f"\nSALVO: {out.resolve()}")
    print("Colunas iguais em todos os sufixos. Depois rode: python3 scripts/juntar_benches.py")


if __name__ == "__main__":
    main()
