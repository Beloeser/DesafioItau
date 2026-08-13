#!/usr/bin/env python3
"""Avaliador COMUM — mesma matematica do teste (abertura, 8 bps, sinal de ontem).

Usa src/avaliar_ganhos.py (execucao na abertura). Nao usa spread.diff() close-to-close.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from avaliar_ganhos import carregar_pipeline, simular_ganhos  # noqa: E402

CAPITAL = 100_000.0
TAXA = 0.0008  # 8 bps
DADOS_SETORES = RAIZ / "data/raw/setores"

SPLITS = [
    ("oficial_3y_1y", "2022-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("walk_2024", "2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("treino_curto", "2023-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("oos_longo", "2022-01-01", "2024-06-30", "2024-07-01", "2025-12-31"),
]


def recorte(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    t0 = pd.Timestamp(a, tz="UTC")
    t1 = pd.Timestamp(b, tz="UTC")
    out = df[(df["data"] >= t0) & (df["data"] <= t1)].copy()
    return out.dropna(subset=["spread_observado"]).reset_index(drop=True)


def simular(df: pd.DataFrame, y: str, x: str, hedge: float, coluna: str, taxa: float) -> dict:
    _, m = simular_ganhos(
        df, y, x, hedge, coluna, CAPITAL, taxa,
        execucao="abertura", dados_setores=DADOS_SETORES,
    )
    return {
        "pnl": float(m["pnl_liquido"]),
        "ret_pct": float(m["retorno_pct"]),
        "max_dd_pct": float(m["max_drawdown_pct"]),
        "pos_pct": float(m["dias_posicionado_pct"]),
        "sharpe_ref": float(m["sharpe_anualizado"]),
        "n_dias": int(len(df)),
        "n_pos_dias": int(round(m["dias_posicionado_pct"] / 100.0 * len(df))),
        "execucao": "abertura",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sufixo", required=True)
    p.add_argument("--entrada", type=Path, action="append", required=True)
    p.add_argument("--saida-dir", type=Path, default=Path("data/processed"))
    args = p.parse_args()

    linhas = []
    for path in args.entrada:
        if not path.exists():
            raise FileNotFoundError(path)
        df, y, x, hedge = carregar_pipeline(path)
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


if __name__ == "__main__":
    main()
