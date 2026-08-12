#!/usr/bin/env python3
"""Compara PnL fechamento vs abertura (Luiz + FinRL se existir)."""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from avaliar_ganhos import carregar_pipeline, simular_ganhos  # noqa: E402
from periodos import DATA_FIM_NEGOCIACAO, DATA_INICIO_NEGOCIACAO  # noqa: E402


def main() -> None:
    entrada = Path("data/processed/pipeline_com_quebras_TAEE_causal.csv")
    if not entrada.exists():
        entrada = Path("data/processed/pipeline_com_quebras_TAEE.csv")
    df, y, x, h = carregar_pipeline(entrada)
    t0 = __import__("pandas").Timestamp(DATA_INICIO_NEGOCIACAO, tz="UTC")
    t1 = __import__("pandas").Timestamp(DATA_FIM_NEGOCIACAO, tz="UTC")
    oos = df[(df["data"] >= t0) & (df["data"] <= t1)].copy()
    colunas = ["sinal"]
    if "sinal_finrl" in oos.columns:
        colunas.append("sinal_finrl")
    if "sinal_hibrido" in oos.columns:
        colunas.append("sinal_hibrido")

    print(f"Comparacao execucao | {y}/{x} | OOS {t0.date()}..{t1.date()} | 8 bps")
    print("-" * 72)
    for col in colunas:
        for modo in ("fechamento", "abertura"):
            _, m = simular_ganhos(
                oos, y, x, h, col, capital=100_000.0, taxa=0.0008, execucao=modo
            )
            print(
                f"{col:14s} {modo:12s}  PnL R$ {m['pnl_liquido']:9,.0f}  "
                f"pos {m['dias_posicionado_pct']:5.1f}%  DD {m['max_drawdown_pct']:6.2f}%"
            )


if __name__ == "__main__":
    main()
