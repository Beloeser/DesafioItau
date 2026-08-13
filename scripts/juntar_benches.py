#!/usr/bin/env python3
"""Junta todos os data/processed/bench_*.csv num unico arquivo para comparar."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PASTA = Path("data/processed")
SAIDA = PASTA / "bench_TODOS.csv"


def main() -> None:
    arquivos = sorted(PASTA.glob("bench_*.csv"))
    arquivos = [a for a in arquivos if a.name != "bench_TODOS.csv"]
    if not arquivos:
        raise SystemExit("Nenhum bench_*.csv em data/processed/")
    partes = [pd.read_csv(a) for a in arquivos]
    tab = pd.concat(partes, ignore_index=True)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(SAIDA, index=False)
    print(f"Juntou {len(arquivos)} arquivos -> {SAIDA} ({len(tab)} linhas)")
    print("Arquivos:", ", ".join(a.name for a in arquivos))
    cols = ["sufixo", "par", "split", "coluna_sinal", "pnl", "pos_pct", "max_dd_pct"]
    print(tab[cols].to_string(index=False))


if __name__ == "__main__":
    main()
