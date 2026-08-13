"""Testa execucao na abertura (exemplo 100 -> 103 -> 105)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from avaliar_ganhos import simular_ganhos  # noqa: E402
from execucao_pnl import pnl_transicao_abertura  # noqa: E402


def testa_exemplo_usuario() -> None:
    """Segunda close 100, terca open 103 close 105: lucro ~2, nao 5."""
    df = pd.DataFrame({
        "data": pd.date_range("2025-06-02", periods=2, freq="B", tz="UTC"),
        "Y": [100.0, 105.0],
        "X": [0.0, 0.0],
        "spread_observado": [100.0, 105.0],
        "spread_abertura": [100.0, 103.0],
        "sinal": [1, 0],
    })
    _, m_ab = simular_ganhos(df, "Y", "X", 0.0, capital=100.0, taxa=0.0, execucao="abertura")
    _, m_fc = simular_ganhos(df, "Y", "X", 0.0, capital=100.0, taxa=0.0, execucao="fechamento")
    # n_y = 50/100 = 0.5; abertura: pos[1]=1, pnl = 0.5*(105-103)=1
    assert abs(m_ab["pnl_liquido"] - 1.0) < 1e-9, m_ab["pnl_liquido"]
    # fechamento: pos[1]=1, pnl = 0.5*(105-100)=2.5
    assert abs(m_fc["pnl_liquido"] - 2.5) < 1e-9, m_fc["pnl_liquido"]
    assert m_fc["pnl_liquido"] > m_ab["pnl_liquido"]
    print("OK  abertura lucra menos que fechamento no gap overnight")


def testa_transicao_entrada() -> None:
    sc = np.array([100.0, 105.0])
    so = np.array([100.0, 103.0])
    pnl = pnl_transicao_abertura(1, 0, 1, sc, so, n_y=1.0)
    assert abs(pnl - 2.0) < 1e-9
    print("OK  pnl_transicao entrada open->close")


if __name__ == "__main__":
    testa_exemplo_usuario()
    testa_transicao_entrada()
    print("\ntest_execucao_abertura: TODOS OS TESTES PASSARAM")
