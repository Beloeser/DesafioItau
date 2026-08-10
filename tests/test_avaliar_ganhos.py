"""Testa a matematica de PnL do avaliar_ganhos (usada tambem pelo FinRL).

Roda com o Python base:
    python3 tests/test_avaliar_ganhos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from avaliar_ganhos import simular_ganhos  # noqa: E402


def base_sintetica() -> pd.DataFrame:
    """Par com hedge=1: spread sobe 1.0 por dia nos dias 2-4."""
    n = 6
    return pd.DataFrame({
        "data": pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC"),
        "Y": [10.0, 10.0, 11.0, 12.0, 13.0, 13.0],
        "X": [5.0] * n,
        "spread_observado": [5.0, 5.0, 6.0, 7.0, 8.0, 8.0],
        "sinal": [1, 1, 1, 0, 0, 0],
    })


def testa_execucao_t_mais_1() -> None:
    """Sinal de hoje so captura o retorno de AMANHA (sem look-ahead)."""
    df = base_sintetica()
    dados, m = simular_ganhos(df, "Y", "X", 1.0, capital=100.0, taxa=0.0)
    # n_y = 50/10 = 5 acoes. Posicao efetiva (shift): dias 1-3 long.
    # PnL = 5 * (dspread dia1 + dia2 + dia3) = 5 * (0 + 1 + 1) = 10.
    assert abs(m["pnl_liquido"] - 10.0) < 1e-9, m["pnl_liquido"]
    print("OK  execucao t+1 (PnL nao inclui o dia do sinal)")


def testa_lookahead_zero() -> None:
    """Sinal ligado SO no ultimo dia nao pode gerar PnL nenhum."""
    df = base_sintetica()
    df["sinal"] = [0, 0, 0, 0, 0, 1]
    _, m = simular_ganhos(df, "Y", "X", 1.0, capital=100.0, taxa=0.0)
    assert m["pnl_liquido"] == 0.0
    print("OK  sinal no ultimo dia gera PnL zero (sem vazamento)")


def testa_custos_por_mudanca() -> None:
    """Custo cobrado a cada mudanca de posicao (abre e fecha)."""
    df = base_sintetica()
    _, sem = simular_ganhos(df, "Y", "X", 1.0, capital=100.0, taxa=0.0)
    _, com = simular_ganhos(df, "Y", "X", 1.0, capital=100.0, taxa=0.01)
    # 2 mudancas (0->1 no dia 1; 1->0 no dia 4). Notional = 5*Y + 5*X.
    esperado = 0.01 * (5 * 10.0 + 5 * 5.0) + 0.01 * (5 * 13.0 + 5 * 5.0)
    assert abs((sem["pnl_liquido"] - com["pnl_liquido"]) - esperado) < 1e-9
    assert com["custos"] > 0
    print("OK  custos cobrados nas 2 pontas (abrir/fechar)")


def testa_short() -> None:
    """Posicao short lucra quando o spread cai."""
    df = base_sintetica()
    df["spread_observado"] = [8.0, 8.0, 7.0, 6.0, 5.0, 5.0]
    df["Y"] = [13.0, 13.0, 12.0, 11.0, 10.0, 10.0]
    df["sinal"] = [-1, -1, -1, 0, 0, 0]
    _, m = simular_ganhos(df, "Y", "X", 1.0, capital=100.0, taxa=0.0)
    # n_y = 50/13; PnL = -n_y * (0 -1 -1) = +2*n_y
    esperado = 2 * (50.0 / 13.0)
    assert abs(m["pnl_liquido"] - esperado) < 1e-9
    print("OK  short lucra com queda do spread")


def testa_trades_e_winrate() -> None:
    df = base_sintetica()
    _, m = simular_ganhos(df, "Y", "X", 1.0, capital=100.0, taxa=0.0)
    assert m["trades_fechados"] == 1
    assert m["win_rate_pct"] == 100.0
    print("OK  contagem de trades e win rate")


if __name__ == "__main__":
    testa_execucao_t_mais_1()
    testa_lookahead_zero()
    testa_custos_por_mudanca()
    testa_short()
    testa_trades_e_winrate()
    print("\ntest_avaliar_ganhos: TODOS OS TESTES PASSARAM")
