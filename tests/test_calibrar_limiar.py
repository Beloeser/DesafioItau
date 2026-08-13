"""Testa calibracao do limiar Luiz (substituto do FinRL)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

mod = importlib.import_module("03_calibrar_limiar")
gerar_sinal_com_limiar = mod.gerar_sinal_com_limiar
grid_search_limiar = mod.grid_search_limiar
score_formacao = mod.score_formacao


def base_mr() -> pd.DataFrame:
    n = 8
    mr = pd.Series([0.0, -0.2, -0.5, -0.1, 0.3, 0.5, 0.1, -0.4])
    std = pd.Series([0.1] * n)
    return pd.DataFrame({
        "data": pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC"),
        "Y": [10.0] * n,
        "X": [5.0] * n,
        "spread_observado": [5.0, 4.8, 4.5, 4.9, 5.3, 5.5, 5.1, 4.6],
        "spread_abertura": [5.0, 4.85, 4.55, 4.88, 5.28, 5.48, 5.08, 4.58],
        "mr_filtrado": mr,
        "std_mr": std,
    })


def testa_limiar_alto_entra_menos() -> None:
    df = base_mr()
    s_baixo = gerar_sinal_com_limiar(df, 0.5)
    s_alto = gerar_sinal_com_limiar(df, 2.0)
    assert (s_baixo != 0).sum() >= (s_alto != 0).sum()
    print("OK  limiar maior -> menos dias posicionados")


def testa_score_penaliza_poucos_trades() -> None:
    assert score_formacao({"pnl_liquido": 1000, "max_drawdown_pct": -1, "trades_fechados": 1}, 3) == float("-inf")
    sc = score_formacao({"pnl_liquido": 1000, "max_drawdown_pct": -2, "trades_fechados": 5}, 3)
    assert sc == 500.0
    print("OK  score Calmar na formacao")


def testa_grid_escolhe_limiar() -> None:
    df = base_mr()
    melhor, tab = grid_search_limiar(
        df, "Y", "X", 1.0,
        limiares=[0.5, 1.0, 2.5],
        capital=100.0,
        taxa=0.0,
        execucao="fechamento",
        dados_setores=None,
        min_trades=1,
    )
    assert melhor in [0.5, 1.0, 2.5]
    assert len(tab) == 3
    print(f"OK  grid search retornou limiar={melhor}")


if __name__ == "__main__":
    testa_limiar_alto_entra_menos()
    testa_score_penaliza_poucos_trades()
    testa_grid_escolhe_limiar()
    print("\ntest_calibrar_limiar: TODOS OS TESTES PASSARAM")
