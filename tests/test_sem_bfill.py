"""Testa a remocao do bfill no pipeline de cointegracao parcial.

Roda com o Python base:
    python3 tests/test_sem_bfill.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from pipeline_cointegracao_parcial import (  # noqa: E402
    calcular_parametros_dinamicos,
    filtro_kalman_cointegracao_parcial,
    gerar_sinais_artigo,
)


def serie_sintetica(n: int = 400, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    mr = np.zeros(n)
    for i in range(1, n):
        mr[i] = 0.7 * mr[i - 1] + rng.normal(0, 0.5)
    rw = np.cumsum(rng.normal(0, 0.2, n))
    return pd.Series(mr + rw)


def testa_aquecimento_sem_futuro() -> None:
    """As primeiras barras NAO podem receber parametros do futuro (bfill)."""
    spread = serie_sintetica()
    janela = 120
    parametros = calcular_parametros_dinamicos(spread, janela)

    # diff(3).rolling(120) so fecha a primeira janela na barra 122.
    assert parametros["rho"].iloc[: janela - 1].isna().all(), (
        "Aquecimento deveria ser NaN (sem bfill copiando o futuro)."
    )
    assert parametros["rho"].iloc[janela + 5 :].notna().all(), (
        "Apos a primeira janela completa, ffill deve preencher tudo."
    )
    print("OK  aquecimento sem futuro (bfill removido)")


def testa_ffill_e_causal() -> None:
    """Mudar o FUTURO da serie nao pode mudar os parametros do passado."""
    spread = serie_sintetica()
    janela = 120
    base = calcular_parametros_dinamicos(spread, janela)

    alterada = spread.copy()
    alterada.iloc[300:] += 50.0  # choque so no futuro
    depois = calcular_parametros_dinamicos(alterada, janela)

    pd.testing.assert_frame_equal(base.iloc[:290], depois.iloc[:290])
    print("OK  parametros do passado nao mudam com choque no futuro")


def testa_kalman_finito_com_nan() -> None:
    """O filtro precisa produzir saida finita mesmo com aquecimento NaN."""
    spread = serie_sintetica()
    parametros = calcular_parametros_dinamicos(spread, 120)
    filtrado = filtro_kalman_cointegracao_parcial(spread, parametros)

    for coluna in ("mr_filtrado", "rw_filtrado", "std_mr"):
        assert np.isfinite(filtrado[coluna].to_numpy()).all(), (
            f"{coluna} tem valores nao finitos com aquecimento NaN."
        )
    assert (filtrado["std_mr"] > 0).all()
    print("OK  Kalman finito com aquecimento NaN")


def testa_sinais_validos() -> None:
    """Sinais devem ser apenas -1, 0, +1 e reagir ao z-score."""
    spread = serie_sintetica()
    parametros = calcular_parametros_dinamicos(spread, 120)
    filtrado = filtro_kalman_cointegracao_parcial(spread, parametros)
    sinais = gerar_sinais_artigo(filtrado["mr_filtrado"], filtrado["std_mr"], 1.25)

    assert set(sinais["sinal"].unique()).issubset({-1, 0, 1})
    assert (sinais["sinal"] != 0).any(), "Nenhum sinal gerado na serie sintetica."
    print("OK  sinais validos (-1/0/+1) e nao degenerados")


if __name__ == "__main__":
    testa_aquecimento_sem_futuro()
    testa_ffill_e_causal()
    testa_kalman_finito_com_nan()
    testa_sinais_validos()
    print("\ntest_sem_bfill: TODOS OS TESTES PASSARAM")
