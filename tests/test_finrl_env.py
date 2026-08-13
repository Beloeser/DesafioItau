"""Testa o ambiente FinRL (Modulo 3): contrato Gym, recompensa em R$ e causalidade.

Roda com o Python do ambiente FinRL (precisa de gymnasium):
    .venv-finrl/bin/python tests/test_finrl_env.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

modulo = importlib.import_module("03_finrl_trading")
PairsTradingFinRLEnv = modulo.PairsTradingFinRLEnv


def base_sintetica(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    spread = np.cumsum(rng.normal(0, 0.5, n)) + 5.0
    return pd.DataFrame({
        "data": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
        "Y": 10.0 + spread,
        "X": np.full(n, 5.0),
        "spread_observado": spread,
        "zscore_mr": rng.normal(0, 1, n),
        "prob_quebra": rng.uniform(0, 1, n),
    })


def testa_contrato_gym() -> None:
    env = PairsTradingFinRLEnv(base_sintetica(), "Y", "X", 1.0, taxa=0.0, execucao="fechamento")
    obs, _ = env.reset()
    assert obs.shape == (4,)
    assert env.action_space.n == 3
    obs, recompensa, terminado, truncado, info = env.step(1)
    assert obs.shape == (4,) and isinstance(recompensa, float) and not terminado
    print("OK  contrato gym (reset/step/espacos)")


def testa_recompensa_igual_pnl() -> None:
    """Recompensa acumulada (sem custo) = PnL do avaliar_ganhos com mesmo sinal."""
    from avaliar_ganhos import simular_ganhos

    df = base_sintetica()
    capital = 100_000.0
    env = PairsTradingFinRLEnv(df, "Y", "X", 1.0, capital=capital, taxa=0.0, execucao="fechamento")

    rng = np.random.default_rng(11)
    acoes = rng.integers(0, 3, len(df) - 1)

    env.reset()
    total_env = 0.0
    posicoes = np.zeros(len(df), dtype=int)
    for i, acao in enumerate(acoes):
        posicoes[i] = PairsTradingFinRLEnv.ACAO_PARA_POSICAO[int(acao)]
        _, recompensa, terminado, _, _ = env.step(int(acao))
        total_env += recompensa / 100.0 * capital
        if terminado:
            break

    df_aval = df.copy()
    df_aval["sinal"] = posicoes
    _, m = simular_ganhos(df_aval, "Y", "X", 1.0, capital=capital, taxa=0.0, execucao="fechamento")
    assert abs(total_env - m["pnl_liquido"]) < 1e-6, (total_env, m["pnl_liquido"])
    print("OK  recompensa do env = PnL do avaliar_ganhos (mesma matematica)")


def testa_execucao_t_mais_1() -> None:
    """Acao no dia t so recebe o retorno de t -> t+1 (sem look-ahead)."""
    df = base_sintetica()
    env = PairsTradingFinRLEnv(df, "Y", "X", 1.0, taxa=0.0, execucao="fechamento")
    env.reset()
    _, recompensa, _, _, _ = env.step(1)  # long no dia 0
    dspread = df["spread_observado"].iloc[1] - df["spread_observado"].iloc[0]
    esperado = env.n_y * dspread / env.capital * 100.0
    assert abs(recompensa - esperado) < 1e-9
    print("OK  execucao t+1 dentro do env")


def testa_custo_no_treino() -> None:
    """Com taxa > 0, girar posicao custa; ficar flat nao custa nada."""
    df = base_sintetica()
    env = PairsTradingFinRLEnv(df, "Y", "X", 1.0, taxa=0.01, execucao="fechamento")
    env.reset()
    _, recompensa_flat, _, _, _ = env.step(0)
    assert recompensa_flat == 0.0

    env.reset()
    _, recompensa_long, _, _, _ = env.step(1)
    dspread = df["spread_observado"].iloc[1] - df["spread_observado"].iloc[0]
    custo = 0.01 * env.notional[0] * 1
    esperado = (env.n_y * dspread - custo) / env.capital * 100.0
    assert abs(recompensa_long - esperado) < 1e-9
    print("OK  custo de transacao cobrado na mudanca de posicao")


def testa_split_causal() -> None:
    """Treino e teste usam janelas distintas (formacao vs negociacao)."""
    from periodos import DATA_FIM_FORMACAO, DATA_INICIO_NEGOCIACAO

    df = base_sintetica(100)
    df["data"] = pd.date_range("2024-11-01", periods=100, freq="D", tz="UTC")
    treino, teste = modulo.split_formacao_negociacao(
        df,
        "2024-11-01",
        DATA_FIM_FORMACAO,
        DATA_INICIO_NEGOCIACAO,
        "2025-12-31",
    )
    assert treino["data"].max() < pd.Timestamp(DATA_INICIO_NEGOCIACAO, tz="UTC")
    assert teste["data"].min() >= pd.Timestamp(DATA_INICIO_NEGOCIACAO, tz="UTC")
    print("OK  split formacao/negociacao (mesmo recorte da SWANet)")


if __name__ == "__main__":
    testa_contrato_gym()
    testa_recompensa_igual_pnl()
    testa_execucao_t_mais_1()
    testa_custo_no_treino()
    testa_split_causal()
    print("\ntest_finrl_env: TODOS OS TESTES PASSARAM")
