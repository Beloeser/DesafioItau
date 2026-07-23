"""Filtro de Kalman com transicao de estado baseada em Ornstein-Uhlenbeck.

Este arquivo contem apenas o filtro. Ele recebe:
- spread observado;
- theta previsto;
- mu previsto;
- sigma previsto.

A transicao usada e:
    x_t = mu_t + phi_t * (x_{t-1} - mu_t) + ruido
    phi_t = exp(-theta_t * dt)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COLUNAS_ENTRADA = [
    "spread_observado",
    "theta_previsto",
    "mu_previsto",
    "sigma_previsto",
]

COLUNAS_SAIDA = [
    "spread_filtrado",
    "incerteza_estado",
    "zscore_kalman",
]


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aplica somente o Filtro de Kalman OU em um CSV ja calibrado."
    )
    parser.add_argument(
        "--entrada",
        type=Path,
        required=True,
        help=(
            "CSV com spread_observado, theta_previsto, mu_previsto e "
            "sigma_previsto."
        ),
    )
    parser.add_argument(
        "--saida",
        type=Path,
        required=True,
        help="CSV de saida com as colunas filtradas adicionadas.",
    )
    parser.add_argument(
        "--custo-medicao",
        type=float,
        default=0.05,
        help="Variancia minima do ruido de medicao.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Passo temporal do modelo OU.",
    )
    return parser.parse_args()


def validar_entradas(spread_observado: pd.Series, parametros_ou: pd.DataFrame) -> None:
    colunas_faltantes = [coluna for coluna in ["theta", "mu", "sigma"] if coluna not in parametros_ou]
    if colunas_faltantes:
        raise ValueError(f"Parametros OU ausentes: {colunas_faltantes}")

    if len(spread_observado) != len(parametros_ou):
        raise ValueError("Spread observado e parametros OU precisam ter o mesmo tamanho.")

    if spread_observado.empty:
        raise ValueError("Spread observado esta vazio.")


def filtro_kalman_ou(
    spread_observado: pd.Series,
    parametros_ou: pd.DataFrame,
    custo_medicao: float = 0.05,
    dt: float = 1.0,
) -> pd.DataFrame:
    """Filtra o spread usando parametros OU dinamicos.

    A matriz de transicao e escalar e muda no tempo:
        F_t = exp(-theta_t * dt)

    O estado latente representa o spread limpo. A observacao e o spread bruto.
    """
    validar_entradas(spread_observado, parametros_ou)

    spread = pd.to_numeric(spread_observado, errors="coerce").ffill().bfill()
    parametros = parametros_ou[["theta", "mu", "sigma"]].apply(pd.to_numeric, errors="coerce")
    parametros = parametros.ffill().bfill()
    parametros["theta"] = parametros["theta"].clip(lower=1e-6)
    parametros["sigma"] = parametros["sigma"].clip(lower=1e-6)

    x_filtrado = np.zeros(len(spread))
    p_filtrado = np.zeros(len(spread))
    zscore = np.zeros(len(spread))

    x_anterior = float(spread.iloc[0])
    variancia_inicial = np.nanvar(spread.iloc[: min(50, len(spread))])
    p_anterior = float(variancia_inicial if variancia_inicial > 0 else 1.0)

    for i, observado in enumerate(spread.to_numpy()):
        theta = float(parametros["theta"].iloc[i])
        mu = float(parametros["mu"].iloc[i])
        sigma = float(parametros["sigma"].iloc[i])

        phi = float(np.exp(-theta * dt))
        variancia_processo = sigma ** 2 * (1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta)
        variancia_processo = max(float(variancia_processo), 1e-10)
        variancia_medicao = max(float(custo_medicao), variancia_processo * 0.25, 1e-10)

        x_predito = mu + phi * (x_anterior - mu)
        p_predito = phi ** 2 * p_anterior + variancia_processo

        ganho_kalman = p_predito / (p_predito + variancia_medicao)
        x_atual = x_predito + ganho_kalman * (float(observado) - x_predito)
        p_atual = (1.0 - ganho_kalman) * p_predito

        x_filtrado[i] = x_atual
        p_filtrado[i] = p_atual
        zscore[i] = (x_atual - mu) / max(np.sqrt(p_atual + variancia_processo), 1e-6)

        x_anterior = x_atual
        p_anterior = max(float(p_atual), 1e-10)

    return pd.DataFrame(
        {
            "spread_filtrado": x_filtrado,
            "incerteza_estado": p_filtrado,
            "zscore_kalman": zscore,
        },
        index=spread_observado.index,
    )


def filtrar_csv(entrada: Path, saida: Path, custo_medicao: float, dt: float) -> pd.DataFrame:
    dados = pd.read_csv(entrada)
    faltantes = [coluna for coluna in COLUNAS_ENTRADA if coluna not in dados]
    if faltantes:
        raise ValueError(f"Colunas ausentes no CSV de entrada: {faltantes}")

    parametros = dados[["theta_previsto", "mu_previsto", "sigma_previsto"]].rename(
        columns={
            "theta_previsto": "theta",
            "mu_previsto": "mu",
            "sigma_previsto": "sigma",
        }
    )
    filtrado = filtro_kalman_ou(
        dados["spread_observado"],
        parametros,
        custo_medicao=custo_medicao,
        dt=dt,
    )

    resultado = pd.concat(
        [dados.drop(columns=[coluna for coluna in COLUNAS_SAIDA if coluna in dados]), filtrado],
        axis=1,
    )
    saida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(saida, index=False)
    return resultado


def main() -> pd.DataFrame:
    args = argumentos()
    resultado = filtrar_csv(args.entrada, args.saida, args.custo_medicao, args.dt)
    print(resultado[COLUNAS_SAIDA].tail(10))
    print(f"\nFiltro salvo em {args.saida}")
    return resultado


if __name__ == "__main__":
    main()
