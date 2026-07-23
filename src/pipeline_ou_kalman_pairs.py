"""Pipeline principal para Pairs Trading com ML, OU e Filtro de Kalman.

Arquitetura:
1. Le um par cointegrado ja escolhido pelo arquivo de Engle-Granger.
2. Constroi o spread: spread_t = Y_t - hedge_ratio * X_t.
3. Estima parametros OU historicos em janelas moveis:
       dS_t = theta * (mu - S_t) * dt + sigma * dW_t
4. Treina um Random Forest para prever theta, mu e sigma a partir de
   features historicas do spread e da volatilidade.
5. Injeta theta, mu e sigma previstos na transicao de estado do Kalman:
       x_t = mu_t + phi_t * (x_{t-1} - mu_t) + ruido
       phi_t = exp(-theta_t * dt)
6. Salva o spread observado, estado filtrado, parametros OU dinamicos,
   z-score e sinal operacional.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from filtro_kalman_ou import filtro_kalman_ou


COLUNAS_RESULTADO = [
    "data",
    "setor",
    "ativo_y",
    "ativo_x",
    "spread_observado",
    "spread_filtrado",
    "theta_previsto",
    "mu_previsto",
    "sigma_previsto",
    "zscore_kalman",
    "sinal",
]


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa prototipo ML + OU + Kalman para um par de pairs trading."
    )
    parser.add_argument(
        "--cointegracao",
        type=Path,
        default=Path("data/processed/cointegracao_engle_granger.csv"),
        help="CSV com pares cointegrados gerado pelo Engle-Granger.",
    )
    parser.add_argument(
        "--dados-setores",
        type=Path,
        default=Path("data/raw/setores"),
        help="Diretorio com os CSVs brutos por setor.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("data/processed/pipeline_ou_kalman_pairs.csv"),
        help="Arquivo CSV de saida do pipeline.",
    )
    parser.add_argument("--setor", default=None, help="Setor do par escolhido.")
    parser.add_argument("--ativo-y", default=None, help="Ativo dependente Y.")
    parser.add_argument("--ativo-x", default=None, help="Ativo explicativo X.")
    parser.add_argument(
        "--janela-ou",
        type=int,
        default=120,
        help="Janela movel para estimar alvos OU historicos.",
    )
    parser.add_argument(
        "--janela-feature",
        type=int,
        default=24,
        help="Janela usada nas features de volatilidade e z-score.",
    )
    parser.add_argument(
        "--entrada-zscore",
        type=float,
        default=2.0,
        help="Limiar absoluto para abrir posicao.",
    )
    parser.add_argument(
        "--saida-zscore",
        type=float,
        default=0.5,
        help="Limiar absoluto para encerrar/zerar posicao.",
    )
    parser.add_argument(
        "--custo-medicao",
        type=float,
        default=0.05,
        help="Variancia minima do ruido de medicao do Kalman.",
    )
    return parser.parse_args()


def escolher_par(caminho: Path, setor: str | None, ativo_y: str | None, ativo_x: str | None) -> pd.Series:
    pares = pd.read_csv(caminho)
    if pares.empty:
        raise ValueError(f"Nenhum par cointegrado encontrado em {caminho}")

    filtro = pd.Series(True, index=pares.index)
    if setor:
        filtro &= pares["Setor"].eq(setor)
    if ativo_y:
        filtro &= pares["Ativo Y"].eq(ativo_y)
    if ativo_x:
        filtro &= pares["Ativo X"].eq(ativo_x)

    candidatos = pares[filtro].sort_values("P-Value")
    if candidatos.empty:
        raise ValueError("Nenhum par encontrado com os filtros informados.")

    return candidatos.iloc[0]


def carregar_spread(
    dados_setores: Path,
    setor: str,
    ativo_y: str,
    ativo_x: str,
    hedge_ratio: float,
) -> pd.DataFrame:
    caminho = dados_setores / f"{setor}.csv"
    dados = pd.read_csv(caminho, usecols=["data", "ticker", "fechamento"])
    dados["data"] = pd.to_datetime(dados["data"], utc=True)
    dados["fechamento"] = pd.to_numeric(dados["fechamento"], errors="coerce")

    precos = dados.pivot_table(
        index="data",
        columns="ticker",
        values="fechamento",
        aggfunc="last",
    ).sort_index()

    serie = precos[[ativo_y, ativo_x]].dropna().copy()
    serie["spread"] = serie[ativo_y] - hedge_ratio * serie[ativo_x]
    return serie.reset_index()[["data", ativo_y, ativo_x, "spread"]]


def estimar_ou_janela(spread: pd.Series, janela: int, dt: float = 1.0) -> pd.DataFrame:
    """Estima theta, mu e sigma por AR(1) em janelas moveis do spread."""
    resultados = pd.DataFrame(index=spread.index, columns=["theta", "mu", "sigma"], dtype=float)

    for fim in range(janela, len(spread)):
        amostra = spread.iloc[fim - janela:fim].dropna()
        if len(amostra) < janela * 0.8:
            continue

        y = amostra.iloc[1:].to_numpy()
        x = amostra.iloc[:-1].to_numpy()
        matriz = np.column_stack([np.ones_like(x), x])

        try:
            intercepto, phi = np.linalg.lstsq(matriz, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        phi = float(np.clip(phi, 1e-6, 0.999999))
        theta = -np.log(phi) / dt
        mu = intercepto / (1.0 - phi)
        residuos = y - (intercepto + phi * x)
        sigma_eps = float(np.std(residuos, ddof=1))
        sigma = sigma_eps * np.sqrt((2.0 * theta) / (1.0 - phi ** 2))

        if np.isfinite(theta) and np.isfinite(mu) and np.isfinite(sigma):
            resultados.loc[spread.index[fim], ["theta", "mu", "sigma"]] = [
                max(theta, 1e-6),
                mu,
                max(sigma, 1e-6),
            ]

    return resultados


def montar_features(spread: pd.Series, janela_feature: int) -> pd.DataFrame:
    retorno = spread.diff()
    media = spread.rolling(janela_feature).mean()
    desvio = spread.rolling(janela_feature).std()

    features = pd.DataFrame(index=spread.index)
    features["spread"] = spread
    features["spread_lag_1"] = spread.shift(1)
    features["spread_lag_2"] = spread.shift(2)
    features["retorno_spread"] = retorno
    features["volatilidade"] = retorno.rolling(janela_feature).std()
    features["media_movel"] = media
    features["zscore"] = (spread - media) / desvio.replace(0, np.nan)
    features["momento_6"] = spread - spread.shift(6)
    features["momento_24"] = spread - spread.shift(24)
    return features


def treinar_calibrador_ml(
    features: pd.DataFrame,
    alvos_ou: pd.DataFrame,
) -> tuple[RandomForestRegressor, pd.DataFrame, dict[str, float]]:
    base = features.join(alvos_ou).replace([np.inf, -np.inf], np.nan).dropna()
    if len(base) < 200:
        raise ValueError("Amostra insuficiente para treinar o calibrador ML.")

    colunas_x = list(features.columns)
    colunas_y = ["theta", "mu", "sigma"]
    x = base[colunas_x]
    y = base[colunas_y]

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x,
        y,
        test_size=0.2,
        shuffle=False,
    )

    modelo = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(x_treino, y_treino)

    previsao_teste = pd.DataFrame(
        modelo.predict(x_teste),
        columns=colunas_y,
        index=x_teste.index,
    )
    metricas = {
        f"mae_{coluna}": mean_absolute_error(y_teste[coluna], previsao_teste[coluna])
        for coluna in colunas_y
    }

    parametros = pd.DataFrame(
        modelo.predict(features[colunas_x].replace([np.inf, -np.inf], np.nan).ffill().bfill()),
        columns=colunas_y,
        index=features.index,
    )
    parametros["theta"] = parametros["theta"].clip(lower=1e-6)
    parametros["sigma"] = parametros["sigma"].clip(lower=1e-6)
    return modelo, parametros, metricas


def gerar_sinais(zscore: pd.Series, entrada: float, saida: float) -> pd.Series:
    sinais = []
    posicao = 0

    for valor in zscore:
        if posicao == 0:
            if valor >= entrada:
                posicao = -1
            elif valor <= -entrada:
                posicao = 1
        elif posicao == 1 and valor >= -saida:
            posicao = 0
        elif posicao == -1 and valor <= saida:
            posicao = 0

        sinais.append(posicao)

    return pd.Series(sinais, index=zscore.index, name="sinal")


def executar_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    par = escolher_par(args.cointegracao, args.setor, args.ativo_y, args.ativo_x)
    setor = str(par["Setor"])
    ativo_y = str(par["Ativo Y"])
    ativo_x = str(par["Ativo X"])
    hedge_ratio = float(par["Hedge Ratio"])

    dados = carregar_spread(args.dados_setores, setor, ativo_y, ativo_x, hedge_ratio)
    spread = dados["spread"]

    alvos_ou = estimar_ou_janela(spread, args.janela_ou)
    features = montar_features(spread, args.janela_feature)
    _, parametros_ou, metricas = treinar_calibrador_ml(features, alvos_ou)
    kalman = filtro_kalman_ou(spread, parametros_ou, args.custo_medicao)
    sinais = gerar_sinais(kalman["zscore_kalman"], args.entrada_zscore, args.saida_zscore)

    resultado = pd.DataFrame(
        {
            "data": dados["data"],
            "setor": setor,
            "ativo_y": ativo_y,
            "ativo_x": ativo_x,
            "spread_observado": spread,
            "spread_filtrado": kalman["spread_filtrado"],
            "theta_previsto": parametros_ou["theta"],
            "mu_previsto": parametros_ou["mu"],
            "sigma_previsto": parametros_ou["sigma"],
            "zscore_kalman": kalman["zscore_kalman"],
            "sinal": sinais,
        }
    )

    print(
        f"Par escolhido: {ativo_y} x {ativo_x} | setor={setor} | "
        f"hedge_ratio={hedge_ratio:.6f}"
    )
    print(
        "MAE do calibrador ML: "
        + ", ".join(f"{nome}={valor:.6f}" for nome, valor in metricas.items())
    )

    return resultado[COLUNAS_RESULTADO]


def main() -> pd.DataFrame:
    args = argumentos()
    resultado = executar_pipeline(args)

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(args.saida, index=False)

    print("\nAmostra final do pipeline:")
    print(resultado.tail(10))
    print(f"\nResultado salvo em {args.saida}")
    return resultado


if __name__ == "__main__":
    main()
