"""Pipeline Definitivo: Pairs Trading com Cointegracao Parcial e Kalman.

Baseado no artigo "Kalman Filtering Applied to Investment Portfolio Management".
Substitui a estimativa de Ornstein-Uhlenbeck por solucoes analiticas
(Equacoes 19 a 23 do artigo) para obter os parametros dinamicamente.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

MPLCONFIGDIR = Path("data/processed/.matplotlib").resolve()
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline Integrado de Cointegracao Parcial e Filtro de Kalman."
    )
    parser.add_argument(
        "--cointegracao",
        type=Path,
        default=Path("data/processed/cointegracao_engle_granger.csv"),
        help="CSV com pares cointegrados (saida do Engle-Granger).",
    )
    parser.add_argument(
        "--dados-setores",
        type=Path,
        default=Path("data/raw/setores"),
        help="Diretorio com os CSVs brutos de precos por setor.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("data/processed/pipeline_cointegracao_parcial.csv"),
        help="Arquivo CSV de saida do pipeline final.",
    )
    parser.add_argument(
        "--grafico",
        type=Path,
        default=Path("data/processed/grafico_cointegracao_parcial.png"),
        help="Arquivo PNG de saida do grafico operacional.",
    )
    parser.add_argument("--setor", default=None, help="Filtrar por setor especifico.")
    parser.add_argument("--ativo-y", default=None, help="Ativo dependente Y.")
    parser.add_argument("--ativo-x", default=None, help="Ativo explicativo X.")
    parser.add_argument(
        "--janela-parametros",
        type=int,
        default=120,
        help="Janela movel para calcular rho, var_mr e var_rw dinamicamente.",
    )
    parser.add_argument(
        "--limiar-entrada",
        type=float,
        default=1.25,
        help="Z-score do componente Mean-Reverting para abrir posicao.",
    )
    return parser.parse_args()


def escolher_par(
    caminho: Path,
    setor: str | None,
    ativo_y: str | None,
    ativo_x: str | None,
) -> pd.Series:
    """Carrega o resumo do Engle-Granger e seleciona o par de interesse."""
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {caminho}. Execute o script de cointegracao primeiro."
        )

    pares = pd.read_csv(caminho)
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
    """Extrai os precos brutos e calcula o spread observado historico."""
    caminho = dados_setores / f"{setor}.csv"
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de setor nao encontrado: {caminho}")

    dados = pd.read_csv(caminho, usecols=["data", "ticker", "fechamento"])
    dados["data"] = pd.to_datetime(dados["data"], utc=True)
    dados["fechamento"] = pd.to_numeric(dados["fechamento"], errors="coerce")

    precos = dados.pivot_table(
        index="data", columns="ticker", values="fechamento", aggfunc="last"
    ).sort_index()

    ativos_ausentes = [ticker for ticker in (ativo_y, ativo_x) if ticker not in precos.columns]
    if ativos_ausentes:
        raise ValueError(
            "Ativo(s) ausente(s) nos dados do setor "
            f"{setor}: {', '.join(ativos_ausentes)}"
        )

    serie = precos[[ativo_y, ativo_x]].dropna().copy()
    if serie.empty:
        raise ValueError(f"Nenhuma data comum encontrada para {ativo_y} e {ativo_x}.")

    serie["spread_observado"] = serie[ativo_y] - hedge_ratio * serie[ativo_x]
    return serie.reset_index()[["data", ativo_y, ativo_x, "spread_observado"]]


def calcular_parametros_dinamicos(spread: pd.Series, janela: int) -> pd.DataFrame:
    """
    Calcula rho, var_mr e var_rw utilizando a variancia das diferencas do spread.
    Implementa as Equacoes 19 a 23 do artigo.
    """
    if janela < 4:
        raise ValueError("--janela-parametros deve ser maior ou igual a 4.")
    if len(spread) <= janela:
        raise ValueError(
            f"Serie com {len(spread)} observacoes; a janela {janela} e grande demais."
        )

    v1 = spread.diff(1).rolling(window=janela).var()
    v2 = spread.diff(2).rolling(window=janela).var()
    v3 = spread.diff(3).rolling(window=janela).var()

    denom_rho = (2 * v1 - v2).replace(0, np.nan)
    rho = -(v1 - 2 * v2 + v3) / denom_rho
    rho = rho.clip(-0.999, 0.999)

    var_mr = 0.5 * ((rho + 1) / (rho - 1)) * (v2 - 2 * v1)
    var_mr = var_mr.clip(lower=1e-6)

    var_rw = 0.5 * (v2 - 2 * var_mr)
    var_rw = var_rw.clip(lower=1e-6)

    parametros = pd.DataFrame({"rho": rho, "var_mr": var_mr, "var_rw": var_rw})
    parametros = parametros.bfill().ffill()
    if parametros.isna().any().any():
        raise ValueError("Nao foi possivel calcular parametros dinamicos validos.")

    return parametros


def filtro_kalman_cointegracao_parcial(
    spread: pd.Series,
    parametros: pd.DataFrame,
    var_medicao: float = 1e-4,
) -> pd.DataFrame:
    """Aplica o filtro de Kalman separando os estados ocultos."""
    n = len(spread)

    mr_filtrado = np.zeros(n)
    rw_filtrado = np.zeros(n)
    std_mr = np.zeros(n)

    h = np.array([[1.0, 1.0]])
    r = np.array([[var_medicao]])

    z_hat = np.array([[0.0], [spread.iloc[0]]])
    p = np.eye(2)

    for i in range(n):
        y = spread.iloc[i]
        rho = parametros["rho"].iloc[i]
        var_mr = parametros["var_mr"].iloc[i]
        var_rw = parametros["var_rw"].iloc[i]

        f = np.array([[rho, 0.0], [0.0, 1.0]])
        q = np.array([[var_mr, 0.0], [0.0, var_rw]])

        z_pred = f @ z_hat
        p_pred = f @ p @ f.T + q

        y_residual = y - (h @ z_pred)[0, 0]
        s = h @ p_pred @ h.T + r
        k = p_pred @ h.T @ np.linalg.inv(s)

        z_hat = z_pred + k * y_residual
        p = (np.eye(2) - k @ h) @ p_pred

        mr_filtrado[i] = z_hat[0, 0]
        rw_filtrado[i] = z_hat[1, 0]
        std_mr[i] = np.sqrt(p[0, 0] + var_mr)

    return pd.DataFrame(
        {
            "mr_filtrado": mr_filtrado,
            "rw_filtrado": rw_filtrado,
            "std_mr": std_mr,
        },
        index=spread.index,
    )


def gerar_sinais_artigo(
    mr_filtrado: pd.Series,
    std_mr: pd.Series,
    limiar_entrada: float,
) -> pd.DataFrame:
    """Gera sinais de trading pelo cruzamento do Z-Score do componente MR."""
    zscore_mr = mr_filtrado / std_mr.replace(0, np.nan)
    sinais = []
    posicao = 0

    for mr, zscore in zip(mr_filtrado, zscore_mr):
        if posicao == 0:
            if zscore <= -limiar_entrada:
                posicao = 1
            elif zscore >= limiar_entrada:
                posicao = -1
        elif posicao == 1 and mr >= 0:
            posicao = 0
        elif posicao == -1 and mr <= 0:
            posicao = 0

        sinais.append(posicao)

    return pd.DataFrame({"zscore_mr": zscore_mr, "sinal": sinais}, index=mr_filtrado.index)


def salvar_grafico_operacional(
    df_resultado: pd.DataFrame,
    limiar: float,
    caminho_saida: Path,
) -> None:
    """Salva um grafico do componente MR, bandas dinamicas e sinais de trade."""
    df_plot = df_resultado.dropna(subset=["mr_filtrado"]).copy()
    if df_plot.empty:
        raise ValueError("Nao ha dados validos para gerar o grafico.")

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True,
    )

    ax1.plot(
        df_plot["data"],
        df_plot["spread_observado"],
        label="Spread Total (Observado)",
        color="lightgray",
        alpha=0.7,
    )
    ax1.plot(
        df_plot["data"],
        df_plot["rw_filtrado"],
        label="Passeio Aleatorio (RW)",
        color="orange",
        linestyle="--",
        alpha=0.8,
    )
    ax1.plot(
        df_plot["data"],
        df_plot["mr_filtrado"],
        label="Reversao a Media (MR)",
        color="blue",
        linewidth=1.5,
    )

    banda_superior = limiar * df_plot["std_mr"]
    banda_inferior = -limiar * df_plot["std_mr"]
    ax1.plot(
        df_plot["data"],
        banda_superior,
        color="red",
        linestyle=":",
        label=f"Banda Superior (+{limiar} std)",
    )
    ax1.plot(
        df_plot["data"],
        banda_inferior,
        color="green",
        linestyle=":",
        label=f"Banda Inferior (-{limiar} std)",
    )
    ax1.axhline(0, color="black", linewidth=1, linestyle="-")

    entradas_long = df_plot[(df_plot["sinal"] == 1) & (df_plot["sinal"].shift(1) == 0)]
    entradas_short = df_plot[(df_plot["sinal"] == -1) & (df_plot["sinal"].shift(1) == 0)]
    saidas = df_plot[(df_plot["sinal"] == 0) & (df_plot["sinal"].shift(1).isin([1, -1]))]

    ax1.scatter(
        entradas_long["data"],
        entradas_long["mr_filtrado"],
        marker="^",
        color="green",
        s=100,
        label="Entrada Long",
        zorder=5,
    )
    ax1.scatter(
        entradas_short["data"],
        entradas_short["mr_filtrado"],
        marker="v",
        color="red",
        s=100,
        label="Entrada Short",
        zorder=5,
    )
    ax1.scatter(
        saidas["data"],
        saidas["mr_filtrado"],
        marker="x",
        color="black",
        s=80,
        label="Saida (Zero-Crossing)",
        zorder=5,
    )

    ax1.set_title("Filtro de Kalman: Componentes do Spread e Sinais de Operacao")
    ax1.set_ylabel("Valor do Spread")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.plot(df_plot["data"], df_plot["zscore_mr"], color="purple", label="Z-Score MR")
    ax2.axhline(limiar, color="red", linestyle="--")
    ax2.axhline(-limiar, color="green", linestyle="--")
    ax2.axhline(0, color="black", linestyle="-")
    ax2.set_title("Dinamica do Z-Score")
    ax2.set_ylabel("Z-Score")
    ax2.set_xlabel("Data")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def main() -> None:
    args = argumentos()

    print("Buscando melhor par cointegrado...")
    par = escolher_par(args.cointegracao, args.setor, args.ativo_y, args.ativo_x)

    df_dados = carregar_spread(
        args.dados_setores,
        str(par["Setor"]),
        str(par["Ativo Y"]),
        str(par["Ativo X"]),
        float(par["Hedge Ratio"]),
    )
    spread = df_dados["spread_observado"]

    print(f"Par selecionado: {par['Ativo Y']} e {par['Ativo X']} (Setor: {par['Setor']})")
    print(
        "Calculando parametros matematicos "
        f"(rho, variancias) em janela de {args.janela_parametros} observacoes..."
    )
    parametros_dinamicos = calcular_parametros_dinamicos(spread, args.janela_parametros)

    print("Aplicando Filtro de Kalman (separando MR e RW)...")
    df_filtrado = filtro_kalman_cointegracao_parcial(spread, parametros_dinamicos)

    print(f"Gerando sinais (limiar = {args.limiar_entrada} desvios do componente MR)...")
    df_sinais = gerar_sinais_artigo(
        df_filtrado["mr_filtrado"],
        df_filtrado["std_mr"],
        args.limiar_entrada,
    )

    df_resultado = pd.concat([df_dados, parametros_dinamicos, df_filtrado, df_sinais], axis=1)

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    df_resultado.to_csv(args.saida, index=False)

    print("\n--- Amostra do Resultado Final ---")
    colunas_exibicao = [
        "data",
        "spread_observado",
        "rho",
        "mr_filtrado",
        "rw_filtrado",
        "zscore_mr",
        "sinal",
    ]
    print(df_resultado[colunas_exibicao].tail(10))
    print(f"\nCSV salvo com sucesso em: {args.saida}")

    print("\nGerando grafico operacional...")
    salvar_grafico_operacional(df_resultado, args.limiar_entrada, args.grafico)
    print(f"Grafico salvo com sucesso em: {args.grafico}")


if __name__ == "__main__":
    main()
