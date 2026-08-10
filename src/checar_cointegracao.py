"""Testa cointegracao por Engle-Granger entre pares de ativos por setor.

Roteiro analitico:
1. Para cada setor, transforma o CSV longo em uma matriz data x ticker.
2. Para cada par ordenado (Ativo Y, Ativo X), estima por OLS:
       Y_t = alpha + beta * X_t + erro_t
   O spread residual e o erro_t. O hedge ratio e o beta.
3. Aplica o teste ADF nos residuos. Se os residuos forem estacionarios,
   o spread tem evidencia estatistica de reversao a media.
4. Mantem apenas pares com p-value menor que 0.05.
"""

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from periodos import DATA_FIM_FORMACAO, DATA_INICIO_FORMACAO


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Checa cointegracao Engle-Granger em todos os pares de ativos "
            "dentro de cada setor."
        )
    )
    parser.add_argument(
        "--entrada",
        type=Path,
        default=Path("data/raw/setores"),
        help="Diretorio com os CSVs por setor (padrao: data/raw/setores).",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("data/processed/cointegracao_engle_granger.csv"),
        help="CSV final com pares cointegrados.",
    )
    parser.add_argument(
        "--coluna-preco",
        default="fechamento",
        help="Coluna de preco usada no teste (padrao: fechamento).",
    )
    parser.add_argument(
        "--p-value-maximo",
        type=float,
        default=0.05,
        help="P-value maximo aceito no ADF dos residuos (padrao: 0.05).",
    )
    parser.add_argument(
        "--min-observacoes",
        type=int,
        default=252,
        help="Minimo de observacoes alinhadas por par (padrao: 252).",
    )
    parser.add_argument(
        "--setor",
        action="append",
        help=(
            "Nome de um setor especifico para testar. Pode repetir. "
            "Por padrao, testa todos os CSVs do diretorio."
        ),
    )
    parser.add_argument(
        "--salvar-todos",
        action="store_true",
        help="Tambem salva pares reprovados em um arquivo *_todos.csv.",
    )
    parser.add_argument(
        "--inicio-formacao",
        type=str,
        default=DATA_INICIO_FORMACAO,
        help="Inicio do periodo usado no teste de cointegracao.",
    )
    parser.add_argument(
        "--fim-formacao",
        type=str,
        default=DATA_FIM_FORMACAO,
        help="Fim do periodo usado no teste de cointegracao.",
    )
    return parser.parse_args()


def arquivos_setores(diretorio: Path, setores: list[str] | None) -> list[Path]:
    if setores:
        return [diretorio / f"{setor}.csv" for setor in setores]

    return sorted(
        caminho
        for caminho in diretorio.glob("*.csv")
        if caminho.name != "resumo_coleta.csv"
    )


def carregar_precos(
    caminho: Path,
    coluna_preco: str,
    data_inicio: str | None,
    data_fim: str | None,
) -> pd.DataFrame:
    dados = pd.read_csv(caminho, usecols=["data", "ticker", coluna_preco])
    dados["data"] = pd.to_datetime(dados["data"], utc=True)
    dados[coluna_preco] = pd.to_numeric(dados[coluna_preco], errors="coerce")

    if data_inicio is not None:
        inicio = pd.Timestamp(data_inicio, tz="UTC")
        dados = dados[dados["data"] >= inicio]

    if data_fim is not None:
        fim = pd.Timestamp(data_fim, tz="UTC")
        dados = dados[dados["data"] <= fim]

    precos = dados.pivot_table(
        index="data",
        columns="ticker",
        values=coluna_preco,
        aggfunc="last",
    )
    precos = precos.sort_index()
    return precos.dropna(axis=1, how="all")


def testar_par(
    precos: pd.DataFrame,
    ativo_y: str,
    ativo_x: str,
    min_observacoes: int,
) -> dict[str, float | int | str] | None:
    serie = precos[[ativo_y, ativo_x]].dropna()
    if len(serie) < min_observacoes:
        return None

    y = serie[ativo_y]
    x = sm.add_constant(serie[ativo_x], has_constant="add")
    modelo = sm.OLS(y, x).fit()
    residuos = modelo.resid.dropna()

    try:
        estatistica_adf, p_value, *_ = adfuller(residuos, autolag="AIC")
    except ValueError:
        return None

    return {
        "Ativo Y": ativo_y,
        "Ativo X": ativo_x,
        "T-Score": estatistica_adf,
        "P-Value": p_value,
        "Hedge Ratio": modelo.params[ativo_x],
        "Observacoes": len(serie),
    }


def testar_setor(
    caminho: Path,
    coluna_preco: str,
    min_observacoes: int,
    data_inicio: str | None,
    data_fim: str | None,
) -> pd.DataFrame:
    setor = caminho.stem
    precos = carregar_precos(caminho, coluna_preco, data_inicio, data_fim)
    tickers = list(precos.columns)
    resultados = []

    for ativo_y, ativo_x in permutations(tickers, 2):
        resultado = testar_par(precos, ativo_y, ativo_x, min_observacoes)
        if resultado is None:
            continue
        resultado["Setor"] = setor
        resultados.append(resultado)

    colunas = [
        "Setor",
        "Ativo Y",
        "Ativo X",
        "T-Score",
        "P-Value",
        "Hedge Ratio",
        "Observacoes",
    ]
    return pd.DataFrame(resultados, columns=colunas)


def main() -> pd.DataFrame:
    args = argumentos()
    arquivos = arquivos_setores(args.entrada, args.setor)
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV de setor encontrado em {args.entrada}")

    tabelas = []
    for indice, arquivo in enumerate(arquivos, start=1):
        if not arquivo.exists():
            print(f"[{indice}/{len(arquivos)}] Pulando arquivo ausente: {arquivo}")
            continue

        print(
            f"[{indice}/{len(arquivos)}] Testando setor {arquivo.stem} "
            f"de {args.inicio_formacao or 'inicio'} ate {args.fim_formacao or 'fim'}..."
        )
        tabela = testar_setor(
            arquivo,
            args.coluna_preco,
            args.min_observacoes,
            args.inicio_formacao,
            args.fim_formacao,
        )
        print(f"  {len(tabela):,} regressao(oes) Engle-Granger avaliadas.")
        tabelas.append(tabela)

    if tabelas:
        todos = pd.concat(tabelas, ignore_index=True)
    else:
        todos = pd.DataFrame(
            columns=[
                "Setor",
                "Ativo Y",
                "Ativo X",
                "T-Score",
                "P-Value",
                "Hedge Ratio",
                "Observacoes",
            ]
        )

    todos = todos.sort_values("P-Value", ignore_index=True)
    validos = todos[todos["P-Value"] < args.p_value_maximo].reset_index(drop=True)

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    validos.to_csv(args.saida, index=False)

    if args.salvar_todos:
        arquivo_todos = args.saida.with_name(f"{args.saida.stem}_todos.csv")
        todos.to_csv(arquivo_todos, index=False)
        print(f"Resultado completo salvo em {arquivo_todos}")

    print("\nModelo de saida:")
    print(validos[["Ativo Y", "Ativo X", "T-Score", "P-Value", "Hedge Ratio"]].head(20))
    print(
        f"\n{len(validos):,} par(es) valido(s) com p-value < {args.p_value_maximo} "
        f"salvos em {args.saida}"
    )

    return validos


if __name__ == "__main__":
    main()
