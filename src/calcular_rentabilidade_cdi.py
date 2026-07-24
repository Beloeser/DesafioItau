"""Calcula a rentabilidade acumulada do CDI em um periodo definido no arquivo.

Fonte: Banco Central do Brasil, SGS serie 12 (CDI diario, % a.d.).
Edite DATA_INICIO e DATA_FIM abaixo para alterar o periodo padrao.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


DATA_INICIO = "2024-01-01"
DATA_FIM = "2024-12-31"
SAIDA = Path("data/raw/cdi/rentabilidade_cdi.csv")

SERIE_CDI_SGS = 12
URL_SGS = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{SERIE_CDI_SGS}/dados"


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baixa o CDI diario no Banco Central e calcula a rentabilidade "
            "acumulada no periodo."
        )
    )
    parser.add_argument(
        "--inicio",
        default=DATA_INICIO,
        help=f"Data inicial inclusiva (AAAA-MM-DD). Padrao: {DATA_INICIO}.",
    )
    parser.add_argument(
        "--fim",
        default=DATA_FIM,
        help=f"Data final inclusiva (AAAA-MM-DD). Padrao: {DATA_FIM}.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=SAIDA,
        help=f"Arquivo CSV de saida. Padrao: {SAIDA}.",
    )
    return parser.parse_args()


def converter_data_para_sgs(data_iso: str) -> str:
    """Converte AAAA-MM-DD para DD/MM/AAAA, formato esperado pela API SGS."""
    return date.fromisoformat(data_iso).strftime("%d/%m/%Y")


def baixar_cdi(inicio: str, fim: str) -> pd.DataFrame:
    """Baixa a taxa diaria do CDI no SGS/BCB para o periodo informado."""
    data_inicio = date.fromisoformat(inicio)
    data_fim = date.fromisoformat(fim)
    if data_inicio > data_fim:
        raise ValueError("DATA_INICIO nao pode ser posterior a DATA_FIM.")

    parametros = urlencode(
        {
            "formato": "json",
            "dataInicial": converter_data_para_sgs(inicio),
            "dataFinal": converter_data_para_sgs(fim),
        }
    )
    url = f"{URL_SGS}?{parametros}"

    try:
        with urlopen(url, timeout=30) as resposta:
            registros = json.load(resposta)
    except HTTPError as erro:
        raise RuntimeError(f"Erro HTTP ao consultar o SGS/BCB: {erro.code}") from erro
    except URLError as erro:
        raise RuntimeError(f"Falha de conexao ao consultar o SGS/BCB: {erro.reason}") from erro

    if not registros:
        raise ValueError("Nenhum dado de CDI retornado para o periodo informado.")

    dados = pd.DataFrame(registros)
    dados["data"] = pd.to_datetime(dados["data"], format="%d/%m/%Y")
    dados["taxa_cdi_dia_percentual"] = (
        dados["valor"].str.replace(",", ".", regex=False).astype(float)
    )
    return dados[["data", "taxa_cdi_dia_percentual"]].sort_values("data")


def calcular_rentabilidade(dados: pd.DataFrame) -> pd.DataFrame:
    """Calcula fator diario, fator acumulado e rentabilidade acumulada do CDI."""
    resultado = dados.copy()
    resultado["fator_cdi_dia"] = 1 + resultado["taxa_cdi_dia_percentual"] / 100
    resultado["fator_cdi_acumulado"] = resultado["fator_cdi_dia"].cumprod()
    resultado["rentabilidade_cdi_acumulada_percentual"] = (
        resultado["fator_cdi_acumulado"] - 1
    ) * 100
    resultado["data"] = resultado["data"].dt.date
    return resultado


def salvar_resultado(resultado: pd.DataFrame, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(caminho, index=False, float_format="%.10f")


def main() -> pd.DataFrame:
    args = argumentos()
    dados = baixar_cdi(args.inicio, args.fim)
    resultado = calcular_rentabilidade(dados)
    salvar_resultado(resultado, args.saida)

    rentabilidade_final = resultado["rentabilidade_cdi_acumulada_percentual"].iloc[-1]
    data_geracao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Arquivo salvo em: {args.saida}")
    print(f"Periodo: {args.inicio} a {args.fim}")
    print(f"Dias uteis com CDI: {len(resultado)}")
    print(f"Rentabilidade acumulada CDI: {rentabilidade_final:.6f}%")
    print(f"Gerado em: {data_geracao}")
    return resultado


if __name__ == "__main__":
    main()
