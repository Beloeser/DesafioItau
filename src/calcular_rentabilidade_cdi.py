"""Baixa e calcula a rentabilidade acumulada do CDI no periodo global.

Fonte: Banco Central do Brasil, SGS serie 12 (CDI diario, % a.d.).
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

from periodos import DATA_FIM_NEGOCIACAO, DATA_INICIO_NEGOCIACAO


SAIDA = Path("data/raw/cdi/rentabilidade_cdi.csv")
SERIE_CDI_SGS = 12
URL_SGS = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{SERIE_CDI_SGS}/dados"


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa o CDI diario e calcula a rentabilidade acumulada."
    )
    parser.add_argument("--inicio", default=DATA_INICIO_NEGOCIACAO)
    parser.add_argument("--fim", default=DATA_FIM_NEGOCIACAO)
    parser.add_argument("--saida", type=Path, default=SAIDA)
    return parser.parse_args()


def converter_data_para_sgs(data_iso: str) -> str:
    return date.fromisoformat(data_iso).strftime("%d/%m/%Y")


def baixar_cdi(inicio: str, fim: str) -> pd.DataFrame:
    data_inicio = date.fromisoformat(inicio)
    data_fim = date.fromisoformat(fim)
    if data_inicio > data_fim:
        raise ValueError("A data inicial nao pode ser posterior a data final.")

    parametros = urlencode(
        {
            "formato": "json",
            "dataInicial": converter_data_para_sgs(inicio),
            "dataFinal": converter_data_para_sgs(fim),
        }
    )
    try:
        with urlopen(f"{URL_SGS}?{parametros}", timeout=30) as resposta:
            registros = json.load(resposta)
    except HTTPError as erro:
        raise RuntimeError(
            f"Erro HTTP ao consultar o SGS/BCB: {erro.code}"
        ) from erro
    except URLError as erro:
        raise RuntimeError(
            f"Falha de conexao ao consultar o SGS/BCB: {erro.reason}"
        ) from erro

    if not registros:
        raise ValueError("Nenhum dado de CDI retornado para o periodo.")

    dados = pd.DataFrame(registros)
    dados["data"] = pd.to_datetime(dados["data"], format="%d/%m/%Y")
    dados["taxa_cdi_dia_percentual"] = pd.to_numeric(
        dados["valor"].str.replace(",", ".", regex=False),
        errors="raise",
    )
    return dados[["data", "taxa_cdi_dia_percentual"]].sort_values("data")


def calcular_rentabilidade(dados: pd.DataFrame) -> pd.DataFrame:
    resultado = dados.copy()
    resultado["fator_cdi_dia"] = (
        1.0 + resultado["taxa_cdi_dia_percentual"] / 100.0
    )
    resultado["fator_cdi_acumulado"] = resultado["fator_cdi_dia"].cumprod()
    resultado["rentabilidade_cdi_acumulada_percentual"] = (
        resultado["fator_cdi_acumulado"] - 1.0
    ) * 100.0
    resultado["data"] = resultado["data"].dt.date
    return resultado


def main() -> pd.DataFrame:
    args = argumentos()
    resultado = calcular_rentabilidade(baixar_cdi(args.inicio, args.fim))
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(args.saida, index=False, float_format="%.10f")

    rentabilidade = resultado["rentabilidade_cdi_acumulada_percentual"].iloc[-1]
    print(f"Arquivo salvo em: {args.saida}")
    print(f"Periodo: {args.inicio} a {args.fim}")
    print(f"Dias uteis com CDI: {len(resultado)}")
    print(f"Rentabilidade acumulada CDI: {rentabilidade:.6f}%")
    print(f"Gerado em: {datetime.now():%Y-%m-%d %H:%M:%S}")
    return resultado


if __name__ == "__main__":
    main()
