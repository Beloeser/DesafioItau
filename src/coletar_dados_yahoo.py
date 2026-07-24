"""Baixa dados brutos de acoes da B3 no Yahoo Finance, separados por setor."""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


SETORES: dict[str, list[str]] = {
    "financeiro": """
        ABCB4 B3SA3 BAZA3 BBAS3 BBSE3 BMEB3 BMEB4 BMIN3 BMIN4 BNBR3 BPAC11
        BPAC3 BPAC5 BRSR3 BRSR5 BRSR6 BSLI3 BSLI4 CGRA3 CGRA4 CXSE3
        INBR32 IRBR3 ITSA3 ITSA4 ITUB3 ITUB4 PABY11 PINE4 PSSA3 RPAD3 RPAD5
        RPAD6 SANB11
        SANB3 SANB4 WIZC3
    """.split(),
    "mineracao_siderurgia": """
        BRAP3 BRAP4 CBAV3 CMIN3 CSNA3 FESA3 FESA4 GGBR3 GGBR4 GOAU3 GOAU4
        PMAM3 USIM3 USIM5 USIM6 VALE3
    """.split(),
    "energia_eletrica": """
        ALUP11 ALUP3 ALUP4 AURE3 CBEE3 CEBR3 CEBR5 CEBR6 CGAS3 CGAS5
        CMIG3 CMIG4 COCE3 COCE5 CPFE3 CPLE3 EGIE3 EMAE4 ENGI11 ENGI3 ENGI4
        ENMT3 ENMT4 EQPA3 EQPA5 EQPA6 EQPA7 EQTL3 LIGT3 REDE3 RNEW3 RNEW4
        TAEE11 TAEE3 TAEE4
    """.split(),
    "oleo_gas": """
        BRAV3 CSAN3 DEXP3 DEXP4 LUXM4 OSXB3 PETR3 PETR4 PRIO3 PTNT3
        PTNT4 RAIZ4 RECV3 UGPA3 VBBR3
    """.split(),
    "agro_alimentos": """
        ABEV3 AGRO3 BEEF3 CAML3 JALL3 LAND3 MDIA3 SLCE3 SMTO3 SOJA3 TTEN3
    """.split(),
    "industria": """
        AERI3 ETER3 FRAS3 INEP3 INEP4 JSLG3 KEPL3 LOGN3 LUPA3 MYPK3 PLAS3
        POMO3 POMO4 PTBL3 RAIL3 RAPT3 RAPT4 ROMI3 SHUL4 TGMA3 TUPY3 VLID3
        WEGE3
    """.split(),
}

CAMPOS_YAHOO = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa OHLCV da B3 pelo Yahoo Finance em um CSV por setor."
    )
    parser.add_argument(
        "--inicio",
        default=None,
        help="Data inicial (AAAA-MM-DD). Por padrao, usa 2015-01-01 para dados diarios.",
    )
    parser.add_argument(
        "--fim", default=None,
        help="Data final exclusiva (AAAA-MM-DD). Por padrao, usa a data atual."
    )
    parser.add_argument(
        "--intervalo", default="1d", choices=["1h", "1d", "1wk", "1mo"],
        help="Frequencia das observacoes (padrao: 1d)."
    )
    parser.add_argument(
        "--saida", type=Path, default=Path("data/raw/setores"),
        help="Diretorio dos CSVs (padrao: data/raw/setores)."
    )
    parser.add_argument(
        "--setor", action="append", choices=sorted(SETORES),
        help="Baixa somente este setor. Pode ser informado mais de uma vez."
    )
    parser.add_argument(
        "--pausa", type=float, default=1.0,
        help="Segundos de espera entre setores (padrao: 1)."
    )
    parser.add_argument(
        "--tentativas", type=int, default=2,
        help="Tentativas individuais para downloads ausentes (padrao: 2)."
    )
    parser.add_argument(
        "--pausa-tentativa", type=float, default=1.0,
        help="Espera entre novas tentativas, em segundos (padrao: 1)."
    )
    return parser.parse_args()


def extrair_ativo(dados: pd.DataFrame, ticker_yahoo: str) -> pd.DataFrame:
    """Extrai as colunas OHLCV de um ticker de uma resposta do yfinance."""
    if dados.empty:
        return pd.DataFrame()

    if isinstance(dados.columns, pd.MultiIndex):
        primeiro_nivel = dados.columns.get_level_values(0)
        if ticker_yahoo in primeiro_nivel:
            ativo = dados[ticker_yahoo].copy()
        elif set(CAMPOS_YAHOO).intersection(primeiro_nivel):
            # O yfinance usa (campo, ticker) em algumas respostas de um unico ativo.
            ativo = dados.xs(ticker_yahoo, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        ativo = dados.copy()

    return ativo.reindex(columns=CAMPOS_YAHOO).dropna(how="all")


def baixar_setor(
    tickers_b3: list[str],
    inicio: str | None,
    fim: str | None,
    intervalo: str,
    tentativas: int,
    pausa_tentativa: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Baixa um setor e retorna dados em formato longo e tickers ausentes."""
    tickers_yahoo = [f"{ticker}.SA" for ticker in tickers_b3]
    dados = yf.download(
        tickers=tickers_yahoo,
        start=inicio,
        end=fim,
        interval=intervalo,
        group_by="ticker",
        auto_adjust=False,
        actions=False,
        threads=True,
        progress=False,
    )

    tabelas: list[pd.DataFrame] = []
    sem_dados: list[str] = []

    for ticker_b3, ticker_yahoo in zip(tickers_b3, tickers_yahoo):
        ativo = extrair_ativo(dados, ticker_yahoo)
        if ativo.empty:
            sem_dados.append(ticker_b3)
            continue

        ativo.index.name = "data"
        ativo = ativo.reset_index()
        ativo.insert(1, "ticker", ticker_b3)
        ativo.columns = [
            "data", "ticker", "abertura", "maxima", "minima", "fechamento",
            "fechamento_ajustado", "volume",
        ]
        tabelas.append(ativo)

    # Mensagens como "no timezone found" podem ser falhas transitorias do Yahoo.
    # Repetir apenas os ausentes reduz falsos positivos sem refazer todo o setor.
    ainda_ausentes: list[str] = []
    for posicao, ticker_b3 in enumerate(sem_dados):
        ticker_yahoo = f"{ticker_b3}.SA"
        ativo = pd.DataFrame()
        for tentativa in range(1, tentativas + 1):
            if pausa_tentativa > 0:
                time.sleep(pausa_tentativa)
            print(
                f"  Nova tentativa {tentativa}/{tentativas}: {ticker_yahoo}",
                flush=True,
            )
            resposta = yf.download(
                tickers=ticker_yahoo,
                start=inicio,
                end=fim,
                interval=intervalo,
                group_by="ticker",
                auto_adjust=False,
                actions=False,
                threads=False,
                progress=False,
            )
            ativo = extrair_ativo(resposta, ticker_yahoo)
            if not ativo.empty:
                break

        if ativo.empty:
            ainda_ausentes.append(ticker_b3)
            continue

        ativo.index.name = "data"
        ativo = ativo.reset_index()
        ativo.insert(1, "ticker", ticker_b3)
        ativo.columns = [
            "data", "ticker", "abertura", "maxima", "minima", "fechamento",
            "fechamento_ajustado", "volume",
        ]
        tabelas.append(ativo)
        print(f"  Recuperado: {ticker_yahoo}")

    if not tabelas:
        return pd.DataFrame(), ainda_ausentes

    resultado = pd.concat(tabelas, ignore_index=True)
    resultado = resultado.sort_values(["data", "ticker"], ignore_index=True)
    return resultado, ainda_ausentes


def main() -> list[str]:
    """Executa a coleta e retorna os tickers que permaneceram sem dados."""
    args = argumentos()
    if args.intervalo == "1h":
        fim_referencia = date.fromisoformat(args.fim) if args.fim else date.today()
        inicio_minimo = fim_referencia - timedelta(days=729)
        if args.inicio is None or date.fromisoformat(args.inicio) < inicio_minimo:
            if args.inicio is not None:
                print(
                    "Aviso: o Yahoo fornece no maximo 730 dias para o intervalo 1h; "
                    f"a data inicial foi ajustada para {inicio_minimo.isoformat()}."
                )
            args.inicio = inicio_minimo.isoformat()
    elif args.inicio is None:
        args.inicio = "2015-01-01"

    setores = args.setor or list(SETORES)
    args.saida.mkdir(parents=True, exist_ok=True)
    resumo: list[dict[str, object]] = []
    falhas_totais: list[tuple[str, str]] = []

    for indice, setor in enumerate(setores):
        tickers = SETORES[setor]
        print(f"[{indice + 1}/{len(setores)}] Baixando {setor} ({len(tickers)} tickers)...")
        dados, sem_dados = baixar_setor(
            tickers,
            args.inicio,
            args.fim,
            args.intervalo,
            args.tentativas,
            args.pausa_tentativa,
        )
        arquivo = args.saida / f"{setor}.csv"

        if not dados.empty:
            formato_data = "%Y-%m-%d %H:%M:%S%z" if args.intervalo == "1h" else "%Y-%m-%d"
            dados.to_csv(arquivo, index=False, date_format=formato_data)
            baixados = dados["ticker"].nunique()
            print(f"  {baixados} tickers e {len(dados):,} linhas salvos em {arquivo}")
        else:
            baixados = 0
            print("  Nenhum dado retornado; CSV nao foi criado.")

        if sem_dados:
            print(f"  Sem dados no Yahoo: {', '.join(sem_dados)}")
            falhas_totais.extend((setor, ticker) for ticker in sem_dados)

        resumo.append({
            "setor": setor,
            "tickers_solicitados": len(tickers),
            "tickers_baixados": baixados,
            "linhas": len(dados),
            "tickers_sem_dados": ",".join(sem_dados),
        })
        if indice < len(setores) - 1 and args.pausa > 0:
            time.sleep(args.pausa)

    arquivo_resumo = args.saida / "resumo_coleta.csv"
    pd.DataFrame(resumo).to_csv(arquivo_resumo, index=False)
    print(f"Resumo salvo em {arquivo_resumo}")

    print("\n" + "=" * 60)
    print("RESUMO FINAL DE TICKERS QUE FALHARAM")
    print("=" * 60)
    tickers_com_falha = sorted({f"{ticker}.SA" for _, ticker in falhas_totais})
    if tickers_com_falha:
        for setor, ticker in falhas_totais:
            print(f"- {ticker}.SA ({setor})")
        print(f"Total: {len(tickers_com_falha)} ticker(s) sem dados.")
        print(f"Lista retornada: {tickers_com_falha}")
    else:
        print("Todos os tickers foram baixados com sucesso.")

    return tickers_com_falha


if __name__ == "__main__":
    main()
