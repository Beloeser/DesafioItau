"""Avalia ganhos (R$) dos sinais gerados pelo pipeline de cointegracao parcial.

Nao altera nenhuma etapa do pipeline: le o CSV salvo por
``pipeline_cointegracao_parcial.py`` (ou ``01_swanet_quebras.py``) e simula o
resultado financeiro APENAS no periodo de negociacao, com execucao no dia
seguinte ao sinal (sem look-ahead).

Modelo de PnL:
    - posicao efetiva no dia t = sinal do fechamento de t-1;
    - long spread  (+1): compra Y, vende hedge*X;
    - short spread (-1): vende Y, compra hedge*X;
    - tamanho fixo: metade do capital na perna Y no primeiro dia;
    - custo opcional (taxa one-way sobre o notional movimentado).

Uso:
    python3 src/avaliar_ganhos.py --entrada data/processed/pipeline_com_quebras.csv
    python3 src/avaliar_ganhos.py --entrada ... --coluna-sinal sinal_finrl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from periodos import DATA_FIM_NEGOCIACAO, DATA_INICIO_NEGOCIACAO


def carregar_pipeline(entrada: Path) -> tuple[pd.DataFrame, str, str, float]:
    """Le o CSV do pipeline e infere tickers e hedge ratio.

    O hedge ratio e recuperado algebricamente de
    spread = Y - hedge * X  =>  hedge = (Y - spread) / X.
    """
    df = pd.read_csv(entrada)
    df["data"] = pd.to_datetime(df["data"], utc=True)
    ativo_y, ativo_x = df.columns[1], df.columns[2]
    linha = df.iloc[0]
    hedge = float((linha[ativo_y] - linha["spread_observado"]) / linha[ativo_x])
    return df, ativo_y, ativo_x, hedge


def simular_ganhos(
    df: pd.DataFrame,
    ativo_y: str,
    ativo_x: str,
    hedge: float,
    coluna_sinal: str = "sinal",
    capital: float = 100_000.0,
    taxa: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """Simula PnL em R$ no DataFrame ja recortado para o periodo de negociacao."""
    dados = df.reset_index(drop=True).copy()
    preco_y = dados[ativo_y].astype(float)
    preco_x = dados[ativo_x].astype(float)
    spread = dados["spread_observado"].astype(float)

    # Execucao t+1: sinal do fechamento de ontem vira posicao de hoje.
    posicao = dados[coluna_sinal].fillna(0).astype(int).shift(1).fillna(0).astype(int)

    n_y = (capital / 2.0) / max(float(preco_y.iloc[0]), 1e-9)
    n_x = abs(hedge) * n_y

    pnl_bruto = posicao * n_y * spread.diff().fillna(0.0)

    notional = n_y * preco_y + n_x * preco_x
    mudanca = posicao.diff().abs().fillna(abs(float(posicao.iloc[0])))
    custos = taxa * notional * mudanca

    pnl_liquido = pnl_bruto - custos
    equity = capital + pnl_liquido.cumsum()

    # Trades fechados: blocos consecutivos de posicao != 0.
    resultados_trades: list[float] = []
    acumulado = 0.0
    em_trade = False
    for pos, pnl in zip(posicao, pnl_liquido):
        if pos != 0:
            em_trade = True
            acumulado += float(pnl)
        elif em_trade:
            resultados_trades.append(acumulado)
            acumulado = 0.0
            em_trade = False
    if em_trade:
        resultados_trades.append(acumulado)

    retornos = pnl_liquido / capital
    sharpe = 0.0
    if retornos.std(ddof=1) > 0:
        sharpe = float(retornos.mean() / retornos.std(ddof=1) * np.sqrt(252))
    topo = equity.cummax()
    max_dd = float(((equity - topo) / topo).min() * 100)

    metricas = {
        "pnl_bruto": float(pnl_bruto.sum()),
        "custos": float(custos.sum()),
        "pnl_liquido": float(pnl_liquido.sum()),
        "retorno_pct": float(pnl_liquido.sum() / capital * 100),
        "sharpe_anualizado": sharpe,
        "max_drawdown_pct": max_dd,
        "trades_fechados": len(resultados_trades),
        "win_rate_pct": (
            100.0 * sum(1 for r in resultados_trades if r > 0) / len(resultados_trades)
            if resultados_trades
            else 0.0
        ),
        "dias_posicionado_pct": float((posicao != 0).mean() * 100),
        "acoes_y": n_y,
        "acoes_x": n_x,
    }

    dados["posicao"] = posicao
    dados["pnl_liquido"] = pnl_liquido
    dados["equity"] = equity
    return dados, metricas


def pnl_mensal(dados: pd.DataFrame) -> pd.Series:
    return dados.set_index("data")["pnl_liquido"].resample("ME").sum()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ganhos em R$ do pipeline parcial.")
    parser.add_argument(
        "--entrada",
        type=Path,
        default=Path("data/processed/pipeline_com_quebras.csv"),
    )
    parser.add_argument("--coluna-sinal", default="sinal")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--inicio-negociacao", default=DATA_INICIO_NEGOCIACAO)
    parser.add_argument("--fim-negociacao", default=DATA_FIM_NEGOCIACAO)
    parser.add_argument(
        "--taxas",
        type=float,
        nargs="*",
        default=[0.0, 0.0008],
        help="Cenarios de custo one-way (padrao: 0 e 8 bps).",
    )
    parser.add_argument("--saida", type=Path, default=None)
    args = parser.parse_args()

    df, ativo_y, ativo_x, hedge = carregar_pipeline(args.entrada)

    inicio = pd.to_datetime(args.inicio_negociacao).tz_localize("UTC")
    fim = pd.to_datetime(args.fim_negociacao).tz_localize("UTC")
    negociacao = df[(df["data"] >= inicio) & (df["data"] <= fim)]
    if negociacao.empty:
        raise ValueError(f"Sem dados de negociacao entre {inicio} e {fim}.")

    print("=" * 62)
    print(f"GANHOS — {ativo_y} x {ativo_x} | hedge={hedge:.4f}")
    print(f"Sinal: '{args.coluna_sinal}' | Negociacao: {inicio.date()} a {fim.date()}")
    print(f"Dias uteis: {len(negociacao)} | Capital: R$ {args.capital:,.0f}")
    print("=" * 62)

    linhas_resumo = []
    for taxa in args.taxas:
        dados, m = simular_ganhos(
            negociacao,
            ativo_y,
            ativo_x,
            hedge,
            coluna_sinal=args.coluna_sinal,
            capital=args.capital,
            taxa=taxa,
        )
        rotulo = "SEM CUSTOS" if taxa == 0 else f"CUSTO {taxa * 1e4:.0f} bps one-way"
        print(f"\n--- {rotulo} ---")
        for chave, valor in m.items():
            if isinstance(valor, float):
                print(f"  {chave:<22}: {valor:>12,.2f}")
            else:
                print(f"  {chave:<22}: {valor:>12}")
        print("  PnL mensal (R$):")
        for data, valor in pnl_mensal(dados).items():
            simbolo = "+" if valor >= 0 else "-"
            print(f"    {data.strftime('%Y-%m')}: {valor:>+10,.2f} [{simbolo}]")
        linhas_resumo.append({"taxa": taxa, "par": f"{ativo_y}/{ativo_x}", **m})

    if args.saida:
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(linhas_resumo).to_csv(args.saida, index=False)
        print(f"\nResumo salvo em {args.saida}")


if __name__ == "__main__":
    main()
