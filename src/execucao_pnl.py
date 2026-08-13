"""Modelos de execucao para backtest (fechamento vs abertura).

Fechamento (legado):
    sinal no close de t-1 -> posicao no dia t; PnL = close[t]-close[t-1].
    Assume entrada no close de t-1 (otimista).

Abertura (padrao honesto):
    sinal no close de t-1 -> executa na abertura de t;
    entrada: open[t]->close[t]; continuacao: close[t-1]->close[t];
    saida: close[t-1]->open[t].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def pnl_transicao_abertura(
    t: int,
    pos_anterior: int,
    pos_alvo: int,
    spread_close: np.ndarray,
    spread_open: np.ndarray,
    n_y: float,
) -> float:
    """PnL de um dia t quando a posicao muda (ou nao) na abertura."""
    if t <= 0:
        if pos_alvo == pos_anterior:
            return 0.0
        if pos_alvo != 0:
            return float(pos_alvo * n_y * (spread_close[t] - spread_open[t]))
        return 0.0

    pnl = 0.0
    if pos_alvo == pos_anterior:
        if pos_alvo != 0:
            pnl = float(pos_alvo * n_y * (spread_close[t] - spread_close[t - 1]))
    else:
        if pos_anterior != 0:
            pnl += float(pos_anterior * n_y * (spread_open[t] - spread_close[t - 1]))
        if pos_alvo != 0:
            pnl += float(pos_alvo * n_y * (spread_close[t] - spread_open[t]))
    return pnl


def pnl_diario_abertura(
    posicao: np.ndarray,
    spread_close: np.ndarray,
    spread_open: np.ndarray,
    n_y: float,
) -> np.ndarray:
    """Vetor de PnL bruto diario com execucao na abertura."""
    n = len(posicao)
    out = np.zeros(n, dtype=float)
    prev = 0
    for t in range(n):
        p = int(posicao[t])
        out[t] = pnl_transicao_abertura(t, prev, p, spread_close, spread_open, n_y)
        prev = p
    return out


def pnl_diario_fechamento(
    posicao: np.ndarray,
    spread_close: np.ndarray,
    n_y: float,
) -> np.ndarray:
    """Vetor de PnL bruto diario (modo legado close->close)."""
    diff = np.diff(spread_close, prepend=spread_close[0])
    diff[0] = 0.0
    return posicao.astype(float) * n_y * diff


def carregar_spread_abertura(
    df: pd.DataFrame,
    ativo_y: str,
    ativo_x: str,
    hedge: float,
    dados_setores: Path | None = None,
) -> np.ndarray:
    """Retorna spread na abertura alinhado a `df`.

    Usa colunas `{Y}_abertura`/`{X}_abertura` ou `spread_abertura` se existirem;
    senao busca `abertura` nos CSVs em `data/raw/setores/`.
    """
    if "spread_abertura" in df.columns:
        return df["spread_abertura"].astype(float).to_numpy()

    y_col, x_col = f"{ativo_y}_abertura", f"{ativo_x}_abertura"
    if y_col in df.columns and x_col in df.columns:
        return (
            df[y_col].astype(float).to_numpy()
            - hedge * df[x_col].astype(float).to_numpy()
        )

    if dados_setores is None:
        dados_setores = Path("data/raw/setores")
    if not dados_setores.exists():
        raise ValueError(
            "Execucao na abertura exige spread_abertura no CSV ou pasta data/raw/setores/"
        )

    tickers = {ativo_y, ativo_x}
    partes: list[pd.DataFrame] = []
    for caminho in sorted(dados_setores.glob("*.csv")):
        if caminho.name == "resumo_coleta.csv":
            continue
        bloco = pd.read_csv(caminho, usecols=["data", "ticker", "abertura"])
        bloco = bloco[bloco["ticker"].isin(tickers)]
        if not bloco.empty:
            partes.append(bloco)

    if not partes:
        raise ValueError(f"Aberturas nao encontradas para {ativo_y}/{ativo_x}")

    raw = pd.concat(partes, ignore_index=True)
    raw["data"] = pd.to_datetime(raw["data"], utc=True)
    raw["abertura"] = pd.to_numeric(raw["abertura"], errors="coerce")
    aberturas = raw.pivot_table(
        index="data", columns="ticker", values="abertura", aggfunc="last"
    ).sort_index()

    base = df[["data"]].copy()
    base["data"] = pd.to_datetime(base["data"], utc=True)
    merged = base.merge(aberturas[[ativo_y, ativo_x]], on="data", how="left")
    if merged[ativo_y].isna().any() or merged[ativo_x].isna().any():
        missing = int(merged[ativo_y].isna().sum() + merged[ativo_x].isna().sum())
        raise ValueError(
            f"Faltam {missing} aberturas para {ativo_y}/{ativo_x} — "
            "reexecute a coleta ou use --execucao fechamento."
        )
    return (
        merged[ativo_y].astype(float).to_numpy()
        - hedge * merged[ativo_x].astype(float).to_numpy()
    )
