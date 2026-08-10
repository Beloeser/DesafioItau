"""
Pipeline Definitivo: Pairs Trading com Cointegracao Parcial e Kalman.
Modificado para Backtest Isolado (Fase de Negociação).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from periodos import (
    DATA_FIM_FORMACAO,
    DATA_FIM_NEGOCIACAO,
    DATA_INICIO_FORMACAO,
    DATA_INICIO_NEGOCIACAO,
)


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline Integrado de Cointegracao Parcial e Kalman.")
    parser.add_argument("--cointegracao", type=Path, default=Path("data/processed/cointegracao_engle_granger.csv"))
    parser.add_argument("--dados-setores", type=Path, default=Path("data/raw/setores"))
    parser.add_argument("--saida", type=Path, default=Path("data/processed/pipeline_cointegracao_parcial.csv"))
    parser.add_argument("--setor", default=None)
    parser.add_argument("--ativo-y", default=None)
    parser.add_argument("--ativo-x", default=None)
    
    # --- Parâmetros de Backtest ---
    # Coloque aqui o período de NEGOCIAÇÃO (os 6 meses APÓS a sua avaliação)
    parser.add_argument(
        "--inicio-formacao",
        type=str,
        default=DATA_INICIO_FORMACAO,
        help="Inicio do periodo de formacao usado na cointegracao e SWANet.",
    )
    parser.add_argument(
        "--fim-formacao",
        type=str,
        default=DATA_FIM_FORMACAO,
        help="Fim do periodo de formacao usado na cointegracao e SWANet.",
    )
    parser.add_argument(
        "--inicio-negociacao",
        type=str,
        default=DATA_INICIO_NEGOCIACAO,
        help="Inicio do periodo de negociacao/backtest.",
    )
    parser.add_argument(
        "--fim-negociacao",
        type=str,
        default=DATA_FIM_NEGOCIACAO,
        help="Fim do periodo de negociacao/backtest.",
    )
    
    parser.add_argument("--janela-parametros", type=int, default=120)
    parser.add_argument("--limiar-entrada", type=float, default=1.25)
    return parser.parse_args()


def escolher_par(caminho: Path, setor: str | None, ativo_y: str | None, ativo_x: str | None) -> pd.Series:
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}.")
        
    pares = pd.read_csv(caminho)
    filtro = pd.Series(True, index=pares.index)
    if setor: filtro &= pares["Setor"].eq(setor)
    if ativo_y: filtro &= pares["Ativo Y"].eq(ativo_y)
    if ativo_x: filtro &= pares["Ativo X"].eq(ativo_x)

    candidatos = pares[filtro].sort_values("P-Value")
    if candidatos.empty:
        raise ValueError("Nenhum par encontrado.")
    return candidatos.iloc[0]


def carregar_spread(dados_setores: Path, setor: str, ativo_y: str, ativo_x: str, hedge_ratio: float, 
                    inicio_serie: str, fim_serie: str, janela_dias: int) -> pd.DataFrame:
    """Extrai os precos já filtrados para a janela de tempo necessária."""
    caminho = dados_setores / f"{setor}.csv"
    
    dados = pd.read_csv(caminho, usecols=["data", "ticker", "fechamento"])
    dados["data"] = pd.to_datetime(dados["data"], utc=True)
    
    # Converte as datas limites
    dt_inicio = pd.to_datetime(inicio_serie).tz_localize('UTC')
    dt_fim = pd.to_datetime(fim_serie).tz_localize('UTC')
    
    # Retrocede a data de início para garantir o "aquecimento" do filtro (dias corridos)
    dt_aquecimento = dt_inicio - pd.Timedelta(days=janela_dias * 2) 
    
    # Recorta brutalmente a base para não carregar anos inúteis de dados
    dados = dados[(dados["data"] >= dt_aquecimento) & (dados["data"] <= dt_fim)]
    
    dados["fechamento"] = pd.to_numeric(dados["fechamento"], errors="coerce")
    precos = dados.pivot_table(index="data", columns="ticker", values="fechamento", aggfunc="last").sort_index()

    serie = precos[[ativo_y, ativo_x]].dropna().copy()
    serie["spread_observado"] = serie[ativo_y] - hedge_ratio * serie[ativo_x]
    
    return serie.reset_index()[["data", ativo_y, ativo_x, "spread_observado"]]


def calcular_parametros_dinamicos(spread: pd.Series, janela: int) -> pd.DataFrame:
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
    # Sem bfill: bfill copiava parametros FUTUROS para o aquecimento (look-ahead).
    # As barras iniciais ficam NaN e sao tratadas de forma neutra no filtro.
    return parametros.ffill()


def filtro_kalman_cointegracao_parcial(spread: pd.Series, parametros: pd.DataFrame, var_medicao: float = 1e-4) -> pd.DataFrame:
    n = len(spread)
    mr_filtrado, rw_filtrado, std_mr = np.zeros(n), np.zeros(n), np.zeros(n)

    H = np.array([[1.0, 1.0]])  
    R = np.array([[var_medicao]])
    Z_hat = np.array([[0.0], [spread.iloc[0]]]) 
    P = np.eye(2)

    for i in range(n):
        y = spread.iloc[i]
        rho = parametros["rho"].iloc[i]
        var_mr = parametros["var_mr"].iloc[i]
        var_rw = parametros["var_rw"].iloc[i]

        # Aquecimento sem bfill: parametros NaN viram valores neutros ate a
        # primeira janela completa (essas barras sao descartadas no recorte).
        if not np.isfinite(rho):
            rho = 0.0
        if not np.isfinite(var_mr) or var_mr <= 0:
            var_mr = 1e-6
        if not np.isfinite(var_rw) or var_rw <= 0:
            var_rw = 1e-6

        F = np.array([[rho, 0.0], [0.0, 1.0]])
        Q = np.array([[var_mr, 0.0], [0.0, var_rw]])

        Z_pred = F @ Z_hat
        P_pred = F @ P @ F.T + Q

        y_residual = y - (H @ Z_pred)[0, 0]
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        Z_hat = Z_pred + K * y_residual
        P = (np.eye(2) - K @ H) @ P_pred

        mr_filtrado[i] = Z_hat[0, 0]
        rw_filtrado[i] = Z_hat[1, 0]
        std_mr[i] = np.sqrt(P[0, 0] + var_mr)

    return pd.DataFrame({"mr_filtrado": mr_filtrado, "rw_filtrado": rw_filtrado, "std_mr": std_mr}, index=spread.index)


def gerar_sinais_artigo(mr_filtrado: pd.Series, std_mr: pd.Series, limiar_entrada: float) -> pd.DataFrame:
    zscore_mr = mr_filtrado / std_mr.replace(0, np.nan)
    sinais = []
    posicao = 0

    for mr, zscore in zip(mr_filtrado, zscore_mr):
        if posicao == 0:
            if zscore <= -limiar_entrada: posicao = 1
            elif zscore >= limiar_entrada: posicao = -1
        elif posicao == 1 and mr >= 0: posicao = 0
        elif posicao == -1 and mr <= 0: posicao = 0
        sinais.append(posicao)

    return pd.DataFrame({"zscore_mr": zscore_mr, "sinal": sinais}, index=mr_filtrado.index)


def main():
    args = argumentos()

    par = escolher_par(args.cointegracao, args.setor, args.ativo_y, args.ativo_x)
    print(
        f"Par selecionado: {par['Ativo Y']} x {par['Ativo X']} | "
        f"Formacao: {args.inicio_formacao} a {args.fim_formacao} | "
        f"Negociacao: {args.inicio_negociacao} a {args.fim_negociacao}"
    )
    
    # O carregamento agora inclui a lógica de aquecimento invisível
    df_dados = carregar_spread(
        args.dados_setores, str(par["Setor"]), str(par["Ativo Y"]), str(par["Ativo X"]), 
        float(par["Hedge Ratio"]), args.inicio_formacao, args.fim_negociacao, args.janela_parametros
    )
    
    spread = df_dados["spread_observado"]
    parametros_dinamicos = calcular_parametros_dinamicos(spread, args.janela_parametros)
    df_filtrado = filtro_kalman_cointegracao_parcial(spread, parametros_dinamicos)
    df_sinais = gerar_sinais_artigo(df_filtrado["mr_filtrado"], df_filtrado["std_mr"], args.limiar_entrada)

    # Consolida os dados gerados
    df_resultado = pd.concat([df_dados, parametros_dinamicos, df_filtrado, df_sinais], axis=1)
    
    # =========================================================
    # RECORTE ESTRITO PARA SALVAMENTO (REMOVENDO O AQUECIMENTO)
    # =========================================================
    dt_inicio_formacao = pd.to_datetime(args.inicio_formacao).tz_localize('UTC')
    dt_fim_negociacao = pd.to_datetime(args.fim_negociacao).tz_localize('UTC')
    df_resultado = df_resultado[
        (df_resultado["data"] >= dt_inicio_formacao)
        & (df_resultado["data"] <= dt_fim_negociacao)
    ]
    
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    df_resultado.to_csv(args.saida, index=False)
    
    print(
        f"\nSalvo com sucesso em: {args.saida} "
        f"({len(df_resultado)} dias uteis de formacao + negociacao)"
    )

if __name__ == "__main__":
    main()
