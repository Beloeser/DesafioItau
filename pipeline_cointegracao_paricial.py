"""Pipeline Definitivo: Pairs Trading com Cointegracao Parcial e Kalman.

Baseado no artigo "Kalman Filtering Applied to Investment Portfolio Management".
Substitui a estimativa de Ornstein-Uhlenbeck por solucoes analiticas 
(Equacoes 19 a 23 do artigo) para obter os parametros dinamicamente.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
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
    parser.add_argument("--setor", default=None, help="Filtrar por setor especifico.")
    parser.add_argument("--ativo-y", default=None, help="Ativo dependente Y.")
    parser.add_argument("--ativo-x", default=None, help="Ativo explicativo X.")
    
    # Parametros da Estrategia (Artigo)
    parser.add_argument(
        "--janela-parametros",
        type=int,
        default=120,
        help="Janela movel (dias) para calcular rho, var_mr e var_rw dinamicamente.",
    )
    parser.add_argument(
        "--limiar-entrada",
        type=float,
        default=1.25,
        help="Z-score do componente Mean-Reverting para abrir posicao (Long/Short).",
    )
    return parser.parse_args()


def escolher_par(caminho: Path, setor: str | None, ativo_y: str | None, ativo_x: str | None) -> pd.Series:
    """Carrega o resumo do Engle-Granger e seleciona o par de interesse."""
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}. Execute o script de cointegracao primeiro.")
        
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


def carregar_spread(dados_setores: Path, setor: str, ativo_y: str, ativo_x: str, hedge_ratio: float) -> pd.DataFrame:
    """Extrai os precos brutos e calcula o spread observado histórico."""
    caminho = dados_setores / f"{setor}.csv"
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de setor não encontrado: {caminho}")

    dados = pd.read_csv(caminho, usecols=["data", "ticker", "fechamento"])
    dados["data"] = pd.to_datetime(dados["data"], utc=True)
    dados["fechamento"] = pd.to_numeric(dados["fechamento"], errors="coerce")

    precos = dados.pivot_table(
        index="data", columns="ticker", values="fechamento", aggfunc="last"
    ).sort_index()

    serie = precos[[ativo_y, ativo_x]].dropna().copy()
    serie["spread_observado"] = serie[ativo_y] - hedge_ratio * serie[ativo_x]
    
    return serie.reset_index()[["data", ativo_y, ativo_x, "spread_observado"]]


def calcular_parametros_dinamicos(spread: pd.Series, janela: int) -> pd.DataFrame:
    """
    Calcula rho, var_mr e var_rw utilizando a variancia das diferencas do spread.
    Implementa as Equacoes 19 a 23 do artigo.
    """
    # vk = Variancia das diferencas do spread (Lag 1, 2 e 3)
    v1 = spread.diff(1).rolling(window=janela).var()
    v2 = spread.diff(2).rolling(window=janela).var()
    v3 = spread.diff(3).rolling(window=janela).var()

    # Evita divisao por zero no denominador de rho (Eq. 20)
    denom_rho = (2 * v1 - v2).replace(0, np.nan)

    # Equacao 20: Estimação de rho
    rho = -(v1 - 2 * v2 + v3) / denom_rho
    # Restringe rho a ser estacionario (< 1)
    rho = rho.clip(-0.999, 0.999) 

    # Equacao 21: Variancia do Mean Reverting
    var_mr = 0.5 * ((rho + 1) / (rho - 1)) * (v2 - 2 * v1)
    var_mr = var_mr.clip(lower=1e-6) # Protecao matematica

    # Equacao 22: Variancia do Random Walk
    var_rw = 0.5 * (v2 - 2 * var_mr)
    var_rw = var_rw.clip(lower=1e-6) # Protecao matematica

    # Preenche os dias iniciais da janela movel (onde é NaN) com o primeiro valor util
    parametros = pd.DataFrame({"rho": rho, "var_mr": var_mr, "var_rw": var_rw})
    parametros = parametros.bfill().ffill()
    
    return parametros


def filtro_kalman_cointegracao_parcial(spread: pd.Series, parametros: pd.DataFrame, var_medicao: float = 1e-4) -> pd.DataFrame:
    """Aplica o filtro de Kalman separando os estados ocultos."""
    n = len(spread)
    
    mr_filtrado = np.zeros(n)
    rw_filtrado = np.zeros(n)
    std_mr = np.zeros(n)

    # Matriz de Observacao Constante (Eq 17)
    H = np.array([[1.0, 1.0]])  
    R = np.array([[var_medicao]])

    # Inicializacao
    Z_hat = np.array([[0.0], [spread.iloc[0]]]) # Inicia assumindo que tudo é random walk
    P = np.eye(2)

    for i in range(n):
        y = spread.iloc[i]
        
        # Pega parametros dinamicos do dia atual
        rho = parametros["rho"].iloc[i]
        var_mr = parametros["var_mr"].iloc[i]
        var_rw = parametros["var_rw"].iloc[i]

        # Matrizes do dia (Eq 18)
        F = np.array([[rho, 0.0],
                      [0.0, 1.0]])
        Q = np.array([[var_mr, 0.0],
                      [0.0, var_rw]])

        # 1. Predicao
        Z_pred = F @ Z_hat
        P_pred = F @ P @ F.T + Q

        # 2. Atualizacao
        y_residual = y - (H @ Z_pred)[0, 0]
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        Z_hat = Z_pred + K * y_residual
        P = (np.eye(2) - K @ H) @ P_pred

        # Guarda Resultados
        mr_filtrado[i] = Z_hat[0, 0]
        rw_filtrado[i] = Z_hat[1, 0]
        std_mr[i] = np.sqrt(P[0, 0] + var_mr)

    return pd.DataFrame({
        "mr_filtrado": mr_filtrado,
        "rw_filtrado": rw_filtrado,
        "std_mr": std_mr
    }, index=spread.index)


def gerar_sinais_artigo(mr_filtrado: pd.Series, std_mr: pd.Series, limiar_entrada: float) -> pd.DataFrame:
    """Lógica de trading baseada no cruzamento do Z-Score do componente de reversao."""
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


def plotar_grafico_operacional(df_resultado: pd.DataFrame, limiar: float):
    """Gera um gráfico do componente Mean Reverting, bandas dinâmicas e sinais de trade."""
    
    # Filtra dados vazios da inicialização (janela móvel inicial)
    df_plot = df_resultado.dropna(subset=["mr_filtrado"]).copy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # --- Gráfico 1: Spread Observado vs Componentes ---
    ax1.plot(df_plot["data"], df_plot["spread_observado"], label="Spread Total (Observado)", color='lightgray', alpha=0.7)
    ax1.plot(df_plot["data"], df_plot["rw_filtrado"], label="Passeio Aleatório (RW)", color='orange', linestyle='--', alpha=0.8)
    ax1.plot(df_plot["data"], df_plot["mr_filtrado"], label="Reversão à Média (MR)", color='blue', linewidth=1.5)
    
    # Bandas Dinâmicas do Kalman
    banda_superior = limiar * df_plot["std_mr"]
    banda_inferior = -limiar * df_plot["std_mr"]
    ax1.plot(df_plot["data"], banda_superior, color='red', linestyle=':', label=f"Banda Superior (+{limiar} std)")
    ax1.plot(df_plot["data"], banda_inferior, color='green', linestyle=':', label=f"Banda Inferior (-{limiar} std)")
    ax1.axhline(0, color='black', linewidth=1, linestyle='-')

    # Identificando pontos de Entrada e Saída
    # Mudanças no status sinalizam entradas/saídas
    entradas_long = df_plot[(df_plot["sinal"] == 1) & (df_plot["sinal"].shift(1) == 0)]
    entradas_short = df_plot[(df_plot["sinal"] == -1) & (df_plot["sinal"].shift(1) == 0)]
    saidas = df_plot[(df_plot["sinal"] == 0) & (df_plot["sinal"].shift(1).isin([1, -1]))]

    ax1.scatter(entradas_long["data"], entradas_long["mr_filtrado"], marker='^', color='green', s=100, label="Entrada Long", zorder=5)
    ax1.scatter(entradas_short["data"], entradas_short["mr_filtrado"], marker='v', color='red', s=100, label="Entrada Short", zorder=5)
    ax1.scatter(saidas["data"], saidas["mr_filtrado"], marker='x', color='black', s=80, label="Saída (Zero-Crossing)", zorder=5)
    
    ax1.set_title("Filtro de Kalman: Componentes do Spread e Sinais de Operação", fontsize=14)
    ax1.set_ylabel("Valor do Spread")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- Gráfico 2: Z-Score ---
    ax2.plot(df_plot["data"], df_plot["zscore_mr"], color='purple', label="Z-Score MR")
    ax2.axhline(limiar, color='red', linestyle='--')
    ax2.axhline(-limiar, color='green', linestyle='--')
    ax2.axhline(0, color='black', linestyle='-')
    ax2.set_title("Dinâmica do Z-Score", fontsize=12)
    ax2.set_ylabel("Z-Score")
    ax2.set_xlabel("Data")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    args = argumentos()

    # 1. Obter Serie de Precos e Spread
    print(f"Buscando melhor par cointegrado...")
    par = escolher_par(args.cointegracao, args.setor, args.ativo_y, args.ativo_x)
    
    df_dados = carregar_spread(args.dados_setores, str(par["Setor"]), str(par["Ativo Y"]), str(par["Ativo X"]), float(par["Hedge Ratio"]))
    spread = df_dados["spread_observado"]

    print(f"Par selecionado: {par['Ativo Y']} e {par['Ativo X']} (Setor: {par['Setor']})")
    
    # 2. Estimativa Dinâmica (Eqs 19-23)
    print(f"Calculando parâmetros matemáticos (rho, variâncias) em janela de {args.janela_parametros} dias...")
    parametros_dinamicos = calcular_parametros_dinamicos(spread, args.janela_parametros)

    # 3. Filtro de Kalman Parcial
    print("Aplicando Filtro de Kalman (separando MR e RW)...")
    df_filtrado = filtro_kalman_cointegracao_parcial(spread, parametros_dinamicos)

    # 4. Gerar Sinais
    print(f"Gerando Sinais (Limiar = {args.limiar_entrada} desvios do componente MR)...")
    df_sinais = gerar_sinais_artigo(df_filtrado["mr_filtrado"], df_filtrado["std_mr"], args.limiar_entrada)

    # 5. Consolidar e Salvar
    df_resultado = pd.concat([df_dados, parametros_dinamicos, df_filtrado, df_sinais], axis=1)
    
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    df_resultado.to_csv(args.saida, index=False)
    
    print("\n--- Amostra do Resultado Final ---")
    colunas_exibicao = ["data", "spread_observado", "rho", "mr_filtrado", "rw_filtrado", "zscore_mr", "sinal"]
    print(df_resultado[colunas_exibicao].tail(10))
    print(f"\nSalvo com sucesso em: {args.saida}")

    print("\nGerando Gráfico Operacional...")
    plotar_grafico_operacional(df_resultado, args.limiar_entrada)


if __name__ == "__main__":
    main()