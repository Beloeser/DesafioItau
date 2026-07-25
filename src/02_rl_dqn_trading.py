"""
Módulo 2 (Definitivo): Deep Q-Network com Separação Treino/Teste.
Elimina o viés de overfitting separando estritamente os dados in-sample e out-of-sample.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN

from periodos import DATA_FIM_NEGOCIACAO, DATA_INICIO_NEGOCIACAO

CAMINHO_CDI_PADRAO = Path("data/raw/cdi/rentabilidade_cdi.csv")

class TradingFronteirasMDP(gym.Env):
    def __init__(self, df: pd.DataFrame, taxa_corretagem: float = 0.0):
        super(TradingFronteirasMDP, self).__init__()
        self.df = df.reset_index(drop=True)
        self.taxa_corretagem = taxa_corretagem 
        
        # Acoes: grid de fronteiras (Entrada, StopLoss)
        self.fronteiras = [
            {"entrada": 1.00, "stop": 3.0},
            {"entrada": 1.25, "stop": 3.5},
            {"entrada": 1.50, "stop": 4.0},
        ]
        self.action_space = spaces.Discrete(4)
        
        # Estados: [Z-Score, Posicao, Prob Quebra, Tempo Restante]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        self.current_step = 0
        self.posicao = 0
        self.zscore_entrada = 0.0
        self.spread_entrada = 0.0  # CORREÇÃO Bug #6: rastreia spread real

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.posicao = 0
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        tempo_restante = (len(self.df) - self.current_step) / len(self.df)
        return np.array([
            row["zscore_mr"], 
            self.posicao, 
            row.get("prob_quebra", 0.0), 
            tempo_restante
        ], dtype=np.float32)

    def step(self, action):
        row = self.df.iloc[self.current_step]
        z_atual = row["zscore_mr"]
        spread_atual = row["spread_observado"]
        recompensa = 0.0
        done = False
        
        if action < 3:
            limite_entrada = self.fronteiras[action]["entrada"]
            limite_stop = self.fronteiras[action]["stop"]
        else:
            limite_entrada = np.inf
            limite_stop = 0.0 

        # Logica baseada no sinal isolado de Reversao a Media
        if self.posicao == 0 and action < 3:
            if z_atual <= -limite_entrada:
                self.posicao = 1 
                self.zscore_entrada = z_atual
                self.spread_entrada = spread_atual
                recompensa -= self.taxa_corretagem
            elif z_atual >= limite_entrada:
                self.posicao = -1
                self.zscore_entrada = z_atual
                self.spread_entrada = spread_atual
                recompensa -= self.taxa_corretagem

        elif self.posicao == 1: 
            if z_atual >= 0 or z_atual <= -limite_stop or action == 3: 
                # CORREÇÃO Bug #6: Recompensa baseada em variação normalizada
                # do spread, não do z-score, para alinhar treino com backtest.
                std_mr = row.get("std_mr", 1.0)
                if std_mr > 0:
                    recompensa += (spread_atual - self.spread_entrada) / std_mr
                recompensa -= self.taxa_corretagem
                self.posicao = 0

        elif self.posicao == -1:
            if z_atual <= 0 or z_atual >= limite_stop or action == 3:
                std_mr = row.get("std_mr", 1.0)
                if std_mr > 0:
                    recompensa += (self.spread_entrada - spread_atual) / std_mr
                recompensa -= self.taxa_corretagem
                self.posicao = 0

        prob_quebra = row.get("prob_quebra", 0.0)
        if self.posicao != 0:
            recompensa -= (prob_quebra * 0.05) 

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
            
        return self._get_obs(), float(recompensa), done, False, {}

def calcular_cdi_periodo(caminho_cdi: Path, inicio, fim) -> float | None:
    """Retorna a rentabilidade acumulada do CDI no periodo do backtest."""
    if not caminho_cdi.exists():
        print(f"Aviso: arquivo de CDI nao encontrado em {caminho_cdi}.")
        return None

    cdi = pd.read_csv(caminho_cdi)
    colunas_necessarias = {"data", "fator_cdi_dia"}
    if not colunas_necessarias.issubset(cdi.columns):
        print(
            "Aviso: arquivo de CDI sem as colunas esperadas "
            "('data' e 'fator_cdi_dia')."
        )
        return None

    cdi["data"] = pd.to_datetime(cdi["data"], utc=True)
    cdi_periodo = cdi[(cdi["data"] >= inicio) & (cdi["data"] <= fim)]
    if cdi_periodo.empty:
        print("Aviso: nao ha dados de CDI dentro do periodo avaliado.")
        return None

    fator_acumulado = cdi_periodo["fator_cdi_dia"].prod()
    return (fator_acumulado - 1) * 100


def executar_backtest_financeiro(
    modelo,
    env_teste,
    capital_inicial=10000.0,
    caminho_cdi: Path | None = CAMINHO_CDI_PADRAO,
):
    """Simula operações de pairs trading com P&L financeiro realista.

    CORREÇÃO Bug #4: O P&L de um pairs trade é calculado como:
        lucro = quantidade × (spread_saida - spread_entrada)   [long]
        lucro = quantidade × (spread_entrada - spread_saida)   [short]
    onde quantidade = capital / custo_nocional_do_par.

    A versão anterior usava variação PERCENTUAL do spread, que explode
    quando o spread está perto de zero (divisão por número pequeno).
    """
    obs, _ = env_teste.reset()
    terminou = False
    
    saldo = capital_inicial
    posicao_anterior = 0
    spread_entrada = 0.0
    preco_y_entrada = 0.0
    preco_x_entrada = 0.0
    trades_realizados = 0
    lucro_bruto_acumulado = 0.0

    while not terminou:
        acao, _states = modelo.predict(obs, deterministic=True)
        row = env_teste.df.iloc[env_teste.current_step]
        spread_atual = row["spread_observado"]
        
        obs, _, terminou, _, _ = env_teste.step(acao)

        if env_teste.posicao != posicao_anterior:
            # IA Mandou Abrir Posicao
            if env_teste.posicao in [1, -1] and posicao_anterior == 0:
                spread_entrada = spread_atual
                # Captura preços individuais para calcular nocional
                preco_y_entrada = row.get(env_teste.df.columns[1], 1.0) if len(env_teste.df.columns) > 1 else 1.0
                preco_x_entrada = row.get(env_teste.df.columns[2], 1.0) if len(env_teste.df.columns) > 2 else 1.0
                # Nocional = preco_Y + |hedge_ratio| * preco_X
                # Como não temos hedge_ratio aqui, usamos o valor absoluto
                # médio do spread como proxy conservador do nocional.
                
            # IA Mandou Fechar Posicao
            elif env_teste.posicao == 0 and posicao_anterior != 0:
                # CORREÇÃO Bug #4: P&L = variação ABSOLUTA do spread × quantidade
                # A quantidade é limitada ao capital alocado / nocional do par.
                # Usamos o spread médio como escala para converter em retorno %.
                escala_nocional = max(
                    abs(preco_y_entrada) + abs(preco_x_entrada),
                    1.0  # piso para evitar divisão por zero
                )
                quantidade = capital_inicial / escala_nocional

                if posicao_anterior == 1:  # Long spread
                    lucro_trade = quantidade * (spread_atual - spread_entrada)
                else:                      # Short spread
                    lucro_trade = quantidade * (spread_entrada - spread_atual)


                
                saldo += lucro_trade
                lucro_bruto_acumulado += lucro_trade
                trades_realizados += 1
                
        posicao_anterior = env_teste.posicao

    rentabilidade_pct = ((saldo - capital_inicial) / capital_inicial) * 100
    data_inicio = env_teste.df["data"].iloc[0]
    data_fim = env_teste.df["data"].iloc[-1]
    rentabilidade_cdi_pct = (
        calcular_cdi_periodo(caminho_cdi, data_inicio, data_fim)
        if caminho_cdi is not None
        else None
    )
    
    print("\n" + "="*50)
    print("RELATÓRIO FINANCEIRO: OUT-OF-SAMPLE (Teste)")
    print("="*50)
    print(f"Período Avaliado:      {data_inicio} a {data_fim}")
    print(f"Capital Alocado:       R$ {capital_inicial:.2f}")
    print(f"Saldo Final:           R$ {saldo:.2f}")
    print(f"Total de Trades:       {trades_realizados}")
    print(f"Custos de Transação:   desconsiderados")
    print(f"Rentabilidade:         {rentabilidade_pct:.2f}%")
    if rentabilidade_cdi_pct is not None:
        diferenca_cdi_pct = rentabilidade_pct - rentabilidade_cdi_pct
        print(f"Rentabilidade CDI:     {rentabilidade_cdi_pct:.2f}%")
        print(f"Diferença vs CDI:      {diferenca_cdi_pct:.2f} p.p.")
        if rentabilidade_cdi_pct != 0:
            percentual_cdi = (rentabilidade_pct / rentabilidade_cdi_pct) * 100
            print(f"Percentual do CDI:     {percentual_cdi:.2f}%")
        else:
            print("Percentual do CDI:     indisponivel (CDI igual a 0%)")
    print("="*50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrada", type=Path, default=Path("data/processed/pipeline_com_quebras.csv"))
    parser.add_argument(
        "--inicio-negociacao",
        type=str,
        default=DATA_INICIO_NEGOCIACAO,
        help="Inicio do periodo exato de negociacao/backtest.",
    )
    parser.add_argument(
        "--fim-negociacao",
        type=str,
        default=DATA_FIM_NEGOCIACAO,
        help="Fim do periodo exato de negociacao/backtest.",
    )
    parser.add_argument(
        "--cdi",
        type=Path,
        default=CAMINHO_CDI_PADRAO,
        help=f"CSV com a rentabilidade diaria do CDI. Padrao: {CAMINHO_CDI_PADRAO}.",
    )
    args = parser.parse_args()

    print(f"Carregando {args.entrada}...")
    df = pd.read_csv(args.entrada).dropna()
    df["data"] = pd.to_datetime(df["data"], utc=True)

    dt_inicio_negociacao = pd.to_datetime(args.inicio_negociacao).tz_localize("UTC")
    dt_fim_negociacao = pd.to_datetime(args.fim_negociacao).tz_localize("UTC")
    
    df_treino = df[df["data"] < dt_inicio_negociacao]
    df_teste = df[
        (df["data"] >= dt_inicio_negociacao)
        & (df["data"] <= dt_fim_negociacao)
    ]

    if df_treino.empty or df_teste.empty:
        raise ValueError(
            "Erro ao dividir os dados. O arquivo precisa conter dados antes do inicio "
            "da negociacao para treino e dados dentro do periodo de negociacao para backtest."
        )

    print(f"\nDados de TREINO (In-Sample): {len(df_treino)} dias")
    print(f"Dados de TESTE (Out-of-Sample): {len(df_teste)} dias")

    # 2. TREINAMENTO
    env_treino = TradingFronteirasMDP(df=df_treino)
    print("\nTreinando a IA com o histórico passado...")
    modelo_dqn = DQN("MlpPolicy", env_treino, verbose=0, learning_rate=1e-3, exploration_fraction=0.2)
    modelo_dqn.learn(total_timesteps=30000)

    # 3. VALIDAÇÃO OUT-OF-SAMPLE
    print("Treinamento concluído. Iniciando backtest cego nos 6 meses seguintes...")
    env_teste = TradingFronteirasMDP(df=df_teste)
    executar_backtest_financeiro(modelo_dqn, env_teste, caminho_cdi=args.cdi)

if __name__ == "__main__":
    main()
