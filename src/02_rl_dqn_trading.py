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
                recompensa -= self.taxa_corretagem
            elif z_atual >= limite_entrada:
                self.posicao = -1
                self.zscore_entrada = z_atual
                recompensa -= self.taxa_corretagem

        elif self.posicao == 1: 
            if z_atual >= 0 or z_atual <= -limite_stop or action == 3: 
                recompensa += (z_atual - self.zscore_entrada) - self.taxa_corretagem
                self.posicao = 0

        elif self.posicao == -1:
            if z_atual <= 0 or z_atual >= limite_stop or action == 3:
                recompensa += (self.zscore_entrada - z_atual) - self.taxa_corretagem
                self.posicao = 0

        prob_quebra = row.get("prob_quebra", 0.0)
        if self.posicao != 0:
            recompensa -= (prob_quebra * 0.05) 

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
            
        return self._get_obs(), float(recompensa), done, False, {}

def executar_backtest_financeiro(modelo, env_teste, capital_inicial=10000.0):
    """Simula as operacoes no mundo real utilizando as decisoes da IA."""
    obs, _ = env_teste.reset()
    terminou = False
    
    saldo = capital_inicial
    posicao_anterior = 0
    preco_entrada = 0.0
    trades_realizados = 0
    lucro_bruto_acumulado = 0.0

    while not terminou:
        acao, _states = modelo.predict(obs, deterministic=True)
        row = env_teste.df.iloc[env_teste.current_step]
        preco_real_spread = row["spread_observado"]
        
        obs, _, terminou, _, _ = env_teste.step(acao)

        if env_teste.posicao != posicao_anterior:
            # IA Mandou Abrir Posicao
            if env_teste.posicao in [1, -1] and posicao_anterior == 0:
                preco_entrada = preco_real_spread
                
            # IA Mandou Fechar Posicao
            elif env_teste.posicao == 0 and posicao_anterior != 0:
                if posicao_anterior == 1: # Long
                    variacao = (preco_real_spread - preco_entrada) / abs(preco_entrada) if preco_entrada != 0 else 0
                else:                     # Short
                    variacao = (preco_entrada - preco_real_spread) / abs(preco_entrada) if preco_entrada != 0 else 0
                
                lucro_trade = capital_inicial * variacao
                
                saldo += lucro_trade
                lucro_bruto_acumulado += lucro_trade
                trades_realizados += 1
                
        posicao_anterior = env_teste.posicao

    rentabilidade_pct = ((saldo - capital_inicial) / capital_inicial) * 100
    
    print("\n" + "="*50)
    print("RELATÓRIO FINANCEIRO: OUT-OF-SAMPLE (Teste)")
    print("="*50)
    print(f"Período Avaliado:      {env_teste.df['data'].iloc[0]} a {env_teste.df['data'].iloc[-1]}")
    print(f"Capital Alocado:       R$ {capital_inicial:.2f}")
    print(f"Saldo Final:           R$ {saldo:.2f}")
    print(f"Total de Trades:       {trades_realizados}")
    print(f"Custos de Transação:   desconsiderados")
    print(f"Rentabilidade:         {rentabilidade_pct:.2f}%")
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
    executar_backtest_financeiro(modelo_dqn, env_teste)

if __name__ == "__main__":
    main()
