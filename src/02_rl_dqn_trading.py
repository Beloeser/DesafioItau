"""
Módulo 2: Otimização da Estratégia via Reinforcement Learning (Seção 7)
Arquitetura: Deep Q-Network (DQN) modelando a carteira como um Processo de Decisão de Markov.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN

class TradingFronteirasMDP(gym.Env):
    """
    O Agente não envia apenas ordem de compra/venda. Ele escolhe o 'conjunto de fronteiras'
    para o Z-Score do Kalman a cada momento, conforme descrito no artigo.
    """
    def __init__(self, df: pd.DataFrame, custo_transacao: float = 0.004):
        super(TradingFronteirasMDP, self).__init__()
        self.df = df.reset_index(drop=True)
        self.custo_transacao = custo_transacao
        
        # Acoes: O artigo menciona que as acoes definem as fronteiras de entrada e stop loss.
        # Criamos um grid discreto de estrategias possiveis:
        # Acao 0: Entrada (z=1.0), StopLoss (z=3.0)
        # Acao 1: Entrada (z=1.25), StopLoss (z=3.5) -> Recomendado
        # Acao 2: Entrada (z=1.5), StopLoss (z=4.0)
        # Acao 3: Fechar posicoes (Forcar saida)
        self.fronteiras = [
            {"entrada": 1.00, "stop": 3.0},
            {"entrada": 1.25, "stop": 3.5},
            {"entrada": 1.50, "stop": 4.0},
        ]
        self.action_space = spaces.Discrete(4)
        
        # Estados: [Z-Score Atual, Posicao Atual, Probabilidade de Quebra (SWANet), Tempo para fechar]
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
        
        # Define os limites baseados na acao da DQN
        if action < 3:
            limite_entrada = self.fronteiras[action]["entrada"]
            limite_stop = self.fronteiras[action]["stop"]
        else:
            limite_entrada = np.inf # Nao permite entrada se acao = 3 (Zerar)
            limite_stop = 0.0       # Forca stop

        # --- MECÂNICA DE TRADE (Eqs do Q-Value são maximizadas indiretamente pela recompensa) ---
        if self.posicao == 0 and action < 3:
            # Avalia condicao de entrada
            if z_atual <= -limite_entrada:
                self.posicao = 1 # Long
                self.zscore_entrada = z_atual
                recompensa -= self.custo_transacao
            elif z_atual >= limite_entrada:
                self.posicao = -1 # Short
                self.zscore_entrada = z_atual
                recompensa -= self.custo_transacao

        elif self.posicao == 1: # Se está Long
            if z_atual >= 0: # Saida normal (cruzou o zero)
                recompensa += (z_atual - self.zscore_entrada) - self.custo_transacao
                self.posicao = 0
            elif z_atual <= -limite_stop or action == 3: # Stop Loss ou Saida Forcada
                recompensa += (z_atual - self.zscore_entrada) - self.custo_transacao
                self.posicao = 0

        elif self.posicao == -1: # Se está Short
            if z_atual <= 0: # Saida normal
                recompensa += (self.zscore_entrada - z_atual) - self.custo_transacao
                self.posicao = 0
            elif z_atual >= limite_stop or action == 3: # Stop Loss ou Saida Forcada
                recompensa += (self.zscore_entrada - z_atual) - self.custo_transacao
                self.posicao = 0

        # Penalidade por manter posicoes quando SWANet avisa quebra estrutural iminente
        prob_quebra = row.get("prob_quebra", 0.0)
        if self.posicao != 0:
            recompensa -= (prob_quebra * 0.05) 

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
            
        return self._get_obs(), float(recompensa), done, False, {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrada", type=Path, default=Path("data/processed/pipeline_com_quebras.csv"))
    args = parser.parse_args()

    print(f"Iniciando Módulo RL (MDP). Carregando {args.entrada}...")
    df = pd.read_csv(args.entrada).dropna()

    # Cria o ambiente
    env = TradingFronteirasMDP(df=df)

    # Inicia e treina o agente Q-Network
    # O stable-baselines3 gerencia matematicamente as Equacoes 26 (Loss) e 27/28 (Atualizacao)
    print("Treinando Deep Q-Network para encontrar os limiares operacionais ótimos...")
    modelo_dqn = DQN("MlpPolicy", env, verbose=0, learning_rate=1e-3, exploration_fraction=0.2)
    modelo_dqn.learn(total_timesteps=20000)

    print("Treinamento concluído. Testando a inteligência da rede no mesmo ambiente...")
    
    # Backtest avaliando acoes tomadas
    obs, _ = env.reset()
    recompensa_total = 0.0
    terminou = False
    
    while not terminou:
        acao, _states = modelo_dqn.predict(obs, deterministic=True)
        obs, recompensa, terminou, _, _ = env.step(acao)
        recompensa_total += recompensa

    print(f"Recompensa Total Acumulada no teste pelo Agente RL: {recompensa_total:.4f} unidades Z")
    print("Pipeline de Inteligência Artificial finalizado com sucesso.")

if __name__ == "__main__":
    main()