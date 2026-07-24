"""
Módulo 1: SWANet - Predição de Quebra Estrutural (Seção 6)
Arquitetura: Continuous Wavelet Transform (Ricker) -> CNN + LSTM -> Probabilidade de Quebra.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import signal
from torch.utils.data import DataLoader, TensorDataset

class SWANet(nn.Module):
    """Arquitetura baseada em Lu et al. combinando Wavelet CNN e LSTM."""
    def __init__(self, seq_length=24):
        super(SWANet, self).__init__()
        
        # Ramo 1: CNN para as features 2D da Wavelet
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # Ramo 2: LSTM para a série temporal do spread
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        
        # Camadas totalmente conectadas fundindo CNN e LSTM
        # O tamanho linear depende do tamanho da escala wavelet. Assumindo escalas=16, seq=24
        # Apos dois MaxPool2d (diminuindo pela metade 2x), a matriz 16x24 vira 4x6. 
        # Flatten de 16 canais * 4 * 6 = 384
        self.fc = nn.Sequential(
            nn.Linear(384 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid() # Saida como probabilidade (0 a 1)
        )

    def forward(self, x_wavelet, x_time):
        out_cnn = self.cnn(x_wavelet)
        out_lstm, _ = self.lstm(x_time)
        out_lstm = out_lstm[:, -1, :] # Pega o ultimo hidden state
        
        # Concatena as features de tempo e frequencia
        out_fused = torch.cat((out_cnn, out_lstm), dim=1)
        prob_quebra = self.fc(out_fused)
        return prob_quebra

def calcular_cwt_ricker(serie: np.ndarray, escalas: np.ndarray):
    """Extrai features de tempo e frequencia usando a wavelet Ricker (Artigo Sec 6)."""
    # Usando cwt da biblioteca scipy.signal com wavelet Ricker
    matriz_cwt = signal.cwt(serie, signal.ricker, escalas)
    return matriz_cwt

def preparar_dados(df: pd.DataFrame, seq_length: int = 24):
    """Gera matrizes para a rede neural a partir do historico de Kalman."""
    spread = df["mr_filtrado"].to_numpy()
    rw = df["rw_filtrado"].to_numpy()
    n = len(spread)
    
    escalas = np.arange(1, 17) # 16 escalas de frequencia
    X_wavelet, X_time, Y_labels = [], [], []
    
    # Criacao de rotulos (Label = 1 se o Random Walk dominar o Mean Reverting no futuro)
    # Isso simula a "quebra estrutural" descrita no artigo
    rotulos = (np.abs(rw) > np.abs(spread) * 1.5).astype(np.float32)

    for i in range(seq_length, n - 5): # Prevendo 5 dias a frente
        janela = spread[i - seq_length:i]
        
        # Feature 1: CWT (CNN)
        cwt_2d = calcular_cwt_ricker(janela, escalas)
        X_wavelet.append(np.expand_dims(cwt_2d, axis=0)) # (1 canal, escalas, tempo)
        
        # Feature 2: Time Series (LSTM)
        X_time.append(janela.reshape(-1, 1))
        
        # Label: Houve quebra estrutural nos proximos 5 dias?
        Y_labels.append(np.max(rotulos[i:i+5]))
        
    return (torch.tensor(np.array(X_wavelet), dtype=torch.float32), 
            torch.tensor(np.array(X_time), dtype=torch.float32), 
            torch.tensor(np.array(Y_labels), dtype=torch.float32))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrada", type=Path, default=Path("data/processed/pipeline_cointegracao_parcial.csv"))
    parser.add_argument("--saida", type=Path, default=Path("data/processed/pipeline_com_quebras.csv"))
    args = parser.parse_args()

    print("Carregando dados do Filtro de Kalman...")
    df = pd.read_csv(args.entrada).dropna(subset=["mr_filtrado"])
    
    print("Processando Transformada Continua de Wavelet (Ricker) e Tensores...")
    seq_length = 24
    x_wav, x_time, y_labels = preparar_dados(df, seq_length=seq_length)
    
    # Simulação de Treinamento Rápido (Minimizando Cross-Entropy)
    modelo = SWANet(seq_length=seq_length)
    criterio = nn.BCELoss() # Binary Cross Entropy (conforme artigo)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
    
    print("Treinando SWANet para predicao de quebras estruturais (Cross-Entropy)...")
    modelo.train()
    for epoca in range(3): # Reduzido para fins de pipeline
        otimizador.zero_grad()
        saidas = modelo(x_wav, x_time).squeeze()
        perda = criterio(saidas, y_labels)
        perda.backward()
        otimizador.step()
        print(f"Época {epoca+1}/3 | Perda: {perda.item():.4f}")

    print("Gerando probabilidades de quebra estrutural (prob_quebra)...")
    modelo.eval()
    comprimento_valido = len(x_wav)
    with torch.no_grad():
        probabilidades = modelo(x_wav, x_time).squeeze().numpy()
    
    # Alinhando as probabilidades de volta ao DataFrame (preenchendo NaNs no inicio e fim)
    coluna_prob = np.full(len(df), np.nan)
    coluna_prob[seq_length : seq_length + comprimento_valido] = probabilidades
    coluna_prob = pd.Series(coluna_prob).ffill().bfill()
    df["prob_quebra"] = coluna_prob

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.saida, index=False)
    print(f"Probabilidades salvas. Novo CSV gerado em: {args.saida}")

if __name__ == "__main__":
    main()