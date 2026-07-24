"""
Módulo 1 (Corrigido): SWANet - Predição de Quebra Estrutural
Garante que o treinamento ocorra estritamente no período de formação (ex: 24-25)
e gera as probabilidades para o período de negociação (ex: 2026).
"""
1
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import signal

class SWANet(nn.Module):
    def __init__(self, seq_length=24):
        super(SWANet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(384 + 16, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid() 
        )

    def forward(self, x_wavelet, x_time):
        out_cnn = self.cnn(x_wavelet)
        out_lstm, _ = self.lstm(x_time)
        out_lstm = out_lstm[:, -1, :] 
        out_fused = torch.cat((out_cnn, out_lstm), dim=1)
        return self.fc(out_fused)

def calcular_cwt_ricker(serie: np.ndarray, escalas: np.ndarray):
    return signal.cwt(serie, signal.ricker, escalas)

def preparar_dados(df: pd.DataFrame, seq_length: int = 24):
    spread = df["mr_filtrado"].to_numpy()
    rw = df["rw_filtrado"].to_numpy()
    n = len(spread)
    
    escalas = np.arange(1, 17)
    X_wavelet, X_time, Y_labels, datas_alinhadas = [], [], [], []
    
    # Label: "O Random Walk engoliu o Mean Reverting nos próximos 5 dias?"
    rotulos = (np.abs(rw) > np.abs(spread) * 1.5).astype(np.float32)

    for i in range(seq_length, n - 5):
        janela = spread[i - seq_length:i]
        
        cwt_2d = calcular_cwt_ricker(janela, escalas)
        X_wavelet.append(np.expand_dims(cwt_2d, axis=0))
        X_time.append(janela.reshape(-1, 1))
        Y_labels.append(np.max(rotulos[i:i+5]))
        datas_alinhadas.append(df["data"].iloc[i]) # Rastreia a data do tensor
        
    return (torch.tensor(np.array(X_wavelet), dtype=torch.float32), 
            torch.tensor(np.array(X_time), dtype=torch.float32), 
            torch.tensor(np.array(Y_labels), dtype=torch.float32),
            np.array(datas_alinhadas))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrada", type=Path, default=Path("data/processed/pipeline_cointegracao_parcial.csv"))
    parser.add_argument("--saida", type=Path, default=Path("data/processed/pipeline_com_quebras.csv"))
    parser.add_argument("--corte-teste", type=str, default="2026-01-01", help="Data limite do Treinamento.")
    args = parser.parse_args()

    print(f"Carregando {args.entrada}...")
    df = pd.read_csv(args.entrada).dropna(subset=["mr_filtrado"])
    df["data"] = pd.to_datetime(df["data"], utc=True)
    
    print("Processando Transformada de Wavelet e LSTMs...")
    seq_length = 24
    x_wav, x_time, y_labels, datas = preparar_dados(df, seq_length=seq_length)
    
    # --- ISOLAMENTO DO TREINAMENTO (IMPEDE LOOK-AHEAD BIAS) ---
    dt_corte = pd.to_datetime(args.corte_teste).tz_localize('UTC')
    mask_treino = datas < dt_corte
    
    x_wav_treino = x_wav[mask_treino]
    x_time_treino = x_time[mask_treino]
    y_labels_treino = y_labels[mask_treino]
    
    print(f"Treinando SWANet apenas com dados passados ({len(x_wav_treino)} dias de formacao)...")
    modelo = SWANet(seq_length=seq_length)
    criterio = nn.BCELoss() 
    otimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
    
    modelo.train()
    for epoca in range(5): 
        otimizador.zero_grad()
        saidas = modelo(x_wav_treino, x_time_treino).squeeze()
        perda = criterio(saidas, y_labels_treino)
        perda.backward()
        otimizador.step()
        print(f"Época {epoca+1}/5 | Perda (Loss): {perda.item():.4f}")

    # --- PREVISÃO (OUT-OF-SAMPLE) ---
    print("Treinamento finalizado. Gerando previsoes para o periodo de teste...")
    modelo.eval()
    with torch.no_grad():
        # Prevemos para o vetor todo, pois o CSV final precisa de todas as datas
        probabilidades = modelo(x_wav, x_time).squeeze().numpy()
    
    # Realinha as probabilidades de volta ao DataFrame original
    coluna_prob = np.full(len(df), np.nan)
    coluna_prob[seq_length : seq_length + len(probabilidades)] = probabilidades
    df["prob_quebra"] = pd.Series(coluna_prob).ffill().bfill()

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.saida, index=False)
    print(f"Pipeline de predicao concluida e salva em: {args.saida}")

if __name__ == "__main__":
    main()