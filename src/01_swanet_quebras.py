"""
Módulo 1 (Corrigido): SWANet - Predição de Quebra Estrutural
Garante que o treinamento ocorra estritamente no período de formação (ex: 24-25)
e gera as probabilidades para o período de negociação (ex: 2026).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from periodos import DATA_FIM_FORMACAO, DATA_INICIO_FORMACAO

SEMENTE = 42
EPOCAS_PADRAO = 50
BATCH_SIZE_PADRAO = 64
PACIENCIA_PADRAO = 8
FRACAO_VALIDACAO = 0.2

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
            nn.Linear(64, 1)
        )

    def forward(self, x_wavelet, x_time):
        out_cnn = self.cnn(x_wavelet)
        out_lstm, _ = self.lstm(x_time)
        out_lstm = out_lstm[:, -1, :] 
        out_fused = torch.cat((out_cnn, out_lstm), dim=1)
        return self.fc(out_fused)

def ricker_wavelet(pontos: int, escala: float) -> np.ndarray:
    """Gera a wavelet Ricker, equivalente ao antigo scipy.signal.ricker."""
    centro = (pontos - 1) / 2
    x = np.arange(pontos) - centro
    a2 = escala ** 2
    normalizacao = 2 / (np.sqrt(3 * escala) * np.pi ** 0.25)
    return normalizacao * (1 - x ** 2 / a2) * np.exp(-(x ** 2) / (2 * a2))


def calcular_cwt_ricker(serie: np.ndarray, escalas: np.ndarray):
    matriz_cwt = []
    for escala in escalas:
        pontos = min(int(10 * escala), len(serie))
        wavelet = ricker_wavelet(pontos, escala)
        coeficientes = np.convolve(serie, wavelet[::-1], mode="same")
        matriz_cwt.append(coeficientes)
    return np.array(matriz_cwt)

def preparar_dados(df: pd.DataFrame, seq_length: int = 24,
                   dt_fim_formacao: pd.Timestamp | None = None):
    """Prepara tensores de entrada e labels para a SWANet.

    CORREÇÃO Bug #3: Os labels olham 5 dias à frente. Para amostras cujos
    5 dias seguintes ultrapassam o fim do período de formação, o label
    recebe NaN (será excluído do treino), evitando vazamento de dados
    do período de negociação para o treinamento.
    """
    spread = df["mr_filtrado"].to_numpy()
    rw = df["rw_filtrado"].to_numpy()
    datas_df = df["data"].to_numpy()
    n = len(spread)

    escalas = np.arange(1, 17)
    X_wavelet, X_time, Y_labels, datas_alinhadas = [], [], [], []

    # Rótulo instantâneo: "O RW engoliu o MR neste dia?"
    rotulos = (np.abs(rw) > np.abs(spread) * 1.5).astype(np.float32)

    for i in range(seq_length, n - 5):
        janela = spread[i - seq_length:i]

        cwt_2d = calcular_cwt_ricker(janela, escalas)
        X_wavelet.append(np.expand_dims(cwt_2d, axis=0))
        X_time.append(janela.reshape(-1, 1))

        # CORREÇÃO Bug #3: Se a janela de label [i, i+5) ultrapassa o fim
        # da formação, marca como NaN para que a amostra seja excluída do
        # treino (evita que labels do período de negociação treinem o modelo).
        if dt_fim_formacao is not None and datas_df[i + 4] > dt_fim_formacao:
            Y_labels.append(np.nan)
        else:
            Y_labels.append(np.max(rotulos[i:i+5]))

        datas_alinhadas.append(datas_df[i])

    return (torch.tensor(np.array(X_wavelet), dtype=torch.float32),
            torch.tensor(np.array(X_time), dtype=torch.float32),
            torch.tensor(np.array(Y_labels, dtype=np.float32)),
            np.array(datas_alinhadas))


def dividir_treino_validacao(
    x_wav: torch.Tensor,
    x_time: torch.Tensor,
    y_labels: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    num_amostras = len(y_labels)
    num_validacao = int(num_amostras * FRACAO_VALIDACAO)

    if num_validacao < 10 or (num_amostras - num_validacao) < 10:
        return (x_wav, x_time, y_labels), None

    corte = num_amostras - num_validacao
    treino = (x_wav[:corte], x_time[:corte], y_labels[:corte])
    validacao = (x_wav[corte:], x_time[corte:], y_labels[corte:])
    return treino, validacao


def calcular_pos_weight(y_labels: torch.Tensor) -> torch.Tensor:
    positivos = float(y_labels.sum().item())
    negativos = float(len(y_labels) - positivos)
    if positivos <= 0 or negativos <= 0:
        return torch.tensor([1.0], dtype=torch.float32)
    return torch.tensor([negativos / positivos], dtype=torch.float32)


def avaliar_loss(
    modelo: SWANet,
    criterio: nn.Module,
    x_wav: torch.Tensor,
    x_time: torch.Tensor,
    y_labels: torch.Tensor,
) -> float:
    modelo.eval()
    with torch.no_grad():
        logits = modelo(x_wav, x_time).view(-1)
        perda = criterio(logits, y_labels)
    return float(perda.item())


def treinar_swanet(
    modelo: SWANet,
    x_wav: torch.Tensor,
    x_time: torch.Tensor,
    y_labels: torch.Tensor,
    epocas: int,
    batch_size: int,
    paciencia: int,
) -> None:
    (x_wav_treino, x_time_treino, y_treino), validacao = dividir_treino_validacao(
        x_wav, x_time, y_labels
    )
    criterio = nn.BCEWithLogitsLoss(pos_weight=calcular_pos_weight(y_treino))
    otimizador = torch.optim.Adam(modelo.parameters(), lr=0.001, weight_decay=1e-4)

    dataset = TensorDataset(x_wav_treino, x_time_treino, y_treino)
    gerador = torch.Generator().manual_seed(SEMENTE)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=gerador,
    )

    melhor_estado = None
    melhor_loss = float("inf")
    epocas_sem_melhora = 0

    for epoca in range(1, epocas + 1):
        modelo.train()
        perdas_batch = []

        for lote_wav, lote_time, lote_y in loader:
            otimizador.zero_grad()
            logits = modelo(lote_wav, lote_time).view(-1)
            perda = criterio(logits, lote_y)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()
            perdas_batch.append(float(perda.item()))

        loss_treino = float(np.mean(perdas_batch))
        if validacao is None:
            loss_monitorado = loss_treino
            sufixo = ""
        else:
            loss_validacao = avaliar_loss(modelo, criterio, *validacao)
            loss_monitorado = loss_validacao
            sufixo = f" | Val: {loss_validacao:.4f}"

        print(f"Época {epoca}/{epocas} | Treino: {loss_treino:.4f}{sufixo}")

        if loss_monitorado < melhor_loss - 1e-4:
            melhor_loss = loss_monitorado
            melhor_estado = {
                chave: valor.detach().clone()
                for chave, valor in modelo.state_dict().items()
            }
            epocas_sem_melhora = 0
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= paciencia:
                print(f"Parada antecipada: sem melhora por {paciencia} épocas.")
                break

    if melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrada", type=Path, default=Path("data/processed/pipeline_cointegracao_parcial.csv"))
    parser.add_argument("--saida", type=Path, default=Path("data/processed/pipeline_com_quebras.csv"))
    parser.add_argument(
        "--inicio-formacao",
        type=str,
        default=DATA_INICIO_FORMACAO,
        help="Inicio do periodo usado para treinar a SWANet.",
    )
    parser.add_argument(
        "--fim-formacao",
        type=str,
        default=DATA_FIM_FORMACAO,
        help="Fim do periodo usado para treinar a SWANet.",
    )
    parser.add_argument("--epocas", type=int, default=EPOCAS_PADRAO)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_PADRAO)
    parser.add_argument("--paciencia", type=int, default=PACIENCIA_PADRAO)
    args = parser.parse_args()

    torch.manual_seed(SEMENTE)
    np.random.seed(SEMENTE)

    print(f"Carregando {args.entrada}...")
    df = pd.read_csv(args.entrada).dropna(subset=["mr_filtrado"])
    df["data"] = pd.to_datetime(df["data"], utc=True)
    
    print("Processando Transformada de Wavelet e LSTMs...")
    seq_length = 24
    dt_inicio_formacao = pd.to_datetime(args.inicio_formacao).tz_localize("UTC")
    dt_fim_formacao = pd.to_datetime(args.fim_formacao).tz_localize("UTC")

    # CORREÇÃO Bug #3: Passa dt_fim_formacao para que labels que olham 5 dias
    # além do fim da formação recebam NaN (serão excluídos do treino).
    x_wav, x_time, y_labels, datas = preparar_dados(
        df, seq_length=seq_length, dt_fim_formacao=dt_fim_formacao
    )

    # --- ISOLAMENTO DO TREINAMENTO (IMPEDE LOOK-AHEAD BIAS) ---
    mask_treino = (
        (datas >= dt_inicio_formacao)
        & (datas <= dt_fim_formacao)
        & ~np.isnan(y_labels.numpy())  # Exclui amostras com label vazado
    )
    if mask_treino.sum() == 0:
        raise ValueError(
            "A SWANet ficou sem dados de treino. Verifique se o arquivo de entrada "
            "contem o mesmo periodo de formacao usado na cointegracao."
        )

    x_wav_treino = x_wav[mask_treino]
    x_time_treino = x_time[mask_treino]
    y_labels_treino = y_labels[mask_treino]
    positivos = int(y_labels_treino.sum().item())
    total_treino = len(y_labels_treino)
    taxa_positiva = positivos / total_treino

    print(
        f"Rotulos treino: {positivos}/{total_treino} positivos "
        f"({taxa_positiva:.2%})."
    )

    if positivos == 0 or positivos == total_treino:
        print(
            "A SWANet recebeu apenas uma classe no treino; "
            "pulando treino neural e usando a taxa-base constante."
        )
        df["prob_quebra"] = taxa_positiva
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.saida, index=False)
        print(f"Pipeline de predicao concluida e salva em: {args.saida}")
        return
    
    print(
        "Treinando SWANet apenas com dados passados "
        f"({len(x_wav_treino)} dias de formacao)..."
    )
    print(
        f"Config treino: epocas={args.epocas}, batch_size={args.batch_size}, "
        f"shuffle=True, paciencia={args.paciencia}"
    )
    modelo = SWANet(seq_length=seq_length)
    treinar_swanet(
        modelo,
        x_wav_treino,
        x_time_treino,
        y_labels_treino,
        epocas=args.epocas,
        batch_size=args.batch_size,
        paciencia=args.paciencia,
    )

    # --- PREVISÃO (OUT-OF-SAMPLE) ---
    print("Treinamento finalizado. Gerando previsoes para o periodo de teste...")
    modelo.eval()
    with torch.no_grad():
        # Prevemos para o vetor todo, pois o CSV final precisa de todas as datas
        probabilidades = torch.sigmoid(modelo(x_wav, x_time).view(-1)).numpy()
    
    # Realinha as probabilidades de volta ao DataFrame original
    coluna_prob = np.full(len(df), np.nan)
    coluna_prob[seq_length : seq_length + len(probabilidades)] = probabilidades
    # CORREÇÃO Bug #5: Nunca bfill — pode propagar probabilidades futuras.
    # Valor 0.5 (neutro) para posições sem previsão.
    df["prob_quebra"] = pd.Series(coluna_prob, index=df.index).ffill().fillna(0.5)

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.saida, index=False)
    print(f"Pipeline de predicao concluida e salva em: {args.saida}")

if __name__ == "__main__":
    main()
