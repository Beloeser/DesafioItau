"""Executa o pipeline completo sem coletar dados novamente.

Ordem das etapas:
1. Checagem de cointegracao
2. Pipeline de cointegracao parcial e Kalman
3. SWANet para probabilidade de quebra
4. Rentabilidade do CDI
5. Backtest DQN
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ETAPAS = [
    ("Checagem de cointegracao", "checar_cointegracao.py"),
    ("Cointegracao parcial e Kalman", "pipeline_cointegracao_parcial.py"),
    ("SWANet de quebras", "01_swanet_quebras.py"),
    ("Rentabilidade do CDI", "calcular_rentabilidade_cdi.py"),
    ("Backtest DQN", "02_rl_dqn_trading.py"),
]


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roda todo o pipeline, exceto a coleta de dados."
    )
    parser.add_argument(
        "--pular-cdi",
        action="store_true",
        help="Pula a atualizacao do CDI se o arquivo local ja existir.",
    )
    return parser.parse_args()


def executar_etapa(nome: str, script: Path) -> None:
    print("\n" + "=" * 80, flush=True)
    print(f"ETAPA: {nome}", flush=True)
    print("=" * 80, flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    resultado = subprocess.run([sys.executable, str(script)], check=False, env=env)
    if resultado.returncode != 0:
        raise SystemExit(
            f"\nPipeline interrompido: a etapa '{nome}' falhou "
            f"com codigo {resultado.returncode}."
        )


def main() -> None:
    args = argumentos()
    raiz_projeto = Path(__file__).resolve().parents[1]
    diretorio_src = raiz_projeto / "src"
    caminho_cdi = raiz_projeto / "data/raw/cdi/rentabilidade_cdi.csv"

    print("Rodando pipeline completo sem coletar dados.", flush=True)
    print(f"Python usado: {sys.executable}", flush=True)

    for nome, arquivo in ETAPAS:
        if args.pular_cdi and arquivo == "calcular_rentabilidade_cdi.py":
            if caminho_cdi.exists():
                print("\nPulando CDI: arquivo local ja existe.", flush=True)
                continue
            print("\nArquivo de CDI nao encontrado; calculando CDI mesmo assim.", flush=True)

        executar_etapa(nome, diretorio_src / arquivo)

    print("\n" + "=" * 80, flush=True)
    print("Pipeline finalizado com sucesso.", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
