"""Executa o pipeline completo sem coletar dados novamente."""

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
    ("Backtest DQN", "02_rl_dqn_trading.py"),
]


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roda todo o pipeline, exceto a coleta de dados."
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
    argumentos()
    raiz_projeto = Path(__file__).resolve().parents[1]
    diretorio_src = raiz_projeto / "src"

    print("Rodando pipeline completo sem coletar dados.", flush=True)
    print(f"Python usado: {sys.executable}", flush=True)

    for nome, arquivo in ETAPAS:
        executar_etapa(nome, diretorio_src / arquivo)

    print("\n" + "=" * 80, flush=True)
    print("Pipeline finalizado com sucesso.", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
