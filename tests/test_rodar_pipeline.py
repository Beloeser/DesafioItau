from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
CAMINHO_MODULO = RAIZ_PROJETO / "src/rodar_pipeline.py"


def carregar_modulo_rodar_pipeline():
    spec = importlib.util.spec_from_file_location("rodar_pipeline", CAMINHO_MODULO)
    modulo = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(modulo)
    return modulo


class RodarPipelineTest(unittest.TestCase):
    def test_roda_todas_as_etapas_na_ordem(self):
        rodar_pipeline = carregar_modulo_rodar_pipeline()
        chamadas = []

        def executar_etapa_falsa(nome, script):
            chamadas.append((nome, Path(script).name))

        with patch.object(sys, "argv", ["rodar_pipeline.py"]):
            with patch.object(rodar_pipeline, "executar_etapa", executar_etapa_falsa):
                rodar_pipeline.main()

        self.assertEqual(
            chamadas,
            [
                ("Checagem de cointegracao", "checar_cointegracao.py"),
                ("Cointegracao parcial e Kalman", "pipeline_cointegracao_parcial.py"),
                ("SWANet de quebras", "01_swanet_quebras.py"),
                ("Backtest DQN", "02_rl_dqn_trading.py"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
