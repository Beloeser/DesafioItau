from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"
sys.path.insert(0, str(SRC))


def carregar_modulo(nome: str, arquivo: str):
    spec = importlib.util.spec_from_file_location(nome, SRC / arquivo)
    modulo = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(modulo)
    return modulo


rl = carregar_modulo("rl_dqn", "02_rl_dqn_trading.py")
swanet = carregar_modulo("swanet", "01_swanet_quebras.py")
import calcular_rentabilidade_cdi as cdi
import periodos


def dados_sinteticos() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "data": pd.to_datetime(
                ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"],
                utc=True,
            ),
            "preco_y": [10.0, 10.0, 11.0, 12.0],
            "preco_x": [5.0, 5.0, 5.0, 5.0],
            "spread_observado": [5.0, 5.0, 6.0, 7.0],
            "hedge_ratio": [1.0, 1.0, 1.0, 1.0],
            "zscore_mr": [-2.0, -2.0, 0.0, 0.0],
            "prob_quebra": [0.1, 0.1, 0.1, 0.1],
        }
    )


class TestPeriodos(unittest.TestCase):
    def test_periodo_de_negociacao_e_primeiro_semestre(self):
        self.assertEqual(periodos.DATA_INICIO_NEGOCIACAO, "2025-01-01")
        self.assertEqual(periodos.DATA_FIM_NEGOCIACAO, "2025-06-30")


class TestAmbienteTrading(unittest.TestCase):
    def test_entrada_so_rende_no_intervalo_seguinte(self):
        env = rl.TradingFronteirasMDP(dados_sinteticos())
        env.reset(seed=42)

        _, recompensa, terminou, _, info = env.step(0)
        self.assertFalse(terminou)
        self.assertTrue(info["abriu"])
        self.assertEqual(info["posicao_intervalo"], 0)
        self.assertEqual(info["posicao_final"], 1)
        self.assertEqual(recompensa, 0.0)

        _, recompensa, _, _, info = env.step(0)
        retorno_esperado = 1.0 / 15.0
        self.assertEqual(info["posicao_intervalo"], 1)
        self.assertAlmostEqual(info["retorno_periodo"], retorno_esperado)
        self.assertAlmostEqual(recompensa, retorno_esperado * 100.0)

    def test_backtest_compoe_saldo_e_fecha_posicao(self):
        resultado = rl.executar_backtest_financeiro(
            rl.PoliticaFronteiraFixa(0),
            rl.TradingFronteirasMDP(dados_sinteticos()),
            capital_inicial=10_000.0,
            caminho_saida=None,
            semente=42,
            exibir_relatorio=False,
        )
        saldo_esperado = 10_000.0 * (1.0 + 1.0 / 15.0) * (1.0 + 1.0 / 16.0)
        self.assertAlmostEqual(resultado["saldo_final"], saldo_esperado)
        self.assertEqual(resultado["total_trades"], 1)


class TestSWANet(unittest.TestCase):
    def test_janela_de_entrada_termina_antes_do_rotulo(self):
        datas = pd.date_range("2022-01-03", periods=35, freq="B", tz="UTC")
        zscore = np.zeros(35)
        zscore[26] = 2.5
        df = pd.DataFrame({"data": datas, "zscore_mr": zscore})

        _, x_time, rotulos, datas_alinhadas, datas_fim = swanet.preparar_dados(
            df,
            seq_length=24,
            horizonte=5,
            limiar_quebra=2.0,
        )
        self.assertEqual(datas_alinhadas[0], datas[24])
        self.assertEqual(datas_fim[0], datas[28])
        self.assertEqual(float(x_time[0, -1, 0]), zscore[23])
        self.assertEqual(float(rotulos[0]), 1.0)


class TestCDI(unittest.TestCase):
    def test_rentabilidade_e_composta(self):
        dados = pd.DataFrame(
            {
                "data": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                "taxa_cdi_dia_percentual": [0.04, 0.04],
            }
        )
        resultado = cdi.calcular_rentabilidade(dados)
        self.assertAlmostEqual(
            resultado["rentabilidade_cdi_acumulada_percentual"].iloc[-1],
            0.080016,
        )


if __name__ == "__main__":
    unittest.main()
