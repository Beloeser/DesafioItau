"""Testes de BIAS e sanidade do modulo FinRL (diferentes dos testes de contrato).

Objetivo
--------
Os testes em ``test_finrl_env.py`` checam se o Gym / reward / split "funcionam".
ESTE arquivo procura problemas mais sutis:

1. Vazamento temporal (look-ahead) no reward do env
2. Contaminacao treino/teste (datas sobrepostas)
3. Feature ``tempo_restante`` (vies de fim de episodio)
4. Ultimo dia do ``prever_posicoes`` sempre 0 (artefato)
5. Baselines burros (flat / always-long / random) no OOS real
6. Embaralhar features deve piorar a politica (se ela aprendeu algo real)
7. Label da SWANet no limite formacao→negociacao (5 dias futuros)
8. Hedge inferido da 1a linha vs mediana (consistencia)
9. Politica quase-sempre-posicionada (alerta de perfil, nao de bug)

Como rodar
----------
# Rapido (sem treinar PPO de novo) — ~1s:
    .venv-finrl/bin/python tests/test_bias_finrl.py

# Completo (inclui treinos curtos de PPO) — ~1-2 min:
    .venv-finrl/bin/python tests/test_bias_finrl.py --completo
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

modulo = importlib.import_module("03_finrl_trading")
PairsTradingFinRLEnv = modulo.PairsTradingFinRLEnv
prever_posicoes = modulo.prever_posicoes
split_formacao_negociacao = modulo.split_formacao_negociacao

from avaliar_ganhos import carregar_pipeline, simular_ganhos  # noqa: E402

CSV_TAEE = RAIZ / "data/processed/pipeline_com_quebras_TAEE.csv"
CSV_FINRL = RAIZ / "data/processed/pipeline_finrl_TAEE.csv"
CAPITAL = 100_000.0
TAXA = 0.0008


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_sintetica(n: int = 80, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Mean-reverting-ish spread + ruido
    mr = np.zeros(n)
    for i in range(1, n):
        mr[i] = 0.85 * mr[i - 1] + rng.normal(0, 0.4)
    spread = 5.0 + mr
    return pd.DataFrame({
        "data": pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC"),
        "Y": 10.0 + spread,
        "X": np.full(n, 5.0),
        "spread_observado": spread,
        "zscore_mr": mr / (mr.std() + 1e-9),
        "prob_quebra": rng.uniform(0.1, 0.4, n),
        "sinal": np.sign(mr).astype(int),  # heuristica burra so para CSV completo
    })


def _treinar_rapido(df: pd.DataFrame, timesteps: int = 8_000, seed: int = 42):
    """PPO curto so para testes de bias (nao e o treino de producao)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = PairsTradingFinRLEnv(df, "Y", "X", 1.0, capital=CAPITAL, taxa=TAXA)
    vec = DummyVecEnv([lambda: env])
    modelo = PPO("MlpPolicy", vec, seed=seed, verbose=0, n_steps=512, batch_size=64)
    modelo.learn(total_timesteps=timesteps)
    return modelo


def _pnl_coluna(df: pd.DataFrame, col: str, y: str, x: str, hedge: float) -> float:
    _, m = simular_ganhos(df, y, x, hedge, coluna_sinal=col, capital=CAPITAL, taxa=TAXA)
    return float(m["pnl_liquido"])


# ---------------------------------------------------------------------------
# TESTES RAPIDOS (sem PPO longo)
# ---------------------------------------------------------------------------

def testa_split_sem_overlap_de_datas() -> None:
    """Treino e teste nao podem compartilhar nenhum dia."""
    df = _base_sintetica(200)
    # forca datas que cruzam 2024/2025
    df["data"] = pd.date_range("2024-06-01", periods=200, freq="B", tz="UTC")
    treino, teste = split_formacao_negociacao(
        df, "2024-06-01", "2024-12-31", "2025-01-01", "2025-12-31"
    )
    overlap = set(treino["data"]) & set(teste["data"])
    assert len(overlap) == 0, f"Overlap de {len(overlap)} datas treino/teste"
    assert treino["data"].max() < teste["data"].min()
    print("OK  bias-split: zero overlap de datas entre formacao e negociacao")


def testa_reward_nao_usa_spread_alem_de_t1() -> None:
    """Look-ahead no env: mudar o futuro distante NAO pode mudar o reward de hoje."""
    df = _base_sintetica(40, seed=1)
    env = PairsTradingFinRLEnv(df, "Y", "X", 1.0, taxa=0.0)
    env.reset()
    _, r1, _, _, _ = env.step(1)

    df2 = df.copy()
    # Choque so a partir do dia 10 (bem depois do passo 0->1)
    df2.loc[10:, "spread_observado"] = df2.loc[10:, "spread_observado"] + 50.0
    df2["Y"] = 10.0 + df2["spread_observado"]
    env2 = PairsTradingFinRLEnv(df2, "Y", "X", 1.0, taxa=0.0)
    env2.reset()
    _, r2, _, _, _ = env2.step(1)

    assert abs(r1 - r2) < 1e-12, (r1, r2)
    print("OK  bias-lookahead-env: reward do dia 0 ignora choque no futuro distante")


def testa_reward_MUDA_se_altera_t1() -> None:
    """Sanidade: se eu mudar spread[t+1], o reward de t TEM que mudar."""
    df = _base_sintetica(40, seed=2)
    env = PairsTradingFinRLEnv(df, "Y", "X", 1.0, taxa=0.0)
    env.reset()
    _, r1, _, _, _ = env.step(1)

    df2 = df.copy()
    df2.loc[1, "spread_observado"] = float(df2.loc[1, "spread_observado"]) + 3.0
    df2.loc[1, "Y"] = 10.0 + df2.loc[1, "spread_observado"]
    env2 = PairsTradingFinRLEnv(df2, "Y", "X", 1.0, taxa=0.0)
    env2.reset()
    _, r2, _, _, _ = env2.step(1)

    assert abs(r1 - r2) > 1e-9
    print("OK  bias-lookahead-env: reward do dia 0 depende corretamente de t->t+1")


def testa_artefato_ultimo_dia_preenchido() -> None:
    """Apos o fix, always-long preenche TAMBEM o ultimo dia (nao fica 0 artificial)."""
    df = _base_sintetica(30)

    class PoliticaSempreLong:
        def predict(self, obs, deterministic=True):
            return 1, None

    env = PairsTradingFinRLEnv(df, "Y", "X", 1.0, taxa=0.0)
    pos = prever_posicoes(PoliticaSempreLong(), env)
    assert (pos == 1).all(), f"Esperava todos +1 apos fix do ultimo dia; got tail={pos[-3:]}"
    print("OK  bias-artefato: ultimo dia tambem recebe a acao prevista (fix aplicado)")


def testa_baselines_burros_no_oos_real() -> None:
    """Compara FinRL salvo vs flat / always-long / random no mesmo OOS.

    Nao falha se FinRL perder (pode acontecer). Reporta e alerta se always-long
    empatar/ganhar — sinal de que o PnL veio de tendencia, nao de timing.
    """
    if not CSV_FINRL.exists() or not CSV_TAEE.exists():
        print("SKIP baselines OOS (CSV FinRL/TAEE ausente)")
        return

    df_f, y, x, hedge = carregar_pipeline(CSV_FINRL)
    n = len(df_f)
    rng = np.random.default_rng(0)

    df_f = df_f.copy()
    df_f["flat"] = 0
    df_f["always_long"] = 1
    df_f["always_short"] = -1
    df_f["random"] = rng.choice([-1, 0, 1], size=n)

    pnl_finrl = _pnl_coluna(df_f, "sinal_finrl", y, x, hedge)
    pnl_flat = _pnl_coluna(df_f, "flat", y, x, hedge)
    pnl_long = _pnl_coluna(df_f, "always_long", y, x, hedge)
    pnl_short = _pnl_coluna(df_f, "always_short", y, x, hedge)
    pnl_rand = _pnl_coluna(df_f, "random", y, x, hedge)

    pos_pct = float((df_f["sinal_finrl"] != 0).mean() * 100)

    print("=" * 64)
    print("BASELINES BURROS no OOS (mesmo CSV FinRL, 8 bps)")
    print("=" * 64)
    print(f"  FinRL salvo     : R$ {pnl_finrl:,.2f} | posicionado {pos_pct:.1f}%")
    print(f"  Always LONG     : R$ {pnl_long:,.2f}")
    print(f"  Always SHORT    : R$ {pnl_short:,.2f}")
    print(f"  Random -1/0/+1  : R$ {pnl_rand:,.2f}")
    print(f"  Flat (0)        : R$ {pnl_flat:,.2f}")

    assert pnl_flat == 0.0
    # Alerta (nao assert): se always-long >= 90% do FinRL, o alpha e quase tendencia
    if pnl_long > 0 and pnl_finrl > 0 and pnl_long >= 0.9 * pnl_finrl:
        print(
            "ALERTA  always-long captura >=90% do PnL do FinRL → "
            "ganho pode ser tendencia do spread, nao timing fino"
        )
    else:
        print("OK  bias-baseline: FinRL nao e apenas um always-long disfarçado")

    if pos_pct > 90:
        print(
            f"ALERTA  perfil agressivo: {pos_pct:.1f}% dos dias posicionados "
            "(Luiz baseline costuma ficar ~15-20%)"
        )


def testa_hedge_consistente_no_csv() -> None:
    """Hedge inferido da 1a linha deve ser quase constante ao longo do CSV."""
    if not CSV_TAEE.exists():
        print("SKIP hedge (CSV ausente)")
        return
    df, y, x, hedge0 = carregar_pipeline(CSV_TAEE)
    hedges = (df[y] - df["spread_observado"]) / df[x]
    # tolerancia larga: ruido numerico / arredondamento CSV
    desvio = float((hedges - hedge0).abs().max())
    print(f"  hedge linha0={hedge0:.6f} | max |hedge_t - hedge0| = {desvio:.6e}")
    assert desvio < 1e-6, "spread nao e Y - hedge*X com hedge constante"
    print("OK  bias-hedge: hedge constante ao longo da serie")


def testa_swanet_label_cruza_fronteira() -> None:
    """Diagnostico: janelas cuja label +5d tocaria o OOS.

    Apos o fix em 01_swanet_quebras.py essas janelas sao EXCLUIDAS do treino.
    Elas ainda podem existir no CSV (previsao), mas nao entram no gradient.
    """
    if not CSV_TAEE.exists():
        print("SKIP swanet frontier (CSV ausente)")
        return
    df = pd.read_csv(CSV_TAEE)
    df["data"] = pd.to_datetime(df["data"], utc=True)
    fim_form = pd.Timestamp("2024-12-31", tz="UTC")
    seq = 24
    cruzam = 0
    for i in range(seq, len(df) - 5):
        if df["data"].iloc[i] <= fim_form and df["data"].iloc[i + 4] > fim_form:
            cruzam += 1
    print(f"  Janelas na fronteira formacao→negociacao (label +5d): {cruzam}")
    print(
        "  Nota: o treino SWANet agora remove essas janelas do mask "
        "(ver src/01_swanet_quebras.py). Use pipeline_com_quebras_TAEE_causal.csv "
        "apos retreino."
    )
    assert cruzam < 20, "cruzamento anormalmente grande — revisar datas do CSV"
    print("OK  SWANet frontier diagnosticado (treino causal apos fix)")


def testa_swanet_mask_treino_exclui_fronteira() -> None:
    """Garante no codigo que o mask de treino exige data[i+4] <= fim_formacao."""
    src = (RAIZ / "src/01_swanet_quebras.py").read_text(encoding="utf-8")
    assert "fim_label_ok" in src
    assert "i + 4" in src or "i+4" in src
    assert "cruzava o fim da formacao" in src or "fim_label_ok" in src
    print("OK  codigo SWANet contem exclusao causal do label +5d no treino")


def testa_obs_nao_contem_dspread_futuro() -> None:
    """Observacao so tem zscore, posicao, prob, tempo — nunca o retorno futuro."""
    df = _base_sintetica(25)
    env = PairsTradingFinRLEnv(df, "Y", "X", 1.0)
    obs, _ = env.reset()
    assert obs.shape == (4,)
    # Forca zscore conhecido e verifica que obs[0] == zscore, nao dspread
    assert abs(float(obs[0]) - float(df["zscore_mr"].iloc[0])) < 1e-6
    assert abs(float(obs[2]) - float(df["prob_quebra"].iloc[0])) < 1e-6
    print("OK  bias-obs: observacao nao expoe dspread/retorno futuro")


# ---------------------------------------------------------------------------
# TESTES COMPLETOS (treinam PPO curto)
# ---------------------------------------------------------------------------

def testa_tempo_restante_e_atalho() -> None:
    """Com e sem tempo_restante: se PnL mudar muito, a politica usa o relogio do episodio.

    Implementacao: treina num env normal; avalia num env onde tempo_restante e
    constante (0.5). Se a politica depender do relogio, o comportamento muda.
    """
    df = _base_sintetica(120, seed=9)
    treino = df.iloc[:80].reset_index(drop=True)
    teste = df.iloc[80:].reset_index(drop=True)

    modelo = _treinar_rapido(treino, timesteps=6_000, seed=11)

    env_a = PairsTradingFinRLEnv(teste.copy(), "Y", "X", 1.0, taxa=TAXA)
    pos_a = prever_posicoes(modelo, env_a)

    # Env com tempo congelado: monkeypatch _obs
    env_b = PairsTradingFinRLEnv(teste.copy(), "Y", "X", 1.0, taxa=TAXA)
    def _obs_tempo_fixo(self=env_b):
        return np.array(
            [self.zscore[self.i], self.posicao, self.prob_quebra[self.i], 0.5],
            dtype=np.float32,
        )
    env_b._obs = _obs_tempo_fixo  # type: ignore[method-assign]
    pos_b = prever_posicoes(modelo, env_b)

    iguais = float((pos_a == pos_b).mean() * 100)
    print(f"  Acordo de acoes com tempo_restante real vs fixo=0.5: {iguais:.1f}%")
    if iguais < 85:
        print(
            "ALERTA  tempo_restante: politica muda bastante sem o relogio do episodio "
            "→ risco de vies de fim de episodio (nao existe no pregão real infinito)"
        )
    else:
        print("OK  bias-tempo: politica pouco sensivel ao tempo_restante neste treino curto")


def testa_embaralhar_zscore_piora_treino() -> None:
    """Se embaralharmos zscore no TREINO, a politica deve ir pior no teste sintetico.

    Se embaralhar NAO piorar, a rede pode estar ignorando zscore (ex.: so segue tendencia
    via posicao/custo) — alerta importante.
    """
    df = _base_sintetica(150, seed=21)
    treino = df.iloc[:100].reset_index(drop=True)
    teste = df.iloc[100:].reset_index(drop=True)

    modelo_ok = _treinar_rapido(treino, timesteps=8_000, seed=3)
    env_ok = PairsTradingFinRLEnv(teste.copy(), "Y", "X", 1.0, taxa=TAXA)
    pos_ok = prever_posicoes(modelo_ok, env_ok)
    d_ok = teste.copy()
    d_ok["sinal_finrl"] = pos_ok
    pnl_ok = _pnl_coluna(d_ok, "sinal_finrl", "Y", "X", 1.0)

    treino_shuf = treino.copy()
    rng = np.random.default_rng(99)
    treino_shuf["zscore_mr"] = rng.permutation(treino_shuf["zscore_mr"].to_numpy())
    modelo_bad = _treinar_rapido(treino_shuf, timesteps=8_000, seed=3)
    env_bad = PairsTradingFinRLEnv(teste.copy(), "Y", "X", 1.0, taxa=TAXA)
    pos_bad = prever_posicoes(modelo_bad, env_bad)
    d_bad = teste.copy()
    d_bad["sinal_finrl"] = pos_bad
    pnl_bad = _pnl_coluna(d_bad, "sinal_finrl", "Y", "X", 1.0)

    print(f"  PnL teste c/ zscore real no treino     : R$ {pnl_ok:,.2f}")
    print(f"  PnL teste c/ zscore embaralhado treino : R$ {pnl_bad:,.2f}")
    if pnl_ok + 500 < pnl_bad:
        print(
            "ALERTA  embaralhar zscore NAO piorou → rede pode nao estar usando zscore "
            "(talvez so hold/tendencia)"
        )
    else:
        print("OK  bias-feature: zscore real no treino nao ficou pior que o embaralhado")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Testes de bias do FinRL")
    parser.add_argument(
        "--completo",
        action="store_true",
        help="Inclui treinos PPO curtos (mais lento, ~1-2 min).",
    )
    args = parser.parse_args()

    print("\n>>> TESTES RAPIDOS DE BIAS (FinRL)\n")
    testa_split_sem_overlap_de_datas()
    testa_reward_nao_usa_spread_alem_de_t1()
    testa_reward_MUDA_se_altera_t1()
    testa_artefato_ultimo_dia_preenchido()
    testa_obs_nao_contem_dspread_futuro()
    testa_hedge_consistente_no_csv()
    testa_swanet_label_cruza_fronteira()
    testa_swanet_mask_treino_exclui_fronteira()
    testa_baselines_burros_no_oos_real()

    if args.completo:
        print("\n>>> TESTES COMPLETOS (treino PPO curto)\n")
        testa_tempo_restante_e_atalho()
        testa_embaralhar_zscore_piora_treino()
    else:
        print("\n(Pulei treinos PPO. Rode com --completo para tempo_restante + shuffle zscore.)\n")

    print("\ntest_bias_finrl: suite finalizada (veja ALERTA acima se houver)\n")


if __name__ == "__main__":
    main()
