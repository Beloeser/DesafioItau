"""FinRL hibrido HONESTO: Luiz decide; RL so pode filtrar (com info de ontem).

Ideia
-----
Nao substituir o Luiz por um PPO agressivo com informacao do mesmo fechamento.
Em vez disso:

  sinal_final = 0          se o RL escolher "ficar fora"
  sinal_final = sinal_luiz se o RL escolher "seguir o Luiz"

O RL ve APENAS informacao de ontem (features_lag=1):
  [zscore_ontem, posicao_atual, prob_quebra_ontem, sinal_luiz_ontem]

Assim:
  - o cerebro principal continua sendo a regra Luiz;
  - o RL so tenta melhorar um pouco (evitar trades ruins);
  - nao ha privilegio de operar no mesmo close que acabou de ver.

Uso:
    .venv-finrl/bin/python src/03_finrl_hibrido.py \\
      --entrada data/processed/pipeline_com_quebras_TAEE_causal.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from avaliar_ganhos import carregar_pipeline, simular_ganhos
from execucao_pnl import carregar_spread_abertura, pnl_transicao_abertura
from periodos import (
    DATA_FIM_FORMACAO,
    DATA_FIM_NEGOCIACAO,
    DATA_INICIO_FORMACAO,
    DATA_INICIO_NEGOCIACAO,
)


class HibridoLuizFinRLEnv(gym.Env):
    """2 acoes: 0=forcar flat, 1=seguir sinal do Luiz."""

    def __init__(
        self,
        df: pd.DataFrame,
        ativo_y: str,
        ativo_x: str,
        hedge: float,
        capital: float = 100_000.0,
        taxa: float = 0.0008,
        holding_cost: float = 0.0,
        execucao: str = "abertura",
        dados_setores: Path | None = None,
    ):
        super().__init__()
        if "sinal" not in df.columns:
            raise ValueError("CSV precisa da coluna 'sinal' (baseline Luiz).")

        self.df = df.reset_index(drop=True)
        self.capital = float(capital)
        self.taxa = float(taxa)
        self.holding_cost = float(holding_cost)
        self.execucao = execucao

        preco_y = self.df[ativo_y].astype(float).to_numpy()
        preco_x = self.df[ativo_x].astype(float).to_numpy()
        self.spread_close = self.df["spread_observado"].astype(float).to_numpy()

        self.n_y = (self.capital / 2.0) / max(float(preco_y[0]), 1e-9)
        self.n_x = abs(float(hedge)) * self.n_y
        self.notional = self.n_y * preco_y + self.n_x * preco_x
        self.dspread = np.diff(self.spread_close)
        if execucao == "abertura":
            self.spread_open = carregar_spread_abertura(
                self.df, ativo_y, ativo_x, hedge, dados_setores=dados_setores
            )
        else:
            self.spread_open = None

        # --- so informacao de ONTEM (lag=1) ---
        z = self.df["zscore_mr"].astype(float).to_numpy()
        p = (
            self.df["prob_quebra"].astype(float).to_numpy()
            if "prob_quebra" in self.df.columns
            else np.full(len(self.df), 0.5)
        )
        s = self.df["sinal"].fillna(0).astype(float).to_numpy()
        self.z_lag = np.concatenate([[z[0]], z[:-1]])
        self.p_lag = np.concatenate([[p[0]], p[:-1]])
        self.sinal_luiz_lag = np.concatenate([[s[0]], s[:-1]])
        # sinal Luiz "de hoje" so e usado DEPOIS da acao do RL para montar a posicao,
        # mas a OBSERVACAO nao ve o sinal de hoje — ve o de ontem.
        # Para seguir o Luiz no dia t de forma operacional realista:
        # usamos o sinal Luiz ja conhecido no fechamento de ontem (lag),
        # alinhado com "decidir com info de ontem".
        self.sinal_luiz_exec = self.sinal_luiz_lag.astype(int)

        self.action_space = spaces.Discrete(2)  # 0=flat, 1=seguir Luiz
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.i = 0
        self.posicao = 0

    def _obs(self) -> np.ndarray:
        tempo = (len(self.df) - 1 - self.i) / max(len(self.df) - 1, 1)
        return np.array(
            [
                self.z_lag[self.i],
                float(self.posicao),
                self.p_lag[self.i],
                float(self.sinal_luiz_lag[self.i]),
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.i = 0
        self.posicao = 0
        return self._obs(), {}

    def _pnl_dia(self, alvo: int) -> float:
        t_next = self.i + 1
        if t_next >= len(self.spread_close):
            return 0.0
        if self.execucao == "abertura":
            assert self.spread_open is not None
            return pnl_transicao_abertura(
                t_next, self.posicao, alvo, self.spread_close, self.spread_open, self.n_y
            )
        return float(alvo * self.n_y * self.dspread[self.i])

    def step(self, action):
        # 0 -> flat; 1 -> copia o Luiz (com info ja conhecida / lag)
        alvo = 0 if int(action) == 0 else int(self.sinal_luiz_exec[self.i])
        custo_giro = self.taxa * self.notional[self.i] * abs(alvo - self.posicao)
        custo_hold = self.holding_cost * self.notional[self.i] * abs(alvo)
        pnl = self._pnl_dia(alvo)
        recompensa = (pnl - custo_giro - custo_hold) / self.capital * 100.0
        self.posicao = alvo
        self.i += 1
        terminado = self.i >= len(self.df) - 1
        return self._obs(), float(recompensa), terminado, False, {"pos": alvo}


def treinar(env: HibridoLuizFinRLEnv, timesteps: int, seed: int = 42):
    vec = DummyVecEnv([lambda: env])
    modelo = PPO(
        "MlpPolicy",
        vec,
        seed=seed,
        verbose=0,
        policy_kwargs={"net_arch": dict(pi=[32], vf=[32])},  # rede pequena de proposito
    )
    modelo.learn(total_timesteps=timesteps)
    return modelo


def prever(modelo, env: HibridoLuizFinRLEnv) -> np.ndarray:
    obs, _ = env.reset()
    pos = np.zeros(len(env.df), dtype=int)
    terminado = False
    while not terminado:
        acao, _ = modelo.predict(obs, deterministic=True)
        # grava a posicao efetiva que sera usada (flat ou Luiz)
        if int(acao) == 0:
            pos[env.i] = 0
        else:
            pos[env.i] = int(env.sinal_luiz_exec[env.i])
        obs, _, terminado, _, _ = env.step(acao)
    if env.i < len(env.df):
        acao, _ = modelo.predict(obs, deterministic=True)
        pos[env.i] = 0 if int(acao) == 0 else int(env.sinal_luiz_exec[env.i])
    return pos


def split_df(df, a, b, c, d):
    df = df.dropna(subset=["zscore_mr", "sinal"]).reset_index(drop=True)
    t0, t1 = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
    t2, t3 = pd.Timestamp(c, tz="UTC"), pd.Timestamp(d, tz="UTC")
    treino = df[(df["data"] >= t0) & (df["data"] <= t1)].reset_index(drop=True)
    teste = df[(df["data"] >= t2) & (df["data"] <= t3)].reset_index(drop=True)
    if treino.empty or teste.empty:
        raise ValueError("Split vazio")
    return treino, teste


def main() -> None:
    parser = argparse.ArgumentParser(description="FinRL hibrido: Luiz + filtro RL (info ontem)")
    parser.add_argument("--entrada", type=Path, default=Path("data/processed/pipeline_com_quebras_TAEE_causal.csv"))
    parser.add_argument("--saida", type=Path, default=None)
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--taxa", type=float, default=0.0008)
    parser.add_argument("--holding-cost", type=float, default=0.0)
    parser.add_argument(
        "--execucao",
        choices=("abertura", "fechamento"),
        default="abertura",
    )
    parser.add_argument("--dados-setores", type=Path, default=Path("data/raw/setores"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inicio-formacao", default=DATA_INICIO_FORMACAO)
    parser.add_argument("--fim-formacao", default=DATA_FIM_FORMACAO)
    parser.add_argument("--inicio-negociacao", default=DATA_INICIO_NEGOCIACAO)
    parser.add_argument("--fim-negociacao", default=DATA_FIM_NEGOCIACAO)
    args = parser.parse_args()

    if not args.entrada.exists():
        # fallback
        alt = Path("data/processed/pipeline_com_quebras_TAEE.csv")
        args.entrada = alt if alt.exists() else args.entrada

    df, y, x, hedge = carregar_pipeline(args.entrada)
    if "prob_quebra" not in df.columns:
        df["prob_quebra"] = 0.5

    treino, teste = split_df(
        df,
        args.inicio_formacao,
        args.fim_formacao,
        args.inicio_negociacao,
        args.fim_negociacao,
    )
    print(f"HIBRIDO Luiz+RL | {y}/{x} | treino {len(treino)}d | teste {len(teste)}d")
    print("  Regra: RL so escolhe FLAT ou SEGUIR LUIZ | features = info de ontem")

    env_kw = dict(
        capital=args.capital,
        taxa=args.taxa,
        holding_cost=args.holding_cost,
        execucao=args.execucao,
        dados_setores=args.dados_setores,
    )
    env_tr = HibridoLuizFinRLEnv(treino, y, x, hedge, **env_kw)
    modelo = treinar(env_tr, args.timesteps, seed=args.seed)

    env_te = HibridoLuizFinRLEnv(teste, y, x, hedge, **env_kw)
    teste = teste.copy()
    teste["sinal_hibrido"] = prever(modelo, env_te)

    # Compara no mesmo OOS: Luiz original, Luiz com info de ontem, Hibrido
    teste["sinal_luiz_ontem"] = teste["sinal"].shift(1).fillna(0).astype(int)
    print("\n--- OOS (mesmo periodo, 8 bps) ---")
    for col, nome in (
        ("sinal", "Luiz original (regra atual)"),
        ("sinal_luiz_ontem", "Luiz so com info de ontem"),
        ("sinal_hibrido", "Hibrido Luiz+RL (info ontem)"),
    ):
        _, m = simular_ganhos(
            teste, y, x, hedge, col, args.capital, args.taxa,
            execucao=args.execucao, dados_setores=args.dados_setores,
        )
        print(
            f"  {nome:32s}: PnL R$ {m['pnl_liquido']:,.0f} | "
            f"DD {m['max_drawdown_pct']:.2f}% | pos {m['dias_posicionado_pct']:.1f}% | "
            f"trades {m['trades_fechados']}"
        )

    # Quao diferente do Luiz?
    iguais = float((teste["sinal_hibrido"] == teste["sinal"]).mean() * 100)
    flat_extra = float(((teste["sinal"] != 0) & (teste["sinal_hibrido"] == 0)).mean() * 100)
    print(f"\n  Acordo com Luiz: {iguais:.1f}% dos dias")
    print(f"  Dias em que RL zerou um trade do Luiz: {flat_extra:.1f}%")

    saida = args.saida or args.entrada.with_name(args.entrada.stem + "_hibrido.csv")
    saida.parent.mkdir(parents=True, exist_ok=True)
    teste.to_csv(saida, index=False)
    modelo_path = Path(f"models/finrl_hibrido_{y}_{x}.zip")
    modelo_path.parent.mkdir(parents=True, exist_ok=True)
    modelo.save(str(modelo_path))
    print(f"\nSalvo: {saida}")
    print(f"Modelo: {modelo_path}")


if __name__ == "__main__":
    main()
