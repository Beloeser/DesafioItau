"""FinRL (Modulo 3): PPO com treino/avaliacao honestos.

Padrao: features_lag=1 (info de ontem), execucao=abertura (preco realista).

Treino : FORMACAO (2022-2024).  Teste : NEGOCIACAO OOS (2025).

Uso:
    .venv-finrl/bin/python src/03_finrl_trading.py \\
      --entrada data/processed/pipeline_com_quebras_TAEE_causal.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from avaliar_ganhos import carregar_pipeline, simular_ganhos
from execucao_pnl import carregar_spread_abertura, pnl_transicao_abertura
from periodos import (
    DATA_FIM_FORMACAO,
    DATA_FIM_NEGOCIACAO,
    DATA_INICIO_FORMACAO,
    DATA_INICIO_NEGOCIACAO,
)


class PairsTradingFinRLEnv(gym.Env):
    """Obs: [zscore_mr, posicao, prob_quebra, tempo_restante]. Acao: flat/long/short."""

    ACAO_PARA_POSICAO = {0: 0, 1: 1, 2: -1}

    def __init__(
        self,
        df: pd.DataFrame,
        ativo_y: str,
        ativo_x: str,
        hedge: float,
        capital: float = 100_000.0,
        taxa: float = 0.0008,
        features_lag: int = 1,
        holding_cost: float = 0.0,
        execucao: str = "abertura",
        dados_setores: Path | None = None,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.capital = float(capital)
        self.taxa = float(taxa)
        self.features_lag = int(features_lag)
        self.holding_cost = float(holding_cost)
        self.execucao = execucao

        preco_y = self.df[ativo_y].astype(float).to_numpy()
        preco_x = self.df[ativo_x].astype(float).to_numpy()
        self.spread_close = self.df["spread_observado"].astype(float).to_numpy()

        self.n_y = (self.capital / 2.0) / max(float(preco_y[0]), 1e-9)
        self.n_x = abs(float(hedge)) * self.n_y
        self.notional = self.n_y * preco_y + self.n_x * preco_x
        self.dspread = np.diff(self.spread_close)

        z = self.df["zscore_mr"].astype(float).to_numpy()
        p = (
            self.df["prob_quebra"].astype(float).to_numpy()
            if "prob_quebra" in self.df.columns
            else np.full(len(self.df), 0.5, dtype=float)
        )
        for _ in range(max(self.features_lag, 0)):
            z = np.concatenate([z[:1], z[:-1]])
            p = np.concatenate([p[:1], p[:-1]])
        self.zscore = z
        self.prob_quebra = p

        if execucao == "abertura":
            self.spread_open = carregar_spread_abertura(
                self.df, ativo_y, ativo_x, hedge, dados_setores=dados_setores
            )
        else:
            self.spread_open = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.i = 0
        self.posicao = 0

    def _obs(self) -> np.ndarray:
        tempo = (len(self.df) - 1 - self.i) / max(len(self.df) - 1, 1)
        return np.array(
            [self.zscore[self.i], self.posicao, self.prob_quebra[self.i], tempo],
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
        alvo = self.ACAO_PARA_POSICAO[int(action)]
        custo_giro = self.taxa * self.notional[self.i] * abs(alvo - self.posicao)
        custo_hold = self.holding_cost * self.notional[self.i] * abs(alvo)
        pnl = self._pnl_dia(alvo)
        recompensa = (pnl - custo_giro - custo_hold) / self.capital * 100.0
        self.posicao = alvo
        self.i += 1
        terminado = self.i >= len(self.df) - 1
        return self._obs(), float(recompensa), terminado, False, {"pnl": pnl - custo_giro - custo_hold}


def treinar_ppo(
    env_treino: PairsTradingFinRLEnv,
    timesteps: int,
    seed: int = 42,
    net_arch: list[int] | None = None,
):
    from stable_baselines3.common.vec_env import DummyVecEnv

    env_vec = DummyVecEnv([lambda: env_treino])
    policy_kwargs = None
    if net_arch is not None:
        policy_kwargs = {"net_arch": dict(pi=list(net_arch), vf=list(net_arch))}

    try:
        from finrl.agents.stablebaselines3.models import DRLAgent

        if policy_kwargs is not None:
            raise RuntimeError("net_arch custom -> SB3")
        agente = DRLAgent(env=env_vec)
        modelo = agente.get_model("ppo", seed=seed, verbose=0)
        return agente.train_model(model=modelo, tb_log_name="ppo_pairs", total_timesteps=timesteps)
    except Exception as exc:
        print(f"  [aviso] PPO/SB3 direto ({type(exc).__name__})")
        from stable_baselines3 import PPO

        kwargs = {"policy_kwargs": policy_kwargs} if policy_kwargs else {}
        modelo = PPO("MlpPolicy", env_vec, seed=seed, verbose=0, **kwargs)
        modelo.learn(total_timesteps=timesteps)
        return modelo


def prever_posicoes(modelo, env: PairsTradingFinRLEnv) -> np.ndarray:
    obs, _ = env.reset()
    posicoes = np.zeros(len(env.df), dtype=int)
    terminado = False
    while not terminado:
        acao, _ = modelo.predict(obs, deterministic=True)
        posicoes[env.i] = PairsTradingFinRLEnv.ACAO_PARA_POSICAO[int(acao)]
        obs, _, terminado, _, _ = env.step(acao)
    if env.i < len(env.df):
        acao_final, _ = modelo.predict(obs, deterministic=True)
        posicoes[env.i] = PairsTradingFinRLEnv.ACAO_PARA_POSICAO[int(acao_final)]
    return posicoes


def split_formacao_negociacao(
    df: pd.DataFrame,
    inicio_formacao: str,
    fim_formacao: str,
    inicio_negociacao: str,
    fim_negociacao: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.dropna(subset=["zscore_mr"]).reset_index(drop=True)
    t0 = pd.to_datetime(inicio_formacao).tz_localize("UTC")
    t1 = pd.to_datetime(fim_formacao).tz_localize("UTC")
    t2 = pd.to_datetime(inicio_negociacao).tz_localize("UTC")
    t3 = pd.to_datetime(fim_negociacao).tz_localize("UTC")
    treino = df[(df["data"] >= t0) & (df["data"] <= t1)].reset_index(drop=True)
    teste = df[(df["data"] >= t2) & (df["data"] <= t3)].reset_index(drop=True)
    if treino.empty or teste.empty:
        raise ValueError("Split formacao/negociacao vazio.")
    return treino, teste


def main() -> None:
    parser = argparse.ArgumentParser(description="FinRL PPO honesto.")
    parser.add_argument("--entrada", type=Path, default=Path("data/processed/pipeline_com_quebras.csv"))
    parser.add_argument("--saida", type=Path, default=None)
    parser.add_argument("--modelo-saida", type=Path, default=None)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--taxa", type=float, default=0.0008)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features-lag", type=int, default=1)
    parser.add_argument("--execucao", choices=("abertura", "fechamento"), default="abertura")
    parser.add_argument("--dados-setores", type=Path, default=Path("data/raw/setores"))
    parser.add_argument("--holding-cost", type=float, default=0.0)
    parser.add_argument("--net-arch", type=int, nargs="*", default=None)
    parser.add_argument("--inicio-formacao", default=DATA_INICIO_FORMACAO)
    parser.add_argument("--fim-formacao", default=DATA_FIM_FORMACAO)
    parser.add_argument("--inicio-negociacao", default=DATA_INICIO_NEGOCIACAO)
    parser.add_argument("--fim-negociacao", default=DATA_FIM_NEGOCIACAO)
    args = parser.parse_args()

    df, ativo_y, ativo_x, hedge = carregar_pipeline(args.entrada)
    df_treino, df_teste = split_formacao_negociacao(
        df, args.inicio_formacao, args.fim_formacao,
        args.inicio_negociacao, args.fim_negociacao,
    )

    env_kw = dict(
        capital=args.capital, taxa=args.taxa, features_lag=args.features_lag,
        holding_cost=args.holding_cost, execucao=args.execucao,
        dados_setores=args.dados_setores,
    )
    print(
        f"FINRL {ativo_y}/{ativo_x} | treino {len(df_treino)}d | teste {len(df_teste)}d | "
        f"lag={args.features_lag} exec={args.execucao}"
    )

    modelo = treinar_ppo(
        PairsTradingFinRLEnv(df_treino, ativo_y, ativo_x, hedge, **env_kw),
        args.timesteps, seed=args.seed, net_arch=args.net_arch,
    )

    modelo_path = args.modelo_saida or Path(f"models/finrl_ppo_{ativo_y}_{ativo_x}.zip")
    modelo_path.parent.mkdir(parents=True, exist_ok=True)
    modelo.save(str(modelo_path))

    env_teste = PairsTradingFinRLEnv(df_teste, ativo_y, ativo_x, hedge, **env_kw)
    df_teste = df_teste.copy()
    df_teste["sinal_finrl"] = prever_posicoes(modelo, env_teste)

    saida = args.saida or args.entrada.with_name(args.entrada.stem + "_finrl.csv")
    saida.parent.mkdir(parents=True, exist_ok=True)
    df_teste.to_csv(saida, index=False)

    for taxa in (0.0, args.taxa):
        _, m = simular_ganhos(
            df_teste, ativo_y, ativo_x, hedge, "sinal_finrl",
            args.capital, taxa, execucao=args.execucao, dados_setores=args.dados_setores,
        )
        rotulo = "sem custo" if taxa == 0 else f"{taxa * 1e4:.0f} bps"
        print(f"  OOS ({rotulo}): PnL R$ {m['pnl_liquido']:,.0f} | Sharpe {m['sharpe_anualizado']:.2f}")


if __name__ == "__main__":
    main()
