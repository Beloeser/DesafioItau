"""Ablacoes simples: Luiz baseline vs FinRL, dados, rede PPO, treino vs backtest.

Roda com:
    .venv-finrl/bin/python tests/test_ablacao_finrl.py

Objetivo: evidencias numericas para decidir se vale
  (1) usar mais historico Yahoo ja baixado,
  (2) aumentar a rede PPO (weights),
  (3) aumentar timesteps de treino,
  (4) alongar formacao vs alongar backtest OOS.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

modulo = importlib.import_module("03_finrl_trading")
PairsTradingFinRLEnv = modulo.PairsTradingFinRLEnv
prever_posicoes = modulo.prever_posicoes
from avaliar_ganhos import carregar_pipeline, simular_ganhos

TAXA = 0.0008
CAPITAL = 100_000.0
SEED = 42
CSV_PADRAO = RAIZ / "data/processed/pipeline_com_quebras_TAEE.csv"
CSV_LONGO = RAIZ / "data/processed/pipeline_parcial_TAEE_desde_2019.csv"
CSV_FINRL = RAIZ / "data/processed/pipeline_finrl_TAEE.csv"
SAIDA = RAIZ / "data/processed/ablacao_finrl_resultados.csv"


def _avaliar(df_teste: pd.DataFrame, ativo_y: str, ativo_x: str, hedge: float, coluna: str) -> dict:
    _, m = simular_ganhos(
        df_teste, ativo_y, ativo_x, hedge,
        coluna_sinal=coluna, capital=CAPITAL, taxa=TAXA,
    )
    return {
        "pnl_liquido": round(m["pnl_liquido"], 2),
        "sharpe": round(m["sharpe_anualizado"], 3),
        "max_dd_pct": round(m["max_drawdown_pct"], 3),
        "dias_pos_pct": round(m["dias_posicionado_pct"], 1),
        "trades": m["trades_fechados"],
    }


def _preparar_df(caminho: Path) -> tuple[pd.DataFrame, str, str, float]:
    df, y, x, hedge = carregar_pipeline(caminho)
    if "prob_quebra" not in df.columns:
        df = df.copy()
        df["prob_quebra"] = 0.5
    return df, y, x, hedge


def treinar_ppo_custom(
    env: PairsTradingFinRLEnv,
    timesteps: int,
    net_arch: list[int],
    seed: int = SEED,
):
    """PPO direto (SB3) com arquitetura controlavel — isolado do wrapper FinRL."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env_vec = DummyVecEnv([lambda: env])
    modelo = PPO(
        "MlpPolicy",
        env_vec,
        seed=seed,
        verbose=0,
        policy_kwargs={"net_arch": dict(pi=net_arch, vf=net_arch)},
    )
    modelo.learn(total_timesteps=timesteps)
    return modelo


def contar_parametros(modelo) -> int:
    return sum(p.numel() for p in modelo.policy.parameters())


def rodar_experimento(
    df: pd.DataFrame,
    ativo_y: str,
    ativo_x: str,
    hedge: float,
    inicio_treino: str,
    fim_treino: str,
    inicio_teste: str,
    fim_teste: str,
    timesteps: int,
    net_arch: list[int],
    rotulo: str,
) -> dict:
    t0 = pd.Timestamp(inicio_treino, tz="UTC")
    t1 = pd.Timestamp(fim_treino, tz="UTC")
    t2 = pd.Timestamp(inicio_teste, tz="UTC")
    t3 = pd.Timestamp(fim_teste, tz="UTC")
    treino = df[(df["data"] >= t0) & (df["data"] <= t1)].dropna(subset=["zscore_mr"]).reset_index(drop=True)
    teste = df[(df["data"] >= t2) & (df["data"] <= t3)].dropna(subset=["zscore_mr"]).reset_index(drop=True)
    if len(treino) < 60 or len(teste) < 20:
        return {
            "rotulo": rotulo,
            "erro": f"split curto treino={len(treino)} teste={len(teste)}",
        }

    env_tr = PairsTradingFinRLEnv(treino, ativo_y, ativo_x, hedge, capital=CAPITAL, taxa=TAXA)
    t_ini = time.time()
    modelo = treinar_ppo_custom(env_tr, timesteps=timesteps, net_arch=net_arch, seed=SEED)
    segundos = time.time() - t_ini

    env_te = PairsTradingFinRLEnv(teste, ativo_y, ativo_x, hedge, capital=CAPITAL, taxa=TAXA)
    teste = teste.copy()
    teste["sinal_finrl"] = prever_posicoes(modelo, env_te)
    met = _avaliar(teste, ativo_y, ativo_x, hedge, "sinal_finrl")

    # Baseline heuristico no MESMO periodo de teste (coluna sinal, se existir)
    base = {}
    if "sinal" in teste.columns:
        base = _avaliar(teste, ativo_y, ativo_x, hedge, "sinal")

    return {
        "rotulo": rotulo,
        "dias_treino": len(treino),
        "dias_teste": len(teste),
        "timesteps": timesteps,
        "net_arch": str(net_arch),
        "n_params": contar_parametros(modelo),
        "segundos_treino": round(segundos, 1),
        "finrl_pnl": met["pnl_liquido"],
        "finrl_sharpe": met["sharpe"],
        "finrl_dd": met["max_dd_pct"],
        "finrl_pos_pct": met["dias_pos_pct"],
        "base_pnl": base.get("pnl_liquido"),
        "base_sharpe": base.get("sharpe"),
        "delta_pnl_vs_base": (
            None if base.get("pnl_liquido") is None else round(met["pnl_liquido"] - base["pnl_liquido"], 2)
        ),
    }


def comparar_luiz_vs_finrl_existente() -> dict:
    """Usa CSVs ja gerados no projeto (mesmo OOS 2025)."""
    df, y, x, hedge = _preparar_df(CSV_PADRAO)
    oos = df[(df["data"] >= "2025-01-01") & (df["data"] <= "2026-01-01")].reset_index(drop=True)
    base = _avaliar(oos, y, x, hedge, "sinal")

    finrl = None
    if CSV_FINRL.exists():
        df_f, yf, xf, hf = carregar_pipeline(CSV_FINRL)
        finrl = _avaliar(df_f, yf, xf, hf, "sinal_finrl")

    return {"par": f"{y}/{x}", "baseline": base, "finrl_salvo": finrl}


def main() -> None:
    print("=" * 72)
    print("1) COMPARACAO Luiz (baseline) vs Luiz-finrl (PPO) — OOS 2025, 8 bps")
    print("=" * 72)
    comp = comparar_luiz_vs_finrl_existente()
    print(f"Par: {comp['par']}")
    print(f"  Baseline heuristico: PnL R$ {comp['baseline']['pnl_liquido']:,.2f} | "
          f"Sharpe {comp['baseline']['sharpe']} | pos {comp['baseline']['dias_pos_pct']}%")
    if comp["finrl_salvo"]:
        f = comp["finrl_salvo"]
        print(f"  FinRL PPO salvo:     PnL R$ {f['pnl_liquido']:,.2f} | "
              f"Sharpe {f['sharpe']} | pos {f['dias_pos_pct']}%")
        print(f"  Delta FinRL - Base:  R$ {f['pnl_liquido'] - comp['baseline']['pnl_liquido']:,.2f}")
        melhor = "FinRL" if f["pnl_liquido"] > comp["baseline"]["pnl_liquido"] else "Baseline"
        print(f"  => Melhor em PnL liquido: {melhor}")
    else:
        print("  (CSV FinRL salvo nao encontrado; ablations abaixo treinam do zero)")

    resultados: list[dict] = []

    # Dataset padrao (2022+) com SWANet
    df_pad, y, x, hedge = _preparar_df(CSV_PADRAO)

    print("\n" + "=" * 72)
    print("2) ABLACAO: tamanho da rede (weights) — mesmo split 2022-24 / 2025")
    print("=" * 72)
    for arch in ([32], [64, 64], [256, 256]):
        r = rodar_experimento(
            df_pad, y, x, hedge,
            "2022-01-01", "2024-12-31", "2025-01-01", "2026-01-01",
            timesteps=30_000, net_arch=arch,
            rotulo=f"rede_{arch}",
        )
        resultados.append(r)
        print(f"  {r['rotulo']}: params={r.get('n_params')} | "
              f"PnL FinRL R$ {r.get('finrl_pnl')} | Sharpe {r.get('finrl_sharpe')} | "
              f"{r.get('segundos_treino')}s")

    print("\n" + "=" * 72)
    print("3) ABLACAO: timesteps de treino — rede [64,64]")
    print("=" * 72)
    for ts in (10_000, 30_000, 80_000):
        r = rodar_experimento(
            df_pad, y, x, hedge,
            "2022-01-01", "2024-12-31", "2025-01-01", "2026-01-01",
            timesteps=ts, net_arch=[64, 64],
            rotulo=f"timesteps_{ts}",
        )
        resultados.append(r)
        print(f"  {r['rotulo']}: PnL R$ {r.get('finrl_pnl')} | Sharpe {r.get('finrl_sharpe')} | "
              f"{r.get('segundos_treino')}s")

    print("\n" + "=" * 72)
    print("4) ABLACAO: mais historico de treino (Yahoo ja tem desde 2015)")
    print("   Mesmo OOS 2025; so muda quanto do passado entra no treino PPO")
    print("=" * 72)
    # Usa CSV longo se existir; senao usa o padrao (so 2022+)
    caminho_hist = CSV_LONGO if CSV_LONGO.exists() else CSV_PADRAO
    df_hist, yh, xh, hh = _preparar_df(caminho_hist)
    print(f"  Fonte: {caminho_hist.name} | dias totais={len(df_hist)} | "
          f"{df_hist['data'].min().date()} -> {df_hist['data'].max().date()}")
    for ini, rot in (
        ("2023-01-01", "treino_1y_2023_2024"),
        ("2022-01-01", "treino_3y_2022_2024"),
        ("2020-01-01", "treino_5y_2020_2024"),
        ("2019-01-01", "treino_6y_2019_2024"),
    ):
        # pula se nao houver barras apos o inicio pedido
        if df_hist["data"].max() < pd.Timestamp(ini, tz="UTC"):
            print(f"  {rot}: pulado (CSV termina antes de {ini})")
            continue
        if (df_hist["data"] >= pd.Timestamp(ini, tz="UTC")).sum() < 60:
            print(f"  {rot}: pulado (poucos dias apos {ini})")
            continue
        r = rodar_experimento(
            df_hist, yh, xh, hh,
            ini, "2024-12-31", "2025-01-01", "2026-01-01",
            timesteps=30_000, net_arch=[64, 64],
            rotulo=rot,
        )
        resultados.append(r)
        print(f"  {r['rotulo']}: dias_treino={r.get('dias_treino')} | "
              f"PnL R$ {r.get('finrl_pnl')} | Sharpe {r.get('finrl_sharpe')} | "
              f"delta_vs_base={r.get('delta_pnl_vs_base')}")

    print("\n" + "=" * 72)
    print("5) ABLACAO: maior treino vs maior backtest (mesmo universo temporal)")
    print("=" * 72)
    # Universo: 2022-01-01 -> 2025-12-31. Move a fronteira de split.
    splits = [
        ("2022-01-01", "2023-12-31", "2024-01-01", "2025-12-31", "mais_backtest_2y_treino_2y_teste"),
        ("2022-01-01", "2024-06-30", "2024-07-01", "2025-12-31", "meio_termo_2.5y_1.5y"),
        ("2022-01-01", "2024-12-31", "2025-01-01", "2025-12-31", "atual_3y_treino_1y_teste"),
        ("2022-01-01", "2025-06-30", "2025-07-01", "2025-12-31", "mais_treino_3.5y_0.5y_teste"),
    ]
    for a, b, c, d, rot in splits:
        r = rodar_experimento(
            df_pad, y, x, hedge, a, b, c, d,
            timesteps=30_000, net_arch=[64, 64], rotulo=rot,
        )
        resultados.append(r)
        print(f"  {r['rotulo']}: treino={r.get('dias_treino')}d teste={r.get('dias_teste')}d | "
              f"FinRL PnL R$ {r.get('finrl_pnl')} Sharpe {r.get('finrl_sharpe')} | "
              f"Base PnL R$ {r.get('base_pnl')}")

    out = pd.DataFrame(resultados)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SAIDA, index=False)
    print(f"\nCSV salvo: {SAIDA}")

    # ---- Vereditos automaticos com regras simples ----
    print("\n" + "=" * 72)
    print("VEREDITOS (regras simples sobre as ablations acima)")
    print("=" * 72)

    redes = [r for r in resultados if str(r.get("rotulo", "")).startswith("rede_")]
    if redes:
        melhor_rede = max(redes, key=lambda r: r.get("finrl_pnl") or -1e18)
        pior_rede = min(redes, key=lambda r: r.get("finrl_pnl") or 1e18)
        ganho = (melhor_rede.get("finrl_pnl") or 0) - (pior_rede.get("finrl_pnl") or 0)
        print(f"  Rede: melhor={melhor_rede['rotulo']} (R$ {melhor_rede.get('finrl_pnl')}) | "
              f"pior={pior_rede['rotulo']} (R$ {pior_rede.get('finrl_pnl')}) | "
              f"amplitude R$ {ganho:,.2f}")
        if abs(ganho) < 2000:
            print("  => Aumentar weights da MLP NAO parece prioridade (ganho < R$ 2k).")
        else:
            print(f"  => Vale considerar rede {melhor_rede['net_arch']} (ganho material).")

    ts_rows = [r for r in resultados if str(r.get("rotulo", "")).startswith("timesteps_")]
    if ts_rows:
        melhor_ts = max(ts_rows, key=lambda r: r.get("finrl_pnl") or -1e18)
        print(f"  Timesteps: melhor={melhor_ts['rotulo']} PnL R$ {melhor_ts.get('finrl_pnl')}")
        ordenado = sorted(ts_rows, key=lambda r: r.get("timesteps") or 0)
        if len(ordenado) >= 2:
            delta = (ordenado[-1].get("finrl_pnl") or 0) - (ordenado[0].get("finrl_pnl") or 0)
            if delta < 2000:
                print("  => Mais timesteps alem do atual tende a saturar (ganho pequeno).")
            else:
                print("  => Mais timesteps ainda melhora — vale subir o treino.")

    hist = [r for r in resultados if str(r.get("rotulo", "")).startswith("treino_")]
    if len(hist) >= 2:
        melhor_h = max(hist, key=lambda r: r.get("finrl_pnl") or -1e18)
        mais_curto = min(hist, key=lambda r: r.get("dias_treino") or 0)
        mais_longo = max(hist, key=lambda r: r.get("dias_treino") or 0)
        print(f"  Historico: melhor={melhor_h['rotulo']} (R$ {melhor_h.get('finrl_pnl')})")
        print(f"    curto {mais_curto['dias_treino']}d -> R$ {mais_curto.get('finrl_pnl')} | "
              f"longo {mais_longo['dias_treino']}d -> R$ {mais_longo.get('finrl_pnl')}")
        if (mais_longo.get("finrl_pnl") or 0) > (mais_curto.get("finrl_pnl") or 0) + 2000:
            print("  => Usar MAIS historico Yahoo (ja baixado) ajuda o PPO.")
        else:
            print("  => Mais anos de Yahoo NAO melhorou de forma clara neste par/OOS.")

    splits_r = [r for r in resultados if "treino" in str(r.get("rotulo", "")) and "teste" in str(r.get("rotulo", ""))]
    # filtro pelos rotulos da secao 5
    splits_r = [r for r in resultados if str(r.get("rotulo", "")).startswith(("mais_", "meio_", "atual_"))]
    if splits_r:
        # Escolhe pelo Sharpe do FinRL (PnL bruto de periodos diferentes nao e comparavel)
        melhor_s = max(splits_r, key=lambda r: r.get("finrl_sharpe") or -1e18)
        print(f"  Split (por Sharpe OOS): melhor={melhor_s['rotulo']} "
              f"Sharpe={melhor_s.get('finrl_sharpe')} PnL={melhor_s.get('finrl_pnl')} "
              f"(treino {melhor_s.get('dias_treino')}d / teste {melhor_s.get('dias_teste')}d)")
        atual = next((r for r in splits_r if r["rotulo"].startswith("atual_")), None)
        mais_bt = next((r for r in splits_r if r["rotulo"].startswith("mais_backtest")), None)
        mais_tr = next((r for r in splits_r if r["rotulo"].startswith("mais_treino")), None)
        if atual and mais_bt and mais_tr:
            print("  Nota: PnL absoluto muda com o tamanho do OOS; compare Sharpe/retorno%.")
            print(f"    atual     Sharpe={atual.get('finrl_sharpe')} PnL={atual.get('finrl_pnl')}")
            print(f"    +backtest Sharpe={mais_bt.get('finrl_sharpe')} PnL={mais_bt.get('finrl_pnl')}")
            print(f"    +treino   Sharpe={mais_tr.get('finrl_sharpe')} PnL={mais_tr.get('finrl_pnl')}")

    print("\nOK — ablacao concluida.")


if __name__ == "__main__":
    main()
