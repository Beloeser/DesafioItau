#!/usr/bin/env python3
"""Compara 3 estrategias NO MESMO pipeline causal, sem vazamento.

  1) luiz          — regra fixa limiar 1.25 (coluna sinal)
  2) luiz_calibrado — grid do limiar na formacao (coluna sinal_calibrado)
  3) finrl_honesto  — PPO lag=1 + execucao abertura (coluna sinal_finrl)

Fluxo completo (opcional --skip-pipeline se CSV causal ja existe):
  cointegracao -> pipeline parcial -> SWANet causal -> treinos -> OOS 2025

Saida:
  data/processed/comparacao_tres_<PAR>.csv   (1 linha por estrategia)
  data/processed/comparacao_tres_<PAR>.json  (metadados)

Uso:
  python3 scripts/run_comparar_tres.py --par TAEE3 TAEE11 --setor energia_eletrica
  python3 scripts/run_comparar_tres.py --entrada data/processed/pipeline_com_quebras_TAEE_causal.csv
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
SRC = RAIZ / "src"
PY = sys.executable
VENV = RAIZ / ".venv-finrl" / "bin" / "python"

sys.path.insert(0, str(SRC))

from avaliar_ganhos import carregar_pipeline, simular_ganhos  # noqa: E402
from periodos import (  # noqa: E402
    DATA_FIM_FORMACAO,
    DATA_FIM_NEGOCIACAO,
    DATA_INICIO_FORMACAO,
    DATA_INICIO_NEGOCIACAO,
)

CAPITAL = 100_000.0
TAXA = 0.0008


def rodar(cmd: list[str], use_venv: bool = False) -> None:
    exe = str(VENV if use_venv else PY)
    if use_venv and not VENV.exists():
        raise FileNotFoundError("Falta .venv-finrl para FinRL/SWANet.")
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run([exe, *cmd], cwd=RAIZ, check=True)


def pipeline_completo(setor: str, y: str, x: str, pasta: Path) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    coint = pasta / "cointegracao.csv"
    parcial = pasta / "pipeline_parcial.csv"
    quebras = pasta / "pipeline_com_quebras_causal.csv"

    rodar([
        str(SRC / "checar_cointegracao.py"),
        "--inicio-formacao", DATA_INICIO_FORMACAO,
        "--fim-formacao", DATA_FIM_FORMACAO,
        "--saida", str(coint),
    ])
    rodar([
        str(SRC / "pipeline_cointegracao_parcial.py"),
        "--cointegracao", str(coint),
        "--setor", setor, "--ativo-y", y, "--ativo-x", x,
        "--inicio-formacao", DATA_INICIO_FORMACAO,
        "--fim-formacao", DATA_FIM_FORMACAO,
        "--inicio-negociacao", DATA_INICIO_NEGOCIACAO,
        "--fim-negociacao", DATA_FIM_NEGOCIACAO,
        "--saida", str(parcial),
    ])
    rodar([
        str(SRC / "01_swanet_quebras.py"),
        "--entrada", str(parcial),
        "--saida", str(quebras),
        "--inicio-formacao", DATA_INICIO_FORMACAO,
        "--fim-formacao", DATA_FIM_FORMACAO,
    ], use_venv=True)
    return quebras


def oos(df: pd.DataFrame) -> pd.DataFrame:
    t0 = pd.Timestamp(DATA_INICIO_NEGOCIACAO, tz="UTC")
    t1 = pd.Timestamp(DATA_FIM_NEGOCIACAO, tz="UTC")
    return df[(df["data"] >= t0) & (df["data"] <= t1)].copy()


def metricas(df, y, x, h, col: str) -> dict:
    _, m = simular_ganhos(
        df, y, x, h, col, CAPITAL, TAXA, execucao="abertura",
        dados_setores=RAIZ / "data/raw/setores",
    )
    return {
        "estrategia": col,
        "pnl_liquido": round(m["pnl_liquido"], 2),
        "retorno_pct": round(m["retorno_pct"], 2),
        "sharpe": round(m["sharpe_anualizado"], 3),
        "max_dd_pct": round(m["max_drawdown_pct"], 3),
        "pos_pct": round(m["dias_posicionado_pct"], 1),
        "trades": int(m["trades_fechados"]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--par", nargs=2, metavar=("Y", "X"))
    p.add_argument("--setor", default="energia_eletrica")
    p.add_argument("--entrada", type=Path, default=None, help="CSV causal pronto")
    p.add_argument("--skip-pipeline", action="store_true")
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.entrada:
        quebras = args.entrada
        y = x = None
    else:
        if not args.par:
            p.error("Informe --par Y X ou --entrada CSV")
        y, x = args.par
        slug = f"{y}_{x}"
        pasta = RAIZ / "data/processed" / f"comparacao_{slug}"
        quebras = pasta / "pipeline_com_quebras_causal.csv"
        if not args.skip_pipeline or not quebras.exists():
            quebras = pipeline_completo(args.setor, y, x, pasta)

    df, ay, ax, hedge = carregar_pipeline(quebras)
    y = y or ay
    x = x or ax
    neg = oos(df)

    print("\n" + "=" * 62)
    print(f"COMPARACAO TRES | {ay}/{ax} | OOS {DATA_INICIO_NEGOCIACAO} .. {DATA_FIM_NEGOCIACAO}")
    print("Pipeline:", quebras)
    print("Execucao: ABERTURA | Custo: 8 bps | Capital: 100k")
    print("=" * 62)

    # 1) Luiz fixo 1.25 — ja vem no CSV
    linhas = [metricas(neg, ay, ax, hedge, "sinal")]
    linhas[0]["estrategia"] = "luiz_limiar_1.25"

    # 2) Calibrar limiar
    cal_mod = importlib.import_module("03_calibrar_limiar")
    form = df[(df["data"] >= pd.Timestamp(DATA_INICIO_FORMACAO, tz="UTC"))
              & (df["data"] <= pd.Timestamp(DATA_FIM_FORMACAO, tz="UTC"))]
    melhor, _ = cal_mod.grid_search_limiar(
        form.reset_index(drop=True), ay, ax, hedge,
        [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
        CAPITAL, TAXA, "abertura", RAIZ / "data/raw/setores", 3, "calmar",
    )
    full = df.copy()
    full["sinal_calibrado"] = cal_mod.gerar_sinal_com_limiar(full, melhor)
    neg_cal = oos(full)
    m_cal = metricas(neg_cal, ay, ax, hedge, "sinal_calibrado")
    m_cal["estrategia"] = f"luiz_calibrado_{melhor:.2f}"
    m_cal["limiar_formacao"] = melhor
    linhas.append(m_cal)

    # 3) FinRL honesto (via .venv-finrl — precisa gymnasium)
    finrl_out = quebras.parent / f"{quebras.stem}_finrl_comparacao.csv"
    rodar([
        str(SRC / "03_finrl_trading.py"),
        "--entrada", str(quebras),
        "--saida", str(finrl_out),
        "--timesteps", str(args.timesteps),
        "--seed", str(args.seed),
        "--features-lag", "1",
        "--execucao", "abertura",
    ], use_venv=True)
    df_fin, _, _, _ = carregar_pipeline(finrl_out)
    m_fin = metricas(oos(df_fin), ay, ax, hedge, "sinal_finrl")
    m_fin["estrategia"] = "finrl_honesto_lag1_abertura"
    linhas.append(m_fin)

    tab = pd.DataFrame(linhas)
    print("\n--- RESULTADO OOS (leia esta tabela) ---")
    print(tab.to_string(index=False))

    out_csv = quebras.parent / f"comparacao_tres_{ay}_{ax}.csv"
    out_json = out_csv.with_suffix(".json")
    tab.to_csv(out_csv, index=False)
    meta = {
        "par": f"{ay}/{ax}",
        "pipeline": str(quebras),
        "formacao": [DATA_INICIO_FORMACAO, DATA_FIM_FORMACAO],
        "oos": [DATA_INICIO_NEGOCIACAO, DATA_FIM_NEGOCIACAO],
        "execucao": "abertura",
        "taxa": TAXA,
        "finrl": {"lag": 1, "timesteps": args.timesteps, "seed": args.seed},
        "limiar_calibrado": melhor,
    }
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSALVO: {out_csv}")
    print(f"META:  {out_json}")


if __name__ == "__main__":
    main()
