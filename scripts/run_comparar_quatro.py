#!/usr/bin/env python3
"""Compara 4 estrategias no MESMO pipeline Luiz limpo (sem bfill / sem vazamento).

Nao faz git checkout de 4 branches. Motivo: Luiz e Luiz-finrl originais
ainda misturavam bug (bfill, label SWANet, PnL close-to-close) com estrategia.
A comparacao honesta e: um CSV causal + quatro geradores de sinal + um avaliador.

Mapeamento branch -> estrategia (codigo ja portado para Luiz-new):

  1) Luiz              -> luiz_limiar_1.25     coluna sinal
  2) Luiz-new          -> luiz_calibrado       coluna sinal_calibrado
  3) Luiz-finrl        -> finrl_puro           coluna sinal_finrl
                         (PPO substitui o Luiz; lag=1 + abertura)
  4) Luiz-new_finrl    -> finrl_sobre_1.25     coluna sinal_hibrido
                         (PPO so pode FLAT ou seguir o Luiz 1.25)

Avaliacao unica: execucao na abertura, 8 bps, capital R$100k, OOS 2025.

Uso:
  python3 scripts/run_comparar_quatro.py --par PLAS3 INEP4
  python3 scripts/run_comparar_quatro.py --top 5
  python3 scripts/run_comparar_quatro.py --top 3 --skip-pipeline
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
LIMIARES = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]


def rodar(cmd: list[str], use_venv: bool = False) -> None:
    exe = str(VENV if use_venv else PY)
    if use_venv and not VENV.exists():
        raise FileNotFoundError("Falta .venv-finrl para SWANet/FinRL.")
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run([exe, *cmd], cwd=RAIZ, check=True)


def oos(df: pd.DataFrame) -> pd.DataFrame:
    t0 = pd.Timestamp(DATA_INICIO_NEGOCIACAO, tz="UTC")
    t1 = pd.Timestamp(DATA_FIM_NEGOCIACAO, tz="UTC")
    return df[(df["data"] >= t0) & (df["data"] <= t1)].copy()


def metricas(df: pd.DataFrame, y: str, x: str, h: float, col: str, nome: str) -> dict:
    _, m = simular_ganhos(
        df, y, x, h, col, CAPITAL, TAXA, execucao="abertura",
        dados_setores=RAIZ / "data/raw/setores",
    )
    return {
        "estrategia": nome,
        "coluna": col,
        "branch": {
            "sinal": "Luiz",
            "sinal_calibrado": "Luiz-new",
            "sinal_finrl": "Luiz-finrl",
            "sinal_hibrido": "Luiz-new_finrl",
        }.get(col, ""),
        "pnl_liquido": round(m["pnl_liquido"], 2),
        "retorno_pct": round(m["retorno_pct"], 2),
        "sharpe": round(m["sharpe_anualizado"], 3),
        "max_dd_pct": round(m["max_drawdown_pct"], 3),
        "pos_pct": round(m["dias_posicionado_pct"], 1),
        "trades": int(m["trades_fechados"]),
    }


def pares_unicos_top(coint: Path, n: int) -> list[tuple[str, str, str]]:
    """Top N pares por p-value, sem contar a direcao inversa (Y/X e X/Y)."""
    df = pd.read_csv(coint).sort_values("P-Value")
    vistos: set[frozenset[str]] = set()
    out: list[tuple[str, str, str]] = []
    for _, row in df.iterrows():
        chave = frozenset((str(row["Ativo Y"]), str(row["Ativo X"])))
        if chave in vistos:
            continue
        vistos.add(chave)
        out.append((str(row["Setor"]), str(row["Ativo Y"]), str(row["Ativo X"])))
        if len(out) >= n:
            break
    return out


def garantir_cointegracao(pasta: Path) -> Path:
    coint = pasta / "cointegracao.csv"
    if not coint.exists():
        rodar([
            str(SRC / "checar_cointegracao.py"),
            "--inicio-formacao", DATA_INICIO_FORMACAO,
            "--fim-formacao", DATA_FIM_FORMACAO,
            "--saida", str(coint),
        ])
    return coint


def pipeline_par(setor: str, y: str, x: str, pasta: Path, coint: Path) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    parcial = pasta / "pipeline_parcial.csv"
    quebras = pasta / "pipeline_com_quebras.csv"
    if quebras.exists():
        return quebras
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


def comparar_par(
    quebras: Path,
    timesteps_finrl: int,
    timesteps_hibrido: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    df, ay, ax, hedge = carregar_pipeline(quebras)
    neg = oos(df)
    linhas: list[dict] = []

    # 1) Luiz tradicional 1.25
    m1 = metricas(neg, ay, ax, hedge, "sinal", "luiz_limiar_1.25")
    m1["limiar_formacao"] = 1.25
    linhas.append(m1)

    # 2) Luiz calibrado (grid so na formacao)
    cal_mod = importlib.import_module("03_calibrar_limiar")
    form = df[
        (df["data"] >= pd.Timestamp(DATA_INICIO_FORMACAO, tz="UTC"))
        & (df["data"] <= pd.Timestamp(DATA_FIM_FORMACAO, tz="UTC"))
    ]
    melhor, _ = cal_mod.grid_search_limiar(
        form.reset_index(drop=True), ay, ax, hedge, LIMIARES,
        CAPITAL, TAXA, "abertura", RAIZ / "data/raw/setores", 3, "calmar",
    )
    full = df.copy()
    full["sinal_calibrado"] = cal_mod.gerar_sinal_com_limiar(full, melhor)
    m2 = metricas(oos(full), ay, ax, hedge, "sinal_calibrado", f"luiz_calibrado_{melhor:.2f}")
    m2["limiar_formacao"] = melhor
    linhas.append(m2)

    # 3) FinRL puro (PPO substitui o Luiz)
    finrl_out = quebras.parent / f"{quebras.stem}_finrl.csv"
    rodar([
        str(SRC / "03_finrl_trading.py"),
        "--entrada", str(quebras),
        "--saida", str(finrl_out),
        "--timesteps", str(timesteps_finrl),
        "--seed", str(seed),
        "--features-lag", "1",
        "--execucao", "abertura",
    ], use_venv=True)
    df_fin, _, _, _ = carregar_pipeline(finrl_out)
    m3 = metricas(oos(df_fin), ay, ax, hedge, "sinal_finrl", "finrl_puro_lag1")
    linhas.append(m3)

    # 4) FinRL sobre o Luiz 1.25 (hibrido: flat ou seguir)
    hib_out = quebras.parent / f"{quebras.stem}_hibrido.csv"
    rodar([
        str(SRC / "03_finrl_hibrido.py"),
        "--entrada", str(quebras),
        "--saida", str(hib_out),
        "--timesteps", str(timesteps_hibrido),
        "--seed", str(seed),
        "--execucao", "abertura",
    ], use_venv=True)
    df_hib, _, _, _ = carregar_pipeline(hib_out)
    m4 = metricas(oos(df_hib), ay, ax, hedge, "sinal_hibrido", "finrl_sobre_1.25")
    linhas.append(m4)

    tab = pd.DataFrame(linhas)
    tab.insert(0, "par", f"{ay}/{ax}")
    meta = {
        "par": f"{ay}/{ax}",
        "pipeline": str(quebras),
        "formacao": [DATA_INICIO_FORMACAO, DATA_FIM_FORMACAO],
        "oos": [DATA_INICIO_NEGOCIACAO, DATA_FIM_NEGOCIACAO],
        "execucao": "abertura",
        "taxa": TAXA,
        "limiar_calibrado": melhor,
        "finrl": {"lag": 1, "timesteps": timesteps_finrl, "seed": seed},
        "hibrido": {"timesteps": timesteps_hibrido, "seed": seed},
    }
    return tab, meta


def main() -> None:
    p = argparse.ArgumentParser(description="Compara Luiz 1.25 vs calibrado vs FinRL puro vs FinRL-sobre-1.25")
    p.add_argument("--par", nargs=2, metavar=("Y", "X"))
    p.add_argument("--setor", default=None)
    p.add_argument("--top", type=int, default=0, help="N pares mais cointegrados (sem direcao inversa)")
    p.add_argument("--entrada", type=Path, default=None, help="CSV de pipeline ja pronto")
    p.add_argument("--skip-pipeline", action="store_true")
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--timesteps-hibrido", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    raiz_out = RAIZ / "data/processed" / "comparacao_quatro"
    raiz_out.mkdir(parents=True, exist_ok=True)
    coint = garantir_cointegracao(raiz_out)

    alvos: list[tuple[str, str, str, Path | None]] = []
    if args.entrada:
        df0, ay, ax, _ = carregar_pipeline(args.entrada)
        alvos.append(("", ay, ax, args.entrada))
    elif args.par:
        y, x = args.par
        pares = pd.read_csv(coint)
        hit = pares[pares["Ativo Y"].eq(y) & pares["Ativo X"].eq(x)]
        if hit.empty:
            raise ValueError(f"Par {y}/{x} nao esta no CSV de cointegracao.")
        setor = str(hit.iloc[0]["Setor"])
        if args.setor and args.setor != setor:
            print(f"  [aviso] --setor {args.setor!r} ignorado; par esta em {setor!r}")
        alvos.append((setor, y, x, None))
    else:
        n = args.top if args.top > 0 else 5
        print(f"Top {n} pares unicos por p-value (formacao {DATA_INICIO_FORMACAO}..{DATA_FIM_FORMACAO})")
        for setor, y, x in pares_unicos_top(coint, n):
            print(f"  {y}/{x}  ({setor})")
            alvos.append((setor, y, x, None))

    todas: list[pd.DataFrame] = []
    metas: list[dict] = []
    for setor, y, x, csv_pronto in alvos:
        print("\n" + "=" * 62)
        print(f"COMPARACAO QUATRO | {y}/{x}")
        print("Execucao: ABERTURA | Custo: 8 bps | Capital: 100k")
        print("=" * 62)
        if csv_pronto is not None:
            quebras = csv_pronto
        else:
            pasta = raiz_out / f"{y}_{x}"
            if args.skip_pipeline and (pasta / "pipeline_com_quebras.csv").exists():
                quebras = pasta / "pipeline_com_quebras.csv"
            else:
                quebras = pipeline_par(setor, y, x, pasta, coint)
        tab, meta = comparar_par(quebras, args.timesteps, args.timesteps_hibrido, args.seed)
        print("\n--- RESULTADO OOS ---")
        print(tab.to_string(index=False))
        out_csv = quebras.parent / f"comparacao_quatro_{y}_{x}.csv"
        tab.to_csv(out_csv, index=False)
        out_csv.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"SALVO: {out_csv}")
        todas.append(tab)
        metas.append(meta)

    consolidado = pd.concat(todas, ignore_index=True)
    cons_path = raiz_out / "comparacao_quatro_consolidado.csv"
    consolidado.to_csv(cons_path, index=False)
    (raiz_out / "comparacao_quatro_consolidado.json").write_text(
        json.dumps(metas, indent=2), encoding="utf-8"
    )
    print("\n" + "=" * 62)
    print("CONSOLIDADO")
    print("=" * 62)
    print(consolidado.to_string(index=False))
    print(f"\nSALVO: {cons_path}")


if __name__ == "__main__":
    main()
