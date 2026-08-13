#!/usr/bin/env python3
"""Roda o pipeline completo da branch Luiz com 4 datas.

Uso:
    python3 run_all.py 2022-01-01 2024-12-31 2025-01-01 2026-01-01
    python3 run_all.py 01-01-2022 31-12-2024 01-01-2025 01-01-2026

Argumentos (nesta ordem):
    1. inicio_formacao   — cointegracao + treino SWANet + FinRL sobre Luiz 1.25
    2. fim_formacao
    3. inicio_negociacao — backtest cego (OOS)
    4. fim_negociacao

Opcoes:
    --par TAEE3 TAEE11     forca um par (senao usa o melhor p-value)
    --sem-calibracao       pula o hibrido (Modulo 3)
    --sem-finrl            alias de --sem-calibracao
    --sem-swanet           pula a SWANet
    --com-dqn              roda tambem o DQN original (Modulo 2)
    --timesteps 30000      timesteps do PPO hibrido

Isolamento temporal (sem look-ahead):
    Etapa 1 cointegracao     -> SO formacao
    Etapa 2 sinais parciais  -> formacao + negociacao (Kalman causal, sem bfill)
    Etapa 3 SWANet           -> TREINA formacao | PREVE formacao + negociacao
    Etapa 4 hibrido          -> PPO so pode FLAT ou seguir Luiz 1.25 (lag=1, abertura)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
SRC = RAIZ / "src"
DADOS = RAIZ / "data"
VENV_PY = RAIZ / ".venv-finrl" / "bin" / "python"
PY = sys.executable


def parse_data(texto: str) -> str:
    """Converte para YYYY-MM-DD. Aceita YYYY-MM-DD ou DD-MM-YYYY."""
    texto = texto.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", texto):
        return texto
    if re.match(r"^\d{2}-\d{2}-\d{4}$", texto):
        d, m, a = texto.split("-")
        return f"{a}-{m}-{d}"
    raise ValueError(f"Data invalida: {texto!r}. Use YYYY-MM-DD ou DD-MM-YYYY.")


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def info(msg: str) -> None:
    print(f"  --> {msg}")


def etapa(num: int, titulo: str) -> None:
    print(f"\n{'=' * 62}\nETAPA {num}: {titulo}\n{'=' * 62}")


def rodar(cmd: list[str], *, precisa_finrl: bool = False) -> str:
    executavel = str(VENV_PY if precisa_finrl else PY)
    if precisa_finrl and not VENV_PY.exists():
        raise FileNotFoundError(
            "Ambiente .venv-finrl nao encontrado. "
            "Instale torch/gymnasium/stable-baselines3/finrl no .venv-finrl."
        )
    proc = subprocess.run(
        [executavel, *cmd[1:]],
        cwd=RAIZ,
        capture_output=True,
        text=True,
    )
    if proc.stdout.strip():
        print(proc.stdout.rstrip())
    if proc.returncode != 0:
        print(proc.stderr.rstrip(), file=sys.stderr)
        raise RuntimeError(f"Comando falhou: {' '.join(cmd)}")
    return proc.stdout


def escolher_par(cointegracao: Path, setor: str | None, y: str | None, x: str | None):
    import pandas as pd

    pares = pd.read_csv(cointegracao)
    if pares.empty:
        raise ValueError("Nenhum par cointegrado encontrado.")
    if y and x:
        filtro = (pares["Ativo Y"] == y) & (pares["Ativo X"] == x)
        if setor:
            filtro &= pares["Setor"] == setor
        candidatos = pares[filtro]
        if candidatos.empty:
            raise ValueError(f"Par {y}/{x} nao encontrado no CSV de cointegracao.")
        return candidatos.iloc[0]
    return pares.sort_values("P-Value").iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline Luiz-new_finrl: cointegracao -> parcial -> SWANet -> FinRL sobre 1.25.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("inicio_formacao", help="Inicio da formacao (treino IA)")
    parser.add_argument("fim_formacao", help="Fim da formacao (treino IA)")
    parser.add_argument("inicio_negociacao", help="Inicio da negociacao (OOS)")
    parser.add_argument("fim_negociacao", help="Fim da negociacao (OOS)")
    parser.add_argument("--par", nargs=2, metavar=("Y", "X"))
    parser.add_argument("--setor", default=None)
    parser.add_argument("--sem-calibracao", action="store_true")
    parser.add_argument("--sem-finrl", action="store_true", help="Alias de --sem-calibracao.")
    parser.add_argument("--sem-swanet", action="store_true")
    parser.add_argument("--com-dqn", action="store_true", help="Roda o DQN original (Modulo 2).")
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--capital", type=float, default=100_000.0)
    args = parser.parse_args()
    sem_cal = args.sem_calibracao or args.sem_finrl
    t0 = parse_data(args.inicio_formacao)
    t1 = parse_data(args.fim_formacao)
    t2 = parse_data(args.inicio_negociacao)
    t3 = parse_data(args.fim_negociacao)

    if datetime.fromisoformat(t1) >= datetime.fromisoformat(t2):
        raise ValueError(
            f"fim_formacao ({t1}) deve ser ANTERIOR a inicio_negociacao ({t2})."
        )

    slug = f"{t0}_{t1}__{t2}_{t3}".replace("-", "")
    pasta = DADOS / "processed" / f"run_{slug}"
    pasta.mkdir(parents=True, exist_ok=True)

    coint_csv = pasta / "cointegracao.csv"
    parcial_csv = pasta / "pipeline_parcial.csv"
    quebras_csv = pasta / "pipeline_com_quebras.csv"
    hib_csv = pasta / "pipeline_hibrido.csv"
    resumo_csv = pasta / "resumo_ganhos.csv"

    print("=" * 62)
    print("PIPELINE LUIZ — run_all.py")
    print("=" * 62)
    print(f"  Formacao (treino IA) : {t0}  ->  {t1}")
    print(f"  Negociacao (OOS)     : {t2}  ->  {t3}")
    print(f"  Saidas em            : {pasta}")
    print()
    print("  Isolamento temporal:")
    print("    cointegracao  -> so formacao")
    print("    SWANet        -> treina formacao | prevê tudo")
    print("    hibrido       -> PPO filtra o Luiz 1.25 | lag=1 abertura")
    print("    SWANet e hibrido usam o MESMO periodo de treino.")

    # --- ETAPA 1: Cointegracao ---
    etapa(1, "Cointegracao Engle-Granger (SO formacao)")
    info(f"Testando pares de {t0} ate {t1}...")
    rodar([
        PY, str(SRC / "checar_cointegracao.py"),
        "--inicio-formacao", t0,
        "--fim-formacao", t1,
        "--saida", str(coint_csv),
        "--salvar-todos",
    ])
    par = escolher_par(coint_csv, args.setor, *(args.par or (None, None)))
    ok(f"Par escolhido: {par['Ativo Y']}/{par['Ativo X']} ({par['Setor']}) p={par['P-Value']:.2e}")

    # --- ETAPA 2: Pipeline parcial ---
    etapa(2, "Cointegracao parcial + sinais heuristcos")
    info("Kalman MR+RW (bfill removido). Gera coluna 'sinal'.")
    rodar([
        PY, str(SRC / "pipeline_cointegracao_parcial.py"),
        "--cointegracao", str(coint_csv),
        "--setor", str(par["Setor"]),
        "--ativo-y", str(par["Ativo Y"]),
        "--ativo-x", str(par["Ativo X"]),
        "--inicio-formacao", t0,
        "--fim-formacao", t1,
        "--inicio-negociacao", t2,
        "--fim-negociacao", t3,
        "--saida", str(parcial_csv),
    ])
    ok(f"CSV salvo: {parcial_csv.name}")

    # --- ETAPA 3: SWANet ---
    if args.sem_swanet:
        etapa(3, "SWANet — PULADA (--sem-swanet)")
        import shutil
        shutil.copy(parcial_csv, quebras_csv)
        info("prob_quebra ausente; FinRL usara zeros.")
    else:
        etapa(3, "SWANet — TREINA na formacao, PREVE formacao+negociacao")
        info(f"Treino SWANet: {t0} .. {t1} (mesmo recorte da calibracao)")
        rodar([
            PY, str(SRC / "01_swanet_quebras.py"),
            "--entrada", str(parcial_csv),
            "--saida", str(quebras_csv),
            "--inicio-formacao", t0,
            "--fim-formacao", t1,
        ], precisa_finrl=True)
        ok(f"prob_quebra adicionada -> {quebras_csv.name}")

    # --- ETAPA 4: Baseline heuristco (ganhos R$) ---
    etapa(4, "Baseline heuristco — ganhos OOS (coluna 'sinal')")
    sys.path.insert(0, str(SRC))
    from avaliar_ganhos import simular_ganhos, carregar_pipeline  # noqa: E402
    import pandas as pd  # noqa: E402

    df_base, ay, ax, hedge = carregar_pipeline(quebras_csv)
    inicio = pd.Timestamp(t2, tz="UTC")
    fim = pd.Timestamp(t3, tz="UTC")
    neg = df_base[(df_base["data"] >= inicio) & (df_base["data"] <= fim)]
    _, m_base_sem = simular_ganhos(neg, ay, ax, hedge, "sinal", args.capital, 0.0, execucao="abertura")
    _, m_base_com = simular_ganhos(neg, ay, ax, hedge, "sinal", args.capital, 0.0008, execucao="abertura")
    ok(f"Baseline sem custo: PnL R$ {m_base_sem['pnl_liquido']:,.0f} | Sharpe {m_base_sem['sharpe_anualizado']:.2f}")
    ok(f"Baseline 8 bps    : PnL R$ {m_base_com['pnl_liquido']:,.0f} | Sharpe {m_base_com['sharpe_anualizado']:.2f}")

    linhas_resumo = [
        {"estrategia": "baseline_heuristico", "taxa": 0.0, **m_base_sem},
        {"estrategia": "baseline_heuristico", "taxa": 0.0008, **m_base_com},
    ]

    # --- ETAPA 5: DQN (opcional) ---
    if args.com_dqn:
        etapa(5, "DQN original (Modulo 2) — opcional")
        info("Recompensa em Z (nao R$). Mantido intacto da branch Luiz.")
        rodar([
            PY, str(SRC / "02_rl_dqn_trading.py"),
            "--entrada", str(quebras_csv),
            "--inicio-negociacao", t2,
            "--fim-negociacao", t3,
        ], precisa_finrl=True)
        ok("DQN concluido (ver relatorio acima).")
        num_cal = 6
    else:
        num_cal = 5

    # --- ETAPA FinRL sobre Luiz 1.25 ---
    if sem_cal:
        etapa(num_cal, "Hibrido FinRL-sobre-1.25 — PULADO (--sem-calibracao)")
    else:
        etapa(num_cal, "Hibrido FinRL-sobre-1.25 — TREINA formacao, TESTA negociacao")
        info(f"PPO so pode FLAT ou seguir Luiz 1.25 | lag=1 | abertura | {t0} .. {t1}")
        rodar([
            PY, str(SRC / "03_finrl_hibrido.py"),
            "--entrada", str(quebras_csv),
            "--saida", str(hib_csv),
            "--inicio-formacao", t0,
            "--fim-formacao", t1,
            "--inicio-negociacao", t2,
            "--fim-negociacao", t3,
            "--timesteps", str(args.timesteps),
            "--capital", str(args.capital),
            "--execucao", "abertura",
        ], precisa_finrl=True)

        df_hib, _, _, _ = carregar_pipeline(hib_csv)
        neg_hib = df_hib[(df_hib["data"] >= inicio) & (df_hib["data"] <= fim)]
        _, m_hib_sem = simular_ganhos(neg_hib, ay, ax, hedge, "sinal_hibrido", args.capital, 0.0, execucao="abertura")
        _, m_hib_com = simular_ganhos(neg_hib, ay, ax, hedge, "sinal_hibrido", args.capital, 0.0008, execucao="abertura")
        ok(f"Hibrido sem custo: PnL R$ {m_hib_sem['pnl_liquido']:,.0f} | Sharpe {m_hib_sem['sharpe_anualizado']:.2f}")
        ok(f"Hibrido 8 bps    : PnL R$ {m_hib_com['pnl_liquido']:,.0f} | Sharpe {m_hib_com['sharpe_anualizado']:.2f}")
        linhas_resumo.extend([
            {"estrategia": "finrl_sobre_1.25", "taxa": 0.0, **m_hib_sem},
            {"estrategia": "finrl_sobre_1.25", "taxa": 0.0008, **m_hib_com},
        ])

    pd.DataFrame(linhas_resumo).to_csv(resumo_csv, index=False)

    print(f"\n{'=' * 62}")
    print("PIPELINE CONCLUIDO")
    print("=" * 62)
    print(f"  Par          : {par['Ativo Y']}/{par['Ativo X']}")
    print(f"  Formacao     : {t0} -> {t1}  (SWANet + FinRL sobre Luiz 1.25)")
    print(f"  Negociacao   : {t2} -> {t3}  (backtest cego, abertura)")
    print(f"  Resumo       : {resumo_csv}")
    print(f"  Hibrido usa os MESMOS dados de formacao que a SWANet: SIM")


if __name__ == "__main__":
    main()
