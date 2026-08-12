"""Modulo 3: calibra o limiar de entrada do Luiz na formacao (substitui FinRL).

Luiz continua IGUAL (mesma regra, mesmo Kalman, mesmo z-score).
So muda o numero do limiar (padrao fixo 1.25 -> melhor valor na formacao).

Treino  : grid search do limiar no periodo de FORMACAO (sem olhar OOS).
Teste   : aplica o limiar escolhido na NEGOCIACAO (backtest cego).

Saida:
  - coluna ``sinal``           = Luiz original (limiar 1.25 do pipeline)
  - coluna ``sinal_calibrado`` = mesma regra, limiar otimizado na formacao

Uso:
    python3 src/03_calibrar_limiar.py \\
      --entrada data/processed/pipeline_com_quebras_TAEE_causal.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from avaliar_ganhos import carregar_pipeline, simular_ganhos
from periodos import (
    DATA_FIM_FORMACAO,
    DATA_FIM_NEGOCIACAO,
    DATA_INICIO_FORMACAO,
    DATA_INICIO_NEGOCIACAO,
)
from pipeline_cointegracao_parcial import gerar_sinais_artigo


def gerar_sinal_com_limiar(df: pd.DataFrame, limiar: float) -> pd.Series:
    """Reaplica a regra Luiz com outro limiar (mr_filtrado/std_mr ja calculados)."""
    if "mr_filtrado" not in df.columns or "std_mr" not in df.columns:
        raise ValueError("CSV precisa de mr_filtrado e std_mr (rode o pipeline parcial).")
    out = gerar_sinais_artigo(
        df["mr_filtrado"].astype(float),
        df["std_mr"].astype(float),
        float(limiar),
    )
    return out["sinal"].astype(int)


def recortar(df: pd.DataFrame, inicio: str, fim: str) -> pd.DataFrame:
    t0 = pd.to_datetime(inicio).tz_localize("UTC")
    t1 = pd.to_datetime(fim).tz_localize("UTC")
    return df[(df["data"] >= t0) & (df["data"] <= t1)].reset_index(drop=True)


def score_formacao(metricas: dict, min_trades: int = 3) -> float:
    """Funcao objetivo conservadora: Calmar com penalidade se poucos trades."""
    pnl = float(metricas["pnl_liquido"])
    dd = abs(float(metricas["max_drawdown_pct"]))
    trades = int(metricas["trades_fechados"])
    if trades < min_trades:
        return -np.inf
    if dd < 1e-6:
        return pnl
    return pnl / dd


def grid_search_limiar(
    df_formacao: pd.DataFrame,
    ativo_y: str,
    ativo_x: str,
    hedge: float,
    limiares: list[float],
    capital: float,
    taxa: float,
    execucao: str,
    dados_setores: Path | None,
    min_trades: int,
    criterio: str = "calmar",
) -> tuple[float, pd.DataFrame]:
    """Testa cada limiar so na formacao; retorna o melhor e tabela de resultados."""

    def pontua(m: dict) -> float:
        trades = int(m["trades_fechados"])
        if trades < min_trades:
            return -np.inf
        if criterio == "pnl":
            return float(m["pnl_liquido"])
        dd = abs(float(m["max_drawdown_pct"]))
        pnl = float(m["pnl_liquido"])
        if dd < 1e-6:
            return pnl
        return pnl / dd

    linhas = []
    melhor_limiar = limiares[0]
    melhor_score = -np.inf

    for lim in limiares:
        df_try = df_formacao.copy()
        df_try["sinal_try"] = gerar_sinal_com_limiar(df_try, lim)
        _, m = simular_ganhos(
            df_try,
            ativo_y,
            ativo_x,
            hedge,
            coluna_sinal="sinal_try",
            capital=capital,
            taxa=taxa,
            execucao=execucao,
            dados_setores=dados_setores,
        )
        sc = pontua(m)
        linhas.append({"limiar": lim, "score": sc, **m})
        if sc > melhor_score:
            melhor_score = sc
            melhor_limiar = lim

    return melhor_limiar, pd.DataFrame(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibra limiar Luiz na formacao (grid search, sem FinRL)."
    )
    parser.add_argument("--entrada", type=Path, default=Path("data/processed/pipeline_com_quebras.csv"))
    parser.add_argument("--saida", type=Path, default=None)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--taxa", type=float, default=0.0008)
    parser.add_argument(
        "--limiares",
        type=float,
        nargs="+",
        default=[0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
    )
    parser.add_argument(
        "--criterio",
        choices=("calmar", "pnl"),
        default="calmar",
        help="calmar=pnl/|DD| na formacao; pnl=so lucro.",
    )
    parser.add_argument("--min-trades", type=int, default=3)
    parser.add_argument(
        "--execucao",
        choices=("abertura", "fechamento"),
        default="abertura",
    )
    parser.add_argument("--dados-setores", type=Path, default=Path("data/raw/setores"))
    parser.add_argument("--inicio-formacao", default=DATA_INICIO_FORMACAO)
    parser.add_argument("--fim-formacao", default=DATA_FIM_FORMACAO)
    parser.add_argument("--inicio-negociacao", default=DATA_INICIO_NEGOCIACAO)
    parser.add_argument("--fim-negociacao", default=DATA_FIM_NEGOCIACAO)
    parser.add_argument("--limiar-fixo-referencia", type=float, default=1.25)
    args = parser.parse_args()

    df, ativo_y, ativo_x, hedge = carregar_pipeline(args.entrada)
    if "mr_filtrado" not in df.columns:
        raise ValueError(f"{args.entrada} sem mr_filtrado — rode pipeline_cointegracao_parcial.py")

    formacao = recortar(df, args.inicio_formacao, args.fim_formacao)
    negociacao = recortar(df, args.inicio_negociacao, args.fim_negociacao)
    if formacao.empty or negociacao.empty:
        raise ValueError("Recorte formacao ou negociacao vazio.")

    print(
        f"CALIBRAR LIMIAR {ativo_y}/{ativo_x} | formacao {len(formacao)}d | "
        f"OOS {len(negociacao)}d | exec={args.execucao} | grid={args.limiares}"
    )

    melhor, tabela = grid_search_limiar(
        formacao,
        ativo_y,
        ativo_x,
        hedge,
        args.limiares,
        args.capital,
        args.taxa,
        args.execucao,
        args.dados_setores,
        args.min_trades,
        criterio=args.criterio,
    )

    print(f"\n--- Formacao (escolha do limiar, criterio={args.criterio}) ---")
    for _, row in tabela.sort_values("limiar").iterrows():
        marca = " <-- MELHOR" if row["limiar"] == melhor else ""
        print(
            f"  limiar {row['limiar']:4.2f}  PnL R$ {row['pnl_liquido']:8,.0f}  "
            f"DD {row['max_drawdown_pct']:6.2f}%  trades {int(row['trades_fechados']):2d}  "
            f"score {row['score']:8.2f}{marca}"
        )

    # Aplica limiar escolhido em formacao+negociacao (sinal_calibrado)
    df_out = df.copy()
    df_out["sinal_calibrado"] = gerar_sinal_com_limiar(df_out, melhor)

    # Garante sinal original 1.25 para comparacao justa
    if "sinal" not in df_out.columns or args.limiar_fixo_referencia != 1.25:
        df_out["sinal"] = gerar_sinal_com_limiar(df_out, args.limiar_fixo_referencia)

    neg = recortar(df_out, args.inicio_negociacao, args.fim_negociacao)
    print(f"\n--- OOS (limiar fixo na formacao = {melhor:.2f}) ---")
    for col, nome in (
        ("sinal", f"Luiz fixo {args.limiar_fixo_referencia:.2f}"),
        ("sinal_calibrado", f"Luiz calibrado {melhor:.2f}"),
    ):
        _, m = simular_ganhos(
            neg,
            ativo_y,
            ativo_x,
            hedge,
            col,
            args.capital,
            args.taxa,
            execucao=args.execucao,
            dados_setores=args.dados_setores,
        )
        print(
            f"  {nome:22s} ({args.taxa*1e4:.0f} bps): PnL R$ {m['pnl_liquido']:,.0f} | "
            f"DD {m['max_drawdown_pct']:.2f}% | pos {m['dias_posicionado_pct']:.1f}% | "
            f"trades {m['trades_fechados']}"
        )

    saida = args.saida or args.entrada.with_name(args.entrada.stem + "_limiar.csv")
    saida.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(saida, index=False)

    meta = {
        "par": f"{ativo_y}/{ativo_x}",
        "limiar_escolhido": melhor,
        "limiar_referencia": args.limiar_fixo_referencia,
        "formacao": [args.inicio_formacao, args.fim_formacao],
        "negociacao": [args.inicio_negociacao, args.fim_negociacao],
        "execucao": args.execucao,
        "criterio": args.criterio,
        "grid": args.limiares,
    }
    meta_path = saida.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    grid_path = saida.with_name(saida.stem + "_grid_formacao.csv")
    tabela.to_csv(grid_path, index=False)

    print(f"\nSalvo: {saida}")
    print(f"Grid formacao: {grid_path}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
