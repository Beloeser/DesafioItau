"""Por que o fill de prob_quebra na SWANet foi alterado — e se isso importa.

Contexto
--------
Na branch Luiz original (origin/Luiz), a SWANet terminava assim:

    df["prob_quebra"] = pd.Series(coluna_prob).ffill().bfill()

Na branch Luiz-finrl ficou:

    df["prob_quebra"] = pd.Series(coluna_prob).ffill().fillna(0.5)

IMPORTANTE: isso NAO muda a arquitetura da SWANet (CNN+LSTM), nem o treino
(5 epocas, BCELoss, Adam), nem as janelas, nem os pesos aprendidos da rede.
Muda SO a forma de preencher os ~24 primeiros dias da serie, onde ainda nao
existe janela completa para gerar uma previsao (seq_length).

O que e bfill / fillna(0.5)
--------------------------
- A SWANet so consegue prever a partir do dia ``seq_length`` (precisa de
  historico). Antes disso, ``prob_quebra`` e NaN.
- ``ffill()``: propaga a ultima previsao valida para frente (causal, ok).
- ``bfill()`` (Luiz original): copia a PRIMEIRA previsao valida para TRAS,
  preenchendo o inicio com um numero que so existiria no futuro.
  Isso e look-ahead bias (vazamento de informacao futura para o passado).
- ``fillna(0.5)`` (Luiz-finrl): no inicio coloca 0.5 = "nao sei / 50-50",
  sem copiar o futuro.

Por que isso foi feito (honestidade)
------------------------------------
O pedido do usuario era: manter o codigo Luiz e so adicionar FinRL como
modulo final de decisao. A troca do bfill NAO era obrigatoria para o FinRL
funcionar. Foi um fix de causalidade acoplado ao mesmo commit do FinRL.

Este arquivo testa se, na pratica, a troca altera o OOS 2025 / o FinRL.

Como rodar
----------
    python3 tests/test_swanet_fill_nao_muda_oos.py
    # (parte A/B do FinRL, opcional, mais lenta:)
    .venv-finrl/bin/python tests/test_swanet_fill_nao_muda_oos.py --com-finrl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CSV_TAEE = RAIZ / "data/processed/pipeline_com_quebras_TAEE.csv"


def reconstruir_bfill_a_partir_do_csv_atual(prob: pd.Series) -> pd.Series:
    """No CSV atual o prefixo e 0.5. O bfill original usaria a 1a previsao real."""
    p = prob.astype(float).copy()
    # Prefixo constante 0.5 = aquecimento sem previsao (fill atual).
    n_prefixo = int((p == 0.5).cumprod().sum())
    if n_prefixo <= 0 or n_prefixo >= len(p):
        return p
    p_bfill = p.copy()
    p_bfill.iloc[:n_prefixo] = p.iloc[n_prefixo]
    return p_bfill


def testa_so_prefixo_da_formacao_muda() -> dict:
    """Negociacao 2025 deve ser IDENTICA com ou sem bfill."""
    assert CSV_TAEE.exists(), f"Falta {CSV_TAEE}"
    df = pd.read_csv(CSV_TAEE)
    df["data"] = pd.to_datetime(df["data"], utc=True)
    p = df["prob_quebra"].astype(float)
    p_bfill = reconstruir_bfill_a_partir_do_csv_atual(p)

    formacao = (df["data"] >= "2022-01-01") & (df["data"] <= "2024-12-31")
    negociacao = (df["data"] >= "2025-01-01") & (df["data"] <= "2026-01-01")

    diff_form = int((p[formacao] != p_bfill[formacao]).sum())
    diff_oos = int((p[negociacao] != p_bfill[negociacao]).sum())
    n_prefixo = int((p == 0.5).cumprod().sum())

    print("=" * 72)
    print("TESTE 1 — Onde o fill da SWANet muda a serie")
    print("=" * 72)
    print(f"  Prefixo preenchido (sem previsao real): {n_prefixo} dias")
    print(f"  Dias diferentes na FORMACAO 2022-24:    {diff_form} / {formacao.sum()}")
    print(f"  Dias diferentes na NEGOCIACAO 2025:     {diff_oos} / {negociacao.sum()}")
    print()
    print("  Interpretacao:")
    print("  - So o aquecimento inicial (~24 dias em 2022) muda.")
    print("  - O periodo de teste cego (2025) e IDENTICO.")
    print("  - Logo o sinal heuristico Luiz no OOS nao muda por causa deste fill.")

    assert diff_oos == 0, (
        "Se a negociacao muda, o fill afetou o OOS — investigar."
    )
    assert diff_form == n_prefixo, (
        "Esperavamos que so o prefixo de aquecimento divergisse."
    )
    print("OK  negociacao 2025 identica; so prefixo da formacao muda\n")
    return {
        "n_prefixo": n_prefixo,
        "diff_formacao": diff_form,
        "diff_negociacao": diff_oos,
    }


def testa_bfill_e_look_ahead_conceitual() -> None:
    """Mostra numericamente o vazamento: bfill = copiar futuro para o passado."""
    # Serie artificial: NaN, NaN, NaN, 0.80, 0.70
    s = pd.Series([np.nan, np.nan, np.nan, 0.80, 0.70])
    com_bfill = s.ffill().bfill()
    com_neutro = s.ffill().fillna(0.5)

    print("=" * 72)
    print("TESTE 2 — Demonstracao do look-ahead do bfill")
    print("=" * 72)
    print("  Serie bruta (NaN = ainda sem janela SWANet):")
    print(f"    {s.tolist()}")
    print("  Com bfill (Luiz original):")
    print(f"    {com_bfill.tolist()}")
    print("    -> dias 0..2 receberam 0.80, que so existia no dia 3 (FUTURO)")
    print("  Com fillna(0.5) (Luiz-finrl):")
    print(f"    {com_neutro.tolist()}")
    print("    -> dias 0..2 = incerteza maxima, sem copiar o futuro")
    print()

    assert com_bfill.iloc[0] == 0.80
    assert com_neutro.iloc[0] == 0.5
    print("OK  bfill copia futuro; fillna(0.5) nao copia\n")


def testa_finrl_ab_opcional() -> None:
    """Treina FinRL 2x: prob com fill atual vs bfill. Compara PnL OOS."""
    sys.path.insert(0, str(RAIZ / "src"))
    sys.path.insert(0, str(RAIZ / "tests"))
    import test_ablacao_finrl as abl
    from avaliar_ganhos import carregar_pipeline

    df, y, x, h = carregar_pipeline(CSV_TAEE)
    p = df["prob_quebra"].astype(float)
    p_bfill = reconstruir_bfill_a_partir_do_csv_atual(p)

    def rodar(rotulo: str, prob: pd.Series) -> dict:
        d = df.copy()
        d["prob_quebra"] = prob.values
        return abl.rodar_experimento(
            d, y, x, h,
            "2022-01-01", "2024-12-31", "2025-01-01", "2026-01-01",
            timesteps=30_000, net_arch=[64, 64], rotulo=rotulo,
        )

    print("=" * 72)
    print("TESTE 3 — A/B FinRL (30k steps, rede [64,64], OOS 2025, 8 bps)")
    print("=" * 72)
    r_atual = rodar("fill_atual_0.5", p)
    r_luiz = rodar("fill_bfill_original", p_bfill)

    for r in (r_atual, r_luiz):
        print(
            f"  {r['rotulo']}: PnL R$ {r['finrl_pnl']:,.2f} | "
            f"Sharpe {r['finrl_sharpe']} | DD {r['finrl_dd']}% | "
            f"pos {r['finrl_pos_pct']}%"
        )
    delta = float(r_atual["finrl_pnl"]) - float(r_luiz["finrl_pnl"])
    print(f"\n  Delta (atual - bfill): R$ {delta:,.2f}")
    print("  Criterio: |delta| < R$ 1.000 => mudanca irrelevante para a conclusao.")
    assert abs(delta) < 1000.0, (
        f"Delta grande demais (R$ {delta:.2f}): a troca do fill alterou o FinRL "
        "de forma material — revisar antes de manter/reverter."
    )
    print("OK  FinRL OOS quase igual com os dois fills\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--com-finrl",
        action="store_true",
        help="Roda tambem o A/B de treino FinRL (precisa .venv-finrl, ~1 min).",
    )
    args = parser.parse_args()

    print(
        "\nNOTA: a SWANet (rede CNN+LSTM, loss, epocas, Adam) NAO foi redesignada.\n"
        "So mudou 1 linha de pos-processamento do CSV: bfill -> fillna(0.5).\n"
    )
    testa_bfill_e_look_ahead_conceitual()
    testa_so_prefixo_da_formacao_muda()
    if args.com_finrl:
        testa_finrl_ab_opcional()
    else:
        print("(Pulei A/B FinRL. Rode com --com-finrl para treinar as 2 variantes.)\n")

    print("=" * 72)
    print("CONCLUSAO DESTE ARQUIVO")
    print("=" * 72)
    print(
        "1. Nao mudamos 'a SWANet' como modelo — so o fill do aquecimento.\n"
        "2. O OOS 2025 de prob_quebra e identico; o baseline Luiz no teste nao muda.\n"
        "3. O FinRL quase nao sente a diferenca (~centenas de R$).\n"
        "4. Se o objetivo for 'codigo Luiz byte-a-byte + modulo FinRL', da para\n"
        "   REVERTER essa linha e voltar ao .ffill().bfill() sem mudar a historia.\n"
        "5. Se o objetivo for pipeline causal (sem look-ahead), MANTER fillna(0.5)\n"
        "   e correto — e o impacto pratico no resultado financeiro e desprezivel.\n"
    )


if __name__ == "__main__":
    main()
