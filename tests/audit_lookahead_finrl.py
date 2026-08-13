"""Auditoria dura de look-ahead / timing do pipeline FinRL.

Pergunta central do usuario:
  features[t] -> acao[t] -> retorno[t]   (VAZAMENTO)
ou
  features[t] -> acao[t] -> retorno[t+1] (correto)?

Tambem verifica:
  - alinhamento env.step vs avaliar_ganhos (shift)
  - correlacao acao[t] com retorno do MESMO dia vs dia SEGUINTE
  - se atrasar features em 1 dia destroi o PnL (sintoma de leak)
  - se recompensar com retorno do MESMO dia (bug intencional) explode o Sharpe

Uso:
    .venv-finrl/bin/python tests/audit_lookahead_finrl.py
"""

from __future__ import annotations

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
treinar_ppo = modulo.treinar_ppo
split_formacao_negociacao = modulo.split_formacao_negociacao
from avaliar_ganhos import carregar_pipeline, simular_ganhos  # noqa: E402

CSV = RAIZ / "data/processed/pipeline_com_quebras_TAEE_causal.csv"
if not CSV.exists():
    CSV = RAIZ / "data/processed/pipeline_com_quebras_TAEE.csv"
OOS_SALVO = RAIZ / "models/backtests_sem_vazamento/ppo_small_32_30k_oos.csv"
CAPITAL, TAXA = 100_000.0, 0.0008


def secao(titulo: str) -> None:
    print("\n" + "=" * 72)
    print(titulo)
    print("=" * 72)


def audit_timing_documentado() -> None:
    secao("1) TIMELINE DOCUMENTADA (o que o codigo FAZ)")
    print(
        """
ENV (treino) — src/03_finrl_trading.py
  obs_t  = [zscore[t], pos, prob_quebra[t], tempo]   # zscore usa fechamento de t (Kalman)
  acao_t = politica(obs_t)
  reward = acao_t * n_y * (spread[t+1] - spread[t]) - custo     # retorno t -> t+1
  avanca para t+1

AVALIAR_GANHOS — src/avaliar_ganhos.py
  posicao[t] = sinal[t-1]          # shift(1)
  pnl[t]     = posicao[t] * n_y * (spread[t] - spread[t-1])

PREVER_POSICOES
  grava sinal[t] = acao tomada ao ver obs_t

Juntando:
  sinal[t] (decisao no fechamento t) vira posicao no dia t+1
  e captura (spread[t+1]-spread[t])

Padrao RUIM que procuramos:
  features[t] -> acao[t] -> retorno[t] = spread[t]-spread[t-1]
Padrao ATUAL no codigo:
  features[t] -> acao[t] -> retorno[t+1] = spread[t+1]-spread[t]
"""
    )


def audit_env_igual_avaliar() -> None:
    secao("2) ENV reward acumulado == avaliar_ganhos (com shift)")
    df, y, x, h = carregar_pipeline(CSV)
    treino, teste = split_formacao_negociacao(
        df, "2022-01-01", "2024-12-31", "2025-01-01", "2026-01-01"
    )
    # usa OOS salvo se existir
    if OOS_SALVO.exists():
        dfe, ye, xe, he = carregar_pipeline(OOS_SALVO)
        sinal = dfe["sinal_finrl"].astype(int).to_numpy()
        teste = dfe
        y, x, h = ye, xe, he
    else:
        sinal = teste["sinal"].astype(int).to_numpy()

    env = PairsTradingFinRLEnv(teste, y, x, h, capital=CAPITAL, taxa=TAXA)
    env.reset()
    total = 0.0
    # Replay das acoes gravadas (mapear posicao -> acao)
    mapa = {0: 0, 1: 1, -1: 2}
    for i in range(len(teste) - 1):
        acao = mapa.get(int(sinal[i]), 0)
        _, r, done, _, _ = env.step(acao)
        total += r / 100.0 * CAPITAL
        if done:
            break
    dfa = teste.copy()
    dfa["sinal_finrl"] = sinal
    _, m = simular_ganhos(dfa, y, x, h, "sinal_finrl", CAPITAL, TAXA)
    print(f"  Soma rewards do env (replay): R$ {total:,.4f}")
    print(f"  PnL avaliar_ganhos:           R$ {m['pnl_liquido']:,.4f}")
    print(f"  |delta|: R$ {abs(total - m['pnl_liquido']):,.6f}")
    ok = abs(total - m["pnl_liquido"]) < 1e-4
    print("  =>", "OK alinhados" if ok else "FALHA: env e avaliador divergem")
    return ok


def audit_correlacao_acao_retorno() -> None:
    secao("3) Correlacao acao[t] × retorno mesmo dia vs dia seguinte")
    if not OOS_SALVO.exists():
        print("  SKIP (sem OOS salvo)")
        return
    df, y, x, h = carregar_pipeline(OOS_SALVO)
    spread = df["spread_observado"].astype(float)
    ret_t = spread.diff()  # retorno realizado no dia t (t-1 -> t)
    ret_tp1 = spread.diff().shift(-1)  # retorno t -> t+1
    acao = df["sinal_finrl"].astype(float)
    # Alinhamento: acao[t] deveria prever ret_tp1, NAO ret_t
    c_same = acao.corr(ret_t)
    c_next = acao.corr(ret_tp1)
    print(f"  corr(acao[t], retorno_mesmo_dia[t])     = {c_same:.4f}")
    print(f"  corr(acao[t], retorno_proximo_dia[t+1]) = {c_next:.4f}")
    # zscore tambem correlaciona com retorno do mesmo dia (Kalman atualiza com y_t)
    if "zscore_mr" in df.columns:
        cz = df["zscore_mr"].astype(float).corr(ret_t)
        print(f"  corr(zscore[t], retorno_mesmo_dia[t])   = {cz:.4f}  (esperado != 0: Kalman ve o close de t)")
    print(
        """
  Interpretacao:
  - corr com retorno do MESMO dia pode ser != 0 porque zscore[t] JA incorpora o close de t.
    Isso NAO e vazamento de PnL se o reward/avaliador usa so t->t+1.
  - Se corr com retorno[t] for MUITO maior que com [t+1] E o PnL usasse retorno[t],
    ai sim seria o bug classico. No codigo o PnL usa t->t+1.
"""
    )


def audit_pnl_se_remover_shift() -> None:
    secao("4) E se avaliassemos SEM shift (bug classico features[t]->retorno[t])?")
    if not OOS_SALVO.exists():
        print("  SKIP")
        return
    df, y, x, h = carregar_pipeline(OOS_SALVO)
    # Correto (com shift) — avaliar_ganhos
    _, m_ok = simular_ganhos(df, y, x, h, "sinal_finrl", CAPITAL, TAXA)
    # Bug intencional: posicao[t] = sinal[t] (sem shift)
    dados = df.copy()
    pos = dados["sinal_finrl"].fillna(0).astype(int)
    preco_y = dados[y].astype(float)
    spread = dados["spread_observado"].astype(float)
    n_y = (CAPITAL / 2.0) / float(preco_y.iloc[0])
    pnl_bug = (pos * n_y * spread.diff().fillna(0.0)).sum()
    # custos grosseiros ignorados na comparacao bruta
    print(f"  PnL liquido CORRETO (shift t+1):     R$ {m_ok['pnl_liquido']:,.2f}")
    print(f"  PnL bruto BUG sem shift (mesmo dia): R$ {pnl_bug:,.2f}")
    print(f"  Razao bug/correto: {pnl_bug / m_ok['pnl_liquido'] if m_ok['pnl_liquido'] else float('nan'):.2f}x")
    if abs(pnl_bug) > abs(m_ok["pnl_liquido"]) * 1.5:
        print("  => O bug mesmo-dia DEIXARIA o resultado ainda mais absurdo.")
        print("     O pipeline atual NAO esta nesse modo (usa shift).")
    else:
        print("  => Diferenca moderada; o shift esta ativo no avaliador oficial.")


def audit_atrasar_features() -> None:
    secao("5) Teste duro: atrasar features em 1 dia e retreinar (leak diagnostic)")
    print("  Se houvesse leak features[t]->retorno[t], atrasar features destrói o alpha.")
    print("  Treino curto (15k steps) so para diagnostico...\n")
    df, y, x, h = carregar_pipeline(CSV)
    treino, teste = split_formacao_negociacao(
        df, "2022-01-01", "2024-12-31", "2025-01-01", "2026-01-01"
    )

    def run(tag: str, tr: pd.DataFrame, te: pd.DataFrame) -> dict:
        env_tr = PairsTradingFinRLEnv(tr, y, x, h, capital=CAPITAL, taxa=TAXA)
        modelo = treinar_ppo(env_tr, timesteps=15_000, seed=42, net_arch=[64, 64])
        env_te = PairsTradingFinRLEnv(te.copy(), y, x, h, capital=CAPITAL, taxa=TAXA)
        pos = prever_posicoes(modelo, env_te)
        d = te.copy()
        d["sinal_finrl"] = pos
        _, m = simular_ganhos(d, y, x, h, "sinal_finrl", CAPITAL, TAXA)
        print(
            f"  {tag}: PnL R$ {m['pnl_liquido']:,.0f} | Sharpe {m['sharpe_anualizado']:.2f} | "
            f"pos {m['dias_posicionado_pct']:.1f}%"
        )
        return m

    m_normal = run("features normais (t)", treino, teste)

    tr_d = treino.copy()
    te_d = teste.copy()
    for col in ("zscore_mr", "prob_quebra"):
        tr_d[col] = tr_d[col].shift(1)
        te_d[col] = te_d[col].shift(1)
    tr_d = tr_d.dropna(subset=["zscore_mr"]).reset_index(drop=True)
    te_d = te_d.dropna(subset=["zscore_mr"]).reset_index(drop=True)
    m_delay = run("features ATRASADAS (t-1)", tr_d, te_d)

    # Baseline Luiz no mesmo teste
    _, m_luiz = simular_ganhos(teste, y, x, h, "sinal", CAPITAL, TAXA)
    print(f"  Luiz no mesmo OOS: PnL R$ {m_luiz['pnl_liquido']:,.0f} | Sharpe {m_luiz['sharpe_anualizado']:.2f}")

    print(
        f"""
  Delta PnL normal - atrasado: R$ {m_normal['pnl_liquido'] - m_delay['pnl_liquido']:,.0f}
  Se o atrasado CONTINUA >> Luiz e com Sharpe absurdo, o alpha NAO depende de
  sincronia features[t]/retorno[t] (nao e o bug classico de mesmo dia).
  Se o atrasado COLAPSA, havia dependencia forte de informacao contemporanea
  (ainda pode ser legitima no close, ou leak — interpretar com o resto).
"""
    )


def audit_recompensa_mesmo_dia_intencional() -> None:
    secao("6) Experiencia: e se o ENV pagasse retorno do MESMO dia? (bug forçado)")
    print("  Treina 10k steps com reward = acao * (spread[t]-spread[t-1]) em vez de t->t+1")
    df, y, x, h = carregar_pipeline(CSV)
    treino, teste = split_formacao_negociacao(
        df, "2022-01-01", "2024-12-31", "2025-01-01", "2026-01-01"
    )

    class EnvLeak(PairsTradingFinRLEnv):
        def step(self, action):
            alvo = self.ACAO_PARA_POSICAO[int(action)]
            custo = self.taxa * self.notional[self.i] * abs(alvo - self.posicao)
            # BUG: usa retorno JA realizado no dia t (spread[t]-spread[t-1])
            if self.i == 0:
                ds = 0.0
            else:
                ds = float(self.df["spread_observado"].iloc[self.i] - self.df["spread_observado"].iloc[self.i - 1])
            pnl = alvo * self.n_y * ds
            recompensa = (pnl - custo) / self.capital * 100.0
            self.posicao = alvo
            self.i += 1
            terminado = self.i >= len(self.df) - 1
            return self._obs(), float(recompensa), terminado, False, {"pnl": pnl - custo}

    env_tr = EnvLeak(treino, y, x, h, capital=CAPITAL, taxa=TAXA)
    modelo = treinar_ppo(env_tr, timesteps=10_000, seed=0, net_arch=[32])
    # Avalia no OOS com o avaliador CORRETO (shift) — o que a politica leaky faria "ao vivo"
    env_te = PairsTradingFinRLEnv(teste.copy(), y, x, h, capital=CAPITAL, taxa=TAXA)
    pos = prever_posicoes(modelo, env_te)
    d = teste.copy()
    d["sinal_finrl"] = pos
    _, m = simular_ganhos(d, y, x, h, "sinal_finrl", CAPITAL, TAXA)
    print(f"  Politica treinada com LEAK, avaliada CORRETO: PnL R$ {m['pnl_liquido']:,.0f} | Sharpe {m['sharpe_anualizado']:.2f}")
    print("  (Se o leak fosse o motor do resultado atual, esperaríamos colapso ao avaliar certo.)")


def audit_estatisticas_honestas() -> None:
    secao("7) Clarificar '92-99% positivos' — o que e o que nao e")
    if not OOS_SALVO.exists():
        print("  SKIP")
        return
    df, y, x, h = carregar_pipeline(OOS_SALVO)
    dados, m = simular_ganhos(df, y, x, h, "sinal_finrl", CAPITAL, TAXA)
    r = dados["pnl_liquido"]
    print(f"  % dias POSICIONADOS (nao e win rate): {m['dias_posicionado_pct']:.1f}%")
    print(f"  % dias com pnl > 0: {(r > 0).mean()*100:.1f}%")
    print(f"  % dias com pnl < 0: {(r < 0).mean()*100:.1f}%")
    print(f"  % dias com pnl = 0: {(r == 0).mean()*100:.1f}%")
    mensal = r.groupby(dados["data"].dt.to_period("M")).sum()
    print(f"  % meses com pnl > 0: {(mensal > 0).mean()*100:.1f}%  ({(mensal>0).sum()}/{len(mensal)})")
    print(f"  Max DD: {m['max_drawdown_pct']:.3f}% | Retorno: {m['retorno_pct']:.2f}%")
    print(
        """
  ATENCAO: 92-99% na tabela de splits e '% do TEMPO POSICIONADO', nao '% de trades/dias ganhos'.
  Dias ganhos tipicamente ~60%, nao 99%. Meses positivos 12/12 e que e o padrao estranho.
"""
    )


def audit_kalman_causal() -> None:
    secao("8) Kalman/zscore: usa so passado+presente (filtro forward)?")
    print(
        """
  filtro_kalman_cointegracao_parcial: loop i=0..n-1
    usa spread[i] e parametros[i] (rolling.var ate i, com ffill — sem bfill)
    atualiza estado e grava mr_filtrado[i], zscore[i]

  => zscore[t] depende de closes ate t inclusive. NAO de t+1.
  => Decisao no close de t com zscore[t] e a hipotese 'trade no mesmo close'
     (otimista operacionalmente, mas NAO e reward do mesmo candle).

  Parametros rolling: var ate a barra i (pandas rolling e backward-looking).
  ffill propaga ultimo valor passado; bfill foi removido.
"""
    )


def main() -> None:
    print(f"CSV base: {CSV}")
    audit_timing_documentado()
    audit_kalman_causal()
    audit_env_igual_avaliar()
    audit_correlacao_acao_retorno()
    audit_pnl_se_remover_shift()
    audit_estatisticas_honestas()
    audit_atrasar_features()
    audit_recompensa_mesmo_dia_intencional()
    secao("VEREDITO PROVISORIO")
    print(
        """
  1. O codigo atual NAO implementa o bug classico features[t]->acao[t]->retorno[t].
     Treino e avaliacao usam retorno t->t+1 (env dspread[i] + avaliar shift(1)).

  2. zscore[t] / Kalman VEEM o fechamento de t. Isso e 'same-close decision',
     nao look-ahead de retorno do mesmo dia no PnL.

  3. O padrao Sharpe diario~5 + mensal~7 + DD<1% + 12/12 meses ainda e
     ECONOMICAMENTE suspeito (overfit, regime, custos irrealistas, exposicao 97%).
     Suspeito != prova de look-ahead no timing de retorno.

  4. Proximo passo se quiser endurecer: decidir so com features[t-1]
     (execucao no open/close seguinte) e re-reportar a tabela.
"""
    )


if __name__ == "__main__":
    main()
