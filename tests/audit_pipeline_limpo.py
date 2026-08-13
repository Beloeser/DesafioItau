"""Auditoria estatica: o pipeline Luiz nao pode ter bfill nem label vazando.

Roda:
    python3 tests/audit_pipeline_limpo.py
"""

from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SRC = RAIZ / "src"


def _ler(nome: str) -> str:
    return (SRC / nome).read_text(encoding="utf-8")


def testa_sem_bfill_executavel() -> None:
    """Comentarios podem citar bfill; chamada .bfill() e proibida."""
    for nome in ("pipeline_cointegracao_parcial.py", "01_swanet_quebras.py"):
        src = _ler(nome)
        assert ".bfill()" not in src, f"{nome} ainda chama .bfill()"
    print("OK  nenhum .bfill() em Kalman/SWANet")


def testa_swanet_label_nao_cruza_formacao() -> None:
    src = _ler("01_swanet_quebras.py")
    assert "fim_label_ok" in src
    assert "i + 4" in src or "i+4" in src
    assert "fillna(0.5)" in src
    print("OK  SWANet exclui label +5d que cruza o fim da formacao")


def testa_kalman_sem_copiar_futuro() -> None:
    src = _ler("pipeline_cointegracao_parcial.py")
    assert "return parametros.ffill()" in src
    assert "if not np.isfinite(rho)" in src
    print("OK  Kalman usa ffill + NaN neutro (sem copiar futuro)")


def testa_avaliador_padrao_abertura() -> None:
    src = _ler("avaliar_ganhos.py")
    assert 'default="abertura"' in src
    assert "shift(1)" in src
    print("OK  avaliador padrao = abertura + sinal de ontem")


if __name__ == "__main__":
    testa_sem_bfill_executavel()
    testa_swanet_label_nao_cruza_formacao()
    testa_kalman_sem_copiar_futuro()
    testa_avaliador_padrao_abertura()
    print("\naudit_pipeline_limpo: TODOS OS TESTES PASSARAM")
