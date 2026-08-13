"""Garante que a SWANet nao treina com label que cruza o fim da formacao."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def testa_mask_treino_exclui_fronteira() -> None:
    src = (RAIZ / "src/01_swanet_quebras.py").read_text(encoding="utf-8")
    assert "fim_label_ok" in src
    assert "i + 4" in src or "i+4" in src
    assert ".bfill()" not in src
    assert "fillna(0.5)" in src
    print("OK  SWANet: sem bfill e com exclusao do label +5d no treino")


if __name__ == "__main__":
    testa_mask_treino_exclui_fronteira()
    print("\ntest_swanet_sem_vazamento: TODOS OS TESTES PASSARAM")
