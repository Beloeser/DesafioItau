"""Modulo 3 legado: redireciona para calibracao de limiar (FinRL removido).

Mantido para nao quebrar scripts que ainda chamam 03_finrl_trading.py.
"""

from __future__ import annotations

import sys
import warnings

warnings.warn(
    "03_finrl_trading.py foi substituido por 03_calibrar_limiar.py (grid do limiar 1.25).",
    DeprecationWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    from importlib import import_module

    mod = import_module("03_calibrar_limiar")
    mod.main()
    sys.exit(0)
