"""Study 688 — Concealing Baby Swallow.

Package layout mirrors the rest of the desk:

* :mod:`concealing_baby_swallow.data` — the real Yahoo! tape (a very large, cache-first
  basket) and a deterministic synthetic positive control.
* :mod:`concealing_baby_swallow.strategy` — the four-candle detector (loose + strict
  cuts), the forward-return event study, its base rate, and the honest small-*n*
  discipline (no *t*-stat theatre below :data:`MIN_N_FOR_TEST`).
"""

from . import data, strategy

__all__ = ["data", "strategy"]
