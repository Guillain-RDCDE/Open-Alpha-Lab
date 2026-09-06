"""Study 990 — Counting the Breaks.

Value-at-Risk makes a falsifiable promise: at 99% confidence, the loss should exceed
the number on one day in a hundred. That promise can be checked by counting, and the counting
has a proper statistical apparatus behind it — Kupiec's unconditional-coverage test, Christoffersen's
independence test, and the joint test that combines them. This study runs that apparatus over
five standard VaR models on six assets, and gives as much attention to the *power* of the tests
as to their verdicts, because a backtest that cannot reject a badly wrong model is not a
backtest.

- :mod:`breaks.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`breaks.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
