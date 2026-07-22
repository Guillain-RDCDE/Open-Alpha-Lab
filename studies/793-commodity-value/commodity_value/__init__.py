"""Study 793 — Cross-sectional commodity *value* (AMP 5-year long-horizon reversal).

Two modules:

* :mod:`commodity_value.data` — the real ETF-basket tape (yfinance, cached under the
  study's own ``_cache/`` with both a raw-price series for the value signal and a
  total-return series for the P&L) plus a deterministic seeded synthetic **price** panel
  with a TUNABLE planted long-horizon reversal (null at ``val_edge = 0``).
* :mod:`commodity_value.strategy` — the 5-year value sort (long the cheap/fallen third,
  short the expensive/risen third), the one-lag L/S book, a costed timer (one-way x NAV,
  shorts pay borrow), and the inference primitives (HAC/Newey-West t, one-sample t,
  annualised Sharpe, a random-rank placebo, a sub-period contrast).
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
