"""Study 792 — Cross-sectional commodity momentum (Miffre-Rallis 12-1).

Two modules:

* :mod:`commodity_momentum.data` — the real ETF-basket tape (yfinance, cached under the
  study's own ``_cache/``) plus a deterministic seeded synthetic panel with a TUNABLE
  planted cross-sectional momentum effect (null at ``mom_edge = 0``).
* :mod:`commodity_momentum.strategy` — the 12-1 cross-sectional sort, the one-lag L/S
  book, a costed timer (one-way x NAV, shorts pay borrow), and the inference primitives
  (HAC/Newey-West t, one-sample t, annualised Sharpe, a random-rank placebo, a
  sub-period contrast).
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
