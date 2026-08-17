"""Study 917 — Stale NAV: does a strong US session predict tomorrow's country ETF?

Two modules:

- :mod:`stale_nav.data` — the tape (cached daily total-return closes for SPY and five
  US-listed single-country ETFs, plus a cash leg) and the deterministic synthetic panel.
- :mod:`stale_nav.strategy` — the lagged HAC regressions, the top-decile next-day rule,
  the domestic (SPY-on-SPY) control, the era cut, and the cost / borrow sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
