"""Study 893 — Vol-Target 60/40.

A volatility thermostat on the balanced book: scale the static 60% SPY / 40% IEF blend's total
exposure by inverse realized *portfolio* vol to a constant target, and test whether the re-timed
book beats the static 60/40 on excess-of-cash Sharpe and drawdown, net of the extra turnover.

* :mod:`vt6040.data` — the synthetic (offline, regime-clustered) and real (cached total-return)
  tapes; SPY / IEF / AGG / BIL, as-of 2026-06-30.
* :mod:`vt6040.strategy` — the static blend, the past-only inverse-vol overlay, excess-of-cash
  stats, the HAC *t* on the return difference, a circular block-bootstrap Sharpe-diff CI, a costed
  race, and the synthetic-control detector.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
