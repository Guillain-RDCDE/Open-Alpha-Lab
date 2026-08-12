"""Study 883 — Mid-Cap Sweet Spot.

Is the "forgotten middle" — the mid-cap ETF (IJH / MDY) — a genuine risk-adjusted
sweet spot that beats BOTH large (SPY) and small (IWM) on an excess-of-cash Sharpe
basis, robustly across eras and after costs?

Engine:
  * ``data``     — real ETF total-return tape (yfinance, cached parquet) + a seeded
                   synthetic world with a tunable planted mid-cap Sharpe edge.
  * ``strategy`` — the excess-vs-excess Sharpe race, HAC *t* on the pairwise return
                   difference, a paired block-bootstrap Sharpe-advantage CI, an era
                   cut, a calendar-year table, drawdowns, and the costed spread.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
