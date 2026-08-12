"""Study 900 — Quality-Income.

High dividend YIELD is a notorious value-trap magnet — the fattest yields often mark
distressed payers about to cut. QUALITY-dividend screens (durable, growing payers)
were sold as the fix. We race a **quality-dividend sleeve** (SCHD + NOBL) against a
**raw high-yield sleeve** (SPHD + VYM) and against **SPY**, all on monthly total
returns, all measured **excess of cash** (minus BIL), asking: does screening dividends
for *quality* deliver a better excess-of-cash Sharpe and a shallower drawdown than
*chasing yield*, net of costs?

* ``data``     — the real tape (yfinance daily total-return closes for SCHD, NOBL, VYM,
                 SPHD, SPY, BIL, cached under this study's own ``_cache/``) plus a
                 deterministic seeded synthetic control with a TUNABLE planted
                 quality-over-yield Sharpe edge (null at ``edge=0``).
* ``strategy`` — the equal-weight monthly-rebalanced sleeves, the excess-of-cash Sharpe
                 race, the HAC *t* on the quality-minus-yield monthly difference, a
                 paired block-bootstrap Sharpe-gap CI, max drawdown, the calendar-year
                 table, an era cut, and a costed (turnover x spread) net version.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
