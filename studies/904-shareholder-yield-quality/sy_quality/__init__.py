"""Study 904 — Shareholder-Yield + Quality.

A raw buyback / shareholder-yield screen scoops up every serial repurchaser — including
the ones running dilution theatre (buybacks that only mop up option grants) or buying
back stock at rich prices. The pitch: overlay a QUALITY screen and you keep the *funded,
value-accretive* buyers and drop the theatre, so a **quality-screened shareholder-yield**
sleeve should beat both a raw buyback sleeve and the plain market on risk-adjusted terms.

We race a **quality-screened sleeve** (PKW + QUAL) against **raw buyback** (PKW) and
against **SPY**, all on monthly total returns, all measured **excess of cash** (minus
BIL), asking: does the quality overlay on shareholder yield deliver a better
excess-of-cash Sharpe than raw buybacks and than the market, net of costs?

* ``data``     — the real tape (yfinance daily total-return closes for PKW, QUAL, SPYD,
                 SPY, BIL, BUYB, cached under this study's own ``_cache/``) plus a
                 deterministic seeded synthetic control with a TUNABLE planted
                 quality-over-raw Sharpe edge (null at ``edge=0``).
* ``strategy`` — the equal-weight monthly-rebalanced sleeves, the excess-of-cash Sharpe
                 race, the HAC *t* on the QSY-minus-raw / QSY-minus-SPY monthly
                 differences, a paired block-bootstrap Sharpe-gap CI, max drawdown, the
                 calendar-year table, an era cut, and a costed (turnover x spread) net
                 version.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
