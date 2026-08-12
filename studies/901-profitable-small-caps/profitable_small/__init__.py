"""Study 901 — Profitable Small-Caps ("the size effect, cleaned").

The size premium mostly lives in **profitable** small caps; junky small caps drag it
toward zero (Asness, Frazzini, Israel, Moskowitz & Pedersen 2018, *Size Matters, If You
Control Your Junk*). This study asks the tradable version of that claim: does a
small-cap **quality/profitability** ETF (CALF cash-cow, XSHQ small quality) beat plain
small caps (IWM, IJR) and SPY on an **excess-of-cash Sharpe** basis, once you strip the
size and value tilts and pay real costs?

Two layers, both offline once cached:

* ``data`` — real ETF total-return tape (yfinance) + a deterministic synthetic control.
* ``strategy`` — excess-of-cash Sharpe races, HAC t on the return difference, a bootstrap
  Sharpe CI, drawdowns, a calendar-year table, an era cut, a size/value beta decomposition,
  and a costed net version.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
