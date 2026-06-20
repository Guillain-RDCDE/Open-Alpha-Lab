"""Study 338 — Preferred-Stocks: are preferred shares (PFF) bond-like safety, or
equity-like crash risk wearing a fat coupon?

We pit PFF against a bond proxy (IEF) and an equity proxy (SPY) on yield, volatility,
and — the decisive test — *who it moves with in the crash*. The marketing sells a
bond-with-a-better-yield; the question is whether the drawdown co-movement says
otherwise. Distinct from Study 97 (fixed 60/40) and Study 69 (gold safe-haven): this
is a single-instrument identity test, not an allocation or a hedge race.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
