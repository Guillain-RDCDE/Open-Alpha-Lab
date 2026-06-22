"""Study 355 — Magnificent-Seven (a genuine factor, or selection-after-the-fact?).

Tears down the "Mag 7" basket (AAPL, MSFT, GOOGL/GOOG, AMZN, NVDA, META, TSLA), equal-weight,
raced against the cap-weighted S&P 500 and an equal-weight large-cap basket. The recent spread
is **real** on the tape — but the basket is *named because it won*: the seven are the ex-post
top-of-the-pile of a much larger mega-cap field, chosen with the full sample in view. The honest
control is a **look-ahead placebo**: pick "the 7 best names, in hindsight" from a pre-2015
universe and you manufacture an even bigger spread out of pure selection. So the Signal axis is
``MIXED`` (real recent spread, look-ahead/survivorship selected), Tradability ``MIRAGE`` (you
could not have named these seven in 2015), and "a repeatable strategy?" is ``BUSTED``.

See :mod:`magnificent_seven.data` (real yfinance monthly panel + deterministic synthetic
positive control) and :mod:`magnificent_seven.strategy` (basket vs benchmark race, HAC t-stat,
the ex-post-selection placebo, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
