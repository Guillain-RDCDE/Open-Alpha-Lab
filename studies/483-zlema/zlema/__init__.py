"""Study 483 — Zero-Lag EMA (ZLEMA, Ehlers).

A mechanical, falsifiable encoding of the *zero-lag exponential moving average*. A plain EMA
lags price by ~ (length-1)/2 bars; Ehlers' fix is to feed the EMA a *de-lagged* input —
``price + (price - price[lag])`` — which extrapolates the recent move forward so the smoothed
line "catches up". The folklore says that removing the lag turns a sluggish trend filter into a
timely one: long when ``price > ZLEMA`` (or on the ZLEMA upcross) gets you in earlier and out
earlier, so it should beat a plain EMA of the same length. We test that as a forward-return
study against a random-entry baseline and a de-lag placebo, with costs — and we put the plain
EMA head-to-head as the thing ZLEMA claims to improve on.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
