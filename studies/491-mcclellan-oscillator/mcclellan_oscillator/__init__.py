"""Study 491 — McClellan Oscillator (breadth-momentum).

Sherman & Marian McClellan's oscillator is a breadth-momentum indicator: take the
daily *net advances* (advancing issues minus declining issues) across a market
basket, and compute the difference of two exponential moving averages of that net
breadth — a fast EMA(19) minus a slow EMA(39). The folklore says the oscillator
*forecasts the index*: when it crosses up from negative territory, breadth momentum
has turned and the market is about to rally.

We encode that as a falsifiable forward-return study on SPY (long on the up-cross
from negative, entered the next close), and we test it against the only honest
benchmark on an upward-drifting tape — a drift-matched **random-entry** baseline —
plus a **shuffled-breadth placebo** that destroys the oscillator's geometry while
keeping its marginal. A deterministic synthetic control proves the detector is live.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
