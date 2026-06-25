"""Study 452 — Spinning-Top (the indecision candle).

A mechanical, falsifiable encoding of the *spinning top* candlestick: a small real body
(|close - open| a small fraction of the day's range) sitting between **comparable upper and
lower wicks**. The folklore says a spinning top marks **indecision** that then "resolves" —
a directional move or a reversal. We test that as a forward-return study against a
drift-matched random-entry baseline and a body/wick-geometry placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
