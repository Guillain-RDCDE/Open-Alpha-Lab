"""Study 488 — FRAMA (Fractal Adaptive Moving Average).

A mechanical, falsifiable encoding of John Ehlers' FRAMA: a moving average whose smoothing
constant adapts to the *fractal dimension* of recent price — fast in trends (D->1), slow in
chop (D->2). The folklore says this fractal-adaptive smoothing lets a ``price > FRAMA``
cross-up long catch trends sooner and dodge whipsaws, beating a plain fixed-length EMA. We test
that as a forward-return study against a drift-matched random-entry baseline, a fixed-EMA
comparator, and a shuffled-alpha placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
