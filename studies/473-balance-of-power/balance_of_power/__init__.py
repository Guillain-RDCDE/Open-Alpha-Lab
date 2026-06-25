"""Study 473 — Balance of Power (Igor Livshin).

A mechanical, falsifiable encoding of Igor Livshin's *Balance of Power* (BOP):
BOP = (close - open) / (high - low), a per-bar tug-of-war between buyers and sellers,
smoothed with a moving average. The folklore says smoothed BOP **leads price** — buyers
gaining the upper hand (smoothed BOP turning positive) forecasts a rise. We test that as a
forward-return study: long when smoothed BOP crosses up through zero, entered the next close,
against a drift-matched random-entry baseline and a sign-scramble placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
