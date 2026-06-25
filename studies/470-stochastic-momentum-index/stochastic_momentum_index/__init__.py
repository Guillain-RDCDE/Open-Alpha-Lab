"""Study 470 — Stochastic Momentum Index (Blau).

A mechanical, falsifiable encoding of William Blau's Stochastic Momentum Index (SMI):
the close's distance from the *midpoint* of the N-day high/low range, double-smoothed by
two EMAs and scaled by the (double-smoothed) half-range, bounded in [-100, +100]. The
folklore says the SMI *times turns* — it crosses up out of oversold before price bottoms.
We test that as a forward-return study against a drift-matched random-entry baseline and a
parameter-scramble placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
