"""Study 477 — Choppiness Index (trend vs chop regime timer).

A mechanical, falsifiable encoding of E.W. Dreiss' Choppiness Index:

    CI = 100 * log10( sum_{i=t-N+1..t} ATR_i / (max(high) - min(low)) ) / log10(N)

bounded to roughly 0-100. Low CI ("trending") is folklore-said to *precede* tradable
momentum, high CI ("chop") to precede chop/whipsaw. We test that as a forward-return study:
on a confirmed low-CI reading, go long at the next close and measure the forward return,
against a drift-matched random-entry baseline and a placebo that scrambles the CI's defining
window-geometry while keeping the price marginal, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
