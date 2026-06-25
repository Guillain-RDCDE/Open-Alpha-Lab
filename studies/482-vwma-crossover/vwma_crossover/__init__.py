"""Study 482 — VWMA-Crossover (volume-weighted vs plain moving-average cross).

A mechanical, falsifiable head-to-head: does *volume-weighting* a moving-average
crossover add edge over the same-length **plain SMA** crossover?

  VWMA_N(t) = sum_{i=t-N+1..t} price_i * vol_i / sum_{i=t-N+1..t} vol_i

A long fires when the fast VWMA crosses **above** the slow VWMA; we measure the
forward return and, crucially, race it against (a) the identical-length plain-SMA
cross (the incremental-value test), (b) a drift-matched random-entry baseline, and
(c) a **shuffled-volume placebo** that destroys the volume weighting while keeping
the price path and the volume marginal — the honest "is the volume term doing
anything?" null.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
