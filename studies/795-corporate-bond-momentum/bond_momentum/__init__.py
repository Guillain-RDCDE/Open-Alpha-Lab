"""Study 795 — Corporate-Bond-Momentum.

Cross-sectional momentum across a small basket of credit and Treasury bond ETFs:
each month-end, rank the universe on its trailing 6-month total return, go long the
top group and short the bottom group, hold one month, rebalance. The claim traces to
Jostova, Nikolova, Philipov & Stahel (2013), who document significant momentum in
*high-yield* corporate bonds.

Two entry points:

* ``data`` — the real ETF tape (yfinance, study-local ``_cache/``) plus a deterministic,
  seeded synthetic panel whose planted cross-sectional-momentum knob is null at 0.
* ``strategy`` — the ranking, the long/short book, the costed timer, and the inference
  primitives (HAC one-sample t, Wilson, rank-shuffle placebo).
"""

from __future__ import annotations

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
