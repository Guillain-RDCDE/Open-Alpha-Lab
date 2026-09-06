"""Study 1004 — How Many Stocks.

Evans and Archer (1968) drew a curve of portfolio standard deviation against the
number of holdings, observed that it flattens around ten to fifteen names, and launched a
textbook fact that has survived sixty years. The curve is real. The conclusion drawn from it is
not, for a reason visible the moment you plot a different statistic: **standard deviation is a
statement about the average portfolio, and an investor gets one draw.** The dispersion of
*terminal wealth* across possible portfolios — which is what actually happens to a person — keeps
narrowing long after volatility has stopped.

- :mod:`howmany.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`howmany.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
