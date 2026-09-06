"""Study 997 — The Rebalance Lottery.

A backtest says "rebalance monthly". It does not say *which day* of the month, and
for most published results nobody checked. Run the identical rule twelve times, shifting only
the rebalance date by one day each time, and you get twelve different equity curves. The spread
between them is **timing luck** — pure noise that a reader will mistake for skill, and that
practitioners have been quietly aware of since Blitz, van der Grient & van Vliet (2010) and
Hoffstein's work made it unavoidable.

- :mod:`lottery.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`lottery.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
