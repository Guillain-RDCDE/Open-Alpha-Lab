"""Study 973 — Different Holidays.

A US-listed Japan ETF trades all day in New York while Tokyo has been shut for
fourteen hours, so its price moves on stale news plus whatever New York thinks. Correlations,
betas and portfolio volatilities computed from such a panel are biased downward in a way that
has been known since 1977 and is ignored daily. This study measures the bias and tests the
three standard fixes.

- :mod:`calendar_gap.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`calendar_gap.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
