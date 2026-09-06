"""Study 972 — Adjusted or Not.

Every backtest reads a price series, and almost every one of them inherits an
adjustment convention nobody chose deliberately. This study measures what that choice is worth:
the return it deletes, the ranking it reorders, and the strategy conclusions it flips —
across a universe built to span yields from near zero to eight percent.

- :mod:`adj_mode.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`adj_mode.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
