"""Study 963 — The Half Day.

The NYSE closes at 1 p.m. on a handful of days a year. This study asks whether the
shortened session's return, the day before it and the day after it differ from an
ordinary session, on five tapes, with the early-close dates **derived by rule and then
confirmed against the volume tape** rather than typed in from memory.

- :mod:`half_day.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`half_day.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
