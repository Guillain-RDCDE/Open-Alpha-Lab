"""Study 1000 — The Cycle Hunt.

Take any price series, apply a Fourier transform, and peaks appear. They always do —
the periodogram of pure noise is not flat, it is exponentially distributed around a flat mean,
so the largest of a few hundred frequency bins is always several times the average. Cycle
hunting in markets has a century of history and an almost unbroken record of finding things that
were not there. This study measures how large a spurious peak can be, gives the correct
threshold for calling one real, and applies it.

- :mod:`cyclehunt.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`cyclehunt.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
