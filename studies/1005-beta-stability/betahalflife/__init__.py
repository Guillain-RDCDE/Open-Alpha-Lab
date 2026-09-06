"""Study 1005 — Beta Has a Half-Life.

A beta is an estimate. It is printed on risk reports, fed into cost-of-capital
calculations and used to size hedges, almost always without a standard error beside it and
essentially never with a statement of how long it stays valid. This study measures three things:
how much of a beta persists into the following period, how much of the observed instability is
mere estimation noise rather than real change, and whether the standard fix — Blume's shrinkage
toward one — actually improves out-of-sample prediction or merely looks tidier.

- :mod:`betahalflife.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`betahalflife.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
