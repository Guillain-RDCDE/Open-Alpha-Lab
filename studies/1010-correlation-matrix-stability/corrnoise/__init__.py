"""Study 1010 — Mostly Noise.

A covariance matrix is the input every optimiser depends on and the one nobody
puts a standard error on. For N assets it contains N(N+1)/2 free parameters estimated from N×T
observations, so the ratio q = N/T decides how much of it can possibly be real. Marchenko and
Pastur (1967) give the answer exactly: for pure noise, the sample eigenvalues fall inside a
known band whose width depends only on q. Anything inside that band is consistent with nothing
at all.

This study measures how much of a real equity correlation matrix falls inside the band, tests
whether the estimated matrix predicts the next period's, and then asks the only question that
matters for an allocator — whether cleaning it produces portfolios whose *realised* risk matches
what was forecast.

- :mod:`corrnoise.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`corrnoise.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
