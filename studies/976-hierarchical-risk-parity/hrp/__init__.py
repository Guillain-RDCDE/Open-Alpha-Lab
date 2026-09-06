"""Study 976 — The Family Tree.

Marcos López de Prado's hierarchical risk parity builds a portfolio without ever
inverting the covariance matrix: cluster the assets into a tree, reorder the matrix so related
assets sit together, then split risk recursively down the branches. The claim is lower
out-of-sample volatility and far more stable weights than a quadratic optimiser. This study
tests it against four alternatives on three panels, and against a Monte Carlo where the block
structure it exploits is planted.

- :mod:`hrp.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`hrp.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
