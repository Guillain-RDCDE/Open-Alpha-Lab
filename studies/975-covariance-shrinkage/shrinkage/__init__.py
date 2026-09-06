"""Study 975 — Shrink the Matrix.

Ledoit and Wolf's shrinkage estimator is thirty years old, three lines of algebra and
routinely ignored. This study runs it against the plain sample covariance on two
cross-sections — eleven sectors and forty stocks — out of sample, with the minimum-variance
portfolio each one builds as the scoreboard, and reports the one number that decides whether
it is worth the code: how much realised volatility it saves.

- :mod:`shrinkage.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`shrinkage.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
