"""Study 837 — Look-Ahead Standardization.

A specific, ubiquitous look-ahead leak: z-scoring / normalising a predictive feature with the
**full-sample** mean & std (the whole history, including the future) instead of an **expanding /
point-in-time** window. On a synthetic feature+return panel we show full-sample standardisation
manufactures a large fake IC/Sharpe out of an efficient-market null, while the honest expanding
version reads ~0 — and a planted-edge control proves the expanding machinery still fires on a real
effect.

* ``data``     — three deterministic seeded worlds: a stationary null (no leak — the contrast), a
                 non-stationary random-walk null (the trap), and a planted real edge (the control).
* ``strategy`` — the two standardisations (``full_standardize`` / ``expanding_standardize``), the
                 cross-sectional rank-IC scorer, the inference primitives (one-sample / Welch /
                 Newey-West HAC / Wilson), the long-short fractile book, the costed timer, and the
                 seed-robust aggregation + horizon/length sweeps.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
