"""Study 944 — How Much Leverage: the realised growth-optimal multiple on SPY.

``data``     — the real tape (SPY, BIL, ^IRX from the shared cache) and a deterministic
               synthetic generator with a *known* Kelly multiple.
``strategy`` — the daily-reset constant-leverage engine, the leverage sweep, the Kelly
               estimator, the rolling-optimum instability measures and the inference.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
