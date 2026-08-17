"""Study 915 — K-1 vs 1099: does the tax-friendly commodity wrapper cost performance?

``data``     — the real tape (cached daily total-return closes) + the deterministic
               synthetic wrapper-pair generator.
``strategy`` — the pre-tax wrapper race, the tracking-error decomposition, the
               after-tax bracket sweep, and the inference primitives.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
