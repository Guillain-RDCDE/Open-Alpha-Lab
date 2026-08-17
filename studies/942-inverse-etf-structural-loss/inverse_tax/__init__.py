"""Study 942 — The Inverse Tax.

Two modules:

``data``     — the real tape (SH, PSQ, SDS, SPY, QQQ, BIL, ^IRX from the shared
               ``studies/_cache``, plus price-only SPY/QQQ closes) and a deterministic
               synthetic generator with a *planted* structural tax.
``strategy`` — the honest direct-short replicate, the daily gap, its decomposition into
               dividends / financing / expense ratio / residual, the path-drag table, and
               the inference (HAC t, block bootstrap, era and rate-regime cuts, sweeps).
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
