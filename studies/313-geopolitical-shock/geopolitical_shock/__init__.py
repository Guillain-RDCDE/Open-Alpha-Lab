"""Study 313 — Geopolitical-Shock: do markets really shrug off wars and attacks within days?

An event study of SPY around a curated table of major geopolitical shocks (the GPR index
is network-blocked, so the shock dates are hardcoded). The honest controls — a placebo
distribution of random non-event dates and a block-bootstrap of the abnormal-return path —
decide whether the post-shock drift is a real, tradable signal or just noise.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
