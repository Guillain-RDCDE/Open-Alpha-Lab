"""Study 930 — the when-issued window of a corporate spin-off.

``data``     — the hardcoded event table, the shared-cache real tape, and the
               deterministic synthetic panel.
``strategy`` — the event-window estimator, the hedged daily sleeve, and the
               inference primitives (HAC t, block bootstrap, era cut, sweeps).
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
