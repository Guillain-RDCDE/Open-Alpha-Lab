"""Study 919 — Methodology Shock: do index rule changes move the wrapper?

Package layout:

- ``data``     — the real tape (cached daily total-return ETF closes), the hardcoded
  event calendar of index methodology changes, and the deterministic synthetic
  generators (``synthetic_daily`` / ``synthetic_panel``).
- ``strategy`` — the market-model event study, the placebo randomisation test, the
  costed dollar-neutral pair trade, and the inference primitives.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
