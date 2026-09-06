"""Study 980 — The Silicon Canary.

"Semis lead the market" is one of the most repeated pieces of desk lore there is —
chips go into everything, so chip demand turns first. The claim is testable and is almost never
tested properly, because the obvious version of the test measures the market factor both
sectors share. This study strips that out, measures what is left at four horizons, and prices
the rule it implies.

- :mod:`semi_lead.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`semi_lead.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
