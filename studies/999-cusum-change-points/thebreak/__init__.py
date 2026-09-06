"""Study 999 — The Break.

Everyone agrees markets change regime. Far fewer people ask how long it takes to
notice. A change-point detector is a hypothesis test run repeatedly, and like any test it needs
evidence to accumulate before it fires — which means every detection is late, and the lateness
is not a defect of the algorithm but a property of the information. This study measures the
delay directly, against changes whose true dates are known because they were planted, and then
asks the only question that matters for a practitioner: by the time you know, is there anything
left to do?

- :mod:`thebreak.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`thebreak.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
