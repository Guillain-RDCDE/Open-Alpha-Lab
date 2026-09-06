"""Study 1007 — Time Does Not Diversify.

Two respectable positions, both with charts. The advisers point out that the
*annualised* return of equities converges as the horizon lengthens, so long-horizon investors
face less uncertainty. Samuelson (1963, 1969) replied that terminal *wealth* uncertainty grows
without limit, and that under constant relative risk aversion the optimal equity weight is
independent of horizon entirely. This study measures both quantities on the same data, shows
they genuinely point in opposite directions, and then works out which one an investor's decision
actually depends on — which turns out to be neither, quite.

- :mod:`timediv.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`timediv.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
