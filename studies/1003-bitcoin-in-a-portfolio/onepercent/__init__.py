"""Study 1003 — The 1% Allocation.

"A 1% allocation to bitcoin" is one of the most confidently repeated numbers in
asset allocation, and it competes with equally confident 2%, 5% and 10%. All of them are
derived from an optimiser fed a historical mean. This study runs that optimisation honestly:
in-sample, out-of-sample, and against the question nobody asks first — given bitcoin's
volatility, **how long a history would be needed to tell 1% from 5%?** The answer to the last
question turns out to determine the answers to the others.

- :mod:`onepercent.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`onepercent.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
