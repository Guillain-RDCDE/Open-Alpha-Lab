"""Study 968 — Which Bootstrap.

A confidence interval is a promise: 95% of the intervals built this way contain the
truth. This study checks whether the bootstraps used across this desk keep that promise, by
running each of them thousands of times on simulated data whose truth is known — under
independence, under volatility clustering, and under genuine serial correlation — and then
measuring how much the choice moves the intervals a reader would actually see.

- :mod:`boot_choice.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`boot_choice.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
