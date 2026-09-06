"""Study 998 — The Moving Target.

Every spread trade rests on a hedge ratio, and every hedge ratio is an estimate of a
relationship that will not sit still. The standard answer is a rolling window, which forces an
awkward choice — short enough to adapt, long enough to be stable, and there is no window length
that is both. The Kalman filter offers a different bargain: treat the hedge ratio as a hidden
state that random-walks, and update it optimally as evidence arrives. This study asks whether
the theoretically better answer is actually better, and separates two questions that are
usually conflated: does it **track** more accurately, and does it **trade** more profitably.

- :mod:`movingtarget.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`movingtarget.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
