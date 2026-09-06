"""Study 992 — How Long Is a Storm?.

"Volatility clusters" is the most reliably true statement in empirical finance. The
useful follow-up question is *how long a cluster lasts*, because that is the number that decides
whether a volatility forecast is worth making, how fast a risk model should react, and how long
after a crash you should still be nervous. This study measures that half-life five different
ways and finds the answers disagree by a factor that matters — not because the measurements are
wrong, but because volatility does not have one half-life.

- :mod:`storm.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`storm.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
