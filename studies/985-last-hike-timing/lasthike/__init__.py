"""Study 985 — The Last Hike.

"Buy when the Fed stops hiking" is one of the few pieces of market folklore that
sounds like it should be true: tightening ends, the pressure comes off, risk assets rally. The
problem is not whether returns after the last hike are good. It is that **nobody knows which
hike was the last one** until months of not-hiking have gone by — and by then the "signal" is
old news. This study measures both: what the tape does after a cycle's true final hike, and what
is left for someone who has to decide in real time.

- :mod:`lasthike.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`lasthike.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
