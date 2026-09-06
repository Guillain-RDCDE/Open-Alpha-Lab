"""Study 986 — The Rolling Ladder.

Buy a ten-year Treasury at 4% and hold it to maturity and you will earn 4% a year,
whatever happens to rates in between. Buy a ten-year Treasury *fund* at 4% and you will not:
the fund sells its bonds when they age out and buys new ones, so it never matures, never pulls
to par, and never delivers the yield you bought it at. Both statements are widely known. What
is much less clear is how big the difference is, which direction it runs, and over what horizon
it resolves — which is what this study measures.

- :mod:`ladder.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`ladder.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
