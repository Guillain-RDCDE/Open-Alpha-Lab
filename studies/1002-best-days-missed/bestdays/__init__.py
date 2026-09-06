"""Study 1002 — The Ten Best Days.

Every fund brochure carries a version of it: miss the ten best days of the last
thirty years and your return collapses. The arithmetic is correct. The inference drawn from it —
that you should therefore never be out of the market — does not follow, and this study shows
why with three measurements the brochure never prints: the symmetric statistic for the worst
days, the fact that the best and worst days are neighbours rather than scattered, and the
timing accuracy a real switching strategy would actually need.

- :mod:`bestdays.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`bestdays.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
