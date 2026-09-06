"""Study 984 — A Dollar Off.

A stock that is about to pay a dollar is worth a dollar more than the same stock
after it pays. So on the morning the dividend detaches, the price should open a dollar lower.
Elton and Gruber measured it in 1970, found 78 cents, and launched a fifty-year literature on
taxes, clienteles and arbitrage. This study measures it again on modern data — and spends as
much effort on whether the number can be measured at all as on what it turns out to be.

- :mod:`exday.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`exday.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
