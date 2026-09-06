"""Study 964 — All-Time High.

"Never buy at an all-time high" is one of the most repeated pieces of folk advice in
investing. This study measures the forward returns of money put in on record-high days
against money put in on every other day — and then prices the strategy the advice actually
implies: sit in cash until the market is below its peak.

- :mod:`ath_buy.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`ath_buy.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
