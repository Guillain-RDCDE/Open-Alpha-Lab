"""Study 994 — The Rounding Tax.

Every model portfolio is specified in percentages, and percentages assume you can buy
any fraction of a share. You cannot — or could not, until fractional trading arrived, and still
cannot in most retirement accounts, most non-US brokers, and every account that holds an ETF
through a plan. So the real portfolio is the rounded one. This study measures what the rounding
costs: the tracking error it creates, how it scales with account size, and — the part that
actually matters — how much of the "cost" is drag and how much is simply noise that cancels.

- :mod:`roundingtax.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`roundingtax.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
