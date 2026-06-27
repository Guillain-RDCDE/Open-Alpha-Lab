"""Study 521 — Cash-Based Operating Profitability (Ball-Gerakos-Linnainmaa-Nikolaev 2016).

Public surface:

- :mod:`cash_op.data`     — fundamentals + price pull (yfinance), study-local cache, plus a
                            deterministic synthetic panel with a tunable cash-OP premium.
- :mod:`cash_op.strategy` — the cash-based OP signal, the quintile long-short, one-sample t,
                            label-shuffle placebo, costs x turnover, and the synthetic control.
"""

from __future__ import annotations

from . import data, strategy

__all__ = ["data", "strategy"]
