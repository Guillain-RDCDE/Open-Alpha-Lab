"""Study 1009 — Sortino's Free Lunch.

The Sortino ratio replaces the Sharpe ratio's standard deviation with downside
deviation, on the reasonable ground that investors do not mind upside volatility. It is widely
presented as a strict improvement. This study takes the claim seriously enough to test it
properly: how much do the two actually disagree, is the disagreement informative, and — the
question that decides it — is the Sortino ratio *estimated as precisely* as the Sharpe ratio it
replaces?

- :mod:`sortino.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`sortino.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
