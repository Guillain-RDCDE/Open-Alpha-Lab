"""Study 974 — The Nth Asset.

"Add another asset class" is the default answer to every portfolio question. This
study prices the answer: build equal-weight portfolios of every size from one to twelve, draw
the constituents at random thousands of times, and measure what the *k*-th asset actually buys
in realised volatility, drawdown and Sharpe — then check the same curve against the
correlation-based theory that predicts where it must flatten.

- :mod:`diversify_n.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`diversify_n.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
