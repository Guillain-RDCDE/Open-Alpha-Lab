"""Study 995 — Whose Sharpe Is It?.

"The S&P returned 10% a year with a Sharpe of 0.55." Whose 10%? A Japanese investor
who bought the S&P in 2012 earned far more than an American did, and a Swiss investor earned far
less — same fund, same shares, different money. The Sharpe ratio compounds the problem, because
the currency changes the numerator *and* the denominator *and* the risk-free rate you subtract.
This study measures how much, and finds the spread across six home currencies is wide enough to
change how a fund would be ranked.

- :mod:`whosesharpe.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`whosesharpe.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
