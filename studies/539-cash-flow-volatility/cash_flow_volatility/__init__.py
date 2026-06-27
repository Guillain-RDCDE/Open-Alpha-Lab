"""Study 539 -- Cash-Flow-Volatility (Huang 2009 cash-flow-uncertainty anomaly).

Public surface:

- ``data``     -- synthetic + real (yfinance) panel loaders, cached to the study's ``_cache/``.
- ``strategy`` -- the CF-volatility signal, long-short sort, one-sample t, placebo null,
  costs x turnover (+ borrow), robustness, and a deterministic synthetic positive control.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
