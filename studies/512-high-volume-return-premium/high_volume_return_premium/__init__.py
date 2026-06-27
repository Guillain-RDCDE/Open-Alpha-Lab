"""Study 512 -- High-Volume-Return-Premium (Gervais-Kaniel-Mingelgrin 2001).

Sort the cross-section each week by *abnormal* trading volume (this week's average daily
volume relative to its own trailing average), then measure short-horizon forward returns:
do the high-volume names appreciate and the low-volume names lag, as GKM claim?

Public surface:

- ``data``     -- the synthetic panel (deterministic positive control + null) and the
  real yfinance daily OHLCV pull, cached to this study's own ``_cache/``.
- ``strategy`` -- the abnormal-volume signal, the weekly long-short book, one-sample &
  HAC inference, a label-shuffle placebo null, costs x turnover (+ short borrow), and the
  synthetic positive control plumbing.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
