"""Study 514 -- Pastor-Stambaugh Liquidity Risk.

Pastor & Stambaugh (2003): stocks whose returns load more on *innovations* in
aggregate market liquidity should earn a higher expected return (a liquidity-risk
premium). This is the liquidity-*risk loading* (a beta), NOT the Amihud illiquidity
*level* of a single name (that is Study 140).

Two layers:
- ``data``     -- yfinance daily OHLCV for a fixed large-cap survivor basket, cached
                  to this study's own ``_cache/`` only.
- ``strategy`` -- the aggregate liquidity series, each name's liquidity beta, the
                  high-minus-low long-short sort, one-sample / HAC t, a placebo
                  label-shuffle null, costs x turnover (+ borrow on the short leg),
                  robustness sweeps, and a deterministic synthetic positive control.
"""

from __future__ import annotations

from . import data, strategy

__all__ = ["data", "strategy"]
