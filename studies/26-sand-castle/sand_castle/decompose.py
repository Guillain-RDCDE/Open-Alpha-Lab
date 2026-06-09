"""The teardown that earns the stamps — the reversion is real, the *optimization* is a sand castle.

Three legs:

  1. :func:`optimizer_vs_naive` — the headline. Does inverting the sample covariance beat a naive
     signal-weighted portfolio, net of cost? If the optimizer earns *less* than the un-optimized version,
     ``C^{-1}`` is amplifying estimation noise, not adding information (Michaud's error-maximization).
  2. :func:`weight_instability` — why. The sample covariance is ill-conditioned, so its inverse produces
     extreme, jittery weights; we measure the condition number, the largest weight, and the day-to-day
     weight churn (which is also what makes the book uneconomic to trade).
The "sand castle" is the gap between the optimizer's *gross* in-sample promise and its *net, causal*
reality — measured directly as the optimizer underperforming a naive book it should dominate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .statarb import residual_returns
from .strategy import compare, statarb_returns, summary

TRADING_DAYS_PER_YEAR = 252


def optimizer_vs_naive(panel: pd.DataFrame, market: pd.Series | None = None, cost_bps: float = 5.0,
                       shrink: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """Optimized (``C^{-1}E``) vs naive (``E``) stat-arb, gross and net — does the inversion help?"""
    c = compare(panel, market, cost_bps=cost_bps, shrink=shrink, periods_per_year=periods_per_year, **kw)
    return {
        "optimized_gross_sharpe": c["optimized_gross"]["sharpe"], "optimized_net_sharpe": c["optimized_net"]["sharpe"],
        "naive_gross_sharpe": c["naive_gross"]["sharpe"], "naive_net_sharpe": c["naive_net"]["sharpe"],
        "opt_minus_naive_net": c["opt_minus_naive_net_sharpe"],
        "optimized_turnover": c["optimized_turnover"], "naive_turnover": c["naive_turnover"],
        "optimizer_helps": bool(c["opt_minus_naive_net_sharpe"] > 0),
    }


def weight_instability(panel: pd.DataFrame, market: pd.Series | None = None, window: int = 252) -> dict:
    """How ill-conditioned is the sample covariance, and how extreme are the inverse-driven weights?

    Computes, at a representative recent date, the condition number of the sample ``C`` and the largest
    absolute optimal weight; a huge condition number and a concentrated weight are the fingerprints of an
    error-maximizing inverse.
    """
    from .statarb import reversion_signal, optimal_weights, naive_weights
    res = residual_returns(panel, market)
    E = reversion_signal(res, 5)
    R = res.to_numpy(); Em = E.to_numpy(); n, m = R.shape
    t = n - 2
    win = R[t - window:t, :]
    valid = ~np.isnan(win).any(axis=0) & ~np.isnan(Em[t])
    Wv = win[:, valid]; Ev = Em[t, valid]
    C = np.cov(Wv, rowvar=False)
    cond = float(np.linalg.cond(C))
    w_opt = optimal_weights(Ev, C); w_nai = naive_weights(Ev)
    return {
        "condition_number": cond,
        "max_abs_weight_optimized": float(np.abs(w_opt).max()),
        "max_abs_weight_naive": float(np.abs(w_nai).max()),
        "n_valid": int(valid.sum()),
    }


def gross_vs_net(panel: pd.DataFrame, market: pd.Series | None = None, cost_bps: float = 5.0,
                periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """The optimized book's *gross* (in-sample-looking) Sharpe vs its *net* (tradable) Sharpe.

    Gross, the daily reversion book looks like an edge; the ~daily turnover then erases it. The gap is the
    sand castle — an impressive backtest that does not survive the cost of building it.
    """
    g = summary(statarb_returns(panel, market, optimized=True, cost_bps=0.0, **kw), periods_per_year)["sharpe"]
    n = summary(statarb_returns(panel, market, optimized=True, cost_bps=cost_bps, **kw), periods_per_year)["sharpe"]
    return {"gross_sharpe": float(g), "net_sharpe": float(n), "cost_gap": float(g - n)}
