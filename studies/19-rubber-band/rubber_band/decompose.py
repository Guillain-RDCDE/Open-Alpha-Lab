"""The teardown that earns the stamps — is the bounce real, and does it clear a real spread?

Two questions, two legs:

  1. :func:`mean_tstat_hac` — is the IBS timing return's mean reliably positive under autocorrelation-
     robust (Newey-West) inference? A large *t* says the bounce is a real, repeatable edge, not luck.
     This is the leg that makes Signal `REAL`.
  2. :func:`breakeven_cost` — the per-unit-traded slippage (bps) at which the net Sharpe hits zero.
     Because the book turns over ~daily, this break-even is *small*; the whole tradability verdict is
     whether it clears the bid-ask spread you'd actually pay. This is the leg that makes Tradability
     `MIRAGE`.

A paired :func:`sharpe_bootstrap` puts an interval on the gross edge so the headline isn't a point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import summary, timing_panel_returns

TRADING_DAYS_PER_YEAR = 252


def mean_tstat_hac(returns: pd.Series, lags: int | None = None) -> dict:
    """Newey-West (HAC) t-stat that the mean daily return is non-zero.

    Regress the series on a constant with Bartlett-weighted long-run variance — the textbook HAC mean
    test (same idea as ``quantlab.analytics.mean_tstat_hac``). Reports the annualised mean, the HAC
    standard error and the t-stat; |t| > 2 with a positive mean is the `REAL` signal.
    """
    r = pd.Series(returns).astype(float).dropna().to_numpy()
    n = r.size
    mu = r.mean()
    e = r - mu
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    gamma0 = float(e @ e / n)
    lrv = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        cov = float(e[k:] @ e[:-k] / n)
        lrv += 2.0 * w * cov
    se = np.sqrt(max(lrv, 0.0) / n)
    return {
        "mean_daily": float(mu),
        "mean_ann_pct": float(mu * TRADING_DAYS_PER_YEAR * 100.0),
        "t_stat": float(mu / se) if se > 0 else np.nan,
        "lags": int(lags),
        "n": int(n),
    }


def breakeven_cost(
    basket: dict,
    hi_bps: float = 50.0,
    tol_bps: float = 0.05,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """The cost per unit traded (bps) at which the basket IBS timing net Sharpe crosses zero.

    Bisection on ``cost_bps``. A *low* break-even (a few bps or less) against a real ETF round-trip
    spread is the `MIRAGE`: the gross edge is real but it lives entirely inside transaction costs.
    """
    def sr(c):
        return summary(timing_panel_returns(basket, cost_bps=c), periods_per_year)["sharpe"]

    s0 = sr(0.0)
    if s0 <= 0:
        return {"breakeven_bps": 0.0, "gross_sharpe": float(s0)}
    if sr(hi_bps) > 0:
        return {"breakeven_bps": float(hi_bps), "gross_sharpe": float(s0), "uncrossed": True}
    lo, hi = 0.0, hi_bps
    while hi - lo > tol_bps:
        mid = 0.5 * (lo + hi)
        if sr(mid) > 0:
            lo = mid
        else:
            hi = mid
    return {"breakeven_bps": float(0.5 * (lo + hi)), "gross_sharpe": float(s0)}


def sharpe_bootstrap(
    returns: pd.Series,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Bootstrap CI on the (gross) IBS timing Sharpe — does the edge clear zero before costs?"""
    r = pd.Series(returns).astype(float).dropna().to_numpy()
    n = r.size
    rng = np.random.default_rng(seed)

    def _sr(x):
        sd = x.std(ddof=1)
        return x.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else 0.0

    point = _sr(r)
    boots = np.array([_sr(r[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "sharpe": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "frac_negative": float((boots < 0).mean()),
        "n_boot": int(n_boot),
    }
