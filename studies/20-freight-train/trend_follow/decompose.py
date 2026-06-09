"""The teardown that earns the stamps — is the trend real, is it alpha, and is it still here?

Three legs:

  1. :func:`mean_tstat_hac` — is the TSMOM return's mean reliably positive under Newey-West inference?
     The leg that makes Signal `REAL`.
  2. :func:`basket_alpha` — regress TSMOM on the equal-weight long-only basket. A positive **alpha**
     with a low (or negative) **beta** says the edge is genuine timing, not just disguised long
     exposure — and trend's famously *negative* beta in crises is the diversification it sells.
  3. :func:`rolling_sharpe` / :func:`subsample_sharpe` — is it *still* working? Managed-futures trend
     had a celebrated 2008 and a difficult 2010s; the decay check decides FRAGILE vs INVESTABLE.

A paired :func:`sharpe_bootstrap` puts an interval on the Sharpe gain over the basket.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import summary, tsmom_returns, long_only_basket

TRADING_DAYS_PER_YEAR = 252


def _ols_nw(y: np.ndarray, x: np.ndarray, lags: int | None = None) -> dict:
    """OLS of ``y`` on ``[1, x]`` with Newey-West HAC covariance (intercept + slope t-stats)."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    X = np.column_stack([np.ones_like(x), x])
    n = X.shape[0]
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    u = X * resid[:, None]
    M = u.T @ u
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        G = u[k:].T @ u[:-k]
        M += w * (G + G.T)
    cov = XtX_inv @ M @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return {"alpha": float(beta[0]), "beta": float(beta[1]),
            "alpha_t": float(beta[0] / se[0]) if se[0] > 0 else np.nan,
            "beta_t": float(beta[1] / se[1]) if se[1] > 0 else np.nan, "lags": int(lags), "n": int(n)}


def mean_tstat_hac(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Newey-West t-stat that the TSMOM daily mean is non-zero; |t| > 2 with mean>0 is `REAL`."""
    r = pd.Series(returns).astype(float).dropna().to_numpy()
    n = r.size; mu = r.mean(); e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e / n)
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k] / n)
    se = np.sqrt(max(lrv, 0.0) / n)
    return {"mean_ann_pct": float(mu * periods_per_year * 100.0),
            "t_stat": float(mu / se) if se > 0 else np.nan, "n": int(n)}


def basket_alpha(panel: pd.DataFrame, cost_bps: float = 2.0,
                 periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """Regress TSMOM on the long-only basket: alpha (annualised, HAC t) and beta.

    A positive alpha says trend timing adds return the static basket can't replicate; a low/negative
    beta is the crisis-diversification trend is prized for (it tends to be *short* when the basket
    crashes). Both together are what make trend a portfolio diversifier rather than hidden beta.
    """
    tsm = tsmom_returns(panel, cost_bps=cost_bps, **kw)
    bench = long_only_basket(panel).reindex(tsm.index)
    reg = _ols_nw(tsm.to_numpy(), bench.to_numpy())
    reg["alpha_ann_pct"] = float(reg["alpha"] * periods_per_year * 100.0)
    return reg


def subsample_sharpe(panel: pd.DataFrame, cost_bps: float = 2.0, n_chunks: int = 3, **kw) -> pd.DataFrame:
    """TSMOM net Sharpe over equal-time sub-periods — the decay read (early vs late)."""
    tsm = tsmom_returns(panel, cost_bps=cost_bps, **kw)
    bounds = np.linspace(0, len(tsm), n_chunks + 1).astype(int)
    rows = {}
    for i in range(n_chunks):
        c = tsm.iloc[bounds[i]:bounds[i + 1]]
        s = summary(c)
        rows[i] = {"start": c.index.min().date(), "end": c.index.max().date(), "sharpe": s["sharpe"]}
    out = pd.DataFrame(rows).T
    out.index.name = "chunk"
    return out


def rolling_sharpe(returns: pd.Series, window: int = 252 * 3) -> pd.Series:
    """Rolling annualised Sharpe — to see whether the edge fades in the back half of the sample."""
    r = pd.Series(returns).astype(float)
    m = r.rolling(window).mean(); s = r.rolling(window).std(ddof=1)
    return (m / s * np.sqrt(TRADING_DAYS_PER_YEAR)).rename("rolling_sharpe")


def sharpe_bootstrap(panel: pd.DataFrame, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                     cost_bps: float = 2.0, periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """Paired bootstrap CI on ``Sharpe(TSMOM) − Sharpe(basket)``."""
    tsm = tsmom_returns(panel, cost_bps=cost_bps, **kw)
    bench = long_only_basket(panel).reindex(tsm.index)
    a, b = bench.to_numpy(), tsm.to_numpy()
    n = a.size; rng = np.random.default_rng(seed)

    def _sr(x):
        sd = x.std(ddof=1)
        return x.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else 0.0

    point = _sr(b) - _sr(a)
    boots = np.array([_sr(b[i]) - _sr(a[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe_gain": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((boots < 0).mean()), "n_boot": int(n_boot)}
