"""The teardown that earns the stamps — does rotation beat just holding the sectors?

Three legs:

  1. :func:`vs_equal_weight` — regress the long-only rotation book on the equal-weight sector basket. A
     positive **alpha** under Newey-West errors is the part rotation adds over simply holding everything;
     a beta near 1 with ~0 alpha says you took concentration risk for nothing.
  2. :func:`long_short_tstat` — the pure sector-momentum factor's mean, with a HAC t. This is the direct
     "does the hot sector stay hot" signal, net of the market.
  3. :func:`subsample_sharpe` — decay across the sample (sector momentum is famously fickle).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import rotation_returns, equal_weight, summary

TRADING_DAYS_PER_YEAR = 252


def _ols_nw(y, x, lags=None):
    y = np.asarray(y, float); x = np.asarray(x, float)
    X = np.column_stack([np.ones_like(x), x]); n = X.shape[0]
    XtX_inv = np.linalg.inv(X.T @ X); beta = XtX_inv @ (X.T @ y); resid = y - X @ beta
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    u = X * resid[:, None]; M = u.T @ u
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0); G = u[k:].T @ u[:-k]; M += w * (G + G.T)
    cov = XtX_inv @ M @ XtX_inv; se = np.sqrt(np.diag(cov))
    return {"alpha": float(beta[0]), "beta": float(beta[1]),
            "alpha_t": float(beta[0] / se[0]) if se[0] > 0 else np.nan}


def vs_equal_weight(panel: pd.DataFrame, top_k: int = 3, cost_bps: float = 3.0,
                    periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """Regress long-only rotation on the equal-weight basket: alpha (ann), HAC t, beta."""
    ro = rotation_returns(panel, top_k=top_k, cost_bps=cost_bps, long_short=False, **kw)
    ew = equal_weight(panel).reindex(ro.index)
    reg = _ols_nw(ro.to_numpy(), ew.to_numpy())
    reg["alpha_ann_pct"] = float(reg["alpha"] * periods_per_year * 100.0)
    s = summary(ro, periods_per_year)
    reg.update(rotation_sharpe=s["sharpe"], ew_sharpe=summary(ew, periods_per_year)["sharpe"],
               rotation_maxdd=s["max_drawdown"], ew_maxdd=summary(ew, periods_per_year)["max_drawdown"])
    return reg


def long_short_tstat(panel: pd.DataFrame, top_k: int = 3, cost_bps: float = 3.0,
                     periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """Newey-West t-stat that the long-short (top-minus-bottom) sector-momentum factor has a positive mean."""
    r = rotation_returns(panel, top_k=top_k, cost_bps=cost_bps, long_short=True, **kw).to_numpy()
    n = r.size; mu = r.mean(); e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e / n)
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0); lrv += 2.0 * w * float(e[k:] @ e[:-k] / n)
    se = np.sqrt(max(lrv, 0.0) / n)
    return {"mean_ann_pct": float(mu * periods_per_year * 100.0),
            "t_stat": float(mu / se) if se > 0 else np.nan, "n": int(n)}


def subsample_sharpe(panel: pd.DataFrame, top_k: int = 3, cost_bps: float = 3.0, n_chunks: int = 3, **kw):
    ro = rotation_returns(panel, top_k=top_k, cost_bps=cost_bps, long_short=False, **kw)
    bounds = np.linspace(0, len(ro), n_chunks + 1).astype(int)
    rows = {i: {"start": ro.iloc[bounds[i]:bounds[i+1]].index.min().date(),
                "end": ro.iloc[bounds[i]:bounds[i+1]].index.max().date(),
                "sharpe": summary(ro.iloc[bounds[i]:bounds[i+1]])["sharpe"]} for i in range(n_chunks)}
    out = pd.DataFrame(rows).T; out.index.name = "chunk"
    return out
