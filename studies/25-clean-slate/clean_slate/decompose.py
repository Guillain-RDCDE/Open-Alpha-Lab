"""The teardown that earns the stamps — residual momentum is real, and it sheds the crash.

Three legs:

  1. :func:`capm_alpha` — regress the residual-WML on the market: a positive HAC-significant alpha is the
     `REAL` signal.
  2. :func:`crash_comparison` — the headline of *this* study: put residual-WML next to total-WML on
     skew, worst month and drawdown. Residual momentum is meant to keep the premium while shedding the
     crash (the systematic, beta-driven tail) — this measures whether it does.
  3. :func:`subsample_sharpe` / :func:`sharpe_bootstrap` — decay and an interval on the residual Sharpe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import wml_returns, summary

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


def capm_alpha(panel: pd.DataFrame, market: pd.Series | None = None, cost_bps: float = 5.0,
               periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """CAPM regression of the residual-WML on the equal-weight market: alpha (ann), HAC t, beta."""
    res = wml_returns(panel, market, residual=True, cost_bps=cost_bps, **kw)
    mkt = panel.mean(axis=1).reindex(res.index)
    reg = _ols_nw(res.to_numpy(), mkt.to_numpy())
    reg["alpha_ann_pct"] = float(reg["alpha"] * periods_per_year * 100.0)
    s = summary(res, periods_per_year)
    reg.update(sharpe=s["sharpe"], max_drawdown=s["max_drawdown"], skew=s["skew"])
    return reg


def _crash(r, periods_per_year):
    s = summary(r, periods_per_year)
    monthly = ((1.0 + r).resample("ME").prod() - 1.0).dropna()
    return {"sharpe": s["sharpe"], "skew": float(monthly.skew()),
            "worst_month_pct": float(monthly.min() * 100.0), "max_drawdown_pct": float(s["max_drawdown"] * 100.0)}


def crash_comparison(panel: pd.DataFrame, market: pd.Series | None = None, cost_bps: float = 5.0,
                     periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """Residual-WML vs total-WML on Sharpe, skew, worst month and drawdown — the cleaner-cousin test."""
    res = wml_returns(panel, market, residual=True, cost_bps=cost_bps, **kw)
    tot = wml_returns(panel, market, residual=False, cost_bps=cost_bps, **kw)
    idx = res.index.intersection(tot.index)
    return {"residual": _crash(res.reindex(idx), periods_per_year),
            "total": _crash(tot.reindex(idx), periods_per_year)}


def subsample_sharpe(panel: pd.DataFrame, market=None, cost_bps: float = 5.0, n_chunks: int = 3, **kw) -> pd.DataFrame:
    res = wml_returns(panel, market, residual=True, cost_bps=cost_bps, **kw)
    bounds = np.linspace(0, len(res), n_chunks + 1).astype(int)
    rows = {i: {"start": res.iloc[bounds[i]:bounds[i+1]].index.min().date(),
                "end": res.iloc[bounds[i]:bounds[i+1]].index.max().date(),
                "sharpe": summary(res.iloc[bounds[i]:bounds[i+1]])["sharpe"]} for i in range(n_chunks)}
    out = pd.DataFrame(rows).T; out.index.name = "chunk"
    return out


def sharpe_bootstrap(panel: pd.DataFrame, market=None, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                     cost_bps: float = 5.0, periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    res = wml_returns(panel, market, residual=True, cost_bps=cost_bps, **kw).to_numpy()
    n = res.size; rng = np.random.default_rng(seed)
    def _sr(x):
        sd = x.std(ddof=1); return x.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else 0.0
    point = _sr(res)
    boots = np.array([_sr(res[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((boots < 0).mean())}
