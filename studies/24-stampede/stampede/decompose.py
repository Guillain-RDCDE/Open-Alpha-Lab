"""The teardown that earns the stamps — momentum is real, and it crashes.

Three legs:

  1. :func:`capm_alpha` — regress WML on the market: a large positive **alpha** under Newey-West errors
     is the `REAL` signal (momentum is one of the most significant anomalies in finance).
  2. :func:`crash_profile` — the catch. WML is short the losers, so when crushed names violently rebound
     (a bear-market bottom), the short leg detonates. We measure the negative **skew**, the worst single
     months, and the depth/length of the signature momentum crash. This is what makes it `FRAGILE`.
  3. :func:`subsample_sharpe` — decay across the sample, and a bootstrap CI on the WML Sharpe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import books, summary

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
            "alpha_t": float(beta[0] / se[0]) if se[0] > 0 else np.nan,
            "beta_t": float(beta[1] / se[1]) if se[1] > 0 else np.nan}


def capm_alpha(panel: pd.DataFrame, cost_bps: float = 5.0,
               periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """CAPM regression of WML on the equal-weight market: annualised alpha, HAC t, beta."""
    b = books(panel, cost_bps=cost_bps, **kw)
    reg = _ols_nw(b["wml"].to_numpy(), b["market"].reindex(b["wml"].index).to_numpy())
    reg["alpha_ann_pct"] = float(reg["alpha"] * periods_per_year * 100.0)
    s = summary(b["wml"], periods_per_year)
    reg["sharpe"] = s["sharpe"]; reg["max_drawdown"] = s["max_drawdown"]; reg["skew"] = s["skew"]
    return reg


def crash_profile(panel: pd.DataFrame, cost_bps: float = 5.0, **kw) -> dict:
    """The momentum-crash tail: WML skew, worst months, and the worst drawdown.

    Momentum's left tail is fat and negative — a handful of catastrophic months (loser rebounds) dwarf
    the steady gains. Reports the monthly skew, the single worst month, the average of the worst 5
    months, and the deepest drawdown.
    """
    wml = books(panel, cost_bps=cost_bps, **kw)["wml"]
    monthly = ((1.0 + wml).resample("ME").prod() - 1.0).dropna()
    worst5 = monthly.nsmallest(5)
    eq = (1.0 + wml).cumprod()
    dd = (eq / eq.cummax() - 1.0)
    return {
        "monthly_skew": float(monthly.skew()),
        "worst_month_pct": float(monthly.min() * 100.0),
        "worst5_months_mean_pct": float(worst5.mean() * 100.0),
        "max_drawdown_pct": float(dd.min() * 100.0),
        "best_month_pct": float(monthly.max() * 100.0),
        "n_months": int(len(monthly)),
    }


def subsample_sharpe(panel: pd.DataFrame, cost_bps: float = 5.0, n_chunks: int = 3, **kw) -> pd.DataFrame:
    """WML net Sharpe over equal-time sub-periods — the decay read."""
    wml = books(panel, cost_bps=cost_bps, **kw)["wml"]
    bounds = np.linspace(0, len(wml), n_chunks + 1).astype(int)
    rows = {}
    for i in range(n_chunks):
        c = wml.iloc[bounds[i]:bounds[i + 1]]
        rows[i] = {"start": c.index.min().date(), "end": c.index.max().date(), "sharpe": summary(c)["sharpe"]}
    out = pd.DataFrame(rows).T; out.index.name = "chunk"
    return out


def sharpe_bootstrap(panel: pd.DataFrame, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                     cost_bps: float = 5.0, periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """Bootstrap CI on the WML Sharpe."""
    wml = books(panel, cost_bps=cost_bps, **kw)["wml"].to_numpy()
    n = wml.size; rng = np.random.default_rng(seed)
    def _sr(x):
        sd = x.std(ddof=1); return x.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else 0.0
    point = _sr(wml)
    boots = np.array([_sr(wml[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((boots < 0).mean()), "n_boot": int(n_boot)}
