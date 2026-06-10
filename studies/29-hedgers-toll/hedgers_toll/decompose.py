"""The teardown that earns the stamps — is the hedging-pressure premium real and net of the basket?

Three legs:

  1. :func:`premium_tstat` — Newey-West t-stat that the long-short hedging-pressure factor's mean is
     positive. The `REAL` signal: speculators are paid for absorbing hedgers' risk.
  2. :func:`vs_basket` — does the factor add return over the equal-weight commodity basket? (A commodity
     long-short is market-neutral by construction, so this is mostly about whether the signal works.)
  3. :func:`subsample_sharpe` — decay (the premium has weakened as the trade became well-known).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import hp_returns, equal_weight, summary

WEEKS_PER_YEAR = 52


def premium_tstat(returns: pd.DataFrame, hp: pd.DataFrame, cost_bps: float = 10.0,
                  periods_per_year: int = WEEKS_PER_YEAR, **kw) -> dict:
    """Newey-West t-stat that the long-short hedging-pressure factor has a positive mean."""
    r = hp_returns(returns, hp, cost_bps=cost_bps, long_short=True, **kw).to_numpy()
    n = r.size
    if n < 10:
        return {"mean_ann_pct": np.nan, "t_stat": np.nan, "n_weeks": int(n)}
    mu = r.mean(); e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e / n)
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0); lrv += 2.0 * w * float(e[k:] @ e[:-k] / n)
    se = np.sqrt(max(lrv, 0.0) / n)
    return {"mean_ann_pct": float(mu * periods_per_year * 100.0),
            "t_stat": float(mu / se) if se > 0 else np.nan, "n_weeks": int(n)}


def vs_basket(returns: pd.DataFrame, hp: pd.DataFrame, cost_bps: float = 10.0,
              periods_per_year: int = WEEKS_PER_YEAR, **kw) -> dict:
    """The factor's Sharpe and its correlation/beta to the equal-weight commodity basket."""
    ls = hp_returns(returns, hp, cost_bps=cost_bps, long_short=True, **kw)
    ew = equal_weight(returns).reindex(ls.index)
    x = ew.to_numpy(); y = ls.to_numpy()
    beta = float(np.cov(y, x)[0, 1] / np.var(x)) if np.var(x) > 0 else np.nan
    corr = float(np.corrcoef(y, x)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan
    return {"ls_sharpe": summary(ls, periods_per_year)["sharpe"],
            "basket_sharpe": summary(ew, periods_per_year)["sharpe"],
            "beta_to_basket": beta, "corr_to_basket": corr}


def subsample_sharpe(returns: pd.DataFrame, hp: pd.DataFrame, cost_bps: float = 10.0, n_chunks: int = 3, **kw):
    ls = hp_returns(returns, hp, cost_bps=cost_bps, long_short=True, **kw)
    bounds = np.linspace(0, len(ls), n_chunks + 1).astype(int)
    rows = {i: {"start": ls.iloc[bounds[i]:bounds[i+1]].index.min().date(),
                "end": ls.iloc[bounds[i]:bounds[i+1]].index.max().date(),
                "sharpe": summary(ls.iloc[bounds[i]:bounds[i+1]])["sharpe"]} for i in range(n_chunks)}
    out = pd.DataFrame(rows).T; out.index.name = "chunk"
    return out
