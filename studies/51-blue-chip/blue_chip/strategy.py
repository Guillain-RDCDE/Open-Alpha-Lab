"""The quality long-short — gross profitability, scored annually.

Novy-Marx (2013): gross profitability = gross profit / total assets is the cleanest "quality" signal;
high-GP firms out-earn low-GP firms. The hedge is long the top-``q`` GP/assets, short the bottom,
equal-weight, rebalanced annually. We measure the hedge, both legs, and the comparison to the universe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def gross_profitability(gp: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    """Gross profit ÷ total assets, aligned on the years and names both report."""
    years = gp.index.intersection(assets.index)
    return gp.reindex(years) / assets.reindex(years)


def quantile_hedge(signal: pd.DataFrame, fwd_ret: pd.DataFrame, q: float = 0.2, long_high: bool = True
                   ) -> pd.DataFrame:
    """Annual long-``q`` / short-``q`` hedge on a (year × ticker) ``signal`` vs next-year returns.

    ``long_high=True`` buys the top-``q`` signal (high quality) and shorts the bottom. Returns a frame
    indexed by the return year with ``high``, ``low`` and ``hedge`` (= long − short).
    """
    rows = {}
    for y in signal.index:
        s = signal.loc[y].dropna()
        if len(s) < 20 or y not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[y].dropna()
        hi = s[s >= s.quantile(1.0 - q)].index
        lo = s[s <= s.quantile(q)].index
        rh = nxt.reindex(hi).dropna().mean()
        rl = nxt.reindex(lo).dropna().mean()
        if np.isnan(rh) or np.isnan(rl):
            continue
        long_leg, short_leg = (rh, rl) if long_high else (rl, rh)
        rows[y] = {"high": rh, "low": rl, "hedge": long_leg - short_leg}
    return pd.DataFrame(rows).T.sort_index()


def market_annual(fwd_ret: pd.DataFrame) -> pd.Series:
    return fwd_ret.mean(axis=1).rename("market").dropna()


def summary(annual_returns: pd.Series) -> dict:
    """Annualised stats for a yearly return series (mean, vol, Sharpe, t-stat, hit-rate, max-DD)."""
    r = pd.Series(annual_returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    sr = mean / std if std > 0 else np.nan       # yearly obs → already annual
    return {
        "mean": float(mean),
        "vol": float(std),
        "sharpe": float(sr),
        "tstat": float(sr * np.sqrt(len(r))) if std > 0 else np.nan,
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": float(dd),
        "n": int(len(r)),
    }
