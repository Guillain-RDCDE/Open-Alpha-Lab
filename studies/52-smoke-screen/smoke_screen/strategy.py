"""The accruals long-short — cash-backed earnings beat accrual-heavy ones.

Sloan (1996): balance-sheet accruals = (net income − operating cash flow) / total assets. High accruals
flag earnings the cash doesn't support; those firms underperform. The trade is **long low-accruals,
short high-accruals**, equal-weight, annual. We measure the hedge, both legs, vs the universe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def accruals(net_income: pd.DataFrame, cfo: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    """Balance-sheet accruals ratio: (net income − operating cash flow) ÷ total assets, aligned."""
    years = net_income.index.intersection(cfo.index).intersection(assets.index)
    return (net_income.reindex(years) - cfo.reindex(years)) / assets.reindex(years)


def quantile_hedge(signal: pd.DataFrame, fwd_ret: pd.DataFrame, q: float = 0.2, long_high: bool = False
                   ) -> pd.DataFrame:
    """Annual long-``q`` / short-``q`` hedge on a (year × ticker) ``signal`` vs next-year returns.

    For accruals the trade is ``long_high=False``: long the bottom-``q`` accruals (cash-backed), short
    the top-``q`` (accrual-heavy). Returns a frame with ``high``, ``low`` and ``hedge`` (= long − short).
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
    sr = mean / std if std > 0 else np.nan
    return {
        "mean": float(mean),
        "vol": float(std),
        "sharpe": float(sr),
        "tstat": float(sr * np.sqrt(len(r))) if std > 0 else np.nan,
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": float(dd),
        "n": int(len(r)),
    }
