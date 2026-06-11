"""The asset-growth long-short, scored honestly.

Cooper, Gulen & Schill (2008): sort firms by year-on-year **total-asset growth**; the low-growth
names beat the high-growth ones over the following year. The hedge is long the low-growth quantile,
short the high-growth quantile, equal-weight, rebalanced annually. The honest questions: is the hedge
positive, is it *symmetric* (do high-growth firms really underperform, or is it all the long leg), and
does it beat just owning the market?
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_hedge(ag: pd.DataFrame, fwd_ret: pd.DataFrame, q: float = 0.2) -> pd.DataFrame:
    """Annual long-low / short-high asset-growth hedge.

    ``ag`` is a (year × ticker) frame of fiscal-year asset growth; ``fwd_ret`` is (year × ticker) of
    the *next* calendar year's total return (already aligned so ``fwd_ret.loc[y]`` is what asset growth
    measured in year ``y`` predicts). Each year, the bottom-``q`` asset-growth names are bought and the
    top-``q`` shorted, equal-weight. Returns a frame indexed by the *return* year with ``low_growth``,
    ``high_growth`` and ``hedge`` (= low − high).
    """
    rows = {}
    for y in ag.index:
        g = ag.loc[y].dropna()
        if len(g) < 20 or y not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[y].dropna()
        lo = g[g <= g.quantile(q)].index
        hi = g[g >= g.quantile(1.0 - q)].index
        rl = nxt.reindex(lo).dropna().mean()
        rh = nxt.reindex(hi).dropna().mean()
        if np.isnan(rl) or np.isnan(rh):
            continue
        rows[y] = {"low_growth": rl, "high_growth": rh, "hedge": rl - rh}
    return pd.DataFrame(rows).T.sort_index()


def market_annual(fwd_ret: pd.DataFrame) -> pd.Series:
    """Equal-weight 'own everything' benchmark — the cross-sectional mean return each year."""
    return fwd_ret.mean(axis=1).rename("market").dropna()


def summary(annual_returns: pd.Series) -> dict:
    """Annualised stats for a yearly return series (mean, vol, Sharpe, hit-rate, max-drawdown)."""
    r = pd.Series(annual_returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "hit_rate", "max_drawdown", "n")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    return {
        "mean": float(mean),
        "vol": float(std),
        "sharpe": float(mean / std) if std > 0 else np.nan,  # yearly obs → already annual
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": float(dd),
        "n": int(len(r)),
    }
