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
    the forward total return (already aligned so ``fwd_ret.loc[y]`` is what asset growth measured in
    year ``y`` predicts — for the real panel that is the July(y+1)→June(y+2) window). Each year, the
    bottom-``q`` asset-growth names are bought and the top-``q`` shorted, equal-weight. Returns a frame
    indexed by the **signal (fiscal/formation) year** ``y`` — *not* the year the return is earned —
    with ``low_growth``, ``high_growth`` and ``hedge`` (= low − high).
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


# Two-sided 95% Student-t critical values by degrees of freedom (df 1–30; ≥30 → 1.96).
# A handful of yearly observations is exactly where the normal 1.96 flatters the CI.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
         23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def summary(annual_returns: pd.Series) -> dict:
    """Annualised stats for a yearly return series — and the honest uncertainty around the mean.

    Besides mean/vol/Sharpe/hit-rate/max-drawdown, reports the standard error of the mean, the
    *t*-statistic against zero and a Student-t **95% CI** — non-negotiable when ``n`` is a couple of
    dozen yearly observations and a headline could otherwise read noise as a verdict.
    """
    keys = ("mean", "vol", "sharpe", "hit_rate", "max_drawdown", "n",
            "se", "tstat", "ci95_low", "ci95_high")
    r = pd.Series(annual_returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in keys}
    mean, std = r.mean(), r.std(ddof=1)
    se = std / np.sqrt(len(r))
    tcrit = _T975.get(len(r) - 1, 1.96)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    return {
        "mean": float(mean),
        "vol": float(std),
        "sharpe": float(mean / std) if std > 0 else np.nan,  # yearly obs → already annual
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": float(dd),
        "n": int(len(r)),
        "se": float(se),
        "tstat": float(mean / se) if se > 0 else np.nan,
        "ci95_low": float(mean - tcrit * se),
        "ci95_high": float(mean + tcrit * se),
    }
