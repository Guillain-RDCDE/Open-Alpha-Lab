"""The rotation signal — sector momentum — and the engine: **does the hot sector stay hot?**

Sector rotation ranks the sectors by their trailing return and holds the leaders. The claim is that a
sector that has out-performed keeps out-performing for a while (sector momentum; Moskowitz–Grinblatt
1999). With only ~11 sectors there are no deciles, so the engine reads the **top-minus-bottom** tercile:
sort by trailing momentum, and compare the next period's return of the leaders vs the laggards. If the
hot sectors didn't continue, rotation would just be chasing noise — and paying turnover for it.

Everything here is a pure function of a ``dates x sector`` daily-returns panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def momentum_score(panel: pd.DataFrame, lookback: int = 126, skip: int = 21) -> pd.DataFrame:
    """Trailing ``lookback``-day return excluding the last ``skip`` days — the rotation ranking signal."""
    logr = np.log1p(panel.clip(lower=-0.99))
    cum = logr.rolling(lookback).sum() - logr.rolling(skip).sum()
    return np.expm1(cum)


def rotation_strength(panel: pd.DataFrame, lookback: int = 126, skip: int = 21, top_k: int = 3,
                      horizon: int = 21) -> dict:
    """Do the top-``top_k`` momentum sectors out-earn the bottom ``top_k`` over the next ``horizon`` days?

    At each rebalance, rank sectors by trailing momentum, then average the forward return of the top and
    bottom ``top_k``. A positive top-minus-bottom spread (annualised) is sector momentum; ~0 on a
    no-persistence null.
    """
    score = momentum_score(panel, lookback, skip)
    fwd = (1.0 + panel).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1.0
    top, bot = [], []
    for p in range(lookback, len(panel) - horizon, horizon):
        s = score.iloc[p].dropna()
        if len(s) < 2 * top_k:
            continue
        f = fwd.iloc[p].reindex(s.index)
        order = s.sort_values()
        bot.append(f[order.index[:top_k]].mean())
        top.append(f[order.index[-top_k:]].mean())
    ann = TRADING_DAYS_PER_YEAR / horizon
    hi, lo = float(np.nanmean(top)), float(np.nanmean(bot))
    return {
        "top_ann_pct": float(hi * ann * 100.0), "bottom_ann_pct": float(lo * ann * 100.0),
        "top_minus_bottom_ann_pct": float((hi - lo) * ann * 100.0),
        "momentum_present": bool(hi > lo), "n_rebalances": int(len(top)),
    }
