"""The momentum signal — the 12-1 trailing return — and the engine: **do past winners keep winning?**

Cross-sectional momentum (Jegadeesh & Titman 1993) ranks stocks by their trailing return, skipping the
most recent month (the "12-1" convention, to dodge short-term reversal), and bets the leaders keep
leading. If that weren't true — if the ranking carried no information about the next month — sorting on
it would just be sorting on noise. So before any backtest we measure it directly: sort the cross-section
by 12-1 return into deciles and read the average *forward* return of each.

Everything here is a pure function of a ``dates x stock`` daily-returns panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def momentum_score(panel: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """The 12-1 momentum signal: the trailing ``lookback`` return excluding the last ``skip`` days.

    Compound the return from ``lookback`` days ago up to ``skip`` days ago (so the most recent month is
    excluded). Higher = stronger past winner. The signal at each date uses only past data.
    """
    logr = np.log1p(panel)
    cum = logr.rolling(lookback).sum() - logr.rolling(skip).sum()      # 12m minus the last 1m
    return np.expm1(cum)


def momentum_spread(panel: pd.DataFrame, lookback: int = 252, skip: int = 21, n_buckets: int = 10,
                    horizon: int = 21) -> dict:
    """Sort the cross-section by 12-1 momentum into deciles; read each decile's *forward* return.

    At each rebalance, bucket the names with a defined score, then average the next ``horizon``-day
    return per bucket, pooled over time. A **rising** profile (winners out-earn losers) is momentum; the
    top-minus-bottom spread (annualised) is the winners-minus-losers premium. A no-momentum null gives a
    flat profile and a ~0 spread.
    """
    score = momentum_score(panel, lookback, skip)
    fwd = (1.0 + panel).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1.0
    rebal = range(lookback, len(panel) - horizon, horizon)
    bucket_fwd = {b: [] for b in range(n_buckets)}
    for p in rebal:
        s = score.iloc[p].dropna()
        if len(s) < n_buckets * 2:
            continue
        f = fwd.iloc[p].reindex(s.index)
        ranks = s.rank(method="first")
        buckets = pd.qcut(ranks, n_buckets, labels=False)
        for b in range(n_buckets):
            vals = f[buckets == b].dropna()
            if len(vals):
                bucket_fwd[b].append(vals.mean())
    table = pd.Series({b: np.mean(v) if v else np.nan for b, v in bucket_fwd.items()}).rename("fwd_ret")
    ann = lambda x: x * (TRADING_DAYS_PER_YEAR / horizon)
    lo, hi = float(table.iloc[0]), float(table.iloc[-1])
    return {
        "decile_fwd_ann": (ann(table) * 100.0).rename("fwd_ann_pct"),
        "winners_ann_pct": float(ann(hi) * 100.0),
        "losers_ann_pct": float(ann(lo) * 100.0),
        "wml_ann_pct": float(ann(hi - lo) * 100.0),
        "momentum_present": bool(hi > lo),
        "n_rebalances": int(sum(1 for _ in rebal)),
    }
