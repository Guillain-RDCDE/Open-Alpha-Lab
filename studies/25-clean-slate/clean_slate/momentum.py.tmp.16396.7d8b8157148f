"""The signal — momentum on *residual* returns — and the engine: do residual winners keep winning?

Residual momentum (Blitz, Huij & Martens 2011) is ordinary 12-1 momentum, but the ranking variable is
each stock's **idiosyncratic residual** rather than its total return. Strip out the part of the return
explained by the market (a rolling regression), and rank on what's left. The thesis: the persistent,
tradable part of momentum lives in the idiosyncratic residual, while the *crash* lives in the systematic
(beta) part — so residualising keeps the premium and sheds the tail.

Everything here is a pure function of a ``dates x stock`` panel (and a market series). We use a
**1-factor (market)** residual; the source uses Fama-French 3-factor residuals — a stated simplification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def residual_returns(panel: pd.DataFrame, market: pd.Series | None = None, beta_window: int = 252) -> pd.DataFrame:
    """Idiosyncratic residual ``r_i - beta_i*market`` per stock, with a **trailing** (causal) beta.

    The beta is a rolling-window regression slope of each stock on the market, lagged one day, so the
    residual at ``t`` uses only past data. ``market`` defaults to the equal-weight cross-section.
    """
    mkt = (market if market is not None else panel.mean(axis=1)).astype(float)
    cov = panel.rolling(beta_window).cov(mkt)
    var = mkt.rolling(beta_window).var()
    beta = cov.div(var, axis=0).shift(1)
    return (panel - beta.mul(mkt, axis=0)).rename_axis(columns=panel.columns.name)


def momentum_score(returns: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-1 momentum on whatever return stream is passed (residual or total): trailing sum minus last month."""
    logr = np.log1p(returns.clip(lower=-0.99))
    cum = logr.rolling(lookback).sum() - logr.rolling(skip).sum()
    return np.expm1(cum)


def momentum_spread(panel: pd.DataFrame, market: pd.Series | None = None, lookback: int = 252,
                    skip: int = 21, n_buckets: int = 10, horizon: int = 21, residual: bool = True) -> dict:
    """Sort by residual (or total) 12-1 momentum into deciles; read each decile's forward *total* return.

    The position is taken on the residual ranking, but the return earned is the stock's real (total)
    return — so the spread is what a tradable book would capture. A rising profile (residual winners
    out-earn residual losers) is residual momentum; the top-minus-bottom spread is the premium.
    """
    rr = residual_returns(panel, market) if residual else panel
    score = momentum_score(rr, lookback, skip)
    fwd = (1.0 + panel).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1.0
    rebal = range(lookback, len(panel) - horizon, horizon)
    bucket = {b: [] for b in range(n_buckets)}
    for p in rebal:
        s = score.iloc[p].dropna()
        if len(s) < n_buckets * 2:
            continue
        f = fwd.iloc[p].reindex(s.index)
        buckets = pd.qcut(s.rank(method="first"), n_buckets, labels=False)
        for b in range(n_buckets):
            v = f[buckets == b].dropna()
            if len(v):
                bucket[b].append(v.mean())
    table = pd.Series({b: np.mean(v) if v else np.nan for b, v in bucket.items()})
    ann = lambda x: x * (TRADING_DAYS_PER_YEAR / horizon)
    lo, hi = float(table.iloc[0]), float(table.iloc[-1])
    return {
        "decile_fwd_ann": (ann(table) * 100.0).rename("fwd_ann_pct"),
        "wml_ann_pct": float(ann(hi - lo) * 100.0),
        "momentum_present": bool(hi > lo),
        "n_rebalances": int(sum(1 for _ in rebal)),
    }
