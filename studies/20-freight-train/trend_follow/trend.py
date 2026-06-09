"""The trend signal — and the load-bearing question: **does the past predict the future here?**

Time-series momentum rests on one fact: an asset's own past-T return predicts the sign of its next
return (Moskowitz, Ooi & Pedersen 2012). If that weren't true, taking a position in the direction of
the recent move would just be coin-flipping. So before any backtest we measure the predictability
directly: pool every (past-return, next-return) pair across the basket and read the relationship.

Everything here is a pure function of a ``dates x asset`` daily-returns panel. The strategy's two
inputs — the **sign** of the trailing-T return and the trailing realized **vol** — both live here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def trailing_return(panel: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Compounded trailing-``lookback`` return per asset — the momentum signal's raw input."""
    return (1.0 + panel).rolling(lookback).apply(np.prod, raw=True) - 1.0


def realized_vol(panel: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Trailing per-asset realized volatility (rolling std) — the inverse-vol position scaler."""
    return panel.rolling(window).std(ddof=1)


def predictability(panel: pd.DataFrame, lookback: int = 252, horizon: int = 21) -> dict:
    """Does a positive past-T return predict a positive next-``horizon`` return? Pool and measure.

    For every asset and every non-overlapping ``horizon`` block, pair the *sign* of the trailing-T
    return at the block's start with the block's realized return, pooled across the basket. Reports:

      * ``pooled_t`` — Newey-West t-stat that ``sign(past) * future_return`` has a positive mean (the
        time-series-momentum t-stat of MOP 2012);
      * ``hit_rate`` — share of blocks where the sign was right;
      * ``slope`` — OLS of future return on the *signed* trailing return.

    A trending tape gives ``pooled_t`` well above 2 and a hit rate above 0.5; the driftless null gives
    ``pooled_t`` ≈ 0 and a hit rate ≈ 0.5.
    """
    tr = trailing_return(panel, lookback)
    pairs = []
    for col in panel.columns:
        r = panel[col]
        sig = np.sign(tr[col])
        # non-overlapping forward blocks
        fwd = (1.0 + r).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1.0
        df = pd.DataFrame({"sig": sig, "fwd": fwd}).dropna()
        df = df.iloc[::horizon]                              # non-overlapping
        pairs.append(df)
    allp = pd.concat(pairs, ignore_index=True)
    allp = allp[allp["sig"] != 0]
    signed = (allp["sig"] * allp["fwd"]).to_numpy()

    # Newey-West t on the mean of signed forward returns
    n = signed.size
    mu = signed.mean()
    e = signed - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e / n)
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k] / n)
    se = np.sqrt(max(lrv, 0.0) / n)

    pooled_t = float(mu / se) if se > 0 else np.nan
    return {
        "pooled_t": pooled_t,
        "mean_signed_bps": float(mu * 1e4),
        "hit_rate": float((signed > 0).mean()),
        "n_blocks": int(n),
        "trend_present": bool(mu > 0 and pooled_t > 2.0),
    }
