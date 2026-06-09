"""Internal Bar Strength — the signal, and the load-bearing question: **does a low close bounce?**

The whole strategy rests on one microstructure fact: a bar that closes near its low (IBS ≈ 0) has, on
average, a higher *next-day* return than one that closes near its high (IBS ≈ 1). It is a short-horizon
mean-reversion / liquidity-provision effect — you are effectively buying the day's sellers out and
collecting the bounce (Nuttall; popularised for ETFs by Connors & Alvarez). If that predictive tilt
weren't there, IBS would just be noise and the strategy would trade for nothing.

Everything here is a pure function of a daily OHLC frame. The load-bearing diagnostic measures the
tilt directly: bucket days by IBS and read the average *next-day* return per bucket.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def ibs(ohlc: pd.DataFrame) -> pd.Series:
    """Internal Bar Strength ``(Close − Low) / (High − Low)`` per bar, clipped to [0, 1].

    Days with no range (High == Low) are set to 0.5 (neither cheap nor dear). The result is the *only*
    input the strategy uses, and it is known at each day's close — so a position taken on it earns the
    *next* day's return with no look-ahead.
    """
    rng = (ohlc["High"] - ohlc["Low"]).replace(0.0, np.nan)
    val = ((ohlc["Close"] - ohlc["Low"]) / rng).clip(0.0, 1.0)
    return val.fillna(0.5).rename("ibs")


def next_day_return(ohlc: pd.DataFrame) -> pd.Series:
    """Close-to-close return of the *following* day, aligned to the bar whose IBS would predict it."""
    r = ohlc["Close"].pct_change().shift(-1)
    return r.rename("next_ret")


def reversal_strength(ohlc: pd.DataFrame, n_buckets: int = 5) -> dict:
    """Does a low IBS predict a higher next-day return? Bucket by IBS and read the gradient.

    Sorts days into ``n_buckets`` equal-count IBS buckets (bucket 0 = closed-near-low) and averages the
    *next-day* return in each. A **decreasing** profile — low-IBS days earning more than high-IBS days —
    is the reversal. We also report the low-minus-high next-day return spread (annualised) and the
    slope of a per-day OLS of next-day return on IBS; a **negative** slope is the effect in one number.
    A random-walk null gives a flat profile and a slope ≈ 0.
    """
    df = pd.DataFrame({"ibs": ibs(ohlc), "next_ret": next_day_return(ohlc)}).dropna()
    df["bucket"] = pd.qcut(df["ibs"].rank(method="first"), n_buckets, labels=False)
    table = df.groupby("bucket").agg(
        ibs=("ibs", "mean"),
        next_ret_bps=("next_ret", lambda x: x.mean() * 1e4),
        n=("next_ret", "size"),
    )
    slope, intercept = np.polyfit(df["ibs"].to_numpy(), df["next_ret"].to_numpy(), 1)
    lo = float(df.loc[df["bucket"] == 0, "next_ret"].mean())
    hi = float(df.loc[df["bucket"] == n_buckets - 1, "next_ret"].mean())
    return {
        "table": table,
        "low_minus_high_bps": (lo - hi) * 1e4,
        "low_minus_high_ann": (lo - hi) * TRADING_DAYS_PER_YEAR,
        "ibs_slope": float(slope),
        "reversal_present": bool(slope < 0),
        "n_days": int(len(df)),
    }
