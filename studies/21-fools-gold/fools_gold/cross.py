"""The moving-average crossover — the signal, and the load-bearing question: **does it predict?**

The folk claim isn't just "trends exist" — it's that the *crossover event itself* is informative: once
the fast average is above the slow one ("golden"), forward returns are supposed to be higher than when
it's below ("death"). That is a testable, falsifiable statement, separate from whether the asset drifts
up on its own. So before any backtest we measure it directly: split every day into golden vs death by
yesterday's crossover state, and compare the *next* day's return.

Everything here is a pure function of a daily close series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def moving_averages(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Fast and slow simple moving averages of the close (the classic 50/200 'golden cross' pair)."""
    return pd.DataFrame({"fast": close.rolling(fast).mean(), "slow": close.rolling(slow).mean()})


def cross_state(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """+1 when the fast MA is above the slow ('golden'), −1 when below ('death'), as of each close."""
    ma = moving_averages(close, fast, slow)
    return np.sign(ma["fast"] - ma["slow"]).rename("state")


def signal_value(close: pd.Series, fast: int = 50, slow: int = 200) -> dict:
    """Does the golden state predict a higher next-day return than the death state? Measure it.

    Splits days by yesterday's crossover state (so it's tradable, no look-ahead) and compares the mean
    next-day return in each. Reports the golden-minus-death spread (annualised) and the share of time
    spent golden. A real signal shows a clearly positive spread; on a driftless null it is ≈ 0.
    """
    state = cross_state(close, fast, slow).shift(1)
    fwd = close.pct_change()
    df = pd.DataFrame({"state": state, "fwd": fwd}).dropna()
    g = df.loc[df["state"] > 0, "fwd"]
    d = df.loc[df["state"] < 0, "fwd"]
    spread = (g.mean() - d.mean())
    return {
        "golden_ann_pct": float(g.mean() * TRADING_DAYS_PER_YEAR * 100.0),
        "death_ann_pct": float(d.mean() * TRADING_DAYS_PER_YEAR * 100.0),
        "spread_ann_pct": float(spread * TRADING_DAYS_PER_YEAR * 100.0),
        "frac_golden": float((df["state"] > 0).mean()),
        "n_golden": int((df["state"] > 0).sum()),
        "n_death": int((df["state"] < 0).sum()),
        "signal_present": bool(spread > 0),
    }
