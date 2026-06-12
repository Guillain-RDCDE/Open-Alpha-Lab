"""The four frozen ingredients and their confluence count.

Execution-lag convention (one lag, documented exactly, bench rule): **every signal is
known at the close of day t and earns the return of day t+1** — one ``shift``, applied
once, in :func:`ambush.strategy.book`, never here. The lone subtlety is the
turn-of-the-month leg: the window is a pure function of the exchange calendar, so
"day t+1 is a TOM day" is *already known* at the close of t — ``s_tom`` therefore marks
*tomorrow's* membership (``tom_mask.shift(-1)``) and then receives the same single lag
as everyone else, landing exactly on the [−1, +3] window study 42 showed is the one
that pays (lagging the raw mask instead silently trades [0, +4]).

Thresholds are frozen in ``docs/preregistration.md`` and inherited from the source
studies — IBS ≤ 0.20 (study 19's low bucket), the [−1, +3] TOM window (study 42),
a plain down close (study 13's daily adaptation), VIX ≥ 1.15× its trailing 20-day
mean (study 03's lesson that absolute VIX levels are regime-dependent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

IBS_LOW = 0.20
VIX_RATIO = 1.15
VIX_WINDOW = 20
TOM_LAST, TOM_FIRST = 1, 3

SIGNAL_COLS = ["s_ibs", "s_tom", "s_red", "s_vix"]


def ibs(df: pd.DataFrame) -> pd.Series:
    """Internal Bar Strength ``(C−L)/(H−L)`` in [0, 1]; a zero-range bar reads 0.5."""
    rng = df["High"] - df["Low"]
    out = (df["Close"] - df["Low"]) / rng
    return out.where(rng > 0, 0.5).rename("ibs")


def low_ibs(df: pd.DataFrame, threshold: float = IBS_LOW) -> pd.Series:
    """S1 — the bar closed in the bottom of its range (study 19's bounce bucket)."""
    return (ibs(df) <= threshold).rename("s_ibs")


def tom_mask(index: pd.DatetimeIndex, last: int = TOM_LAST, first: int = TOM_FIRST) -> pd.Series:
    """Turn-of-the-month membership: final ``last`` trading day(s) of a month plus the
    first ``first`` of the next — study 42's [−1, +3] window, holiday-aware."""
    idx = pd.DatetimeIndex(index)
    ym = idx.to_period("M")
    pos_start = pd.Series(np.arange(len(idx)), index=idx).groupby(ym).cumcount() + 1
    pos_end = pd.Series(np.arange(len(idx)), index=idx).groupby(ym).cumcount(ascending=False) + 1
    mask = (pos_start <= first) | (pos_end <= last)
    return pd.Series(mask.values, index=idx, name="tom")


def tom_tomorrow(index: pd.DatetimeIndex) -> pd.Series:
    """S2 — *tomorrow* is a TOM day (calendar-known at today's close; see module note).

    The last row reads False: tomorrow's calendar slot is outside the sample, and an
    armed signal there could never earn a return anyway.
    """
    return tom_mask(index).shift(-1).fillna(False).astype(bool).rename("s_tom")


def red_day(df: pd.DataFrame) -> pd.Series:
    """S3 — today closed below yesterday's close."""
    return (df["Close"].pct_change() < 0).fillna(False).rename("s_red")


def vix_stress(vix: pd.Series, window: int = VIX_WINDOW, ratio: float = VIX_RATIO) -> pd.Series:
    """S4 — VIX closes ≥ ``ratio`` × its trailing ``window``-day mean (incl. today)."""
    ma = vix.rolling(window).mean()
    return (vix >= ratio * ma).fillna(False).rename("s_vix")


def confluence(spy: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """The four booleans and their count, all stamped *at the close of day t*.

    ``vix`` is aligned on SPY's calendar and forward-filled across holiday mismatches
    (a stale VIX close is yesterday's information — never tomorrow's).
    """
    v = vix.reindex(spy.index).ffill()
    out = pd.concat(
        [low_ibs(spy), tom_tomorrow(spy.index), red_day(spy), vix_stress(v)], axis=1
    )
    out["count"] = out[SIGNAL_COLS].sum(axis=1).astype(int)
    return out
