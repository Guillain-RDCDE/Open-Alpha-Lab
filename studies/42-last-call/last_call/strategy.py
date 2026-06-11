"""The turn-of-the-month window, the per-day comparison, and the tradable rule.

The effect (Lakonishok & Smidt 1988; McConnell & Xu 2008): daily returns concentrate in a window
around the month boundary — conventionally the **last trading day** of a month through the **first
three** of the next (the [-1, +3] window). The honest tests: (1) do TOM days really out-earn the rest,
and (2) does *trading* only the window beat just holding the index, after costs?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def tom_mask(index: pd.DatetimeIndex, last: int = 1, first: int = 3) -> pd.Series:
    """Boolean mask of turn-of-the-month days for a daily index.

    Marks the final ``last`` trading day(s) of each month and the first ``first`` of the next — the
    classic [-1, +3] window with the defaults. Uses trading-day position within the calendar month, so
    it adapts to holidays automatically.
    """
    idx = pd.DatetimeIndex(index)
    ym = idx.to_period("M")
    pos_start = pd.Series(np.arange(len(idx)), index=idx).groupby(ym).cumcount() + 1
    pos_end = pd.Series(np.arange(len(idx)), index=idx).groupby(ym).cumcount(ascending=False) + 1
    mask = (pos_start <= first) | (pos_end <= last)
    return pd.Series(mask.values, index=idx, name="tom")


def tom_vs_rest(returns: pd.Series, last: int = 1, first: int = 3) -> dict:
    """Mean daily return (bp) on turn-of-the-month days vs the rest, plus counts and a t-stat."""
    r = pd.Series(returns).astype(float).dropna()
    m = tom_mask(r.index, last, first)
    tom, rest = r[m.values], r[~m.values]
    # Welch t for the difference in means
    s = np.sqrt(tom.var(ddof=1) / len(tom) + rest.var(ddof=1) / len(rest))
    t = (tom.mean() - rest.mean()) / s if s > 0 else np.nan
    return {
        "tom_bp": float(tom.mean() * 1e4),
        "rest_bp": float(rest.mean() * 1e4),
        "all_bp": float(r.mean() * 1e4),
        "n_tom": int(len(tom)),
        "n_rest": int(len(rest)),
        "tom_share": float(m.mean()),
        "welch_t": float(t),
    }


def tom_returns(returns: pd.Series, cost_bps: float = 2.0, last: int = 1, first: int = 3) -> pd.Series:
    """The tradable rule: hold the index only on turn-of-the-month days, cash otherwise.

    Position is the TOM mask lagged one day (you must be in *before* the move); each entry/exit pays
    ``cost_bps``. Returns the net daily return series (mostly zeros — you're in cash ~80% of the time).
    """
    r = pd.Series(returns).astype(float).dropna()
    pos = tom_mask(r.index, last, first).astype(float)
    gross = pos.shift(1).fillna(0.0) * r
    cost = (cost_bps * 1e-4) * pos.diff().abs().fillna(0.0)
    return (gross - cost).rename("tom")


def buy_hold(returns: pd.Series) -> pd.Series:
    return pd.Series(returns).astype(float).dropna().rename("buy_hold")


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown, time-in-market for a daily return series."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "time_in_market", "n")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {
        "sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
        "cagr": float(cagr),
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float(dd),
        "time_in_market": float((r != 0).mean()),
        "n": int(len(r)),
    }
