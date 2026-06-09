"""The investable book — trade the spread's z-score back to its mean, with a causal hedge ratio.

The classic pairs trade: when the spread is stretched far above its mean (z high) it's "rich", so
**short the spread** (short A, long β·B); when far below (z low), **long the spread**; close when it
reverts to the mean. The position is held until the exit band is crossed.

Two honesty choices are baked in. The hedge ratio ``β`` is estimated on a **trailing window** (causal,
no full-sample look-ahead — the trap [Study 22](../../22-crystal-ball/) warns about), and the z-score is
likewise rolling. The spread return earned is ``position · (r_A − β·r_B)``, a dollar-neutral-ish bet,
net of cost charged on each change of position (both legs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .spread import zscore

TRADING_DAYS_PER_YEAR = 252


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    r = pd.Series(returns).astype(float).dropna()
    mean, std = r.mean(), r.std(ddof=1)
    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    years = len(r) / periods_per_year
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and equity.iloc[-1] > 0 else np.nan
    return {
        "sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
        "ann_return": float(mean * periods_per_year),
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "cagr": float(cagr),
        "max_drawdown": float(drawdown.min()),
        "n_days": int(len(r)),
    }


def rolling_hedge_ratio(a: pd.Series, b: pd.Series, window: int = 252) -> pd.Series:
    """Trailing-window OLS slope of ``log A`` on ``log B`` — a *causal* hedge ratio (no look-ahead)."""
    la, lb = np.log(a.astype(float)), np.log(b.astype(float))
    cov = la.rolling(window).cov(lb)
    var = lb.rolling(window).var()
    return (cov / var).rename("beta")


def _positions(z: pd.Series, entry: float, exit: float) -> pd.Series:
    """State machine: enter ∓1 when |z| crosses ``entry``, exit to 0 when |z| < ``exit``."""
    pos = np.zeros(len(z))
    state = 0.0
    zz = z.to_numpy()
    for i, zi in enumerate(zz):
        if np.isnan(zi):
            pos[i] = 0.0; continue
        if state == 0.0:
            if zi > entry:
                state = -1.0                                  # spread rich -> short the spread
            elif zi < -entry:
                state = 1.0                                   # spread cheap -> long the spread
        else:
            if abs(zi) < exit:
                state = 0.0
        pos[i] = state
    return pd.Series(pos, index=z.index, name="position")


def pairs_returns(a: pd.Series, b: pd.Series, entry: float = 1.5, exit: float = 0.3,
                  window: int = 63, beta_window: int = 252, cost_bps: float = 2.0) -> pd.Series:
    """Net daily return of the causal z-score pairs trade on ``(A, B)``.

    Hedge ratio and z-score are both trailing (causal). The spread return is ``pos·(r_A − β·r_B)`` with
    ``pos`` lagged one day; cost is charged on each change of position (across the two legs).
    """
    beta = rolling_hedge_ratio(a, b, beta_window)
    sp = (np.log(a.astype(float)) - beta * np.log(b.astype(float))).rename("spread")
    z = zscore(sp, window)
    pos = _positions(z, entry, exit).shift(1)
    ra, rb = a.pct_change(), b.pct_change()
    spread_ret = pos * (ra - beta.shift(1) * rb)
    cost = (cost_bps * 1e-4) * pos.diff().abs() * (1.0 + beta.shift(1).abs())   # both legs traded
    return (spread_ret - cost).dropna().rename("pairs")


def compare(a: pd.Series, b: pd.Series, cost_bps: float = 2.0, periods_per_year: int = TRADING_DAYS_PER_YEAR,
            **kw) -> dict:
    """Headline stats for the pairs book, plus its trade frequency."""
    r = pairs_returns(a, b, cost_bps=cost_bps, **kw)
    s = summary(r, periods_per_year)
    # count round-trips (position resets to 0)
    beta = rolling_hedge_ratio(a, b, kw.get("beta_window", 252))
    sp = (np.log(a.astype(float)) - beta * np.log(b.astype(float)))
    z = zscore(sp, kw.get("window", 63))
    pos = _positions(z, kw.get("entry", 1.5), kw.get("exit", 0.3))
    trades = int((pos.diff().abs() > 0).sum())
    return {**s, "trades": trades}
