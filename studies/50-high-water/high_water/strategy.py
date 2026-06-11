"""The 52-week-high signal, the momentum control, and the honest comparison.

George & Hwang (2004): rank stocks by **nearness to their 52-week high** (price ÷ trailing-12-month
max), long the near, short the far. Their claim is that this beats — and is distinct from — ordinary
momentum. The tests: (1) does the nearness long-short pay, and (2) is it actually different from the
standard **12-2 momentum** control (Jegadeesh-Titman: trailing 12 months skipping the most recent,
to dodge the 1-month reversal), or ~0.9-correlated to it (a relabelled factor)?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


def _prices(returns: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct a price index per stock from monthly returns (NaN preserved where data is missing)."""
    p = (1.0 + returns.fillna(0.0)).cumprod()
    return p.where(returns.notna())


def nearness(returns: pd.DataFrame, window: int = MONTHS) -> pd.DataFrame:
    """Price ÷ trailing-``window``-month high. 1.0 means the stock sits at its 52-week high."""
    p = _prices(returns)
    return p / p.rolling(window, min_periods=window).max()


def momentum(returns: pd.DataFrame, window: int = MONTHS, skip: int = 1) -> pd.DataFrame:
    """Standard **12-2 momentum**, the control: total return from ``window`` months back to ``skip``
    months back (default 12 → 1), i.e. the trailing year *excluding the most recent month* — the
    Jegadeesh-Titman convention that sidesteps the 1-month reversal. ``skip=0`` recovers the naive
    trailing-12-month return."""
    p = _prices(returns)
    return p.shift(skip) / p.shift(window) - 1.0


def cross_section_hedge(returns: pd.DataFrame, signal: pd.DataFrame, q: float = 0.2,
                        lag: int = 1, min_names: int = 30) -> pd.Series:
    """Monthly long-top-``q`` / short-bottom-``q`` hedge on a cross-sectional ``signal`` (lagged)."""
    sig = signal.shift(lag)
    out = {}
    for t in returns.index:
        s = sig.loc[t].dropna()
        n = returns.loc[t].dropna()
        common = s.index.intersection(n.index)
        if len(common) < min_names:
            continue
        s, n = s.loc[common], n.loc[common]
        lo, hi = s <= s.quantile(q), s >= s.quantile(1.0 - q)
        out[t] = float(n[hi].mean() - n[lo].mean())
    return pd.Series(out, name="hedge").dropna()


def signal_overlap(returns: pd.DataFrame) -> float:
    """Correlation between the 52-week-high hedge and the 12-2 momentum hedge — how redundant they are."""
    h = cross_section_hedge(returns, nearness(returns))
    m = cross_section_hedge(returns, momentum(returns))
    df = pd.concat([h.rename("h"), m.rename("m")], axis=1).dropna()
    return float(df["h"].corr(df["m"])) if len(df) > 2 else np.nan


def net_of_cost(hedge: pd.Series, cost_bps: float, turnover: float = 3.2) -> pd.Series:
    """Net hedge after a per **one-way** trade cost. ``turnover`` is the one-way traded notional per
    month in NAV multiples: replacing ~80% of a two-sided book = 2 sides × 2 trades × 0.8 ≈ **3.2×
    NAV one-way** (an earlier version charged 1.6×, treating the sweep as round-trip)."""
    return (hedge - turnover * cost_bps / 1e4).rename(hedge.name)


def stats(spread: pd.Series, periods_per_year: int = MONTHS) -> dict:
    """Annualised mean, Sharpe, Lo (2002) t-stat, hit-rate for a monthly hedge series."""
    r = pd.Series(spread).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("mean_ann", "sharpe", "tstat", "hit_rate", "n")}
    sr_m = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else np.nan
    se = np.sqrt((1.0 + 0.5 * sr_m**2) / len(r))
    return {
        "mean_ann": float(r.mean() * periods_per_year),
        "sharpe": float(sr_m * np.sqrt(periods_per_year)),
        "tstat": float(sr_m / se) if se > 0 else np.nan,
        "hit_rate": float((r > 0).mean()),
        "n": int(len(r)),
    }
