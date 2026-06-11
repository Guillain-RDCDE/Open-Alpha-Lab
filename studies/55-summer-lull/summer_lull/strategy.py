"""The Halloween seasonal — winter vs summer, and whether sitting out summer actually helps.

Bouman & Jacobsen (2002): equity returns are higher Nov–Apr ("winter") than May–Oct ("summer"). The
honest tests: (1) is the winter-minus-summer gap real and significant (a Welch t), and (2) does the
"sell in May" rule (hold equities Nov–Apr, cash otherwise) beat buy-and-hold — *given that summer
returns are still positive*?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINTER = {11, 12, 1, 2, 3, 4}


def is_winter(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(pd.DatetimeIndex(index).month.isin(WINTER), index=index, name="winter")


def seasonal_split(returns: pd.Series) -> dict:
    """Winter vs summer mean monthly return, annualised halves, and the Welch t for the difference."""
    r = pd.Series(returns).astype(float).dropna()
    w = r[is_winter(r.index).to_numpy()]
    s = r[~is_winter(r.index).to_numpy()]
    se = np.sqrt(w.var(ddof=1) / len(w) + s.var(ddof=1) / len(s))
    return {
        "winter_bp": float(w.mean() * 1e4),
        "summer_bp": float(s.mean() * 1e4),
        "winter_ann": float((1 + w.mean()) ** 6 - 1),
        "summer_ann": float((1 + s.mean()) ** 6 - 1),
        "welch_t": float((w.mean() - s.mean()) / se) if se > 0 else np.nan,
        "n_winter": int(len(w)),
        "n_summer": int(len(s)),
    }


def sell_in_may(returns: pd.Series) -> pd.Series:
    """Hold equities in winter (Nov–Apr), cash in summer (May–Oct)."""
    r = pd.Series(returns).astype(float).dropna()
    return (is_winter(r.index).astype(float) * r).rename("sell_in_may")


def buy_hold(returns: pd.Series) -> pd.Series:
    return pd.Series(returns).astype(float).dropna().rename("buy_hold")


def summary(returns: pd.Series, periods_per_year: int = 12) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown for a monthly return series."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "n")}
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
        "n": int(len(r)),
    }
