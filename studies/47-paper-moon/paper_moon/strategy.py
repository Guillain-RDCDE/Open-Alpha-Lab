"""The Fed Model timing rule, and the test that exposes it.

The rule: hold equities when the earnings yield E/P exceeds the 10-year Treasury yield, else hold
bonds. The honest tests: (1) does the timing beat buy-and-hold, and (2) does the *bond-yield* term add
anything — i.e. does the Fed signal (E/P − yield) forecast returns better than **E/P alone**? If E/P by
itself does as well, the model's defining ingredient is decoration (Asness 2003: it confuses a nominal
bond yield with a real earnings yield).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


def fed_signal(ep: pd.Series, y10: pd.Series) -> pd.Series:
    """The Fed-Model signal: earnings yield minus the 10-year yield (>0 ⇒ 'stocks cheap vs bonds')."""
    return (pd.Series(ep).astype(float) - pd.Series(y10).astype(float)).rename("fed_signal")


def fed_timing(tr: pd.Series, bond: pd.Series, signal: pd.Series) -> pd.Series:
    """Hold equities next month when this month's signal > 0, else bonds. Lagged — no look-ahead."""
    pos = (pd.Series(signal).astype(float).shift(1) > 0).astype(float)
    out = pos * pd.Series(tr).astype(float) + (1.0 - pos) * pd.Series(bond).astype(float).shift(1)
    return out.dropna().rename("fed_timing")


def time_in_stocks(signal: pd.Series) -> float:
    """Fraction of months the rule is in equities (signal > 0) — high when E/P usually beats yields."""
    return float((pd.Series(signal).astype(float).shift(1) > 0).mean())


def buy_hold(tr: pd.Series) -> pd.Series:
    return pd.Series(tr).astype(float).dropna().rename("buy_hold")


def forward_return(tr: pd.Series, horizon: int = 12) -> pd.Series:
    """Cumulative equity return over the next ``horizon`` months (for the predictive-corr test)."""
    r = pd.Series(tr).astype(float)
    fwd = (1.0 + r).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1.0
    return fwd.rename("fwd")


def predictive_corr(predictor: pd.Series, tr: pd.Series, horizon: int = 12) -> float:
    """Correlation of a predictor with the next-``horizon``-month equity return (overlap-aware enough
    for a like-for-like comparison between the Fed signal and E/P alone)."""
    fwd = forward_return(tr, horizon)
    df = pd.concat([pd.Series(predictor).astype(float).rename("p"), fwd], axis=1).dropna()
    return float(df["p"].corr(df["fwd"])) if len(df) > 2 else np.nan


def summary(returns: pd.Series, periods_per_year: int = MONTHS) -> dict:
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
