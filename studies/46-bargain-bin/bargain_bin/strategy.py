"""The value premium (HML), scored across regimes.

HML = value return minus growth return. The questions: is the long-run spread positive and
significant, do value stocks even win on a risk-adjusted basis, and — the heart of the modern debate —
is the premium *stable*, or a regime-switching bet with a decade-long underwater stretch that no real
investor survives? We measure the spread, its Lo (2002) t-stat, each leg's Sharpe, and the regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


def hml(returns: pd.DataFrame, value: str, growth: str) -> pd.Series:
    """The value spread: monthly ``value − growth`` return, dropping months either leg is missing."""
    df = returns[[value, growth]].dropna()
    return (df[value] - df[growth]).rename("hml")


def leg_summary(returns: pd.DataFrame, col: str) -> dict:
    return summary(returns[col].dropna())


def hml_stats(spread: pd.Series, periods_per_year: int = MONTHS) -> dict:
    """Annualised HML mean, Sharpe, the Lo (2002) t-stat, and hit-rate."""
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


def regime_split(spread: pd.Series, breaks=(2007, 2021)) -> pd.DataFrame:
    """HML by regime: before ``breaks[0]``, the ``breaks[0]..breaks[1]-1`` lost decade, and after."""
    r = pd.Series(spread).astype(float).dropna()
    lo, hi = breaks
    segs = [(f"pre-{lo}", r[r.index.year < lo]),
            (f"{lo}-{hi - 1} (lost decade)", r[(r.index.year >= lo) & (r.index.year < hi)]),
            (f"{hi}-on", r[r.index.year >= hi])]
    out = {}
    for lab, sub in segs:
        s = hml_stats(sub)
        out[lab] = {"mean_ann": s["mean_ann"], "sharpe": s["sharpe"], "n": s["n"]}
    return pd.DataFrame(out).T


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
