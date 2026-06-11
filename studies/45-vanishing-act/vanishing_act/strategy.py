"""The size premium (SMB), and the honest way to score a decayed factor.

SMB = small-cap return minus large-cap return. The questions: is the long-run premium distinguishable
from zero, do small-caps even win on a *risk-adjusted* basis, and — the heart of it — has the premium
*decayed* since Banz (1981) put it in print? We measure the spread, its Lo (2002) t-stat, the two legs'
Sharpes, and a pre/post sub-period split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


def smb(returns: pd.DataFrame, small: str, large: str) -> pd.Series:
    """The size spread: monthly ``small − large`` return, dropping months either leg is missing."""
    df = returns[[small, large]].dropna()
    return (df[small] - df[large]).rename("smb")


def leg_summary(returns: pd.DataFrame, col: str) -> dict:
    """Stand-alone stats for one leg (so we can show small-caps trailing on Sharpe, not just SMB)."""
    return summary(returns[col].dropna())


def smb_stats(spread: pd.Series, periods_per_year: int = MONTHS) -> dict:
    """Annualised SMB mean, Sharpe, the Lo (2002) t-stat, and hit-rate."""
    r = pd.Series(spread).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("mean_ann", "sharpe", "tstat", "hit_rate", "n")}
    sr_m = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else np.nan
    se = np.sqrt((1.0 + 0.5 * sr_m**2) / len(r))   # Lo (2002) SE on the per-period Sharpe
    return {
        "mean_ann": float(r.mean() * periods_per_year),
        "sharpe": float(sr_m * np.sqrt(periods_per_year)),
        "tstat": float(sr_m / se) if se > 0 else np.nan,
        "hit_rate": float((r > 0).mean()),
        "n": int(len(r)),
    }


def decay_split(spread: pd.Series, cut_year: int = 2010) -> pd.DataFrame:
    """SMB before vs after ``cut_year`` — the post-publication-decay test."""
    r = pd.Series(spread).astype(float).dropna()
    out = {}
    for lab, sub in [(f"pre-{cut_year}", r[r.index.year < cut_year]),
                     (f"{cut_year}-on", r[r.index.year >= cut_year])]:
        s = smb_stats(sub)
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
