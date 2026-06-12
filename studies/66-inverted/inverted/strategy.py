"""The yield-curve slope as a forward equity signal — conditional returns and the horizon question.

The 10y−3m slope inverts (<0) before recessions. The tests: do equity returns differ after an inversion
vs a normal curve, and at what horizon does the gap appear (the recession lead is ~6-24 months, so the
12-month and 18-month splits tell different stories)?
"""
from __future__ import annotations
import numpy as np, pandas as pd
MONTHS = 12


def curve_slope(long_yield: pd.Series, short_yield: pd.Series) -> pd.Series:
    """10y minus 3m yield (percentage points). < 0 ⇒ inverted."""
    return (pd.Series(long_yield).astype(float) - pd.Series(short_yield).astype(float)).rename("curve")


def forward_return(eq_ret: pd.Series, horizon: int) -> pd.Series:
    r = pd.Series(eq_ret).astype(float)
    return ((1.0 + r).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1.0).rename("fwd")


def conditional_forward(slope: pd.Series, eq_ret: pd.Series, horizon: int = 18) -> dict:
    """Mean forward equity return over ``horizon`` months after an inverted curve vs a normal one."""
    fwd = forward_return(eq_ret, horizon)
    df = pd.concat([pd.Series(slope).astype(float).rename("curve"), fwd], axis=1).dropna()
    inv, norm = df[df["curve"] < 0], df[df["curve"] >= 0]
    return {"horizon": horizon, "inverted_fwd": float(inv["fwd"].mean()), "normal_fwd": float(norm["fwd"].mean()),
            "gap": float(inv["fwd"].mean() - norm["fwd"].mean()),
            "n_inverted": int(len(inv)), "n_normal": int(len(norm)),
            "inverted_share": float((df["curve"] < 0).mean())}


def summary(returns: pd.Series, periods_per_year: int = MONTHS) -> dict:
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "n")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod(); dd = (eq / eq.cummax() - 1.0).min(); years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {"sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan, "cagr": float(cagr),
            "vol_ann": float(std * np.sqrt(periods_per_year)), "max_drawdown": float(dd), "n": int(len(r))}
