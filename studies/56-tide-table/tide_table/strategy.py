"""CAPE as a long-horizon return forecaster — and the horizon caveat that makes it not a timer.

Campbell & Shiller (1998): a high cyclically-adjusted P/E (CAPE) precedes low long-run returns. The
tests: (1) does CAPE forecast the next-10-year real return (correlation, R², valuation buckets), and
(2) the honest limit — at a *1-year* horizon the relationship is far weaker, so CAPE sets long-run
expectations rather than timing the next move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_corr(cape: pd.Series, fwd: pd.Series) -> dict:
    """Correlation of CAPE (and the earnings yield 1/CAPE) with a forward real-return series, + R²."""
    df = pd.concat([pd.Series(cape).astype(float).rename("cape"),
                    pd.Series(fwd).astype(float).rename("fwd")], axis=1).dropna()
    df = df[df["cape"] > 0]
    if len(df) < 3:
        return {"corr_cape": np.nan, "corr_yield": np.nan, "r2": np.nan, "n": len(df)}
    r_cape = float(np.corrcoef(df["cape"], df["fwd"])[0, 1])
    r_yield = float(np.corrcoef(1.0 / df["cape"], df["fwd"])[0, 1])
    return {"corr_cape": r_cape, "corr_yield": r_yield, "r2": float(r_yield ** 2), "n": int(len(df))}


def implied_return(cape: pd.Series, fwd: pd.Series):
    """OLS of the forward real return on the earnings yield (1/CAPE): returns (intercept, slope)."""
    df = pd.concat([(1.0 / pd.Series(cape).astype(float)).rename("y"),
                    pd.Series(fwd).astype(float).rename("fwd")], axis=1).dropna()
    df = df[np.isfinite(df["y"])]
    slope, intercept = np.polyfit(df["y"], df["fwd"], 1)
    return float(intercept), float(slope)


def bucket_returns(cape: pd.Series, fwd: pd.Series, n: int = 3) -> pd.Series:
    """Mean forward real return by CAPE bucket (cheap → expensive) — the monotone valuation ladder."""
    df = pd.concat([pd.Series(cape).astype(float).rename("cape"),
                    pd.Series(fwd).astype(float).rename("fwd")], axis=1).dropna()
    df = df[df["cape"] > 0]
    labels = ["cheap", "mid", "expensive"] if n == 3 else [f"q{i+1}" for i in range(n)]
    df["bucket"] = pd.qcut(df["cape"], n, labels=labels)
    return df.groupby("bucket", observed=True)["fwd"].mean()
