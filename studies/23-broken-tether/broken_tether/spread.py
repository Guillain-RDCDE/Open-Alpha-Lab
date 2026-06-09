"""The spread engine — hedge ratio, z-score, and the load-bearing question: **is the spread stationary?**

A pairs trade only works if the two legs are **cointegrated**: some linear combination ``log A − β·log B``
is *stationary* — it wanders around a fixed mean and pulls back, rather than drifting off. If that's
true, a spread stretched far from its mean is a bet on reversion. If it's false (the legs merely trended
together for a while), the "spread" is itself a random walk and there is nothing to revert to. So before
any backtest we measure the spread's mean-reversion directly via its **half-life**.

Everything here is a pure function of two price series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def hedge_ratio(a: pd.Series, b: pd.Series) -> float:
    """OLS slope ``β`` of ``log A`` on ``log B`` — the units of B that hedge one unit of A."""
    la, lb = np.log(a.astype(float)), np.log(b.astype(float))
    df = pd.DataFrame({"a": la, "b": lb}).dropna()
    x = np.column_stack([np.ones(len(df)), df["b"].to_numpy()])
    beta = np.linalg.lstsq(x, df["a"].to_numpy(), rcond=None)[0]
    return float(beta[1])


def spread(a: pd.Series, b: pd.Series, beta: float | None = None) -> pd.Series:
    """The (log) spread ``log A − β·log B`` — the series a pairs trade bets will mean-revert."""
    if beta is None:
        beta = hedge_ratio(a, b)
    return (np.log(a.astype(float)) - beta * np.log(b.astype(float))).rename("spread")


def zscore(spread_series: pd.Series, window: int = 63) -> pd.Series:
    """Rolling z-score of the spread (the trade trigger): how many σ from its recent mean it sits."""
    m = spread_series.rolling(window).mean()
    s = spread_series.rolling(window).std(ddof=1)
    return ((spread_series - m) / s).rename("zscore")


def half_life(spread_series: pd.Series) -> float:
    """Mean-reversion half-life (days) of the spread via an AR(1) ``Δs_t = a + ρ·s_{t−1} + e``.

    From the regression slope ``ρ`` (which is negative for a reverting series), the half-life is
    ``−ln(2)/ln(1+ρ)``. A *short* half-life (days–weeks) means a tradable, stationary spread; an
    *infinite/huge* half-life means the spread is a random walk (no reversion) — the spurious-pair tell.
    """
    s = pd.Series(spread_series).astype(float).dropna()
    ds = s.diff().dropna()
    lag = s.shift(1).reindex(ds.index)
    x = np.column_stack([np.ones(len(ds)), lag.to_numpy()])
    coef = np.linalg.lstsq(x, ds.to_numpy(), rcond=None)[0]
    rho = coef[1]
    if rho >= 0:
        return float("inf")                                   # not mean-reverting
    return float(-np.log(2) / np.log(1.0 + rho))


def stationarity(a: pd.Series, b: pd.Series, window: int = 63) -> dict:
    """Is the pair tradable? The hedge ratio, the spread half-life, and the z-score's range.

    A real (cointegrated) pair has a finite, short half-life; a spurious pair has a near-infinite one.
    """
    beta = hedge_ratio(a, b)
    sp = spread(a, b, beta)
    hl = half_life(sp)
    z = zscore(sp, window)
    return {
        "hedge_ratio": float(beta),
        "half_life_days": float(hl),
        "is_reverting": bool(np.isfinite(hl) and hl < 252),
        "z_abs_max": float(z.abs().max()),
        "n": int(sp.dropna().shape[0]),
    }
