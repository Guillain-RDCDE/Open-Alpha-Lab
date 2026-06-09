"""The stat-arb engine — the reversion signal, the sample covariance, and the 'optimal' weights.

The §3.18 recipe is mean-variance optimization: given expected stock returns ``E`` and a covariance
matrix ``C``, the Sharpe-maximizing dollar-neutral weights are proportional to ``C^{-1} E``. Here the
expected return is a short-horizon **mean-reversion** signal (yesterday's residual tends to bounce), and
``C`` is the *sample* covariance of recent residual returns. The load-bearing question is two-sided:
does the reversion signal predict the next day (so there's something to harvest), and — the study's real
target — does inverting a noisy sample ``C`` *help*, or does it amplify estimation error into unstable,
self-defeating weights (Michaud's "optimization is error-maximization")?

Everything is a pure function of a ``dates x stock`` panel. Estimation is causal (trailing windows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def residual_returns(panel: pd.DataFrame, market: pd.Series | None = None, beta_window: int = 252) -> pd.DataFrame:
    """Idiosyncratic residual ``r_i - beta_i*market`` with a trailing (causal) beta — the trade's input."""
    mkt = (market if market is not None else panel.mean(axis=1)).astype(float)
    beta = panel.rolling(beta_window).cov(mkt).div(mkt.rolling(beta_window).var(), axis=0).shift(1)
    return panel - beta.mul(mkt, axis=0)


def reversion_signal(residuals: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Expected next return ``E_i = -`` (recent cumulative residual): a stretched-up name is expected to fall."""
    return (-residuals.rolling(lookback).sum())


def optimal_weights(E: np.ndarray, C: np.ndarray, shrink: float = 0.0) -> np.ndarray:
    """Mean-variance weights ``w ∝ C^{-1} E``, dollar-neutral and gross-normalised to 1.

    ``shrink`` blends the sample covariance toward its diagonal (``C_s = (1-s)C + s*diag(C)``) — the
    Ledoit-Wolf-style regulariser the beat-7 complement tests. ``shrink = 0`` is the raw, unstable
    inverse the headline uses.
    """
    d = np.diag(np.diag(C))
    Cs = (1.0 - shrink) * C + shrink * d
    try:
        w = np.linalg.solve(Cs, E)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(Cs, E, rcond=None)[0]
    w = w - w.mean()                                  # dollar-neutral
    gross = np.abs(w).sum()
    return w / gross if gross > 0 else w


def naive_weights(E: np.ndarray) -> np.ndarray:
    """The un-optimized version: dollar-neutral, gross-normalised, proportional to the signal itself."""
    w = E - E.mean()
    gross = np.abs(w).sum()
    return w / gross if gross > 0 else w


def signal_quality(panel: pd.DataFrame, market: pd.Series | None = None, lookback: int = 5) -> dict:
    """Does the reversion signal predict the next-day residual return? The cross-sectional IC.

    Each day, correlate the signal ``E_i`` with the *next* day's residual return across stocks; average
    the daily information coefficient. A positive IC means there is real reversion to harvest (gross);
    ~0 on the null. This is separate from whether the *optimizer* helps — that's :mod:`decompose`.
    """
    res = residual_returns(panel, market)
    E = reversion_signal(res, lookback)
    fwd = res.shift(-1)
    ics = []
    for t in range(lookback + 1, len(panel) - 1):
        e = E.iloc[t].dropna(); f = fwd.iloc[t].reindex(e.index).dropna()
        e = e.reindex(f.index)
        if len(f) > 5 and e.std() > 0 and f.std() > 0:
            ics.append(np.corrcoef(e, f)[0, 1])
    ics = np.array(ics)
    return {"mean_ic": float(ics.mean()), "ic_t": float(ics.mean() / ics.std(ddof=1) * np.sqrt(len(ics)))
            if ics.std(ddof=1) > 0 else np.nan, "n_days": int(len(ics)), "reversion_present": bool(ics.mean() > 0)}
