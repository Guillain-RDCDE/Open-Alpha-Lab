"""Realized-volatility estimators and the load-bearing question: **is variance forecastable?**

The entire overlay rests on one stylized fact — *volatility clusters*. Big moves follow big moves;
calm follows calm (Mandelbrot 1963; Engle 1982, ARCH). If tomorrow's variance were unpredictable
from today's, scaling exposure by ``1/σ̂`` would just add noise and cost. So before any backtest we
measure the thing directly: how strongly does this period's realized variance predict the next?

Everything here works on a plain daily-**return** series and is a pure function of it — no fitting,
no parameters beyond an estimator window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def to_returns(close: pd.Series) -> pd.Series:
    """Simple daily returns from a close series (the tape every diagnostic consumes)."""
    return close.pct_change().dropna().rename("ret")


# --------------------------------------------------------------------------- #
# Estimators — trailing-window and RiskMetrics EWMA realized vol (per-bar)
# --------------------------------------------------------------------------- #

def realized_vol(returns: pd.Series, window: int = 21) -> pd.Series:
    """Trailing **per-bar** realized volatility: the rolling standard deviation over ``window`` days.

    Per-bar (not annualised) because that is the unit the overlay sizes in. ``window=21`` ≈ one
    trading month, the canonical Moreira–Muir horizon. Leading rows are NaN until the window fills.
    """
    return returns.rolling(window).std(ddof=1).rename("rv")


def ewma_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics exponentially-weighted per-bar vol: ``σ²_t = λ σ²_{t−1} + (1−λ)(r_{t−1}−μ)²``.

    The ``lam=0.94`` daily decay is the RiskMetrics standard. EWMA reacts faster to a regime change
    than a flat window while using all history — a useful robustness alternative to
    :func:`realized_vol` for the overlay's variance forecast.
    """
    r = returns.astype(float)
    var = r.ewm(alpha=1.0 - lam, adjust=False).var(bias=True)
    return np.sqrt(var).rename("ewma_vol")


# --------------------------------------------------------------------------- #
# Forecastability — the engine: does today's variance predict tomorrow's?
# --------------------------------------------------------------------------- #

def forecastability(returns: pd.Series, window: int = 21, horizon: int = 21) -> dict:
    """How predictable is variance? The number that decides whether the overlay *can* work.

    We form non-overlapping blocks of ``horizon`` days, take each block's realized variance, and
    measure how this block predicts the next via an AR(1) on **log-variance** (log to tame the
    right skew, the natural scale for a multiplicative process)::

        log var[k+1] = a + ρ · log var[k] + e

    Returns:
      * ``rho`` — the AR(1) persistence of log-variance; ``r2`` — its in-sample fit;
      * ``autocorr_lag1`` — lag-1 autocorrelation of block variance (a model-free echo of ``rho``);
      * ``vol_of_vol`` — std of log block-variance, i.e. how much there is to forecast at all.

    A clustered tape gives a high ``rho`` (≈ persistence of the regime); a flat-vol tape gives
    ``rho ≈ 0`` and ``vol_of_vol ≈ 0`` — nothing to read, and the overlay correctly adds nothing.
    """
    r = returns.astype(float).dropna()
    # Non-overlapping block variances (independent observations for the AR(1)).
    n_blocks = len(r) // horizon
    if n_blocks < 4:
        raise ValueError("need at least 4 horizon-blocks of returns to fit the AR(1)")
    blocks = r.iloc[: n_blocks * horizon].to_numpy().reshape(n_blocks, horizon)
    block_var = blocks.var(axis=1, ddof=1)
    block_var = np.clip(block_var, 1e-12, None)
    logv = np.log(block_var)

    x, y = logv[:-1], logv[1:]
    rho, a = np.polyfit(x, y, 1)
    yhat = a + rho * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # model-free lag-1 autocorr of the block-variance level
    bv = block_var - block_var.mean()
    ac1 = float(bv[:-1] @ bv[1:] / (bv @ bv)) if bv @ bv > 0 else float("nan")

    return {
        "rho": float(rho),
        "r2": float(r2),
        "autocorr_lag1": ac1,
        "vol_of_vol": float(logv.std(ddof=1)),
        "n_blocks": int(n_blocks),
        "horizon": int(horizon),
    }
