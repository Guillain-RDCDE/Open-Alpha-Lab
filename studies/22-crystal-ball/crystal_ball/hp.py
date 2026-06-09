"""The Hodrick-Prescott filter — two-sided (the trap) and one-sided (the honest version).

The HP filter splits a series ``y`` into a smooth **trend** ``tau`` and a **cycle** ``y - tau`` by
solving::

    minimise   sum_t (y_t - tau_t)^2  +  lam * sum_t (tau_{t+1} - 2 tau_t + tau_{t-1})^2

The first term keeps the trend close to the data; the second penalises curvature (``lam`` controls
smoothness; 1600 is the textbook quarterly value, ~10^5-10^6 for daily). The closed-form solution
``tau = (I + lam D'D)^{-1} y`` is **two-sided**: the trend at time ``t`` is a function of the *entire*
series, future included. That is the trap — a strategy that trades the cycle ``y_t - tau_t`` is using
tomorrow's data to decide today's position.

The **one-sided** (causal) filter fixes this: the trend at ``t`` is computed from a window ending at
``t`` (``y[t-window : t+1]``), taking only the endpoint. It uses no future data, so a strategy built on
its cycle is actually tradable. The whole study is the gap between the two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import solveh_banded


def _hp_trend_array(y: np.ndarray, lam: float) -> np.ndarray:
    """Two-sided HP trend of a 1-D array via the pentadiagonal normal equations (uses all of ``y``)."""
    n = y.size
    if n < 5:
        return y.copy()
    # A = I + lam * D'D, with D the (n-2) x n second-difference operator. A is symmetric pentadiagonal.
    main = np.ones(n) + lam * np.array([1.0] + [5.0] + [6.0] * (n - 4) + [5.0] + [1.0])
    off1 = lam * np.array([-2.0] + [-4.0] * (n - 3) + [-2.0])      # length n-1
    off2 = lam * np.ones(n - 2)                                     # length n-2
    ab = np.zeros((3, n))                                           # upper form for solveh_banded
    ab[2, :] = main
    ab[1, 1:] = off1
    ab[0, 2:] = off2
    return solveh_banded(ab, y, lower=False)


def hp_trend_twosided(close_log: pd.Series, lam: float = 129600.0) -> pd.Series:
    """The classic (two-sided) HP trend of a log-price — **uses future data** (the look-ahead trap)."""
    tau = _hp_trend_array(close_log.to_numpy(dtype=float), lam)
    return pd.Series(tau, index=close_log.index, name="trend")


def hp_trend_onesided(close_log: pd.Series, lam: float = 129600.0, window: int = 252) -> pd.Series:
    """The causal (one-sided) HP trend: at each ``t``, the endpoint of an HP filter on a trailing window.

    Uses only ``y[t-window : t+1]`` — no future data — so the cycle built from it is tradable. Returns
    NaN for the warm-up (the first ``window`` points), where no full trailing window exists yet.
    """
    y = close_log.to_numpy(dtype=float)
    n = y.size
    tau = np.full(n, np.nan)
    for t in range(window, n):
        seg = y[t - window: t + 1]
        tau[t] = _hp_trend_array(seg, lam)[-1]                      # endpoint only -> causal
    return pd.Series(tau, index=close_log.index, name="trend_causal")


def cycle(close: pd.Series, lam: float = 129600.0, causal: bool = False, window: int = 252) -> pd.Series:
    """The HP cycle ``log(price) - trend`` — two-sided (``causal=False``) or one-sided (``causal=True``).

    The cycle is the deviation of (log) price from its smooth trend; a mean-reversion strategy trades
    against it (long when the cycle is negative, i.e. price below trend). The ``causal`` flag is the
    whole experiment: ``False`` peeks at the future, ``True`` does not.
    """
    yl = np.log(close.astype(float))
    tau = hp_trend_onesided(yl, lam, window) if causal else hp_trend_twosided(yl, lam)
    return (yl - tau).rename("cycle")
