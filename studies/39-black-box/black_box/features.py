"""Causal feature matrix — strictly lagged price-derived predictors, no look-ahead.

The neural net is fed a small bank of features computed from **past** returns only, all shifted so that
the row aligned to day ``t`` uses information available at the *close of* ``t-1`` (or earlier). The target
is the *sign of day-t's* return — i.e. what a trader standing at the close of ``t-1`` would try to
predict. Keeping the feature/target alignment honest is the single most important guard against the
look-ahead that fabricates spurious backtests (cf. Study 22 Crystal-Ball).

Features (all from a single close series, all lagged):
  * ``ret_1`` … ``ret_k`` — the last ``k`` daily log-returns (the most recent is ``ret_1 = r_{t-1}``).
  * ``mom_5``, ``mom_10`` — trailing 5- and 10-day momentum (sum of past returns).
  * ``vol_10`` — trailing 10-day realised volatility.
  * ``zscore_10`` — last return standardised by trailing-10-day mean/std (a mean-reversion cue).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(close: pd.Series, n_lags: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` for one close series: a causal lagged-feature matrix ``X`` and the binary target
    ``y`` (1 if day-t's log-return is positive, else 0), aligned and de-NaN'd.

    Every column of ``X`` is shifted by one day, so the row for day ``t`` contains only quantities known
    at the close of ``t-1``. The target ``y_t = 1[r_t > 0]`` is the very next return the predictor has not
    yet seen — exactly the tradable forecasting problem.
    """
    close = pd.Series(close).astype(float).dropna()
    r = np.log(close).diff()  # daily log-returns

    feat = {}
    for k in range(1, n_lags + 1):
        feat[f"ret_{k}"] = r.shift(k)               # r_{t-k}, all strictly in the past
    feat["mom_5"] = r.rolling(5).sum().shift(1)
    feat["mom_10"] = r.rolling(10).sum().shift(1)
    feat["vol_10"] = r.rolling(10).std().shift(1)
    roll_mean = r.rolling(10).mean().shift(1)
    roll_std = r.rolling(10).std().shift(1)
    feat["zscore_10"] = (r.shift(1) - roll_mean) / roll_std.replace(0.0, np.nan)

    X = pd.DataFrame(feat, index=close.index)
    y = (r > 0).astype(int).rename("up")            # sign of the CURRENT (next, from predictor's view) day

    data = pd.concat([X, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    return data[X.columns], data["up"]
