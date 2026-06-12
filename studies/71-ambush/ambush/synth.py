"""Synthetic controls — the machinery proof, never market evidence (bench rule).

:func:`synthetic_tape` builds an OHLC tape + VIX series where each of the four
ambush signals armed at the close of *t* adds a known ``plant_bps_per_signal`` to the
*mean* of day *t+1*'s return — a planted, monotone confluence premium. The positive
control demands the harness light up on it (monotone lift table, HAC *t* ≥ 2 on the
armed stream, positive net book); the null (``plant_bps_per_signal=0``) demands it
stay dark. A pipeline that can't bank a planted signal proves nothing by finding
nothing — and quoting a synthetic Sharpe in support of the Signal stamp is circular,
full stop.

The tape is deterministic per seed: bars are built day by day (gap, high, low around
the close-to-close move), the VIX is an AR(1) in logs with occasional spikes,
independent of price; signal definitions match ``ambush.signals`` exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import signals


def synthetic_tape(
    n_days: int = 5000,
    plant_bps_per_signal: float = 8.0,
    daily_vol: float = 0.01,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(spy_like_ohlc, vix_series)`` with the planted confluence premium."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n_days)

    # VIX: AR(1) around log(20), with ~2%/day spike shocks — independent of price.
    lv = np.empty(n_days)
    lv[0] = np.log(20.0)
    for t in range(1, n_days):
        shock = 0.35 if rng.random() < 0.02 else 0.0
        lv[t] = 0.97 * lv[t - 1] + 0.03 * np.log(20.0) + rng.normal(0.0, 0.06) + shock
    vix = pd.Series(np.exp(lv), index=idx, name="vix")

    # Calendar / VIX legs are known before the price loop.
    s_tom = signals.tom_tomorrow(idx).to_numpy()
    s_vix = signals.vix_stress(vix).to_numpy()

    o = np.empty(n_days)
    h = np.empty(n_days)
    lo = np.empty(n_days)
    c = np.empty(n_days)
    prev_close, count_prev = 100.0, 0
    for t in range(n_days):
        mu = plant_bps_per_signal * 1e-4 * count_prev
        r = rng.normal(mu, daily_vol)
        o[t] = prev_close * (1.0 + rng.normal(0.3 * r, 0.3 * daily_vol))
        c[t] = prev_close * (1.0 + r)
        h[t] = max(o[t], c[t]) * (1.0 + abs(rng.normal(0.0, 0.4 * daily_vol)))
        lo[t] = min(o[t], c[t]) * (1.0 - abs(rng.normal(0.0, 0.4 * daily_vol)))

        rng_t = h[t] - lo[t]
        ibs_t = (c[t] - lo[t]) / rng_t if rng_t > 0 else 0.5
        s_ibs = ibs_t <= signals.IBS_LOW
        s_red = c[t] < prev_close
        count_prev = int(s_ibs) + int(s_red) + int(s_tom[t]) + int(s_vix[t])
        prev_close = c[t]

    spy = pd.DataFrame({"Open": o, "High": h, "Low": lo, "Close": c}, index=idx)
    return spy, vix


def flat_rf(index: pd.DatetimeIndex, ann_rate: float = 0.0) -> pd.Series:
    """A constant per-day cash rate for the offline controls."""
    return pd.Series(ann_rate / 252.0, index=index, name="rf")
