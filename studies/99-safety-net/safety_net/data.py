"""Data layer for Study 99 (Safety-Net — the trailing stop-loss).

Three tapes, one shape (a daily frame with a ``close`` column):

- ``synthetic_trending`` — a *deterministic, offline* tape with long, sustained
  bull/bear regimes (a sticky two-state chain). When the market falls, it keeps
  falling for months, so a trailing stop that exits on a drawdown-from-peak genuinely
  cuts the sustained downtrend. This is the tape where stops are *supposed* to help
  (Kaminski & Lo 2014: stops help trending/momentum assets). It is the harness's
  positive control.
- ``synthetic_meanrev`` — a *deterministic, offline* tape that is mean-reverting
  (an Ornstein-Uhlenbeck-style pull back to a slowly-rising fair value). Dips snap
  back, so a trailing stop sells the bottom and misses the rebound — it *whipsaws*
  and hurts. This is the tape where stops are supposed to hurt (Kaminski-Lo's
  mean-reverting case). It is the harness's negative control.
- ``load_real`` — the real daily tape via :mod:`quantlab.data` (Yahoo, cached
  parquet), total-return adjusted so the buy-and-hold benchmark is a fair,
  dividend-inclusive comparison. Cache-first; the network is touched only on a miss.

No look-ahead is baked in here — the one-day execution lag lives in ``strategy.py``
(the stop is triggered by the drawdown known at close *t*, the exit happens at *t+1*).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# Synthetic tape #1 — TRENDING (sticky regimes) — stops should HELP
# ---------------------------------------------------------------------------
def synthetic_trending(
    n_days: int = 6000,
    down_scale: float = 1.0,
    daily_vol: float = 0.011,
    start: str = "2000-01-03",
    seed: int = 99,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily close with long, sustained bull/bear regimes.

    A sticky two-state Markov chain alternates a calm up-drift state with a
    persistent down-drift state. Because declines *persist* (momentum), a trailing
    stop that exits when price has fallen far from its peak escapes most of the
    sustained downtrend and re-enters after the recovery — so it should beat
    buy-and-hold on a risk-adjusted basis. ``down_scale`` scales the bear drift.

    Returns ``(frame, truth)`` where ``frame`` has a single ``close`` column and
    ``truth`` records the planted parameters (incl. the bear-day fraction).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    mu_bull = 0.16 / 252.0  # ~16%/yr calm up-drift
    mu_bear = -down_scale * 0.55 / 252.0  # persistent down-drift
    vol_bull, vol_bear = daily_vol, daily_vol * 1.5
    # Long regimes: bull ~500-day mean run, bear ~120-day mean run -> sustained
    # declines a trailing stop can actually escape.
    stay_bull, stay_bear = 0.998, 0.992
    state = np.empty(n_days, dtype=int)
    s = 0
    rets = np.empty(n_days)
    for t in range(n_days):
        stay = stay_bull if s == 0 else stay_bear
        if rng.random() > stay:
            s = 1 - s
        state[t] = s
        if s == 0:
            rets[t] = rng.normal(mu_bull, vol_bull)
        else:
            rets[t] = rng.normal(mu_bear, vol_bear)

    close = 100.0 * np.exp(np.cumsum(rets))
    frame = pd.DataFrame({"close": close}, index=idx)
    frame.index.name = "Date"
    truth = {
        "tape": "trending",
        "n_days": n_days,
        "down_scale": down_scale,
        "daily_vol": daily_vol,
        "seed": seed,
        "frac_bear": float(state.mean()),
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Synthetic tape #2 — MEAN-REVERTING — stops should HURT (whipsaw)
# ---------------------------------------------------------------------------
def synthetic_meanrev(
    n_days: int = 6000,
    kappa: float = 0.04,
    sigma: float = 0.018,
    drift: float = 0.08,
    start: str = "2000-01-03",
    seed: int = 99,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily close that mean-reverts around a slowly-rising fair value.

    Log-price follows an Ornstein-Uhlenbeck process pulled back (strength ``kappa``)
    toward a fair value that drifts up at ``drift``/yr. Dips snap *back* rather than
    persisting, so a trailing stop sells near local bottoms and re-enters after the
    rebound is already underway — it whipsaws and should *underperform* buy-and-hold.
    This is Kaminski-Lo's mean-reverting case where stops stop nothing useful.

    Returns ``(frame, truth)``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    mu = drift / 252.0  # fair-value drift per day
    logfv = mu * np.arange(n_days)  # rising fair value in log space
    x = np.empty(n_days)  # deviation of log-price from fair value
    x[0] = 0.0
    for t in range(1, n_days):
        x[t] = (1.0 - kappa) * x[t - 1] + rng.normal(0.0, sigma)
    logprice = logfv + x
    close = 100.0 * np.exp(logprice)
    frame = pd.DataFrame({"close": close}, index=idx)
    frame.index.name = "Date"
    truth = {
        "tape": "meanrev",
        "n_days": n_days,
        "kappa": kappa,
        "sigma": sigma,
        "drift": drift,
        "seed": seed,
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — daily total-return close via quantlab, cache-first
# ---------------------------------------------------------------------------
def load_real(
    ticker: str = "SPY",
    start: str = "1993-01-29",
    end: str | None = None,
    mode: str = "total_return",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Real daily close for ``ticker`` as a one-column ``close`` frame.

    Delegates to :func:`quantlab.data.fetch` (cache-first parquet under the repo
    ``_cache/``). ``mode="total_return"`` folds dividends back in so buy-and-hold is a
    fair benchmark for a strategy that sits in cash part of the time. Returns a frame
    with a single lower-case ``close`` column, indexed by date.
    """
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    from quantlab import data as qdata

    raw = qdata.fetch(ticker, start=start, end=end, mode=mode, use_cache=use_cache)
    close = raw["Close"].rename("close")
    return close.to_frame()


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(frame["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
