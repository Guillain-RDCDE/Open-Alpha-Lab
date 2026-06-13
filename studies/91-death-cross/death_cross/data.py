"""Data layer for Study 91 (Death-Cross).

Two tapes, one shape (a daily frame with a ``close`` column):

- ``synthetic_daily`` — a *deterministic, offline* generator. A regime knob
  (``regime``) plants the only thing a trend filter can possibly harvest: persistent
  bull/bear regimes. ``regime=0`` is a constant-drift geometric random walk — there is
  no regime to dodge, so a trend filter cannot add *risk-adjusted* value. ``regime>0``
  switches between a calm up-drift and a violent down-drift, so the death-cross filter
  genuinely sidesteps the drawdown. This is the study's null (and its positive control)
  in a bottle.
- ``load_real`` — the real daily tape via :mod:`quantlab.data` (Yahoo, cached parquet),
  total-return adjusted so the buy-and-hold benchmark is a fair, dividend-inclusive
  comparison. Cache-first; the network is touched only on a cache miss.

No look-ahead is baked in here — the one-day execution lag lives in ``strategy.py``
(the cross is known at close *t*, the position changes at *t+1*).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 6000,
    regime: float = 0.0,
    daily_vol: float = 0.011,
    start: str = "2000-01-03",
    seed: int = 91,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily close series with a tunable amount of regime persistence.

    - ``regime = 0``  → a single constant-drift geometric random walk. There is no bear
      market to dodge, so any trend filter only *removes* time-in-market: it cannot beat
      buy-and-hold on a risk-adjusted basis. This is the study's null.
    - ``regime > 0``  → a two-state Markov chain alternating a calm up-drift state with a
      violent down-drift state; ``regime`` scales the down-drift. A death-cross filter
      that exits in the bear state should now genuinely cut the drawdown — the positive
      control that proves the harness can bank a real timing edge when one exists.

    Returns ``(frame, truth)`` where ``frame`` has a single ``close`` column and ``truth``
    records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    mu_bull = 0.15 / 252.0  # ~15%/yr calm up-drift
    if regime <= 0:
        rets = rng.normal(0.10 / 252.0, daily_vol, n_days)
        state = np.zeros(n_days, dtype=int)
    else:
        mu_bear = -regime * 0.45 / 252.0  # scaled down-drift
        vol_bull, vol_bear = daily_vol, daily_vol * 1.4
        # Asymmetric sticky chain with LONG regimes — a 50/200 cross is a slow filter, so
        # the bear must last many months for it to react. Bull stays 0.998 (~500-day mean
        # run), bear stays 0.991 (~110-day mean run) → ~18% of days bear, a market that
        # rises overall (net ~+4%/yr) but with deep, sustained declines for it to dodge.
        stay_bull, stay_bear = 0.998, 0.991
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
        "regime": regime,
        "n_days": n_days,
        "daily_vol": daily_vol,
        "seed": seed,
        "frac_bear": float(state.mean()),
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
