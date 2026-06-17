"""Data layer for Study 224 (Monday Effect / day-of-week close-to-close returns).

Two tapes, one shape (a daily frame with a close column):

- synthetic_daily -- deterministic offline generator with a tunable dow_effect knob.
  dow_effect=0 is an i.i.d. tape (null); dow_effect>0 pushes Monday returns below
  zero (by -dow_effect) and Tuesday above (by +dow_effect).
- load_real -- real SPY close-to-close total-return via quantlab.data, cache-first
  parquet in the shared _cache/ (same as sibling studies).

Difference from Study 90 (Weekend): that study used overnight (close-to-open) returns to
isolate the weekend gap. This study uses standard close-to-close returns -- the full
intra-day move is included -- to test the classic Monday Effect as measured in the
academic literature (French 1980, who used close-to-close).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def synthetic_daily(
    n_days: int = 6000,
    dow_effect: float = 0.0,
    daily_vol: float = 0.010,
    drift: float = 0.08,
    start: str = "1990-01-02",
    seed: int = 224,
) -> tuple:
    """Reproducible daily close series with tunable day-of-week seasonality.

    dow_effect=0: i.i.d. constant-drift tape (null, no weekday structure).
    dow_effect>0: Monday gets -dow_effect, Tuesday +dow_effect; other days unchanged.
    Returns (frame, truth) where frame has a single close column indexed by business date.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    dow = idx.dayofweek.to_numpy()

    base = rng.normal(drift / 252.0, daily_vol, n_days)
    bump = np.zeros(n_days)
    if dow_effect != 0:
        bump[dow == 0] = -dow_effect
        bump[dow == 1] = +dow_effect
    rets = base + bump

    close = 100.0 * np.exp(np.cumsum(rets))
    frame = pd.DataFrame({"close": close}, index=idx)
    frame.index.name = "Date"

    planted = {WEEKDAYS[d]: float(bump[dow == d].mean()) for d in range(5)}
    truth = {
        "dow_effect": dow_effect,
        "n_days": n_days,
        "daily_vol": daily_vol,
        "drift": drift,
        "seed": seed,
        "planted_bump": planted,
    }
    return frame, truth


def load_real(
    ticker: str = "SPY",
    start: str = "1993-01-29",
    end = None,
    mode: str = "total_return",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Real daily close for ticker as a one-column close frame.

    Delegates to quantlab.data.fetch (cache-first parquet under the shared _cache/).
    Total-return mode folds dividends in.
    """
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    from quantlab import data as qdata

    raw = qdata.fetch(ticker, start=start, end=end, mode=mode, use_cache=use_cache)
    close = raw["Close"].rename("close")
    return close.to_frame()


def fingerprint(frame: pd.DataFrame) -> str:
    """Short content fingerprint of a tape (close column)."""
    h = hashlib.sha1(np.ascontiguousarray(frame["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
