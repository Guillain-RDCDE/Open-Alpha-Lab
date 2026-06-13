"""Data layer for Study 90 (Weekend / day-of-week effect).

Two tapes, one shape (a daily frame with a ``close`` column, indexed by date so the
weekday of every bar is recoverable):

- ``synthetic_daily`` — a *deterministic, offline* generator. A seasonality knob
  (``dow_effect``) plants the only thing a day-of-week harness can possibly harvest: a
  per-weekday bump on the daily mean return. ``dow_effect=0`` is an i.i.d. constant-drift
  tape — there is no weekday structure, so the harness must *not* find one. ``dow_effect>0``
  pushes Mondays down and Tuesdays up by a known amount, so the contrast tests must
  light up. This is the study's null (and its positive control) in a bottle.
- ``load_real`` — the real daily tape via :mod:`quantlab.data` (Yahoo, cached parquet),
  total-return adjusted so dividends are folded in and the mean-return-by-weekday read is
  on a fair, dividend-inclusive series. Cache-first; the network is touched only on a
  cache miss.

No look-ahead is baked in here — the weekday of a bar is *calendar-known*, so the
day-of-week rules in ``strategy.py`` need no execution lag at all (the rule "be long on
Tuesday" is decidable the night before).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Monday=0 ... Friday=4 (pandas ``DatetimeIndex.dayofweek`` convention).
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 6000,
    dow_effect: float = 0.0,
    daily_vol: float = 0.010,
    drift: float = 0.08,
    start: str = "1990-01-02",
    seed: int = 90,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily close series with a tunable amount of day-of-week seasonality.

    - ``dow_effect = 0`` → an i.i.d. constant-drift tape. There is no weekday structure to
      detect, so a per-weekday mean test must *not* reject and a "long-only-on-Tuesday"
      timer must not out-earn buy-and-hold on a risk-adjusted basis. This is the study's null.
    - ``dow_effect > 0`` → the same tape with a planted weekday bump added to each day's
      return: Monday gets ``-dow_effect`` and Tuesday ``+dow_effect`` (other days unchanged),
      so the classic "weekend effect / turnaround Tuesday" pattern is present by construction.
      The positive control that proves the harness can bank a real seasonality when one exists.

    The bump is applied on a **business-day calendar** (Mon-Fri), so weekday counts match the
    real tape. Returns ``(frame, truth)`` where ``frame`` has a single ``close`` column and
    ``truth`` records the planted parameters (including the realised per-weekday mean bump).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    dow = idx.dayofweek.to_numpy()

    base = rng.normal(drift / 252.0, daily_vol, n_days)
    bump = np.zeros(n_days)
    if dow_effect != 0:
        bump[dow == 0] = -dow_effect  # Monday down
        bump[dow == 1] = +dow_effect  # Tuesday up
    rets = base + bump

    close = 100.0 * np.exp(np.cumsum(rets))
    frame = pd.DataFrame({"close": close}, index=idx)
    frame.index.name = "Date"

    # Realised planted bump per weekday (so tests can assert on what was actually added).
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
    ``_cache/``). ``mode="total_return"`` folds dividends back in so the per-weekday mean
    read is on a fair, dividend-inclusive series. Returns a frame with a single lower-case
    ``close`` column, indexed by date (so each bar's weekday is recoverable).
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
