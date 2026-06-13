"""Data layer for Study 95 (Holiday-Cheer).

Three things live here, all sharing one shape (a daily frame with a ``close`` column):

- ``derive_pre_holiday`` - the study's quiet trick. We never load an external holiday
  calendar. Instead we read the market's *own* trading index: a weekday that is ABSENT
  from the trading dates (and is not a Saturday or Sunday) can only be a market holiday -
  the exchange was shut. The immediately preceding trading day is then a **pre-holiday**
  day. This needs no third-party calendar, can't drift out of date, and is exactly the
  set of days the folklore is about. (It does also catch the rare unscheduled close -
  9/11, hurricanes - which is fine: those *are* days the market was closed.)

- ``load_real`` - the real daily tape via :mod:`quantlab.data` (Yahoo, cached parquet).
  Two series are offered as *decisions, not details*: ``SPY`` total-return (dividends
  folded in, the fair tape for a buy-the-day-before rule, but only from 1993) and
  ``^GSPC`` **price-only** (no dividends - state it - but reaching back to 1950, the
  long sample the pre/post-1990 decay story needs).

- ``synthetic_daily`` - a deterministic, offline tape that *engineers* its own market
  holidays (drops the weekday occurrences of a fixed holiday list) and optionally PLANTS
  a pre-holiday return bump on exactly the days the gap-derivation will flag. With
  ``planted=True`` the harness must detect the bump; with ``planted=False`` (a flat iid
  tape) it must not. This is the study's positive/negative control in a bottle.

No look-ahead is baked in: a pre-holiday day is known from the *calendar* in advance, so
the rule needs no execution lag at all (see ``strategy.py`` - it documents this).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# A fixed set of US holidays whose *weekday* occurrences a real exchange closes for.
# Used ONLY by the synthetic generator to carve realistic gaps into an offline tape;
# the real-tape path never consults this - it reads the gaps straight from the index.
_SYNTH_HOLIDAY_MD = [
    (1, 1),    # New Year's Day
    (5, 28),   # (a Memorial-Day-ish late-May date)
    (7, 4),    # Independence Day
    (9, 3),    # (a Labor-Day-ish early-September date)
    (11, 28),  # (a Thanksgiving-ish late-November date)
    (12, 25),  # Christmas
    (12, 31),  # year-end half-session proxy
]


# ---------------------------------------------------------------------------
# The holiday-from-gaps derivation - the heart of the study's data layer
# ---------------------------------------------------------------------------
def derive_market_holidays(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Weekdays absent from the trading index ``idx`` - i.e. market holidays.

    A business day (Mon-Fri) in the span of the tape that has no row in ``idx`` is a day
    the exchange did not trade: a holiday (or, rarely, an unscheduled close). Weekends are
    never holidays by this definition because ``bdate_range`` already excludes them.
    """
    trading = pd.DatetimeIndex(idx.normalize().unique()).sort_values()
    full = pd.bdate_range(trading[0], trading[-1])
    return full.difference(trading)


def derive_pre_holiday(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The trading days immediately *before* each market holiday (the pre-holiday set).

    For every holiday found by :func:`derive_market_holidays`, the pre-holiday day is the
    last trading day on or before it. Returned as a sorted, de-duplicated DatetimeIndex of
    normalised dates.
    """
    trading = pd.DatetimeIndex(idx.normalize().unique()).sort_values()
    holidays = derive_market_holidays(idx)
    if len(holidays) == 0:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    pos = trading.searchsorted(holidays) - 1
    pos = pos[pos >= 0]
    return pd.DatetimeIndex(trading[pos].unique()).sort_values()


def pre_holiday_mask(idx: pd.DatetimeIndex) -> pd.Series:
    """Boolean Series aligned to ``idx``: True on pre-holiday days, else False."""
    pre = derive_pre_holiday(idx)
    return pd.Series(idx.normalize().isin(pre), index=idx)


# ---------------------------------------------------------------------------
# Synthetic tape - the deterministic offline control
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 30,
    bump: float = 0.004,
    planted: bool = True,
    daily_vol: float = 0.01,
    mu: float = 0.0003,
    start: str = "1980-01-01",
    seed: int = 95,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily close series with engineered holidays and an optional bump.

    The generator builds a business-day index, then *removes* the weekday occurrences of
    a fixed holiday list (:data:`_SYNTH_HOLIDAY_MD`) - so the resulting tape has the same
    kind of gaps a real exchange tape has, and :func:`derive_pre_holiday` will recover the
    very days adjacent to those gaps.

    - ``planted=True``  -> a return ``bump`` is added on every derived pre-holiday day.
      The harness must detect this (positive control).
    - ``planted=False`` -> a flat i.i.d. tape with no special pre-holiday behaviour. The
      harness must NOT find a significant pre-holiday effect (negative control).

    Returns ``(frame, truth)`` where ``frame`` has a single ``close`` column and ``truth``
    records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    full = pd.bdate_range(start=start, periods=n_years * 260)

    def is_holiday(d: pd.Timestamp) -> bool:
        return (d.month, d.day) in _SYNTH_HOLIDAY_MD

    trading = pd.DatetimeIndex([d for d in full if not is_holiday(d)])
    pre = derive_pre_holiday(trading)
    is_pre = trading.normalize().isin(pre)

    rets = rng.normal(mu, daily_vol, len(trading))
    if planted:
        rets[is_pre] = rets[is_pre] + bump

    close = 100.0 * np.exp(np.cumsum(rets))
    frame = pd.DataFrame({"close": close}, index=trading)
    frame.index.name = "Date"
    truth = {
        "n_years": n_years,
        "bump": bump if planted else 0.0,
        "planted": planted,
        "n_days": len(trading),
        "n_pre": int(is_pre.sum()),
        "seed": seed,
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape - daily close via quantlab, cache-first
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
    ``_cache/``). Two intended calls:

    - ``load_real("SPY", mode="total_return")`` - dividend-inclusive, fair for a strategy
      that is in cash most of the time, but only back to 1993.
    - ``load_real("^GSPC", start="1950-01-01", mode="split_only")`` - **price-only** (no
      dividends; say so wherever you quote it), back to 1950 for the pre/post-1990 split.

    Returns a frame with a single lower-case ``close`` column, indexed by date.
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
