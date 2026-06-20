"""Data for the Jackson-Hole study — the symposium calendar, an offline synthetic
world, and the cached SPY daily total returns.

The **Jackson Hole Economic Symposium** is the Kansas City Fed's annual conference,
held in late August in Grand Teton National Park since 1978. Its market-relevant
moment is the **keynote speech by the Fed Chair**, traditionally delivered on the
**Friday morning** of the symposium (the conference runs Thu–Sat). Several of these
speeches moved markets — Bernanke's 2010 "QE2" hint, Powell's hawkish 2022 eight
minutes — which is why a "Jackson Hole drift / Jackson Hole effect" rumour circulates.

``JH_SPEECH_DATES`` are the **Friday keynote dates** (the day the Chair speaks), our
event anchor. This is an *annual, single-event* calendar: one observation per year,
~31 events across the SPY era. Sources: Kansas City Fed symposium archive
(kansascityfed.org/research/jackson-hole-economic-symposium/) and contemporaneous
press. A keynote that fell on a non-trading day would snap to the prior session in
``strategy.event_window``; in practice every keynote here is a Friday.

This is decisively distinct from:
  * Study 67 (Fed-Drift) — the *pre-FOMC* announcement drift (8 scheduled FOMC
    statement days/yr); and
  * Study 135 (FOMC-Cycle) — the even-week FOMC inter-meeting cycle.
Jackson Hole is **not an FOMC meeting** and carries no policy decision; it is a
speech-driven, once-a-year media event.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
SHARED_SPY_CACHE = os.path.join(DEFAULT_CACHE, "last_call_spy.parquet")
TRADING_DAYS = 252

# Friday keynote dates of the Jackson Hole symposium (the Fed Chair speaks). One per
# year. Covers the SPY era (SPY listed 1993-01-29). The symposium runs Thu–Sat; the
# Chair's address is the Friday morning headline event.
JH_SPEECH_DATES = pd.to_datetime([
    "1993-08-20", "1994-08-19", "1995-08-25", "1996-08-30", "1997-08-29",
    "1998-08-28", "1999-08-27", "2000-08-25", "2001-08-31", "2002-08-30",
    "2003-08-29", "2004-08-27", "2005-08-26", "2006-08-25", "2007-08-31",
    "2008-08-22", "2009-08-21", "2010-08-27", "2011-08-26", "2012-08-31",
    "2013-08-23", "2014-08-22", "2015-08-28", "2016-08-26", "2017-08-25",
    "2018-08-24", "2019-08-23", "2020-08-27", "2021-08-27", "2022-08-26",
    "2023-08-25", "2024-08-23", "2025-08-22",
])
# Note 2020: the symposium was virtual (COVID); Powell spoke Thu Aug 27. 2021 was also
# virtual. We keep the Chair's speech date as the anchor in both years.


@dataclass(frozen=True)
class WorldTruth:
    """The planted truth of a synthetic tape: the abnormal log-return added on the
    Jackson-Hole speech day. ``bump == 0`` is the null (a pure i.i.d. tape)."""

    bump: float

    @property
    def has_effect(self) -> bool:
        return self.bump != 0.0


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (the positive control + the null)
# ---------------------------------------------------------------------------
def synthetic_world(
    n_years: int = 33,
    bump: float = 0.0,
    seed: int = 314,
    base: float = 0.0003,
    vol: float = 0.011,
) -> tuple[pd.Series, pd.DatetimeIndex, WorldTruth]:
    """A daily-return world with one Jackson-Hole 'speech day' per simulated year.

    Log returns are i.i.d. normal (mean ``base``, sd ``vol``); on each planted speech
    day the return is lifted by ``bump``. ``bump = 0`` is the null (the event day is an
    ordinary day); ``bump > 0`` plants a positive Jackson-Hole drift the engine must
    detect. The calendar is built so each speech day is the last business day of the
    third full week of August — close enough to the real anchor for the harness.

    Returns ``(ret, speech_dates, truth)``. The span stays well under 290 years of
    ns-timestamps (we build a real ``bdate_range``, not a giant ``date_range``).
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("1990-01-01")
    # ~252 trading days/yr → enough business days to cover n_years of late-Augusts.
    n_days = int((n_years + 1) * TRADING_DAYS) + 40
    idx = pd.bdate_range(start, periods=n_days, name="date")
    ret = base + vol * rng.standard_normal(n_days)

    speech_dates = []
    for yr in range(start.year, start.year + n_years):
        # Friday of the last full week of August (a fixed, look-ahead-free rule).
        aug = pd.Timestamp(year=yr, month=8, day=25)
        # snap to the nearest Friday on/just before the 25th
        while aug.weekday() != 4:  # Friday == 4
            aug -= pd.Timedelta(days=1)
        speech_dates.append(aug)
    speech = pd.DatetimeIndex(speech_dates)

    # Apply the bump on the trading session at/just before each speech date.
    pos = idx.searchsorted(speech)
    for p in pos:
        if 0 <= p < n_days:
            ret[p] += bump
    # keep only speech dates that land inside the generated window
    speech = speech[(speech >= idx[0]) & (speech <= idx[-1])]
    return pd.Series(ret, index=idx, name="ret"), speech, WorldTruth(bump)


# ---------------------------------------------------------------------------
# Real tape — cache-first SPY daily total returns (shared parquet)
# ---------------------------------------------------------------------------
def fetch_spy(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.Series:
    """Daily SPY total returns, cache-first from the shared ``last_call_spy`` pull.

    Returns a tz-naive daily-return Series. Network is touched only on an explicit
    ``fetch=True`` AND a cache miss — so the reproducible core and the test-suite never
    go online. On a cache miss with ``fetch=False`` an empty Series is returned, and the
    caller falls back to the synthetic tape.
    """
    cache = os.path.join(cache_dir, "last_call_spy.parquet")
    if os.path.exists(cache):
        ret = pd.read_parquet(cache)["ret"]
        ret.index = pd.DatetimeIndex(ret.index).tz_localize(None)
        ret.name = "ret"
        return ret
    if not fetch:
        return pd.Series(dtype=float, name="ret")
    import yfinance as yf  # lazy: only on an explicit online refresh

    px = yf.download("SPY", period="max", auto_adjust=True, progress=False)["Close"]
    ret = px.pct_change().dropna()
    ret.name = "ret"
    ret.index = pd.DatetimeIndex(ret.index).tz_localize(None)
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_frame().to_parquet(cache)
    return ret


def fingerprint(ret: pd.Series) -> str:
    """A short content fingerprint of a return series, for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(ret.to_numpy()).tobytes())
    return h.hexdigest()[:12]
