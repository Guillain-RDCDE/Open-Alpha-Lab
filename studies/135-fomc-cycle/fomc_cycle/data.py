"""Data layer for Study 135 (FOMC-Cycle).

Two tapes, one logical shape (a daily return Series with an FOMC-cycle week label):

- ``synthetic_cycle`` — a *deterministic, offline* generator. A ``even_premium`` knob
  injects an extra daily return on every even-week (weeks 0, 2, 4) trading day of each
  6-week FOMC cycle. Set ``even_premium=0`` for the null: pure random walk, so the
  even-vs-odd test scores exactly zero. This is the study's null in a bottle.
- ``fetch_panel`` — loads SPY daily total returns from the shared repo cache plus the
  hardcoded FOMC meeting date schedule (1994–2026). Cache-only by default (``fetch=False``
  raises if no cache file exists); ``fetch=True`` hits Yahoo Finance and repopulates.

No look-ahead is baked in here — each daily return is assigned its FOMC-cycle week using
only the *prior* FOMC meeting date, not the next one.

FOMC cycle definition (Cieslak, Morse & Vuolteenaho 2019):
  - Week 0: the 5 trading days *including* and *following* the FOMC statement day.
  - Week 1: the next 5 trading days (days 6-10).
  - Weeks 2-5: successive 5-day blocks (days 11-15, 16-20, 21-25, 26-30).
  - A meeting ~every 45 calendar days / ~6 weeks means week 5 may be truncated.
  Even weeks (0, 2, 4) are the CMV signal period; odd weeks (1, 3, 5) the control.

Reference: Cieslak, Morse & Vuolteenaho (2019), "Stock Returns over the FOMC Cycle",
Journal of Finance 74(5).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")

TRADING_DAYS = 252
CYCLE_DAYS = 5   # 5 trading days per FOMC-cycle week

# ---------------------------------------------------------------------------
# FOMC meeting statement-release dates, 1994-2026 (scheduled meetings only).
# Source: Federal Reserve FOMC historical calendars. Week 0 of the cycle begins
# on the statement day itself (the afternoon announcement).  We reuse the same
# hardcoded table from Study 67 — its FOMC_DATES cover the same schedule.
# ---------------------------------------------------------------------------
FOMC_DATES = pd.to_datetime([
    # 1994
    "1994-02-04", "1994-03-22", "1994-04-18", "1994-05-17",
    "1994-07-06", "1994-08-16", "1994-09-27", "1994-11-15", "1994-12-20",
    # 1995
    "1995-02-01", "1995-03-28", "1995-05-23", "1995-07-06",
    "1995-08-22", "1995-09-26", "1995-11-15", "1995-12-19",
    # 1996
    "1996-01-31", "1996-03-26", "1996-05-21", "1996-07-03",
    "1996-08-20", "1996-09-24", "1996-11-13", "1996-12-17",
    # 1997
    "1997-02-05", "1997-03-25", "1997-05-20", "1997-07-02",
    "1997-08-19", "1997-09-30", "1997-11-12", "1997-12-16",
    # 1998
    "1998-02-04", "1998-03-31", "1998-05-19", "1998-07-01",
    "1998-08-18", "1998-09-29", "1998-10-15", "1998-11-17", "1998-12-22",
    # 1999
    "1999-02-03", "1999-03-30", "1999-05-18", "1999-06-30",
    "1999-08-24", "1999-10-05", "1999-11-16", "1999-12-21",
    # 2000
    "2000-02-02", "2000-03-21", "2000-05-16", "2000-06-28",
    "2000-08-22", "2000-10-03", "2000-11-15", "2000-12-19",
    # 2001
    "2001-01-31", "2001-03-20", "2001-04-18", "2001-05-15",
    "2001-06-27", "2001-08-21", "2001-09-17", "2001-10-02",
    "2001-11-06", "2001-12-11",
    # 2002
    "2002-01-30", "2002-03-19", "2002-05-07", "2002-06-26",
    "2002-08-13", "2002-09-24", "2002-11-06", "2002-12-10",
    # 2003
    "2003-01-29", "2003-03-18", "2003-05-06", "2003-06-25",
    "2003-08-12", "2003-09-16", "2003-10-28", "2003-12-09",
    # 2004
    "2004-01-28", "2004-03-16", "2004-05-04", "2004-06-30",
    "2004-08-10", "2004-09-21", "2004-11-10", "2004-12-14",
    # 2005
    "2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30",
    "2005-08-09", "2005-09-20", "2005-11-01", "2005-12-13",
    # 2006
    "2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29",
    "2006-08-08", "2006-09-20", "2006-10-25", "2006-12-12",
    # 2007
    "2007-01-31", "2007-03-21", "2007-05-09", "2007-06-28",
    "2007-08-07", "2007-09-18", "2007-10-31", "2007-12-11",
    # 2008
    "2008-01-30", "2008-03-18", "2008-04-30", "2008-06-25",
    "2008-08-05", "2008-09-16", "2008-10-29", "2008-12-16",
    # 2009
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24",
    "2009-08-12", "2009-09-23", "2009-11-04", "2009-12-16",
    # 2010
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23",
    "2010-08-10", "2010-09-21", "2010-11-03", "2010-12-14",
    # 2011
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22",
    "2011-08-09", "2011-09-21", "2011-11-02", "2011-12-13",
    # 2012
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20",
    "2012-08-01", "2012-09-13", "2012-10-24", "2012-12-12",
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19",
    "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18",
    "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17",
    "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-18", "2020-04-29", "2020-06-10",
    "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (through our as-of date)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
])


# ---------------------------------------------------------------------------
# FOMC-cycle week assignment
# ---------------------------------------------------------------------------
def assign_cycle_week(index: pd.DatetimeIndex,
                      fomc_dates: pd.DatetimeIndex | None = None) -> pd.Series:
    """Assign each trading day in ``index`` its FOMC-cycle week (0–5).

    Delegates to :func:`assign_cycle_week_fast` — kept as the public name for
    backward compatibility with tests and notebooks.
    """
    return assign_cycle_week_fast(index, fomc_dates)


def assign_cycle_week_fast(index: pd.DatetimeIndex,
                           fomc_dates: pd.DatetimeIndex | None = None) -> pd.Series:
    """Vectorised FOMC-cycle week assignment.

    For each trading day in ``index``, finds the most recent FOMC date on or
    before it, then counts the number of *trading days* elapsed since that FOMC
    day (0-indexed, where 0 = the statement day itself) and divides by 5 to get
    the cycle week (capped at 5 to handle the last block before the next meeting).
    """
    if fomc_dates is None:
        fomc_dates = FOMC_DATES

    idx = pd.DatetimeIndex(index).normalize()
    fomc_sorted = pd.DatetimeIndex(np.sort(pd.DatetimeIndex(fomc_dates).normalize()))

    # Convert to plain numpy datetime64[D] for reliable comparison
    idx_d = np.array(idx.strftime("%Y-%m-%d"), dtype="datetime64[D]")
    fomc_d = np.array(fomc_sorted.strftime("%Y-%m-%d"), dtype="datetime64[D]")

    # For each trading day, find the index of the most recent FOMC date.
    pos = np.searchsorted(fomc_d, idx_d, side="right") - 1

    week_labels = np.full(len(idx), np.nan)
    for i in range(len(idx)):
        if pos[i] < 0:
            continue
        last_fomc = fomc_d[pos[i]]
        # Count trading days from last_fomc up to and including idx[i].
        mask = (idx_d[:i + 1] >= last_fomc)
        day_offset = int(mask.sum()) - 1  # 0 = the FOMC day itself
        week_labels[i] = min(day_offset // CYCLE_DAYS, 5)

    return pd.Series(week_labels, index=pd.DatetimeIndex(index), name="cycle_week")


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_cycle(
    n_days: int = 2000,
    even_premium: float = 0.0,
    base_ret: float = 4e-4,
    daily_vol: float = 0.011,
    seed: int = 135,
    fomc_dates: pd.DatetimeIndex | None = None,
    start: str = "1994-02-01",
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily-return series with a tunable FOMC even-week premium.

    Log returns are i.i.d. normal with mean ``base_ret`` and std ``daily_vol``,
    except on even FOMC-cycle-week (weeks 0, 2, 4) days where the mean is lifted
    by ``even_premium``. Setting ``even_premium=0`` gives a pure random walk —
    the null — so a test can assert the even-vs-odd gap appears only when we plant
    it. Returns ``(frame, truth)`` where ``frame`` has columns ``ret`` and
    ``cycle_week``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days, name="date")
    if fomc_dates is None:
        fomc_dates = FOMC_DATES

    weeks = assign_cycle_week_fast(idx, fomc_dates)
    even_mask = weeks.isin([0, 2, 4])

    rets = base_ret + daily_vol * rng.standard_normal(n_days)
    rets[even_mask.values] += even_premium

    frame = pd.DataFrame({"ret": rets, "cycle_week": weeks.values}, index=idx)
    truth = {
        "even_premium": even_premium,
        "base_ret": base_ret,
        "daily_vol": daily_vol,
        "n_days": n_days,
        "seed": seed,
        "n_even": int(even_mask.sum()),
        "n_odd": int((~even_mask & weeks.notna()).sum()),
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — SPY daily total returns from the shared repo cache
# ---------------------------------------------------------------------------
def _spy_cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "last_call_spy.parquet")


def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "1994-01-01",
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """SPY daily total returns with FOMC-cycle week labels; cache-first.

    Returns ``(frame, fomc_dates)`` where ``frame`` has columns ``ret`` and
    ``cycle_week``, index = trading date. Only trading days from ``start``
    onward are returned (the study's sample begins with the modern FOMC era,
    1994). Network is touched only on an explicit ``fetch=True``.
    """
    cache = _spy_cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(cache):
            raise FileNotFoundError(
                f"No cached SPY data at {cache}. "
                "Call fetch_panel(fetch=True) once to populate, or run "
                "examples/verify.py --fetch."
            )
        raw = pd.read_parquet(cache)
        ret = raw["ret"] if "ret" in raw.columns else raw.iloc[:, 0]
    else:
        import yfinance as yf

        px = yf.download("SPY", period="max", auto_adjust=True, progress=False)
        if px.empty:
            raise RuntimeError("yfinance returned no data for SPY")
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.get_level_values(0)
        close_col = next(c for c in px.columns if str(c).lower() in ("close", "adj close"))
        ret = px[close_col].pct_change().dropna()
        ret.name = "ret"
        os.makedirs(cache_dir, exist_ok=True)
        ret.to_frame().to_parquet(cache)

    ret.index = pd.DatetimeIndex(ret.index).tz_localize(None)
    ret = ret[ret.index >= start].dropna()
    ret.index.name = "date"

    weeks = assign_cycle_week_fast(ret.index, FOMC_DATES)
    frame = pd.DataFrame({"ret": ret.values, "cycle_week": weeks.values}, index=ret.index)
    return frame, FOMC_DATES


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of the return column, for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(frame["ret"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
