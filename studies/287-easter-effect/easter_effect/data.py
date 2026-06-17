"""Data layer for Study 287 (Easter-Effect).

The claim under test is a calendar-folklore one: a "Good-Friday / Easter seasonal"
in the U.S. stock market. Good Friday is one of the handful of days the NYSE is
*closed* (it has been a market holiday in every year of our sample), so the natural
event study is two-pronged:

- the **pre-holiday session** -- the last trading day before Good Friday (which is
  always Maundy Thursday). This is the textbook *pre-holiday effect* (Lakonishok &
  Smidt 1988; Ariel 1990): equity markets tend to drift up into a long weekend.
- the **post-holiday session** -- the first trading day after Easter Sunday, i.e.
  Easter Monday. Folklore says markets are sluggish coming back from the break.

Three components, all offline for the core:

- ``EASTER_SESSIONS`` -- a hardcoded table mapping each year (1950-2025) to the
  computed Easter Sunday, Good Friday, the pre-holiday session date and the
  post-holiday session date, with the weekday of each session recorded so the
  reader can see the (unavoidable) day-of-week structure. Frozen once from the
  ^GSPC daily session calendar; Easter itself is computed by the anonymous
  Gregorian algorithm in :func:`easter_sunday`, so the table is reproducible.

- ``synthetic_daily`` -- a deterministic, offline daily-return generator with an
  optional planted "pre-holiday premium" knob. ``premium_bps = 0`` is the null (no
  effect), so tests can confirm the machinery is truthful before looking at real
  data.

- ``fetch_gspc_daily`` -- real ^GSPC daily closes (price-only) from the repo-level
  cache (``_cache/^GSPC_split_only.parquet``). Cache-only by default
  (``fetch=False``); network (yfinance) only on explicit ``fetch=True``.

No look-ahead: the pre-holiday event return is the close-to-close return *of* the
pre-holiday session itself; a long-only "buy the pre-holiday session" rule enters at
the prior close and exits at the pre-holiday close -- a one-session hold with an
explicit one-day execution lag baked into the date convention. Price-only (no
dividends); the index level is back-revised by the vendor (named on the Signal axis).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))

GSPC_CACHE_NAME = "^GSPC_split_only.parquet"


# ---------------------------------------------------------------------------
# Easter computation -- the anonymous Gregorian algorithm (Meeus/Jones/Butcher)
# ---------------------------------------------------------------------------
def easter_sunday(year: int) -> _dt.date:
    """Return the Gregorian date of Easter Sunday for ``year``.

    Uses the anonymous Gregorian algorithm (a.k.a. Meeus/Jones/Butcher). This is
    exact for all Gregorian years and is the canonical way to place the moveable
    Easter feast (and hence Good Friday = Easter - 2 days, Easter Monday =
    Easter + 1 day) on the calendar.
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month = (h + el - 7 * m + 114) // 31
    day = ((h + el - 7 * m + 114) % 31) + 1
    return _dt.date(year, month, day)


def good_friday(year: int) -> _dt.date:
    """Good Friday = Easter Sunday minus two days."""
    return easter_sunday(year) - _dt.timedelta(days=2)


def easter_monday(year: int) -> _dt.date:
    """Easter Monday = Easter Sunday plus one day."""
    return easter_sunday(year) + _dt.timedelta(days=1)


# ---------------------------------------------------------------------------
# The hardcoded Easter session table -- the deterministic offline core
# ---------------------------------------------------------------------------
# Columns:
#   year         : calendar year
#   easter       : Easter Sunday (YYYY-MM-DD), computed by easter_sunday()
#   good_friday  : Good Friday = Easter - 2 days (always an NYSE holiday in-sample)
#   pre_date     : last ^GSPC trading session strictly before Good Friday
#                  (always Maundy Thursday -- weekday 3)
#   pre_wd       : weekday of pre_date (0=Mon ... 4=Fri)
#   post_date    : first ^GSPC trading session on or after Easter Monday
#                  (always Easter Monday itself -- weekday 0)
#   post_wd      : weekday of post_date
#
# Good Friday is closed for the NYSE in every one of these 76 years, so there is no
# "Good Friday return" to measure; the event sessions are the sessions that bracket
# the closed day. Frozen once from the ^GSPC daily session calendar (1950-2025).
# 76 observations.

EASTER_SESSIONS: list[dict] = [
    {"year": 1950, "easter": "1950-04-09", "good_friday": "1950-04-07", "pre_date": "1950-04-06", "pre_wd": 3, "post_date": "1950-04-10", "post_wd": 0},
    {"year": 1951, "easter": "1951-03-25", "good_friday": "1951-03-23", "pre_date": "1951-03-22", "pre_wd": 3, "post_date": "1951-03-26", "post_wd": 0},
    {"year": 1952, "easter": "1952-04-13", "good_friday": "1952-04-11", "pre_date": "1952-04-10", "pre_wd": 3, "post_date": "1952-04-14", "post_wd": 0},
    {"year": 1953, "easter": "1953-04-05", "good_friday": "1953-04-03", "pre_date": "1953-04-02", "pre_wd": 3, "post_date": "1953-04-06", "post_wd": 0},
    {"year": 1954, "easter": "1954-04-18", "good_friday": "1954-04-16", "pre_date": "1954-04-15", "pre_wd": 3, "post_date": "1954-04-19", "post_wd": 0},
    {"year": 1955, "easter": "1955-04-10", "good_friday": "1955-04-08", "pre_date": "1955-04-07", "pre_wd": 3, "post_date": "1955-04-11", "post_wd": 0},
    {"year": 1956, "easter": "1956-04-01", "good_friday": "1956-03-30", "pre_date": "1956-03-29", "pre_wd": 3, "post_date": "1956-04-02", "post_wd": 0},
    {"year": 1957, "easter": "1957-04-21", "good_friday": "1957-04-19", "pre_date": "1957-04-18", "pre_wd": 3, "post_date": "1957-04-22", "post_wd": 0},
    {"year": 1958, "easter": "1958-04-06", "good_friday": "1958-04-04", "pre_date": "1958-04-03", "pre_wd": 3, "post_date": "1958-04-07", "post_wd": 0},
    {"year": 1959, "easter": "1959-03-29", "good_friday": "1959-03-27", "pre_date": "1959-03-26", "pre_wd": 3, "post_date": "1959-03-30", "post_wd": 0},
    {"year": 1960, "easter": "1960-04-17", "good_friday": "1960-04-15", "pre_date": "1960-04-14", "pre_wd": 3, "post_date": "1960-04-18", "post_wd": 0},
    {"year": 1961, "easter": "1961-04-02", "good_friday": "1961-03-31", "pre_date": "1961-03-30", "pre_wd": 3, "post_date": "1961-04-03", "post_wd": 0},
    {"year": 1962, "easter": "1962-04-22", "good_friday": "1962-04-20", "pre_date": "1962-04-19", "pre_wd": 3, "post_date": "1962-04-23", "post_wd": 0},
    {"year": 1963, "easter": "1963-04-14", "good_friday": "1963-04-12", "pre_date": "1963-04-11", "pre_wd": 3, "post_date": "1963-04-15", "post_wd": 0},
    {"year": 1964, "easter": "1964-03-29", "good_friday": "1964-03-27", "pre_date": "1964-03-26", "pre_wd": 3, "post_date": "1964-03-30", "post_wd": 0},
    {"year": 1965, "easter": "1965-04-18", "good_friday": "1965-04-16", "pre_date": "1965-04-15", "pre_wd": 3, "post_date": "1965-04-19", "post_wd": 0},
    {"year": 1966, "easter": "1966-04-10", "good_friday": "1966-04-08", "pre_date": "1966-04-07", "pre_wd": 3, "post_date": "1966-04-11", "post_wd": 0},
    {"year": 1967, "easter": "1967-03-26", "good_friday": "1967-03-24", "pre_date": "1967-03-23", "pre_wd": 3, "post_date": "1967-03-27", "post_wd": 0},
    {"year": 1968, "easter": "1968-04-14", "good_friday": "1968-04-12", "pre_date": "1968-04-11", "pre_wd": 3, "post_date": "1968-04-15", "post_wd": 0},
    {"year": 1969, "easter": "1969-04-06", "good_friday": "1969-04-04", "pre_date": "1969-04-03", "pre_wd": 3, "post_date": "1969-04-07", "post_wd": 0},
    {"year": 1970, "easter": "1970-03-29", "good_friday": "1970-03-27", "pre_date": "1970-03-26", "pre_wd": 3, "post_date": "1970-03-30", "post_wd": 0},
    {"year": 1971, "easter": "1971-04-11", "good_friday": "1971-04-09", "pre_date": "1971-04-08", "pre_wd": 3, "post_date": "1971-04-12", "post_wd": 0},
    {"year": 1972, "easter": "1972-04-02", "good_friday": "1972-03-31", "pre_date": "1972-03-30", "pre_wd": 3, "post_date": "1972-04-03", "post_wd": 0},
    {"year": 1973, "easter": "1973-04-22", "good_friday": "1973-04-20", "pre_date": "1973-04-19", "pre_wd": 3, "post_date": "1973-04-23", "post_wd": 0},
    {"year": 1974, "easter": "1974-04-14", "good_friday": "1974-04-12", "pre_date": "1974-04-11", "pre_wd": 3, "post_date": "1974-04-15", "post_wd": 0},
    {"year": 1975, "easter": "1975-03-30", "good_friday": "1975-03-28", "pre_date": "1975-03-27", "pre_wd": 3, "post_date": "1975-03-31", "post_wd": 0},
    {"year": 1976, "easter": "1976-04-18", "good_friday": "1976-04-16", "pre_date": "1976-04-15", "pre_wd": 3, "post_date": "1976-04-19", "post_wd": 0},
    {"year": 1977, "easter": "1977-04-10", "good_friday": "1977-04-08", "pre_date": "1977-04-07", "pre_wd": 3, "post_date": "1977-04-11", "post_wd": 0},
    {"year": 1978, "easter": "1978-03-26", "good_friday": "1978-03-24", "pre_date": "1978-03-23", "pre_wd": 3, "post_date": "1978-03-27", "post_wd": 0},
    {"year": 1979, "easter": "1979-04-15", "good_friday": "1979-04-13", "pre_date": "1979-04-12", "pre_wd": 3, "post_date": "1979-04-16", "post_wd": 0},
    {"year": 1980, "easter": "1980-04-06", "good_friday": "1980-04-04", "pre_date": "1980-04-03", "pre_wd": 3, "post_date": "1980-04-07", "post_wd": 0},
    {"year": 1981, "easter": "1981-04-19", "good_friday": "1981-04-17", "pre_date": "1981-04-16", "pre_wd": 3, "post_date": "1981-04-20", "post_wd": 0},
    {"year": 1982, "easter": "1982-04-11", "good_friday": "1982-04-09", "pre_date": "1982-04-08", "pre_wd": 3, "post_date": "1982-04-12", "post_wd": 0},
    {"year": 1983, "easter": "1983-04-03", "good_friday": "1983-04-01", "pre_date": "1983-03-31", "pre_wd": 3, "post_date": "1983-04-04", "post_wd": 0},
    {"year": 1984, "easter": "1984-04-22", "good_friday": "1984-04-20", "pre_date": "1984-04-19", "pre_wd": 3, "post_date": "1984-04-23", "post_wd": 0},
    {"year": 1985, "easter": "1985-04-07", "good_friday": "1985-04-05", "pre_date": "1985-04-04", "pre_wd": 3, "post_date": "1985-04-08", "post_wd": 0},
    {"year": 1986, "easter": "1986-03-30", "good_friday": "1986-03-28", "pre_date": "1986-03-27", "pre_wd": 3, "post_date": "1986-03-31", "post_wd": 0},
    {"year": 1987, "easter": "1987-04-19", "good_friday": "1987-04-17", "pre_date": "1987-04-16", "pre_wd": 3, "post_date": "1987-04-20", "post_wd": 0},
    {"year": 1988, "easter": "1988-04-03", "good_friday": "1988-04-01", "pre_date": "1988-03-31", "pre_wd": 3, "post_date": "1988-04-04", "post_wd": 0},
    {"year": 1989, "easter": "1989-03-26", "good_friday": "1989-03-24", "pre_date": "1989-03-23", "pre_wd": 3, "post_date": "1989-03-27", "post_wd": 0},
    {"year": 1990, "easter": "1990-04-15", "good_friday": "1990-04-13", "pre_date": "1990-04-12", "pre_wd": 3, "post_date": "1990-04-16", "post_wd": 0},
    {"year": 1991, "easter": "1991-03-31", "good_friday": "1991-03-29", "pre_date": "1991-03-28", "pre_wd": 3, "post_date": "1991-04-01", "post_wd": 0},
    {"year": 1992, "easter": "1992-04-19", "good_friday": "1992-04-17", "pre_date": "1992-04-16", "pre_wd": 3, "post_date": "1992-04-20", "post_wd": 0},
    {"year": 1993, "easter": "1993-04-11", "good_friday": "1993-04-09", "pre_date": "1993-04-08", "pre_wd": 3, "post_date": "1993-04-12", "post_wd": 0},
    {"year": 1994, "easter": "1994-04-03", "good_friday": "1994-04-01", "pre_date": "1994-03-31", "pre_wd": 3, "post_date": "1994-04-04", "post_wd": 0},
    {"year": 1995, "easter": "1995-04-16", "good_friday": "1995-04-14", "pre_date": "1995-04-13", "pre_wd": 3, "post_date": "1995-04-17", "post_wd": 0},
    {"year": 1996, "easter": "1996-04-07", "good_friday": "1996-04-05", "pre_date": "1996-04-04", "pre_wd": 3, "post_date": "1996-04-08", "post_wd": 0},
    {"year": 1997, "easter": "1997-03-30", "good_friday": "1997-03-28", "pre_date": "1997-03-27", "pre_wd": 3, "post_date": "1997-03-31", "post_wd": 0},
    {"year": 1998, "easter": "1998-04-12", "good_friday": "1998-04-10", "pre_date": "1998-04-09", "pre_wd": 3, "post_date": "1998-04-13", "post_wd": 0},
    {"year": 1999, "easter": "1999-04-04", "good_friday": "1999-04-02", "pre_date": "1999-04-01", "pre_wd": 3, "post_date": "1999-04-05", "post_wd": 0},
    {"year": 2000, "easter": "2000-04-23", "good_friday": "2000-04-21", "pre_date": "2000-04-20", "pre_wd": 3, "post_date": "2000-04-24", "post_wd": 0},
    {"year": 2001, "easter": "2001-04-15", "good_friday": "2001-04-13", "pre_date": "2001-04-12", "pre_wd": 3, "post_date": "2001-04-16", "post_wd": 0},
    {"year": 2002, "easter": "2002-03-31", "good_friday": "2002-03-29", "pre_date": "2002-03-28", "pre_wd": 3, "post_date": "2002-04-01", "post_wd": 0},
    {"year": 2003, "easter": "2003-04-20", "good_friday": "2003-04-18", "pre_date": "2003-04-17", "pre_wd": 3, "post_date": "2003-04-21", "post_wd": 0},
    {"year": 2004, "easter": "2004-04-11", "good_friday": "2004-04-09", "pre_date": "2004-04-08", "pre_wd": 3, "post_date": "2004-04-12", "post_wd": 0},
    {"year": 2005, "easter": "2005-03-27", "good_friday": "2005-03-25", "pre_date": "2005-03-24", "pre_wd": 3, "post_date": "2005-03-28", "post_wd": 0},
    {"year": 2006, "easter": "2006-04-16", "good_friday": "2006-04-14", "pre_date": "2006-04-13", "pre_wd": 3, "post_date": "2006-04-17", "post_wd": 0},
    {"year": 2007, "easter": "2007-04-08", "good_friday": "2007-04-06", "pre_date": "2007-04-05", "pre_wd": 3, "post_date": "2007-04-09", "post_wd": 0},
    {"year": 2008, "easter": "2008-03-23", "good_friday": "2008-03-21", "pre_date": "2008-03-20", "pre_wd": 3, "post_date": "2008-03-24", "post_wd": 0},
    {"year": 2009, "easter": "2009-04-12", "good_friday": "2009-04-10", "pre_date": "2009-04-09", "pre_wd": 3, "post_date": "2009-04-13", "post_wd": 0},
    {"year": 2010, "easter": "2010-04-04", "good_friday": "2010-04-02", "pre_date": "2010-04-01", "pre_wd": 3, "post_date": "2010-04-05", "post_wd": 0},
    {"year": 2011, "easter": "2011-04-24", "good_friday": "2011-04-22", "pre_date": "2011-04-21", "pre_wd": 3, "post_date": "2011-04-25", "post_wd": 0},
    {"year": 2012, "easter": "2012-04-08", "good_friday": "2012-04-06", "pre_date": "2012-04-05", "pre_wd": 3, "post_date": "2012-04-09", "post_wd": 0},
    {"year": 2013, "easter": "2013-03-31", "good_friday": "2013-03-29", "pre_date": "2013-03-28", "pre_wd": 3, "post_date": "2013-04-01", "post_wd": 0},
    {"year": 2014, "easter": "2014-04-20", "good_friday": "2014-04-18", "pre_date": "2014-04-17", "pre_wd": 3, "post_date": "2014-04-21", "post_wd": 0},
    {"year": 2015, "easter": "2015-04-05", "good_friday": "2015-04-03", "pre_date": "2015-04-02", "pre_wd": 3, "post_date": "2015-04-06", "post_wd": 0},
    {"year": 2016, "easter": "2016-03-27", "good_friday": "2016-03-25", "pre_date": "2016-03-24", "pre_wd": 3, "post_date": "2016-03-28", "post_wd": 0},
    {"year": 2017, "easter": "2017-04-16", "good_friday": "2017-04-14", "pre_date": "2017-04-13", "pre_wd": 3, "post_date": "2017-04-17", "post_wd": 0},
    {"year": 2018, "easter": "2018-04-01", "good_friday": "2018-03-30", "pre_date": "2018-03-29", "pre_wd": 3, "post_date": "2018-04-02", "post_wd": 0},
    {"year": 2019, "easter": "2019-04-21", "good_friday": "2019-04-19", "pre_date": "2019-04-18", "pre_wd": 3, "post_date": "2019-04-22", "post_wd": 0},
    {"year": 2020, "easter": "2020-04-12", "good_friday": "2020-04-10", "pre_date": "2020-04-09", "pre_wd": 3, "post_date": "2020-04-13", "post_wd": 0},
    {"year": 2021, "easter": "2021-04-04", "good_friday": "2021-04-02", "pre_date": "2021-04-01", "pre_wd": 3, "post_date": "2021-04-05", "post_wd": 0},
    {"year": 2022, "easter": "2022-04-17", "good_friday": "2022-04-15", "pre_date": "2022-04-14", "pre_wd": 3, "post_date": "2022-04-18", "post_wd": 0},
    {"year": 2023, "easter": "2023-04-09", "good_friday": "2023-04-07", "pre_date": "2023-04-06", "pre_wd": 3, "post_date": "2023-04-10", "post_wd": 0},
    {"year": 2024, "easter": "2024-03-31", "good_friday": "2024-03-29", "pre_date": "2024-03-28", "pre_wd": 3, "post_date": "2024-04-01", "post_wd": 0},
    {"year": 2025, "easter": "2025-04-20", "good_friday": "2025-04-18", "pre_date": "2025-04-17", "pre_wd": 3, "post_date": "2025-04-21", "post_wd": 0},
]

EASTER_DF: pd.DataFrame = pd.DataFrame(EASTER_SESSIONS)


def easter_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded Easter session table.

    Columns: ``year`` (int), ``easter`` / ``good_friday`` / ``pre_date`` /
    ``post_date`` (str 'YYYY-MM-DD'), ``pre_wd`` / ``post_wd`` (int, 0=Mon..4=Fri).
    76 rows (1950-2025). The ``easter`` column equals :func:`easter_sunday` for the
    year -- the table is self-checking.
    """
    return EASTER_DF.copy()


# ---------------------------------------------------------------------------
# Synthetic daily-return generator -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    start_year: int = 1950,
    end_year: int = 2025,
    premium_bps: float = 0.0,
    base_bps: float = 3.6,
    vol_bps: float = 99.0,
    seed: int = 287,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily-return series with an optional planted pre-holiday premium.

    A synthetic Mon-Fri business-day calendar spanning ``start_year..end_year`` is
    generated. Each day's return is drawn i.i.d. from N(base_bps, vol_bps^2) in
    basis points. On the synthetic pre-Good-Friday session of each year (the last
    business day before the computed Good Friday), an extra drift of ``premium_bps``
    bps is added. ``premium_bps = 0`` is the null: the pre-holiday session is
    statistically identical to every other day.

    Returns ``(df, truth)`` where ``df`` is indexed by date with columns ``ret``
    (daily simple return) and ``is_pre`` (bool, the pre-holiday session flag), and
    ``truth`` records the planted parameters. This is the study's null (and positive
    control) in a bottle.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(f"{start_year}-01-01", f"{end_year}-12-31")
    rets = rng.normal(base_bps * 1e-4, vol_bps * 1e-4, len(idx))

    df = pd.DataFrame({"ret": rets}, index=idx)
    is_pre = pd.Series(False, index=df.index)
    for y in range(start_year, end_year + 1):
        gf = pd.Timestamp(good_friday(y))
        before = df.index[df.index < gf]
        if len(before):
            is_pre.loc[before[-1]] = True
    df["is_pre"] = is_pre.values
    df.loc[df["is_pre"], "ret"] += premium_bps * 1e-4

    truth = {
        "start_year": start_year,
        "end_year": end_year,
        "premium_bps": premium_bps,
        "base_bps": base_bps,
        "vol_bps": vol_bps,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data -- ^GSPC daily closes (repo-level cache, cache-only by default)
# ---------------------------------------------------------------------------
def fetch_gspc_daily(
    fetch: bool = False,
    cache_dir: str = REPO_CACHE,
) -> pd.DataFrame:
    """Load ^GSPC daily close-to-close returns (price-only).

    Cache-only by default: reads ``_cache/^GSPC_split_only.parquet`` (staged in the
    repo). Network (yfinance) is touched only on explicit ``fetch=True``; the lazy
    import keeps the offline core network-free.

    Returns a DataFrame indexed by date with columns ``close`` and ``ret``
    (close-to-close simple return). Price-only -- no dividends. The index level is
    back-revised by the vendor, which we name on the Signal axis.

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False``.
    """
    cache_path = os.path.join(cache_dir, GSPC_CACHE_NAME)

    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"^GSPC cache not found at {cache_path}. "
                "Call fetch_gspc_daily(fetch=True) once, or stage the repo-level cache."
            )
        px = pd.read_parquet(cache_path)
    else:
        import yfinance as yf  # lazy import: network only on fetch=True

        raw = yf.download("^GSPC", start="1927-01-01", auto_adjust=False, progress=False)
        if raw.empty:
            raise RuntimeError("yfinance returned no ^GSPC data")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        px = raw[["Close"]].copy()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        px.to_parquet(cache_path)

    if "Close" in px.columns:
        close = px["Close"]
    elif "close" in px.columns:
        close = px["close"]
    else:
        close = px.iloc[:, 0]
    out = pd.DataFrame({"close": close.astype(float)})
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out["ret"] = out["close"].pct_change()
    return out.dropna(subset=["ret"])


def build_event_frame(
    fetch: bool = False,
    cache_dir: str = REPO_CACHE,
    start_year: int = 1950,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Join the hardcoded Easter session dates with the real ^GSPC daily returns.

    For each year we attach the close-to-close return of the ``pre_date`` (pre-Good-
    Friday session) and the ``post_date`` (Easter Monday session). Returns a
    DataFrame indexed by ``year`` with columns: ``pre_date`` / ``post_date``
    (Timestamp), ``pre_wd`` / ``post_wd`` (int), ``pre_ret`` / ``post_ret`` (the
    event returns), ``pre_up`` / ``post_up`` (bool).

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False``.
    """
    daily = fetch_gspc_daily(fetch=fetch, cache_dir=cache_dir)
    tbl = easter_table()
    tbl = tbl[(tbl["year"] >= start_year) & (tbl["year"] <= end_year)].copy()
    tbl["pre_date"] = pd.to_datetime(tbl["pre_date"])
    tbl["post_date"] = pd.to_datetime(tbl["post_date"])

    rets = daily["ret"]
    tbl["pre_ret"] = tbl["pre_date"].map(lambda d: rets.get(d, np.nan))
    tbl["post_ret"] = tbl["post_date"].map(lambda d: rets.get(d, np.nan))
    tbl["pre_up"] = tbl["pre_ret"] > 0
    tbl["post_up"] = tbl["post_ret"] > 0
    return tbl.set_index("year")


def unconditional_daily(
    fetch: bool = False,
    cache_dir: str = REPO_CACHE,
    start_year: int = 1950,
    end_year: int = 2025,
    weekday: int | None = None,
) -> pd.Series:
    """The unconditional ^GSPC daily-return series over the study window (the baseline).

    If ``weekday`` is given (0=Mon..4=Fri), restrict to that weekday -- used to build
    the day-of-week control (the pre-holiday session is always a Thursday).
    """
    daily = fetch_gspc_daily(fetch=fetch, cache_dir=cache_dir)
    yr = daily.index.year
    mask = (yr >= start_year) & (yr <= end_year)
    if weekday is not None:
        mask = mask & (daily.index.weekday == weekday)
    return daily.loc[mask, "ret"]


def fingerprint(rets) -> str:
    """A short content fingerprint of a return array/Series, for the as-of stamp."""
    vals = np.asarray(pd.Series(rets).dropna().to_numpy(dtype=float))
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
