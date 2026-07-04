"""Data layer for Study 605 — VIX Settlement Day.

Three ingredients, all offline-friendly once cached:

* **Real tape.** Daily ^VIX and ^GSPC OHLC from yfinance (no key), cached as CSV under the
  study's own ``_cache/``. The VIX *index* is where Griffin-Shams' settlement-morning pressure
  should leave daily-bar tracks: the settlement SOQ is computed from the opening SPX option
  auction, and on settlement morning the live VIX uses (approximately) the same option strip —
  so a distorted settlement auction shows up in the index's open print and in how the day
  digests it.

* **The settlement calendar, by rule.** VIX futures/options settle on the Wednesday that is
  30 days before the S&P-500 option expiration of the *following* month (the third Friday,
  or the preceding business day when that Friday is an exchange holiday); if the resulting
  Wednesday is itself a holiday, settlement moves to the preceding business day. We build the
  calendar from that rule (Good Friday via the Gregorian Easter algorithm + Juneteenth from
  2022) and **assert it against 18 known CBOE settlement dates**, including every
  holiday-shifted Tuesday in the sample.

* **Synthetic world.** A deterministic, seeded generator of a mean-reverting "vol index" with
  a TUNABLE planted settlement-morning distortion: on settlement days the morning gap
  *continues* into the close by a planted extra slope (the Griffin-Shams signature), while on
  normal days gaps mean-revert. ``planted = 0`` is the null — the machinery must not
  manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
VIX_CACHE = os.path.join(CACHE_DIR, "vix_ohlc.csv")
SPX_CACHE = os.path.join(CACHE_DIR, "spx_ohlc.csv")

START = "2004-01-01"      # VIX futures launched 2004-03-26; first full settlement year
AS_OF = "2026-06-30"      # last complete month at publication (2026-07-03)
PAPER_SPLIT = "2018-01-01"  # Griffin-Shams RFS publication + manipulation lawsuits year


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2003-06-01", end: str = "2026-07-01") -> None:
    """Download ^VIX and ^GSPC daily OHLC and cache them. Network; run once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for tk, path, adj in (("^VIX", VIX_CACHE, False), ("^GSPC", SPX_CACHE, True)):
        raw = yf.download(tk, start=start, end=end, auto_adjust=adj, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw[["Open", "High", "Low", "Close"]].dropna().to_csv(path)


def have_real() -> bool:
    return os.path.exists(VIX_CACHE) and os.path.exists(SPX_CACHE)


def load_real(start: str = START, asof: str = AS_OF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached (vix, spx) OHLC frames, sliced to [start, asof]."""
    out = []
    for path in (VIX_CACHE, SPX_CACHE):
        df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        out.append(df.loc[(df.index >= start) & (df.index <= asof)].copy())
    return out[0], out[1]


# --------------------------------------------------------------------------- #
# Settlement calendar, by rule
# --------------------------------------------------------------------------- #
def _easter(year: int) -> dt.date:
    """Gregorian Easter Sunday (Anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return dt.date(year, month, day)


def _third_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 15)
    while d.weekday() != 4:
        d += dt.timedelta(days=1)
    return d


def _is_market_holiday(d: dt.date) -> bool:
    """The only exchange holidays that can hit a third Friday (day 15-21) or the settlement
    Wednesday (day 13-21) inside 2004-2026: Good Friday, and Juneteenth (NYSE/CBOE observe it
    since 2022)."""
    if d == _easter(d.year) - dt.timedelta(days=2):
        return True
    if d.year >= 2022 and d.month == 6 and d.day == 19:
        return True
    return False


def settlement_date(year: int, month: int) -> dt.date:
    """Final-settlement date of the VIX contract expiring in (year, month).

    Rule (CBOE VX contract specs): the Wednesday 30 days before the S&P-500 option expiration
    of the following month — the third Friday, stepped back to the preceding business day if
    that Friday is a holiday. If the resulting Wednesday is itself a holiday (Juneteenth on a
    Wednesday, e.g. 2024-06-19), settlement moves to the preceding business day.
    """
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    spx_exp = _third_friday(ny, nm)
    while _is_market_holiday(spx_exp) or spx_exp.weekday() > 4:
        spx_exp -= dt.timedelta(days=1)
    d = spx_exp - dt.timedelta(days=30)
    while _is_market_holiday(d) or d.weekday() > 4:
        d -= dt.timedelta(days=1)
    return d


# Verification sample — known CBOE final-settlement dates, hand-checked against CBOE
# announcements / expiration calendars (sources: cdn.cboe.com futures-settlement calendars;
# macroption.com/vix-expiration-calendar; CBOE product update "Juneteenth Holiday Closure
# Impact on VIX Options and VX Futures", 2024). Includes every holiday-shifted TUESDAY in
# the 2004-2026 sample (Good Fridays 2008/2014/2019/2022/2025 on a third Friday, Juneteenth
# 2024 on the settlement Wednesday itself, Juneteenth 2026 on a third Friday).
KNOWN_SETTLEMENTS = {
    (2008, 2): "2008-02-19",   # Tue — Good Friday 2008-03-21 was the March third Friday
    (2014, 3): "2014-03-18",   # Tue — Good Friday 2014-04-18
    (2018, 1): "2018-01-17",
    (2019, 3): "2019-03-19",   # Tue — Good Friday 2019-04-19
    (2020, 3): "2020-03-18",
    (2022, 3): "2022-03-15",   # Tue — Good Friday 2022-04-15
    (2024, 6): "2024-06-18",   # Tue — Juneteenth 2024-06-19 fell ON the settlement Wednesday
    (2025, 1): "2025-01-22", (2025, 2): "2025-02-19",
    (2025, 3): "2025-03-18",   # Tue — Good Friday 2025-04-18
    (2025, 4): "2025-04-16", (2025, 5): "2025-05-21", (2025, 6): "2025-06-18",
    (2025, 7): "2025-07-16", (2025, 8): "2025-08-20", (2025, 12): "2025-12-17",
    (2026, 5): "2026-05-19",   # Tue — Juneteenth Fri 2026-06-19 (June SPX expiry -> Thu 06-18)
    (2026, 6): "2026-06-17",
}


def verify_calendar() -> int:
    """Assert the rule reproduces every known CBOE date; returns how many were checked."""
    for (y, m), want in KNOWN_SETTLEMENTS.items():
        got = settlement_date(y, m).isoformat()
        assert got == want, f"calendar rule failed for {y}-{m:02d}: got {got}, want {want}"
    return len(KNOWN_SETTLEMENTS)


def settlement_calendar(start: str = START, end: str = AS_OF) -> pd.DatetimeIndex:
    """All monthly VIX final-settlement dates inside [start, end], by rule."""
    lo, hi = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    dates = [settlement_date(y, m)
             for y in range(lo.year, hi.year + 1) for m in range(1, 13)]
    return pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in dates if lo <= d <= hi))


# --------------------------------------------------------------------------- #
# FOMC statement days (the big Wednesday confounder)
# --------------------------------------------------------------------------- #
# Scheduled FOMC statement-release dates 2004-2026. Source: Federal Reserve FOMC historical
# calendars — same hardcoded table as sibling studies 67-fed-drift / 135-fomc-cycle /
# 517-pre-fomc-drift (federalreserve.gov/monetarypolicy/fomccalendars.htm).
FOMC_DATES = pd.to_datetime("""
2004-01-28 2004-03-16 2004-05-04 2004-06-30 2004-08-10 2004-09-21 2004-11-10 2004-12-14
2005-02-02 2005-03-22 2005-05-03 2005-06-30 2005-08-09 2005-09-20 2005-11-01 2005-12-13
2006-01-31 2006-03-28 2006-05-10 2006-06-29 2006-08-08 2006-09-20 2006-10-25 2006-12-12
2007-01-31 2007-03-21 2007-05-09 2007-06-28 2007-08-07 2007-09-18 2007-10-31 2007-12-11
2008-01-30 2008-03-18 2008-04-30 2008-06-25 2008-08-05 2008-09-16 2008-10-29 2008-12-16
2009-01-28 2009-03-18 2009-04-29 2009-06-24 2009-08-12 2009-09-23 2009-11-04 2009-12-16
2010-01-27 2010-03-16 2010-04-28 2010-06-23 2010-08-10 2010-09-21 2010-11-03 2010-12-14
2011-01-26 2011-03-15 2011-04-27 2011-06-22 2011-08-09 2011-09-21 2011-11-02 2011-12-13
2012-01-25 2012-03-13 2012-04-25 2012-06-20 2012-08-01 2012-09-13 2012-10-24 2012-12-12
2013-01-30 2013-03-20 2013-05-01 2013-06-19 2013-07-31 2013-09-18 2013-10-30 2013-12-18
2014-01-29 2014-03-19 2014-04-30 2014-06-18 2014-07-30 2014-09-17 2014-10-29 2014-12-17
2015-01-28 2015-03-18 2015-04-29 2015-06-17 2015-07-29 2015-09-17 2015-10-28 2015-12-16
2016-01-27 2016-03-16 2016-04-27 2016-06-15 2016-07-27 2016-09-21 2016-11-02 2016-12-14
2017-02-01 2017-03-15 2017-05-03 2017-06-14 2017-07-26 2017-09-20 2017-11-01 2017-12-13
2018-01-31 2018-03-21 2018-05-02 2018-06-13 2018-08-01 2018-09-26 2018-11-08 2018-12-19
2019-01-30 2019-03-20 2019-05-01 2019-06-19 2019-07-31 2019-09-18 2019-10-30 2019-12-11
2020-01-29 2020-03-18 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16
2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17
""".split())


# --------------------------------------------------------------------------- #
# Synthetic world — planted settlement-morning continuation
# --------------------------------------------------------------------------- #
def synthetic_world(planted: float = 0.0, seed: int = 605,
                    start: str = "1980-01-01", end: str = "2023-12-31",
                    base_slope: float = -0.20, gap_sig: float = 0.022,
                    intr_sig: float = 0.045) -> pd.DataFrame:
    """Deterministic daily OHLC of a synthetic vol index with a planted settlement signature.

    Every day: an overnight gap ``g ~ N(0, gap_sig)`` and an intraday move
    ``i = slope * g + N(0, intr_sig)``. On NORMAL days ``slope = base_slope`` (gaps
    mean-revert, as the real tape shows). On rule-based SETTLEMENT days
    ``slope = base_slope + planted`` — the TUNABLE knob: ``planted = 0`` is the null world
    (settlement days statistically identical to other Wednesdays); ``planted > 0`` plants the
    Griffin-Shams continuation signature (the distorted morning print carries into the day).

    Business-day index, span << 250 years (safe for pandas ns timestamps).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    sett = set(settlement_calendar(start, end))
    is_sett = np.array([ts in sett for ts in idx])

    gap = rng.normal(0.0, gap_sig, len(idx))
    slope = np.where(is_sett, base_slope + planted, base_slope)
    intr = slope * gap + rng.normal(0.0, intr_sig, len(idx))

    # build O/C so that log(O/C_prev) = gap and log(C/O) = intr EXACTLY — the downstream
    # pipeline recomputes both from the bars, so the planted structure survives byte-for-byte
    # (the level path itself is decorative)
    c = 18.0 * np.exp(np.cumsum(gap + intr))
    o = np.concatenate([[18.0], c[:-1]]) * np.exp(gap)
    hi = np.maximum(o, c) * 1.01
    lo_ = np.minimum(o, c) * 0.99
    return pd.DataFrame({"Open": o, "High": hi, "Low": lo_, "Close": c}, index=idx)
