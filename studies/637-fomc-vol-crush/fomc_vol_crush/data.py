"""Data layer for Study 637 — FOMC Vol Crush.

Three ingredients, all offline-friendly once cached:

* **Real tape.** Daily ^VIX OHLC (the implied-vol index whose decision-day collapse we test),
  daily SPY raw OHLC (the realized high-low range on the same days) and daily SVXY adjusted
  closes (the retail short-vol expression for the third axis), all from yfinance (no key),
  cached as CSV under the study's own ``_cache/``.

* **The scheduled FOMC decision calendar, hardcoded.** Since February 1994 the FOMC announces
  its decision the day the (scheduled) meeting ends — a statement released mid-afternoon ET
  (2:15 pm, moved to 2:00 pm in 2013), i.e. *before* the 4:15 pm ^VIX close. The dates are
  known **years in advance**, so a decision-day flag involves zero look-ahead. Scheduled
  meetings only — no inter-meeting / emergency actions (those are, by definition, not
  "set-your-watch" events). Source: Federal Reserve historical FOMC calendars
  (federalreserve.gov/monetarypolicy/fomccalendars.htm and the historical-year pages) — the
  same hardcoded table as sibling studies 67-fed-drift / 135-fomc-cycle / 322-fomc-blackout /
  517-pre-fomc-drift, extended through the 2026 schedule.

* **Synthetic world.** A deterministic, seeded mean-reverting log-vol index with a TUNABLE
  planted decision-day crush (knob ``crush``, in vol points): on scheduled "decision days"
  (every 32nd business day) the index drops by an extra ``crush`` points. ``crush = 0`` is the
  null world — decision days statistically identical to the rest; the Welch machinery must NOT
  manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
VIX_CACHE = os.path.join(CACHE_DIR, "fvc_vix.csv")
SPY_CACHE = os.path.join(CACHE_DIR, "fvc_spy.csv")
SVXY_CACHE = os.path.join(CACHE_DIR, "fvc_svxy.csv")

START = "1994-01-03"        # first full year of the post-1994 announcement regime
AS_OF = "2026-06-30"        # last complete month at publication (2026-07-03)
PRESSER_SPLIT = "2011-04-27"  # first post-meeting press conference (Bernanke) — the era split
SVXY_DELEV = "2018-02-28"   # SVXY cut its exposure -1x -> -0.5x after Volmageddon (2018-02-05)

# --------------------------------------------------------------------------- #
# Hardcoded scheduled FOMC decision (statement-release) dates, 1994 -> 2026.
# Scheduled meetings only (no inter-meeting/emergency actions). For two-day meetings this is
# the SECOND day — the day the statement hits the tape mid-afternoon ET. Source: Federal
# Reserve historical FOMC calendars (federalreserve.gov/monetarypolicy/fomccalendars.htm);
# identical to the table shared by sibling studies 67/135/322/517.
# Known quirk, kept for calendar consistency with the siblings: the scheduled 2020-03-18
# meeting was pre-empted by the 2020-03-15 emergency action; the scheduled date stays in the
# table (it was still the pre-announced decision date ex ante).
# --------------------------------------------------------------------------- #
FOMC_DATES = pd.to_datetime("""
1994-02-04 1994-03-22 1994-04-18 1994-05-17 1994-07-06 1994-08-16 1994-09-27 1994-11-15 1994-12-20
1995-02-01 1995-03-28 1995-05-23 1995-07-06 1995-08-22 1995-09-26 1995-11-15 1995-12-19
1996-01-31 1996-03-26 1996-05-21 1996-07-03 1996-08-20 1996-09-24 1996-11-13 1996-12-17
1997-02-05 1997-03-25 1997-05-20 1997-07-02 1997-08-19 1997-09-30 1997-11-12 1997-12-16
1998-02-04 1998-03-31 1998-05-19 1998-07-01 1998-08-18 1998-09-29 1998-11-17 1998-12-22
1999-02-03 1999-03-30 1999-05-18 1999-06-30 1999-08-24 1999-10-05 1999-11-16 1999-12-21
2000-02-02 2000-03-21 2000-05-16 2000-06-28 2000-08-22 2000-10-03 2000-11-15 2000-12-19
2001-01-31 2001-03-20 2001-05-15 2001-06-27 2001-08-21 2001-10-02 2001-11-06 2001-12-11
2002-01-30 2002-03-19 2002-05-07 2002-06-26 2002-08-13 2002-09-24 2002-11-06 2002-12-10
2003-01-29 2003-03-18 2003-05-06 2003-06-25 2003-08-12 2003-09-16 2003-10-28 2003-12-09
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


def fomc_calendar(start: str = START, end: str = AS_OF) -> pd.DatetimeIndex:
    """Scheduled FOMC decision dates inside [start, end], sorted."""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return pd.DatetimeIndex(sorted(d for d in FOMC_DATES if lo <= d <= hi))


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "1993-06-01", end: str = "2026-07-01") -> None:
    """Download ^VIX OHLC, SPY raw OHLC and SVXY adjusted closes; cache them. Network; once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    # ^VIX — an index; no adjustment concept, take the bars as printed
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix[["Open", "High", "Low", "Close"]].dropna(how="all").to_csv(VIX_CACHE)

    # SPY — RAW bars for the realized high-low range (unadjusted intraday geometry),
    # plus the adjusted close for any return use
    raw = yf.download("SPY", start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    spy = pd.DataFrame({
        "Open": raw["Open"], "High": raw["High"], "Low": raw["Low"],
        "Close": raw["Close"],
        "AdjClose": raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"],
    }).dropna(how="all")
    spy.to_csv(SPY_CACHE)

    # SVXY — the retail short-vol ETF (inception 2011-10-04); total-return adjusted closes
    sv = yf.download("SVXY", start="2011-10-01", end=end, auto_adjust=True, progress=False)
    if isinstance(sv.columns, pd.MultiIndex):
        sv.columns = sv.columns.get_level_values(0)
    sv[["Close"]].dropna().to_csv(SVXY_CACHE)


def have_real() -> bool:
    return all(os.path.exists(p) for p in (VIX_CACHE, SPY_CACHE, SVXY_CACHE))


def load_real(start: str = START, asof: str = AS_OF
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cached (vix, spy, svxy) frames, sliced to [start, asof]."""
    out = []
    for path in (VIX_CACHE, SPY_CACHE, SVXY_CACHE):
        df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        out.append(df.loc[(df.index >= start) & (df.index <= asof)].copy())
    return out[0], out[1], out[2]


# --------------------------------------------------------------------------- #
# Synthetic world — planted decision-day crush (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(crush: float = 0.0, seed: int = 637,
                    start: str = "1994-01-03", end: str = "2026-06-30",
                    kappa: float = 0.02, mu: float = 19.0, sig: float = 1.15,
                    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Deterministic mean-reverting vol index with a TUNABLE planted decision-day crush.

    The level follows a discrete OU process around ``mu`` (vol clusters, mean-reverts — the
    coarse shape of the real VIX). Every 32nd business day is a scheduled "decision day"; on
    those days the level drops by an EXTRA ``crush`` points (the planted uncertainty
    resolution). ``crush = 0`` is the null world: decision days are statistically identical to
    every other day, and the Welch split must NOT reach significance.

    Business-day index, span ~32 years — far below the 250-year pandas ns-timestamp trap.
    Returns (frame with a Close column, decision-day DatetimeIndex).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    is_dec = np.zeros(n, dtype=bool)
    is_dec[31::32] = True                      # scheduled, evenly spaced pseudo-FOMC days

    lvl = np.empty(n)
    lvl[0] = mu
    for t in range(1, n):
        shock = rng.normal(0.0, sig)
        lvl[t] = lvl[t - 1] + kappa * (mu - lvl[t - 1]) + shock
        if is_dec[t]:
            lvl[t] -= crush                    # the planted decision-day collapse
        lvl[t] = max(lvl[t], 5.0)              # a floor, like the real thing
    close = pd.DataFrame({"Close": lvl}, index=idx)
    return close, idx[is_dec]
