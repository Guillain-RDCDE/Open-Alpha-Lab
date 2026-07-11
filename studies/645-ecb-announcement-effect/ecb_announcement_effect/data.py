"""Data layer for Study 645 — ECB Announcement Effect.

Three ingredients, all offline-friendly once cached:

* **Real tape.** Daily FEZ (SPDR EURO STOXX 50 ETF, the tradable USD-listed euro-area large-cap
  basket) raw OHLC — for the same-day return and the realized high-low range — and daily
  EURUSD=X OHLC (the FX reaction to the decision), both from yfinance (no key), cached as CSV
  under the study's own ``_cache/``.

* **The scheduled ECB Governing Council monetary-policy decision calendar, hardcoded.** Since
  1999 the Governing Council announces its rate decision on a fixed pre-published schedule: a
  press release at 13:45 CET, a press conference at 14:30 CET (14:15 CET since 2012). From
  1999 through 2014 the stance was assessed monthly (as a rule, the first Thursday of the
  month, August usually skipped 2005-2009, added back from 2010); from **January 2015** the
  Governing Council moved to the now-familiar **6-week cycle** (8 decisions/year) — a
  structural change we use as the study's one justified era split. Dates sourced from the
  ECB's own year-ahead schedule press releases (ecb.europa.eu/press/pr/date/…, e.g.
  pr040617/pr060602/pr070330/pr090508/pr110420/pr130517 for the monthly era) and the ECB's
  "monetary policy statements" archive for the 6-week era (2015 → 2026); cross-checked against
  the effective-date jumps in the ECB's own key-rate series (FRED ``ECBDFR``) wherever a
  decision changed a rate, since a rate change always takes effect a few days after the
  Governing Council meeting that decided it, never on an unlisted date.

* **Synthetic world.** A deterministic, seeded mean-reverting-return index with a TUNABLE
  planted decision-day effect (knob ``drift``, in daily-return units): on scheduled "decision
  days" (every 22nd business day, ≈ the 6-week cadence) the index gets an extra ``drift``
  return. ``drift = 0`` is the null world — decision days statistically identical to the rest;
  the Welch machinery must NOT manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
FEZ_CACHE = os.path.join(CACHE_DIR, "eae_fez.csv")
EURUSD_CACHE = os.path.join(CACHE_DIR, "eae_eurusd.csv")

START = "2005-01-03"          # FEZ (inception 2002-10) has a clean full tape from here
AS_OF = "2026-06-30"          # last complete month at publication (2026-07-10)
SIXWEEK_SPLIT = "2015-01-01"  # the Governing Council's own structural change: monthly -> 6-week

# --------------------------------------------------------------------------- #
# Hardcoded scheduled ECB Governing Council monetary-policy decision dates, 2005 -> 2026.
# 2005-2014: monthly era (the stance was, as a rule, assessed only at the first meeting of the
# month; August skipped 2005/2007/2008/2009, held 2006/2010-2014). 2015 onward: the 6-week
# cycle (8 decisions/year), effective from the Governing Council's own July-2014 announcement.
# Source: ECB year-ahead schedule press releases (2005/2007-2014) and the ECB's monetary-policy
# statement archive (2015-2026); every date that also carries a rate CHANGE is cross-checked
# against the effective-date jump in the ECB's key-rate series (FRED ECBDFR), which always
# lands 5-7 calendar days after the Governing Council meeting that decided it.
# --------------------------------------------------------------------------- #
ECB_DATES = pd.to_datetime("""
2005-01-13 2005-02-03 2005-03-03 2005-04-07 2005-05-04 2005-06-02 2005-07-07 2005-09-01 2005-10-06 2005-11-03 2005-12-01
2006-01-12 2006-02-02 2006-03-02 2006-04-06 2006-05-04 2006-06-08 2006-07-06 2006-08-03 2006-09-07 2006-10-05 2006-11-02 2006-12-07
2007-01-11 2007-02-08 2007-03-08 2007-04-12 2007-05-10 2007-06-06 2007-07-05 2007-09-06 2007-10-04 2007-11-08 2007-12-06
2008-01-10 2008-02-07 2008-03-06 2008-04-10 2008-05-08 2008-06-05 2008-07-03 2008-09-04 2008-10-02 2008-11-06 2008-12-04
2009-01-15 2009-02-05 2009-03-05 2009-04-02 2009-05-07 2009-06-04 2009-07-02 2009-09-03 2009-10-08 2009-11-05 2009-12-03
2010-01-14 2010-02-04 2010-03-04 2010-04-08 2010-05-06 2010-06-10 2010-07-08 2010-08-05 2010-09-02 2010-10-07 2010-11-04 2010-12-02
2011-01-13 2011-02-03 2011-03-03 2011-04-07 2011-05-05 2011-06-09 2011-07-07 2011-08-04 2011-09-08 2011-10-06 2011-11-03 2011-12-08
2012-01-12 2012-02-09 2012-03-08 2012-04-04 2012-05-03 2012-06-06 2012-07-05 2012-08-02 2012-09-06 2012-10-04 2012-11-08 2012-12-06
2013-01-10 2013-02-07 2013-03-07 2013-04-04 2013-05-02 2013-06-06 2013-07-04 2013-08-01 2013-09-05 2013-10-02 2013-11-07 2013-12-05
2014-01-09 2014-02-06 2014-03-06 2014-04-03 2014-05-08 2014-06-05 2014-07-03 2014-08-07 2014-09-04 2014-10-02 2014-11-06 2014-12-04
2015-01-22 2015-03-05 2015-04-15 2015-06-03 2015-07-16 2015-09-03 2015-10-22 2015-12-03
2016-01-21 2016-03-10 2016-04-21 2016-06-02 2016-07-21 2016-09-08 2016-10-20 2016-12-08
2017-01-19 2017-03-09 2017-04-27 2017-06-08 2017-07-20 2017-09-07 2017-10-26 2017-12-14
2018-01-25 2018-03-08 2018-04-26 2018-06-14 2018-07-26 2018-09-13 2018-10-25 2018-12-13
2019-01-24 2019-03-07 2019-04-10 2019-06-06 2019-07-25 2019-09-12 2019-10-24 2019-12-12
2020-01-23 2020-03-12 2020-04-30 2020-06-04 2020-07-16 2020-09-10 2020-10-29 2020-12-10
2021-01-21 2021-03-11 2021-04-22 2021-06-10 2021-07-22 2021-09-09 2021-10-28 2021-12-16
2022-02-03 2022-03-10 2022-04-14 2022-06-09 2022-07-21 2022-09-08 2022-10-27 2022-12-15
2023-02-02 2023-03-16 2023-05-04 2023-06-15 2023-07-27 2023-09-14 2023-10-26 2023-12-14
2024-01-25 2024-03-07 2024-04-11 2024-06-06 2024-07-18 2024-09-12 2024-10-17 2024-12-12
2025-01-30 2025-03-06 2025-04-17 2025-06-05 2025-07-24 2025-09-11 2025-10-30 2025-12-18
2026-02-05 2026-03-19 2026-04-30 2026-06-11 2026-07-23 2026-09-10 2026-10-29 2026-12-17
""".split())


def ecb_calendar(start: str = START, end: str = AS_OF) -> pd.DatetimeIndex:
    """Scheduled ECB decision dates inside [start, end], sorted."""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return pd.DatetimeIndex(sorted(d for d in ECB_DATES if lo <= d <= hi))


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2004-06-01", end: str = "2026-07-01") -> None:
    """Download FEZ raw OHLC and EURUSD=X OHLC; cache them. Network; run once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    fez = yf.download("FEZ", start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(fez.columns, pd.MultiIndex):
        fez.columns = fez.columns.get_level_values(0)
    out = pd.DataFrame({
        "Open": fez["Open"], "High": fez["High"], "Low": fez["Low"], "Close": fez["Close"],
        "AdjClose": fez["Adj Close"] if "Adj Close" in fez.columns else fez["Close"],
    }).dropna(how="all")
    out.to_csv(FEZ_CACHE)

    fx = yf.download("EURUSD=X", start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(fx.columns, pd.MultiIndex):
        fx.columns = fx.columns.get_level_values(0)
    fx[["Open", "High", "Low", "Close"]].dropna(how="all").to_csv(EURUSD_CACHE)


def have_real() -> bool:
    return all(os.path.exists(p) for p in (FEZ_CACHE, EURUSD_CACHE))


def load_real(start: str = START, asof: str = AS_OF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached (fez, eurusd) frames, sliced to [start, asof]."""
    out = []
    for path in (FEZ_CACHE, EURUSD_CACHE):
        df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        out.append(df.loc[(df.index >= start) & (df.index <= asof)].copy())
    return out[0], out[1]


# --------------------------------------------------------------------------- #
# Synthetic world — planted decision-day effect (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(drift: float = 0.0, seed: int = 645,
                    start: str = "2005-01-03", end: str = "2026-06-30",
                    mu: float = 0.0, sig: float = 0.011,
                    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Deterministic AR(1)-in-level daily price index with a TUNABLE planted decision-day drift.

    Daily log-returns are i.i.d. Gaussian noise (``sig``, ~FEZ's typical daily vol) around a
    flat mean (``mu``); every 22nd business day is a scheduled "decision day" (approximating
    the ~6-week ECB cadence). On those days the return gets an EXTRA ``drift`` (the planted
    announcement effect). ``drift = 0`` is the null world: decision days are statistically
    identical to every other day, and the Welch split must NOT reach significance.

    Business-day index, span ~21.5 years — far below the 250-year pandas ns-timestamp trap.
    Returns (frame with a Close column, decision-day DatetimeIndex).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    is_dec = np.zeros(n, dtype=bool)
    is_dec[21::22] = True                      # scheduled, evenly spaced pseudo-ECB days

    ret = rng.normal(mu, sig, n)
    ret[is_dec] += drift                       # the planted decision-day effect
    ret[0] = 0.0
    price = 100.0 * np.exp(np.cumsum(ret))
    close = pd.DataFrame({"Close": price}, index=idx)
    return close, idx[is_dec]
