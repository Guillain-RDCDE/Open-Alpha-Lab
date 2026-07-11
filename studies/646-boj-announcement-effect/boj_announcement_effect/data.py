"""Data layer for Study 646 — BoJ Announcement Effect.

Three ingredients, all offline-friendly once cached:

* **Real tape.** Daily EWJ (iShares MSCI Japan ETF, unhedged, USD-denominated) raw OHLC and
  daily USDJPY (``JPY=X``) OHLC, both from yfinance (no key), cached as CSV under the study's
  own ``_cache/``.

* **The BoJ Monetary Policy Meeting (MPM) decision calendar, hardcoded.** Every date on which
  the Bank of Japan's Policy Board released a monetary-policy decision/statement, 2005 -> 2026.
  Unlike the FOMC-vol-crush sibling (which deliberately excludes emergency/inter-meeting
  actions), this table keeps **every** decision/statement date on record, scheduled or not —
  the claim under test explicitly includes the "surprise" NIRP/YCC episodes, so filtering them
  out would remove the very events the folklore is about. Meeting frequency itself is a fact
  worth knowing: the BoJ met roughly monthly (14-18 statements/year) through 2015, then
  compressed to 8 scheduled meetings/year from 2016 onward to align with global peers (plus a
  couple of extra COVID-era emergency meetings in 2020). Source: Bank of Japan official
  archives — "Statements on Monetary Policy" (boj.or.jp/en/mopo/mpmdeci/state_all/) for
  2005-2009 and "Past Monetary Policy Meetings" (boj.or.jp/en/mopo/mpmsche_minu/past.htm) for
  2010-2026 (two-day meetings: the date given is the **second**/announcement day).

* **Named surprise tail events.** A short hand-picked table of the episodes the folklore
  actually points to — NIRP (2016-01-29), the introduction of Yield Curve Control
  (2016-09-21), the December-2022 YCC band-widening shock, the July-2023 and October-2023 YCC
  flexibility tweaks, the March-2024 NIRP/YCC exit and the July-2024 hike that helped trigger
  the August-2024 carry unwind. Used only to *illustrate* the tail, never to certify the
  average-day signal.

* **Synthetic world.** A deterministic, seeded i.i.d.-return world with a TUNABLE planted
  decision-day mean shift (knob ``effect``): on scheduled "decision days" (every 12th business
  day) daily log returns get an added drift. ``effect = 0`` is the null world — decision days
  statistically identical to the rest; the Welch machinery must NOT manufacture significance
  from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
EWJ_CACHE = os.path.join(CACHE_DIR, "boj_ewj.csv")
JPY_CACHE = os.path.join(CACHE_DIR, "boj_jpy.csv")

START = "2005-01-01"       # EWJ (inception 2001) and JPY=X both trade cleanly by here
AS_OF = "2026-06-30"       # last complete month at publication (2026-07-10)
NIRP_YCC_SPLIT = "2016-01-29"  # NIRP announcement — start of the "surprise regime" era

# --------------------------------------------------------------------------- #
# Hardcoded BoJ Monetary Policy Meeting decision/statement dates, 2005 -> 2026.
# Every statement date on record (scheduled AND the rare inter-meeting/emergency ones) — the
# claim under test is explicitly about the surprise-driven eras, so nothing is filtered out.
# For two-day meetings the date given is the SECOND (announcement) day. Source: Bank of Japan
# official archives (boj.or.jp/en/mopo/mpmdeci/state_all/, boj.or.jp/en/mopo/mpmsche_minu/past.htm).
# --------------------------------------------------------------------------- #
BOJ_DATES = pd.to_datetime("""
2005-01-19 2005-02-17 2005-03-16 2005-04-06 2005-04-28 2005-05-20 2005-06-15 2005-07-13
2005-07-27 2005-08-09 2005-09-08 2005-10-12 2005-10-31 2005-11-18 2005-12-16
2006-01-20 2006-02-09 2006-03-09 2006-04-11 2006-04-28 2006-05-19 2006-06-15 2006-07-14
2006-08-11 2006-09-08 2006-10-13 2006-10-31 2006-11-16 2006-12-19
2007-01-18 2007-02-21 2007-03-20 2007-04-10 2007-04-27 2007-05-17 2007-06-15 2007-07-12
2007-08-23 2007-09-19 2007-10-11 2007-10-31 2007-11-13 2007-12-20
2008-01-22 2008-02-15 2008-03-07 2008-04-09 2008-04-30 2008-05-20 2008-06-13 2008-07-15
2008-08-19 2008-09-17 2008-09-18 2008-09-29 2008-10-07 2008-10-14 2008-10-31 2008-11-21
2008-12-02 2008-12-19
2009-01-22 2009-02-19 2009-03-18 2009-04-07 2009-04-30 2009-05-22 2009-06-16 2009-07-15
2009-08-11 2009-09-17 2009-10-14 2009-10-30 2009-11-20 2009-12-01 2009-12-18
2010-01-26 2010-02-18 2010-03-17 2010-04-07 2010-04-30 2010-05-10 2010-05-21 2010-06-15
2010-07-15 2010-08-10 2010-08-30 2010-09-07 2010-10-05 2010-10-28 2010-11-05 2010-12-21
2011-01-25 2011-02-15 2011-03-14 2011-04-07 2011-04-28 2011-05-20 2011-06-14 2011-07-12
2011-08-04 2011-09-07 2011-10-07 2011-10-27 2011-11-16 2011-11-30 2011-12-21
2012-01-24 2012-02-14 2012-03-13 2012-04-10 2012-04-27 2012-05-23 2012-06-15 2012-07-12
2012-08-09 2012-09-19 2012-10-05 2012-10-30 2012-11-20 2012-12-20
2013-01-22 2013-02-14 2013-03-07 2013-04-04 2013-04-26 2013-05-22 2013-06-11 2013-07-11
2013-08-08 2013-09-05 2013-10-04 2013-10-31 2013-11-21 2013-12-20
2014-01-22 2014-02-18 2014-03-11 2014-04-08 2014-04-30 2014-05-21 2014-06-13 2014-07-15
2014-08-08 2014-09-04 2014-10-07 2014-10-31 2014-11-19 2014-12-19
2015-01-21 2015-02-18 2015-03-17 2015-04-08 2015-04-30 2015-05-22 2015-06-19 2015-07-15
2015-08-07 2015-09-15 2015-10-07 2015-10-30 2015-11-19 2015-12-18
2016-01-29 2016-03-15 2016-04-28 2016-06-16 2016-07-29 2016-09-21 2016-11-01 2016-12-20
2017-01-31 2017-03-16 2017-04-27 2017-06-16 2017-07-20 2017-09-21 2017-10-31 2017-12-21
2018-01-23 2018-03-09 2018-04-27 2018-06-15 2018-07-31 2018-09-19 2018-10-31 2018-12-20
2019-01-23 2019-03-15 2019-04-25 2019-06-20 2019-07-30 2019-09-19 2019-10-31 2019-12-19
2020-01-21 2020-03-16 2020-04-27 2020-05-22 2020-06-16 2020-07-15 2020-09-17 2020-10-29
2020-12-18
2021-01-21 2021-03-19 2021-04-27 2021-06-18 2021-07-16 2021-09-22 2021-10-28 2021-12-17
2022-01-18 2022-03-18 2022-04-28 2022-06-17 2022-07-21 2022-09-22 2022-10-28 2022-12-20
2023-01-18 2023-03-10 2023-04-28 2023-06-16 2023-07-28 2023-09-22 2023-10-31 2023-12-19
2024-01-23 2024-03-19 2024-04-26 2024-06-14 2024-07-31 2024-09-20 2024-10-31 2024-12-19
2025-01-24 2025-03-19 2025-05-01 2025-06-17 2025-07-31 2025-09-19 2025-10-30 2025-12-19
2026-01-23 2026-03-19 2026-04-28 2026-06-16
""".split())

# Hand-picked surprise tail events (label, why). Illustrative only — the average-day Welch
# split is the certified number; these are quoted to show *what the tail looks like*.
SURPRISE_EVENTS = [
    ("2016-01-29", "NIRP announced (-0.1% on policy-rate balances)"),
    ("2016-09-21", "QQE with Yield Curve Control introduced"),
    ("2022-12-20", "YCC band widened +-0.25% -> +-0.50% (the \"Kuroda shock\")"),
    ("2023-07-28", "YCC made more flexible (reference, not hard cap)"),
    ("2023-10-31", "YCC further loosened (1.0% a \"reference\" upper bound)"),
    ("2024-03-19", "NIRP and YCC formally ended; first hike since 2007"),
    ("2024-07-31", "Policy rate raised to ~0.25% (helped trigger the Aug-2024 carry unwind)"),
]


def boj_calendar(start: str = START, end: str = AS_OF) -> pd.DatetimeIndex:
    """BoJ decision/statement dates inside [start, end], sorted."""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return pd.DatetimeIndex(sorted(d for d in BOJ_DATES if lo <= d <= hi))


def surprise_calendar() -> pd.DataFrame:
    """The hand-picked tail-event table as a small DataFrame (date, label)."""
    rows = [(pd.Timestamp(d), lbl) for d, lbl in SURPRISE_EVENTS]
    return pd.DataFrame(rows, columns=["date", "label"]).set_index("date")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2004-06-01", end: str = "2026-07-01") -> None:
    """Download EWJ raw OHLC and USDJPY (JPY=X) OHLC; cache them. Network; once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    ewj = yf.download("EWJ", start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(ewj.columns, pd.MultiIndex):
        ewj.columns = ewj.columns.get_level_values(0)
    out = pd.DataFrame({
        "Open": ewj["Open"], "High": ewj["High"], "Low": ewj["Low"], "Close": ewj["Close"],
        "AdjClose": ewj["Adj Close"] if "Adj Close" in ewj.columns else ewj["Close"],
    }).dropna(how="all")
    out.to_csv(EWJ_CACHE)

    jpy = yf.download("JPY=X", start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(jpy.columns, pd.MultiIndex):
        jpy.columns = jpy.columns.get_level_values(0)
    jpy[["Open", "High", "Low", "Close"]].dropna(how="all").to_csv(JPY_CACHE)


def have_real() -> bool:
    return all(os.path.exists(p) for p in (EWJ_CACHE, JPY_CACHE))


def load_real(start: str = START, asof: str = AS_OF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached (ewj, jpy) frames, sliced to [start, asof]."""
    out = []
    for path in (EWJ_CACHE, JPY_CACHE):
        df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        out.append(df.loc[(df.index >= start) & (df.index <= asof)].copy())
    return out[0], out[1]


# --------------------------------------------------------------------------- #
# Synthetic world — planted decision-day mean shift (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(effect: float = 0.0, seed: int = 646,
                    start: str = "2005-01-01", end: str = "2026-06-30",
                    sigma: float = 0.009,
                    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Deterministic i.i.d.-return price world with a TUNABLE planted decision-day mean shift.

    Daily log returns are N(0, sigma). Every 12th business day is a scheduled
    "decision day" (roughly the ~21/yr average BoJ cadence over the full 2005-2026 tape); on
    those days an EXTRA ``effect`` (log-return units) is added. ``effect = 0`` is the null
    world: decision days are statistically identical to every other day, and the Welch split
    must NOT reach significance.

    Business-day index, span ~21.5 years — far below the 250-year pandas ns-timestamp trap.
    Returns (frame with a Close column, decision-day DatetimeIndex).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    is_dec = np.zeros(n, dtype=bool)
    is_dec[11::12] = True                      # scheduled, evenly spaced pseudo-BoJ days

    ret = rng.normal(0.0, sigma, n)
    ret[is_dec] += effect                       # the planted decision-day mean shift
    ret[0] = 0.0
    close = 100.0 * np.exp(np.cumsum(ret))
    frame = pd.DataFrame({"Close": close}, index=idx)
    return frame, idx[is_dec]
