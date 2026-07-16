"""Data layer for Study 780 — Long-Weekend-Drift.

The claim under test: **the pre-holiday session drifts up** — the last trading session
before a US market holiday (the "long-weekend" session) earns an abnormally *positive*
return, so a "buy at the prior close, sell on the pre-holiday close" rule beats an
ordinary day. This is the classic **pre-holiday effect** (Ariel 1990; Lakonishok & Smidt
1988): historically the day before a holiday earned several times the mean daily return.
The interesting question in a modern sample is whether the anomaly *survived* its own
publication, or decayed to folklore the way most calendar effects do.

Why it is a clean calendar test: the NYSE holiday schedule is **published years ahead** —
a "buy K sessions before the holiday, sell on the pre-holiday close" rule is calendar-known
and zero-look-ahead by construction. There is nothing to reconstruct from filings; the
holiday date *is* the event.

Four ingredients:

* **The holiday calendar, hardcoded.** Every NYSE full-day closure 2005->2025 with its real
  date (New Year's, MLK, Presidents/Washington, Good Friday, Memorial, Juneteenth from 2022,
  Independence, Labor, Thanksgiving, Christmas — with the standard weekend-observance shifts).
  Source: NYSE holiday schedule, cross-checked against the actual SPY trading calendar (the
  holiday date is, by construction, a weekday ABSENT from the tape). The pre-holiday
  *session* is then found from the tape as the last SPY close strictly before each holiday,
  so half-days and observance shifts resolve automatically.

* **The tradable instrument (yfinance).** ``SPY`` — the S&P 500 ETF, daily total-return
  (adjusted) close. This is a **single-series, self-benchmarked** calendar effect: there is
  no cross-sectional benchmark leg. "Abnormal" here means *excess over the sample's own mean
  daily return* — i.e. does the pre-holiday day beat an ordinary SPY day, not another asset.

* **No fundamental proxy needed.** The pre-holiday effect is a pure price-path calendar
  claim; the event is a scheduled market closure.

* **Synthetic world.** A deterministic, seeded single-series daily-return world with a
  TUNABLE planted "pre-holiday bump" on synthetic holiday-eve positions. ``bump = 0`` is the
  null world; the one-sample-t machinery must not manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"     # last complete month at publication
INSTRUMENT = "SPY"       # S&P 500 ETF, total-return (adjusted) close

# --------------------------------------------------------------------------- #
# The NYSE full-day-closure calendar, hardcoded: (holiday_date, name) for every scheduled
# market holiday 2005->2025. The published NYSE schedule (known years ahead) makes a
# "buy K sessions before the holiday" rule calendar-known and zero-look-ahead. Dates carry
# the standard weekend-observance shifts (e.g. Jul-4 on a Saturday -> observed Jul-3), and
# Juneteenth appears from 2022 when the NYSE first observed it. The *session* we trade is
# the last SPY close strictly before each date, resolved from the tape (so half-days such
# as an early-close Christmas Eve resolve automatically). Source: NYSE holiday schedule,
# cross-checked against the SPY trading calendar. Note: unscheduled closures (e.g. Hurricane
# Sandy Oct-2012, the Bush/Ford/Carter funeral days) are NOT holidays and are excluded.
# --------------------------------------------------------------------------- #
EVENTS = [
    ("2005-01-17", "MLK"), ("2005-02-21", "Presidents"), ("2005-03-25", "GoodFriday"),
    ("2005-05-30", "Memorial"), ("2005-07-04", "Independence"), ("2005-09-05", "Labor"),
    ("2005-11-24", "Thanksgiving"), ("2005-12-26", "Christmas"), ("2006-01-02", "NewYears"),
    ("2006-01-16", "MLK"), ("2006-02-20", "Presidents"), ("2006-04-14", "GoodFriday"),
    ("2006-05-29", "Memorial"), ("2006-07-04", "Independence"), ("2006-09-04", "Labor"),
    ("2006-11-23", "Thanksgiving"), ("2006-12-25", "Christmas"), ("2007-01-01", "NewYears"),
    ("2007-01-15", "MLK"), ("2007-02-19", "Presidents"), ("2007-04-06", "GoodFriday"),
    ("2007-05-28", "Memorial"), ("2007-07-04", "Independence"), ("2007-09-03", "Labor"),
    ("2007-11-22", "Thanksgiving"), ("2007-12-25", "Christmas"), ("2008-01-01", "NewYears"),
    ("2008-01-21", "MLK"), ("2008-02-18", "Presidents"), ("2008-03-21", "GoodFriday"),
    ("2008-05-26", "Memorial"), ("2008-07-04", "Independence"), ("2008-09-01", "Labor"),
    ("2008-11-27", "Thanksgiving"), ("2008-12-25", "Christmas"), ("2009-01-01", "NewYears"),
    ("2009-01-19", "MLK"), ("2009-02-16", "Presidents"), ("2009-04-10", "GoodFriday"),
    ("2009-05-25", "Memorial"), ("2009-07-03", "Independence"), ("2009-09-07", "Labor"),
    ("2009-11-26", "Thanksgiving"), ("2009-12-25", "Christmas"), ("2010-01-01", "NewYears"),
    ("2010-01-18", "MLK"), ("2010-02-15", "Presidents"), ("2010-04-02", "GoodFriday"),
    ("2010-05-31", "Memorial"), ("2010-07-05", "Independence"), ("2010-09-06", "Labor"),
    ("2010-11-25", "Thanksgiving"), ("2010-12-24", "Christmas"), ("2010-12-31", "NewYears"),
    ("2011-01-17", "MLK"), ("2011-02-21", "Presidents"), ("2011-04-22", "GoodFriday"),
    ("2011-05-30", "Memorial"), ("2011-07-04", "Independence"), ("2011-09-05", "Labor"),
    ("2011-11-24", "Thanksgiving"), ("2011-12-26", "Christmas"), ("2012-01-02", "NewYears"),
    ("2012-01-16", "MLK"), ("2012-02-20", "Presidents"), ("2012-04-06", "GoodFriday"),
    ("2012-05-28", "Memorial"), ("2012-07-04", "Independence"), ("2012-09-03", "Labor"),
    ("2012-11-22", "Thanksgiving"), ("2012-12-25", "Christmas"), ("2013-01-01", "NewYears"),
    ("2013-01-21", "MLK"), ("2013-02-18", "Presidents"), ("2013-03-29", "GoodFriday"),
    ("2013-05-27", "Memorial"), ("2013-07-04", "Independence"), ("2013-09-02", "Labor"),
    ("2013-11-28", "Thanksgiving"), ("2013-12-25", "Christmas"), ("2014-01-01", "NewYears"),
    ("2014-01-20", "MLK"), ("2014-02-17", "Presidents"), ("2014-04-18", "GoodFriday"),
    ("2014-05-26", "Memorial"), ("2014-07-04", "Independence"), ("2014-09-01", "Labor"),
    ("2014-11-27", "Thanksgiving"), ("2014-12-25", "Christmas"), ("2015-01-01", "NewYears"),
    ("2015-01-19", "MLK"), ("2015-02-16", "Presidents"), ("2015-04-03", "GoodFriday"),
    ("2015-05-25", "Memorial"), ("2015-07-03", "Independence"), ("2015-09-07", "Labor"),
    ("2015-11-26", "Thanksgiving"), ("2015-12-25", "Christmas"), ("2016-01-01", "NewYears"),
    ("2016-01-18", "MLK"), ("2016-02-15", "Presidents"), ("2016-03-25", "GoodFriday"),
    ("2016-05-30", "Memorial"), ("2016-07-04", "Independence"), ("2016-09-05", "Labor"),
    ("2016-11-24", "Thanksgiving"), ("2016-12-26", "Christmas"), ("2017-01-02", "NewYears"),
    ("2017-01-16", "MLK"), ("2017-02-20", "Presidents"), ("2017-04-14", "GoodFriday"),
    ("2017-05-29", "Memorial"), ("2017-07-04", "Independence"), ("2017-09-04", "Labor"),
    ("2017-11-23", "Thanksgiving"), ("2017-12-25", "Christmas"), ("2018-01-01", "NewYears"),
    ("2018-01-15", "MLK"), ("2018-02-19", "Presidents"), ("2018-03-30", "GoodFriday"),
    ("2018-05-28", "Memorial"), ("2018-07-04", "Independence"), ("2018-09-03", "Labor"),
    ("2018-11-22", "Thanksgiving"), ("2018-12-25", "Christmas"), ("2019-01-01", "NewYears"),
    ("2019-01-21", "MLK"), ("2019-02-18", "Presidents"), ("2019-04-19", "GoodFriday"),
    ("2019-05-27", "Memorial"), ("2019-07-04", "Independence"), ("2019-09-02", "Labor"),
    ("2019-11-28", "Thanksgiving"), ("2019-12-25", "Christmas"), ("2020-01-01", "NewYears"),
    ("2020-01-20", "MLK"), ("2020-02-17", "Presidents"), ("2020-04-10", "GoodFriday"),
    ("2020-05-25", "Memorial"), ("2020-07-03", "Independence"), ("2020-09-07", "Labor"),
    ("2020-11-26", "Thanksgiving"), ("2020-12-25", "Christmas"), ("2021-01-01", "NewYears"),
    ("2021-01-18", "MLK"), ("2021-02-15", "Presidents"), ("2021-04-02", "GoodFriday"),
    ("2021-05-31", "Memorial"), ("2021-07-05", "Independence"), ("2021-09-06", "Labor"),
    ("2021-11-25", "Thanksgiving"), ("2021-12-24", "Christmas"), ("2021-12-31", "NewYears"),
    ("2022-01-17", "MLK"), ("2022-02-21", "Presidents"), ("2022-04-15", "GoodFriday"),
    ("2022-05-30", "Memorial"), ("2022-06-20", "Juneteenth"), ("2022-07-04", "Independence"),
    ("2022-09-05", "Labor"), ("2022-11-24", "Thanksgiving"), ("2022-12-26", "Christmas"),
    ("2023-01-02", "NewYears"), ("2023-01-16", "MLK"), ("2023-02-20", "Presidents"),
    ("2023-04-07", "GoodFriday"), ("2023-05-29", "Memorial"), ("2023-06-19", "Juneteenth"),
    ("2023-07-04", "Independence"), ("2023-09-04", "Labor"), ("2023-11-23", "Thanksgiving"),
    ("2023-12-25", "Christmas"), ("2024-01-01", "NewYears"), ("2024-01-15", "MLK"),
    ("2024-02-19", "Presidents"), ("2024-03-29", "GoodFriday"), ("2024-05-27", "Memorial"),
    ("2024-06-19", "Juneteenth"), ("2024-07-04", "Independence"), ("2024-09-02", "Labor"),
    ("2024-11-28", "Thanksgiving"), ("2024-12-25", "Christmas"), ("2025-01-01", "NewYears"),
    ("2025-01-20", "MLK"), ("2025-02-17", "Presidents"), ("2025-04-18", "GoodFriday"),
    ("2025-05-26", "Memorial"), ("2025-06-19", "Juneteenth"), ("2025-07-04", "Independence"),
    ("2025-09-01", "Labor"), ("2025-11-27", "Thanksgiving"), ("2025-12-25", "Christmas"),
]


def all_tickers() -> list[str]:
    return [INSTRUMENT]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"longweekend_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2004-06-01", end: str = "2026-07-01", retries: int = 4) -> None:
    """Download adjusted (total-return) daily closes for SPY; cache them.

    Retries with linear backoff — Yahoo rate-limits transient bursts, so a first empty frame
    is usually cured by a short wait rather than a real "no such ticker".
    """
    import time

    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in all_tickers():
        last_err = None
        for attempt in range(retries):
            try:
                d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                d = d[["Close"]].dropna()
                if len(d) > 0:
                    d.to_csv(_cache_path(t))
                    break
                last_err = f"empty frame for {t}"
            except Exception as e:  # noqa: BLE001 -- transient network/rate-limit
                last_err = str(e)
            time.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError(f"fetch failed for {t} after {retries} tries: {last_err}")


def have_real() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in all_tickers())


def load_real(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached {ticker: adjusted-close Series}, each sliced to <= asof."""
    out = {}
    for t in all_tickers():
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        s = df["Close"]
        out[t] = s[s.index <= pd.Timestamp(asof)]
    return out


# --------------------------------------------------------------------------- #
# Synthetic world -- planted pre-holiday bump on a single self-benchmarked series
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, seed: int = 792, n_events: int = 190,
                    n_days: int = 5200, spacing: int = 25,
                    ) -> tuple[pd.Series, list[int]]:
    """Deterministic single-series daily-return world with a planted pre-holiday bump.

    Zero-drift-plus-noise SPY-like daily simple returns; on each synthetic "pre-holiday
    session" (every ``spacing``-th business day) the return gets an EXTRA ``bump`` — a
    planted pre-holiday drift. ``bump = 0`` is the null world. Self-benchmarked: "abnormal"
    is excess over the series' own mean daily return, exactly as on the real tape.

    Business-day integer index (positions 0..n_days). Returns (daily_returns, event_positions)
    where each event position is the pre-holiday session itself.
    """
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.011, n_days)     # mild positive drift + ~1.1% daily vol
    ev_pos = list(range(spacing, n_days - 5, spacing))[:n_events]
    for p in ev_pos:
        r[p] += bump                          # planted pre-holiday drift on the eve session
    idx = pd.RangeIndex(n_days)
    return pd.Series(r, index=idx), ev_pos
