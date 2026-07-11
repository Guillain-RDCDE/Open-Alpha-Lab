"""Data layer for Study 643 — Payrolls-Day-Effect.

Two ingredients, both offline-friendly once cached:

* **Real tape.** Daily SPY raw OHLC (Open/High/Low/Close) plus the adjusted close,
  from yfinance (no key), cached as CSV under the study's own ``_cache/``. Raw bars
  drive the realized high-low range; the adjusted close drives the close-to-close
  return used everywhere else.

* **The Nonfarm Payrolls (Employment Situation) release calendar, hardcoded.** The
  BLS publishes its release schedule **years in advance** — usually the first
  Friday of the month, at 8:30 am ET (before the 9:30 am open), with documented
  exceptions (July-4th-week Thursdays, shutdown delays, one 2026 schedule slip).
  These are the **actual** release dates, not a "first Friday" pattern
  reconstruction: the identical, source-verified table already used by sibling
  study [602-macro-announcement-premium](../../602-macro-announcement-premium/)
  (BLS archived-news-release index + the official ``histreleasedates.pdf``, 19/19
  overlapping dates agree across the two sources), trimmed here to the study's
  as-of. **Table starts 1997-01** (not 1994) — extending it further back would mean
  re-deriving unverified dates instead of reusing the sourced ones; 353 releases
  and 29+ years is still ample sample.

* **Synthetic world.** A deterministic, seeded random-walk daily return series with
  a TUNABLE planted release-day effect (knob ``edge``, in daily log-return units):
  on scheduled "release days" (every 21st business day, mimicking the ~monthly
  cadence) the mean return gets an extra ``edge``. ``edge = 0`` is the null world —
  release days statistically identical to the rest; the Welch machinery must NOT
  manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
SPY_CACHE = os.path.join(CACHE_DIR, "pde_spy.csv")

START = "1997-01-02"        # first NFP release in the hardcoded table is 1997-01-10
AS_OF = "2026-06-30"        # last complete month at publication (2026-07-10)

# --------------------------------------------------------------------------- #
# Hardcoded actual Employment Situation (Nonfarm Payrolls) release dates,
# 1997-01 -> 2026-06. Source: BLS archived-news-release index (empsit) cross-checked
# against the official histreleasedates.pdf (see module docstring). Identical table
# to sibling study 602-macro-announcement-premium's NFP_DATES, trimmed to this
# study's as-of (2026-06-30). 344/353 fall on a Friday; every non-Friday is a
# documented exception (July-4th-week Thursdays, the 1998-11-05 early release,
# shutdown-delayed reports 2013-10-22 / 2025-11-20 / 2025-12-16, and the 2026-02-11
# schedule slip).
# --------------------------------------------------------------------------- #
NFP_DATES = pd.to_datetime("""
1997-01-10 1997-02-07 1997-03-07 1997-04-04 1997-05-02 1997-06-06
1997-07-03 1997-08-01 1997-09-05 1997-10-03 1997-11-07 1997-12-05
1998-01-09 1998-02-06 1998-03-06 1998-04-03 1998-05-08 1998-06-05
1998-07-02 1998-08-07 1998-09-04 1998-10-02 1998-11-05 1998-12-04
1999-01-08 1999-02-05 1999-03-05 1999-04-02 1999-05-07 1999-06-04
1999-07-02 1999-08-06 1999-09-03 1999-10-08 1999-11-05 1999-12-03
2000-01-07 2000-02-04 2000-03-03 2000-04-07 2000-05-05 2000-06-02
2000-07-07 2000-08-04 2000-09-01 2000-10-06 2000-11-03 2000-12-08
2001-01-05 2001-02-02 2001-03-09 2001-04-06 2001-05-04 2001-06-01
2001-07-06 2001-08-03 2001-09-07 2001-10-05 2001-11-02 2001-12-07
2002-01-04 2002-02-01 2002-03-08 2002-04-05 2002-05-03 2002-06-07
2002-07-05 2002-08-02 2002-09-06 2002-10-04 2002-11-01 2002-12-06
2003-01-10 2003-02-07 2003-03-07 2003-04-04 2003-05-02 2003-06-06
2003-07-03 2003-08-01 2003-09-05 2003-10-03 2003-11-07 2003-12-05
2004-01-09 2004-02-06 2004-03-05 2004-04-02 2004-05-07 2004-06-04
2004-07-02 2004-08-06 2004-09-03 2004-10-08 2004-11-05 2004-12-03
2005-01-07 2005-02-04 2005-03-04 2005-04-01 2005-05-06 2005-06-03
2005-07-08 2005-08-05 2005-09-02 2005-10-07 2005-11-04 2005-12-02
2006-01-06 2006-02-03 2006-03-10 2006-04-07 2006-05-05 2006-06-02
2006-07-07 2006-08-04 2006-09-01 2006-10-06 2006-11-03 2006-12-08
2007-01-05 2007-02-02 2007-03-09 2007-04-06 2007-05-04 2007-06-01
2007-07-06 2007-08-03 2007-09-07 2007-10-05 2007-11-02 2007-12-07
2008-01-04 2008-02-01 2008-03-07 2008-04-04 2008-05-02 2008-06-06
2008-07-03 2008-08-01 2008-09-05 2008-10-03 2008-11-07 2008-12-05
2009-01-09 2009-02-06 2009-03-06 2009-04-03 2009-05-08 2009-06-05
2009-07-02 2009-08-07 2009-09-04 2009-10-02 2009-11-06 2009-12-04
2010-01-08 2010-02-05 2010-03-05 2010-04-02 2010-05-07 2010-06-04
2010-07-02 2010-08-06 2010-09-03 2010-10-08 2010-11-05 2010-12-03
2011-01-07 2011-02-04 2011-03-04 2011-04-01 2011-05-06 2011-06-03
2011-07-08 2011-08-05 2011-09-02 2011-10-07 2011-11-04 2011-12-02
2012-01-06 2012-02-03 2012-03-09 2012-04-06 2012-05-04 2012-06-01
2012-07-06 2012-08-03 2012-09-07 2012-10-05 2012-11-02 2012-12-07
2013-01-04 2013-02-01 2013-03-08 2013-04-05 2013-05-03 2013-06-07
2013-07-05 2013-08-02 2013-09-06 2013-10-22 2013-11-08 2013-12-06
2014-01-10 2014-02-07 2014-03-07 2014-04-04 2014-05-02 2014-06-06
2014-07-03 2014-08-01 2014-09-05 2014-10-03 2014-11-07 2014-12-05
2015-01-09 2015-02-06 2015-03-06 2015-04-03 2015-05-08 2015-06-05
2015-07-02 2015-08-07 2015-09-04 2015-10-02 2015-11-06 2015-12-04
2016-01-08 2016-02-05 2016-03-04 2016-04-01 2016-05-06 2016-06-03
2016-07-08 2016-08-05 2016-09-02 2016-10-07 2016-11-04 2016-12-02
2017-01-06 2017-02-03 2017-03-10 2017-04-07 2017-05-05 2017-06-02
2017-07-07 2017-08-04 2017-09-01 2017-10-06 2017-11-03 2017-12-08
2018-01-05 2018-02-02 2018-03-09 2018-04-06 2018-05-04 2018-06-01
2018-07-06 2018-08-03 2018-09-07 2018-10-05 2018-11-02 2018-12-07
2019-01-04 2019-02-01 2019-03-08 2019-04-05 2019-05-03 2019-06-07
2019-07-05 2019-08-02 2019-09-06 2019-10-04 2019-11-01 2019-12-06
2020-01-10 2020-02-07 2020-03-06 2020-04-03 2020-05-08 2020-06-05
2020-07-02 2020-08-07 2020-09-04 2020-10-02 2020-11-06 2020-12-04
2021-01-08 2021-02-05 2021-03-05 2021-04-02 2021-05-07 2021-06-04
2021-07-02 2021-08-06 2021-09-03 2021-10-08 2021-11-05 2021-12-03
2022-01-07 2022-02-04 2022-03-04 2022-04-01 2022-05-06 2022-06-03
2022-07-08 2022-08-05 2022-09-02 2022-10-07 2022-11-04 2022-12-02
2023-01-06 2023-02-03 2023-03-10 2023-04-07 2023-05-05 2023-06-02
2023-07-07 2023-08-04 2023-09-01 2023-10-06 2023-11-03 2023-12-08
2024-01-05 2024-02-02 2024-03-08 2024-04-05 2024-05-03 2024-06-07
2024-07-05 2024-08-02 2024-09-06 2024-10-04 2024-11-01 2024-12-06
2025-01-10 2025-02-07 2025-03-07 2025-04-04 2025-05-02 2025-06-06
2025-07-03 2025-08-01 2025-09-05 2025-11-20 2025-12-16
2026-01-09 2026-02-11 2026-03-06 2026-04-03 2026-05-08 2026-06-05
""".split())


def nfp_calendar(start: str = START, end: str = AS_OF) -> pd.DatetimeIndex:
    """Actual scheduled NFP release dates inside [start, end], sorted."""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return pd.DatetimeIndex(sorted(d for d in NFP_DATES if lo <= d <= hi))


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "1996-06-01", end: str = "2026-07-01") -> None:
    """Download SPY raw OHLC + adjusted close; cache them. Network; run once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    raw = yf.download("SPY", start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    spy = pd.DataFrame({
        "Open": raw["Open"], "High": raw["High"], "Low": raw["Low"],
        "Close": raw["Close"],
        "AdjClose": raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"],
    }).dropna(how="all")
    spy.to_csv(SPY_CACHE)


def have_real() -> bool:
    return os.path.exists(SPY_CACHE)


def load_real(start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Cached SPY raw+adjusted frame, sliced to [start, asof]."""
    df = pd.read_csv(SPY_CACHE, index_col=0, parse_dates=True).sort_index()
    return df.loc[(df.index >= start) & (df.index <= asof)].copy()


# --------------------------------------------------------------------------- #
# Announcement-day mapping — release lands before the open; the trading session
# that "carries" it is the release date itself if it's a trading day, else the
# NEXT trading session (e.g. an NFP falling on a market holiday).
# --------------------------------------------------------------------------- #
def map_to_sessions(index: pd.DatetimeIndex, cal: pd.DatetimeIndex
                    ) -> tuple[pd.DatetimeIndex, int]:
    """Map each release date onto its trading session (next session if a holiday).

    Returns (mapped DatetimeIndex of trading days flagged, count forward-mapped).
    """
    mapped = []
    n_mapped = 0
    lo, hi = index.min(), index.max()
    for d in cal:
        if d < lo or d > hi:
            continue
        pos = index.searchsorted(d, side="left")
        if pos >= len(index):
            continue
        if index[pos] != d:
            n_mapped += 1
        mapped.append(index[pos])
    return pd.DatetimeIndex(sorted(set(mapped))), n_mapped


# --------------------------------------------------------------------------- #
# Synthetic world — planted release-day effect (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(edge: float = 0.0, seed: int = 643,
                    n_days: int = 7600, mu: float = 0.00030, sig: float = 0.011,
                    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Deterministic random-walk daily return series with a TUNABLE planted
    release-day effect.

    Every 21st business day is a scheduled "release day" (mimics the ~monthly NFP
    cadence); on those days the mean daily log return gets an extra ``edge``.
    ``edge = 0`` is the null world: release days are statistically identical to
    every other day, and the Welch split must NOT reach significance.

    Business-day index, span ~30 years — far below the 250-year pandas
    ns-timestamp trap. Returns (frame with Close/High/Low columns, release-day
    DatetimeIndex).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1997-01-02", periods=n_days)
    n = len(idx)
    is_rel = np.zeros(n, dtype=bool)
    is_rel[20::21] = True                      # scheduled, evenly spaced pseudo-NFP days

    ret = rng.normal(mu, sig, size=n)
    ret[is_rel] += edge                        # the planted release-day effect
    hl_spread = np.abs(rng.normal(0.0, sig * 0.6, size=n))  # unconditional chop —
    # the range detector gets no planted edge; it exists only so day_frame-style
    # code runs on the synthetic world, not as a range positive control

    close = 100.0 * np.exp(np.cumsum(ret))
    prev_close = np.roll(close, 1)
    prev_close[0] = 100.0
    high = np.maximum(close, prev_close) * (1.0 + hl_spread)
    low = np.minimum(close, prev_close) * (1.0 - hl_spread)

    df = pd.DataFrame({"Close": close, "High": high, "Low": low,
                       "AdjClose": close}, index=idx)
    return df, idx[is_rel]
