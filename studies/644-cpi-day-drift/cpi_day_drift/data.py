"""Data layer for Study 644 — CPI-Day-Drift.

Three ingredients, all offline-friendly once cached:

* **Real tape.** Daily SPY raw OHLC (Open/High/Low/Close) plus adjusted close, and daily
  TLT raw OHLC plus adjusted close (the 20+yr Treasury ETF — inception 2002-07-30, so its
  sample is shorter than SPY's), both from yfinance (no key), cached as CSV under the
  study's own ``_cache/``. Raw bars drive the realized high-low range; adjusted closes
  drive every close-to-close return.

* **The CPI (Consumer Price Index) release calendar, hardcoded.** The BLS publishes its
  release schedule **years in advance** — 8:30 am ET, before the 9:30 am open. These are
  the identical, source-verified release dates already used by sibling study
  [602-macro-announcement-premium](../../602-macro-announcement-premium/) (BLS archived
  news-release index for CPI, cross-checked against the official ``histreleasedates.pdf``;
  19/19 overlapping dates agree across the two sources), trimmed here to this study's
  as-of. **Table starts 1997-01** (not earlier) — extending it further back would mean
  re-deriving unverified dates instead of reusing the sourced ones; 353 releases and 29+
  years is still ample sample. Known quirk carried from the source table: no November-2025
  release — the September-2025 CPI was delayed to 2025-10-24 by the federal shutdown and
  the October-2025 report was never separately published; the next release (2025-12-18)
  covered both months combined.

* **Synthetic world.** A deterministic, seeded daily-return process with TWO independently
  tunable planted CPI-day effects: a mean-return shift (``mu_shift``, the *direction*
  knob) and a volatility multiplier (``vol_mult``, the *realized-range* knob) — because
  the claim under test genuinely splits into a direction question and a loudness question,
  and the positive control has to prove the harness can catch either one without the other.
  ``mu_shift = 0`` / ``vol_mult = 1`` is the null world: the Welch machinery must NOT
  manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
SPY_CACHE = os.path.join(CACHE_DIR, "cdd_spy.csv")
TLT_CACHE = os.path.join(CACHE_DIR, "cdd_tlt.csv")

START = "1997-01-02"          # first CPI release in the hardcoded table is 1997-01-14
AS_OF = "2026-06-30"           # last complete month at publication (2026-07-10)
TLT_START = "2002-07-30"       # TLT inception
REGIME_SPLIT = "2022-01-01"    # justified split: the Fed's Dec-2021 hawkish pivot (accel-
                                # erated taper + 2022 dot-plot hikes announced at the
                                # 2021-12-15 FOMC) made every CPI print a direct input to
                                # the rate-hike reaction function — the start of the
                                # "CPI day is the biggest day of the month" era. Chosen
                                # from the FOMC calendar, not fit to the return data.

# --------------------------------------------------------------------------- #
# Hardcoded actual CPI (all-urban, "CPI-U") news-release dates, 1997-01 -> 2026-06.
# Source: BLS archived-news-release index (cpi) cross-checked against the official
# histreleasedates.pdf (see module docstring). Identical table to sibling study
# 602-macro-announcement-premium's CPI_DATES, trimmed to this study's as-of
# (2026-06-30). These are actual release days, including shutdown-delayed ones
# (2013-10-30, 2025-10-24) and the missing Nov-2025 release (named above).
# --------------------------------------------------------------------------- #
CPI_DATES = pd.to_datetime("""
1997-01-14 1997-02-19 1997-03-19 1997-04-15 1997-05-15 1997-06-17
1997-07-16 1997-08-14 1997-09-16 1997-10-16 1997-11-18 1997-12-16
1998-01-13 1998-02-24 1998-03-19 1998-04-14 1998-05-14 1998-06-16
1998-07-14 1998-08-18 1998-09-17 1998-10-16 1998-11-17 1998-12-15
1999-01-14 1999-02-19 1999-03-18 1999-04-13 1999-05-14 1999-06-16
1999-07-15 1999-08-17 1999-09-15 1999-10-19 1999-11-17 1999-12-14
2000-01-14 2000-02-18 2000-03-17 2000-04-14 2000-05-16 2000-06-14
2000-07-18 2000-08-16 2000-09-15 2000-10-18 2000-11-16 2000-12-15
2001-01-17 2001-02-21 2001-03-21 2001-04-17 2001-05-16 2001-06-15
2001-07-18 2001-08-16 2001-09-18 2001-10-19 2001-11-16 2001-12-14
2002-01-16 2002-02-20 2002-03-21 2002-04-16 2002-05-15 2002-06-18
2002-07-19 2002-08-16 2002-09-18 2002-10-18 2002-11-19 2002-12-17
2003-01-16 2003-02-21 2003-03-21 2003-04-16 2003-05-16 2003-06-17
2003-07-16 2003-08-15 2003-09-16 2003-10-16 2003-11-18 2003-12-16
2004-01-15 2004-02-20 2004-03-17 2004-04-14 2004-05-14 2004-06-15
2004-07-16 2004-08-17 2004-09-16 2004-10-19 2004-11-17 2004-12-17
2005-01-19 2005-02-23 2005-03-23 2005-04-20 2005-05-18 2005-06-15
2005-07-14 2005-08-16 2005-09-15 2005-10-14 2005-11-16 2005-12-15
2006-01-18 2006-02-22 2006-03-16 2006-04-19 2006-05-17 2006-06-14
2006-07-19 2006-08-16 2006-09-15 2006-10-18 2006-11-16 2006-12-15
2007-01-18 2007-02-21 2007-03-16 2007-04-17 2007-05-15 2007-06-15
2007-07-18 2007-08-15 2007-09-19 2007-10-17 2007-11-15 2007-12-14
2008-01-16 2008-02-20 2008-03-14 2008-04-16 2008-05-14 2008-06-13
2008-07-16 2008-08-14 2008-09-16 2008-10-16 2008-11-19 2008-12-16
2009-01-16 2009-02-20 2009-03-18 2009-04-15 2009-05-15 2009-06-17
2009-07-15 2009-08-14 2009-09-16 2009-10-15 2009-11-18 2009-12-16
2010-01-15 2010-02-19 2010-03-18 2010-04-14 2010-05-19 2010-06-17
2010-07-16 2010-08-13 2010-09-17 2010-10-15 2010-11-17 2010-12-15
2011-01-14 2011-02-17 2011-03-17 2011-04-15 2011-05-13 2011-06-15
2011-07-15 2011-08-18 2011-09-15 2011-10-19 2011-11-16 2011-12-16
2012-01-19 2012-02-17 2012-03-16 2012-04-13 2012-05-15 2012-06-14
2012-07-17 2012-08-15 2012-09-14 2012-10-16 2012-11-15 2012-12-14
2013-01-16 2013-02-21 2013-03-15 2013-04-16 2013-05-16 2013-06-18
2013-07-16 2013-08-15 2013-09-17 2013-10-30 2013-11-20 2013-12-17
2014-01-16 2014-02-20 2014-03-18 2014-04-15 2014-05-15 2014-06-17
2014-07-22 2014-08-19 2014-09-17 2014-10-22 2014-11-20 2014-12-17
2015-01-16 2015-02-26 2015-03-24 2015-04-17 2015-05-22 2015-06-18
2015-07-17 2015-08-19 2015-09-16 2015-10-15 2015-11-17 2015-12-15
2016-01-20 2016-02-19 2016-03-16 2016-04-14 2016-05-17 2016-06-16
2016-07-15 2016-08-16 2016-09-16 2016-10-18 2016-11-17 2016-12-15
2017-01-18 2017-02-15 2017-03-15 2017-04-14 2017-05-12 2017-06-14
2017-07-14 2017-08-11 2017-09-14 2017-10-13 2017-11-15 2017-12-13
2018-01-12 2018-02-14 2018-03-13 2018-04-11 2018-05-10 2018-06-12
2018-07-12 2018-08-10 2018-09-13 2018-10-11 2018-11-14 2018-12-12
2019-01-11 2019-02-13 2019-03-12 2019-04-10 2019-05-10 2019-06-12
2019-07-11 2019-08-13 2019-09-12 2019-10-10 2019-11-13 2019-12-11
2020-01-14 2020-02-13 2020-03-11 2020-04-10 2020-05-12 2020-06-10
2020-07-14 2020-08-12 2020-09-11 2020-10-13 2020-11-12 2020-12-10
2021-01-13 2021-02-10 2021-03-10 2021-04-13 2021-05-12 2021-06-10
2021-07-13 2021-08-11 2021-09-14 2021-10-13 2021-11-10 2021-12-10
2022-01-12 2022-02-10 2022-03-10 2022-04-12 2022-05-11 2022-06-10
2022-07-13 2022-08-10 2022-09-13 2022-10-13 2022-11-10 2022-12-13
2023-01-12 2023-02-14 2023-03-14 2023-04-12 2023-05-10 2023-06-13
2023-07-12 2023-08-10 2023-09-13 2023-10-12 2023-11-14 2023-12-12
2024-01-11 2024-02-13 2024-03-12 2024-04-10 2024-05-15 2024-06-12
2024-07-11 2024-08-14 2024-09-11 2024-10-10 2024-11-13 2024-12-11
2025-01-15 2025-02-12 2025-03-12 2025-04-10 2025-05-13 2025-06-11
2025-07-15 2025-08-12 2025-09-11 2025-10-24 2025-12-18
2026-01-13 2026-02-13 2026-03-11 2026-04-10 2026-05-12 2026-06-10
""".split())


def cpi_calendar(start: str = START, end: str = AS_OF) -> pd.DatetimeIndex:
    """Actual CPI release dates inside [start, end], sorted."""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return pd.DatetimeIndex(sorted(d for d in CPI_DATES if lo <= d <= hi))


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "1996-06-01", end: str = "2026-07-01") -> None:
    """Download SPY + TLT raw OHLC and adjusted closes; cache them. Network; once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for ticker, path in ((("SPY", SPY_CACHE)), (("TLT", TLT_CACHE))):
        raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        frame = pd.DataFrame({
            "Open": raw["Open"], "High": raw["High"], "Low": raw["Low"],
            "Close": raw["Close"],
            "AdjClose": raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"],
        }).dropna(how="all")
        frame.to_csv(path)


def have_real() -> bool:
    return os.path.exists(SPY_CACHE) and os.path.exists(TLT_CACHE)


def load_real(start: str = START, asof: str = AS_OF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached (SPY, TLT) raw+adjusted frames, sliced to [start, asof]."""
    out = []
    for path in (SPY_CACHE, TLT_CACHE):
        df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        out.append(df.loc[(df.index >= start) & (df.index <= asof)].copy())
    return out[0], out[1]


# --------------------------------------------------------------------------- #
# Announcement-day mapping — CPI lands before the open; the trading session that
# "carries" it is the release date itself if it's a trading day, else the NEXT
# trading session (e.g. a CPI print falling on a market holiday).
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
# Synthetic world — TWO independent planted CPI-day effects (direction + range)
# --------------------------------------------------------------------------- #
def synthetic_world(mu_shift: float = 0.0, vol_mult: float = 1.0, seed: int = 644,
                    n_days: int = 7300, mu: float = 0.00030, sig: float = 0.010,
                    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Deterministic daily-return process with a TUNABLE planted CPI-day return shift
    (``mu_shift``) and a TUNABLE planted CPI-day volatility multiplier (``vol_mult``),
    independently switchable — because this study measures a direction claim and a
    loudness claim separately, and the positive control must prove the harness can
    catch either one without manufacturing the other.

    Every 21st business day is a scheduled "CPI day" (mimics the ~monthly cadence). On
    those days the mean daily log return gets an extra ``mu_shift`` and the daily
    volatility is scaled by ``vol_mult`` (feeding both the return series and a
    Parkinson-style high-low range proxy). ``mu_shift = 0`` / ``vol_mult = 1`` is the
    null world: CPI days are statistically identical to every other day on both
    dimensions, and the Welch splits must NOT reach significance on either.

    Business-day index, span ~28 years — far below the 250-year pandas ns-timestamp
    trap. Returns (frame with Close/High/Low/AdjClose columns, CPI-day DatetimeIndex).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1997-01-02", periods=n_days)
    n = len(idx)
    is_cpi = np.zeros(n, dtype=bool)
    is_cpi[20::21] = True                      # scheduled, evenly spaced pseudo-CPI days

    sigma = np.where(is_cpi, sig * vol_mult, sig)
    ret = rng.normal(0.0, 1.0, size=n) * sigma + mu
    ret[is_cpi] += mu_shift                    # the planted CPI-day mean shift

    close = 100.0 * np.exp(np.cumsum(ret))
    prev_close = np.roll(close, 1)
    prev_close[0] = 100.0
    hl_spread = np.abs(rng.normal(0.0, 1.0, size=n)) * sigma * 1.2  # Parkinson-like proxy
    high = np.maximum(close, prev_close) * (1.0 + hl_spread)
    low = np.minimum(close, prev_close) * (1.0 - hl_spread)

    df = pd.DataFrame({"Close": close, "High": high, "Low": low,
                       "AdjClose": close}, index=idx)
    return df, idx[is_cpi]
