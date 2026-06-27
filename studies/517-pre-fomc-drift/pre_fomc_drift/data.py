"""Data layer for Study 517 — Pre-FOMC-Drift.

Two sources, both offline-friendly once cached:

* **Real tape.** Daily adjusted OHLC for SPY and a fixed survivor basket (yfinance, no key),
  cached under ``_cache/pfd_prices.csv`` (adjusted closes, wide) and ``_cache/pfd_ohlc.csv``
  (SPY open/close, for the overnight-vs-intraday split). The **FOMC announcement calendar**
  is hardcoded below (``FOMC_DATES``) — scheduled meeting *decision* dates, 1994 → 2026.

* **Synthetic.** A deterministic, fixed-seed daily world with a *planted* pre-FOMC drift on
  the day before each (synthetic) meeting. It is the positive control: with ``edge = 0`` the
  pre-FOMC-vs-other test must NOT manufacture significance; with a planted edge it must light
  up. Used only as a faithful-engine / power check — never to support a real-tape stamp.

Pure numpy + pandas + stdlib offline. ``fetch_panel`` (network) builds the cache once and is
never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "pfd_prices.csv")
OHLC_CACHE = os.path.join(CACHE_DIR, "pfd_ohlc.csv")

# Lucca-Moench published in 2015; the standard split tests whether the effect decayed once
# the paper was public. We split at the start of 2012 (post-paper-circulation era).
PUBLICATION_SPLIT = "2012-01-01"

# A transparent fixed basket of large, long-listed US large-caps with deep clean histories
# on yfinance. SURVIVORS — all still trading in 2026 — named on the Signal axis: a fixed
# surviving-names basket cannot include firms delisted/blown up, a mild upward tilt.
BASKET = [
    "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "IBM", "CVX",
    "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT", "MMM",
    "HON", "GE", "ORCL", "PEP", "ABT", "TXN", "WFC", "T", "VZ", "GS",
]

# --------------------------------------------------------------------------- #
# Hardcoded FOMC scheduled-announcement (decision) calendar, 1994 -> 2026.
# These are the days the FOMC *released its decision*. Lucca-Moench measure the return on the
# session that ENDS just before the announcement; in modern practice (post-1994) the statement
# is released ~14:15-14:30 ET on the second day of a two-day meeting, so the "pre-FOMC window"
# is essentially the session up to that release. We tag the announcement date itself; the
# strategy can shift to the prior session for a strict no-look-ahead variant.
# Source: Federal Reserve historical FOMC calendars. Scheduled meetings only (no inter-meeting
# / emergency actions, to keep the calendar clean and pre-known).
# --------------------------------------------------------------------------- #
FOMC_DATES = [
    # 1994
    "1994-02-04", "1994-03-22", "1994-04-18", "1994-05-17", "1994-07-06",
    "1994-08-16", "1994-09-27", "1994-11-15", "1994-12-20",
    # 1995
    "1995-02-01", "1995-03-28", "1995-05-23", "1995-07-06", "1995-08-22",
    "1995-09-26", "1995-11-15", "1995-12-19",
    # 1996
    "1996-01-31", "1996-03-26", "1996-05-21", "1996-07-03", "1996-08-20",
    "1996-09-24", "1996-11-13", "1996-12-17",
    # 1997
    "1997-02-05", "1997-03-25", "1997-05-20", "1997-07-02", "1997-08-19",
    "1997-09-30", "1997-11-12", "1997-12-16",
    # 1998
    "1998-02-04", "1998-03-31", "1998-05-19", "1998-07-01", "1998-08-18",
    "1998-09-29", "1998-11-17", "1998-12-22",
    # 1999
    "1999-02-03", "1999-03-30", "1999-05-18", "1999-06-30", "1999-08-24",
    "1999-10-05", "1999-11-16", "1999-12-21",
    # 2000
    "2000-02-02", "2000-03-21", "2000-05-16", "2000-06-28", "2000-08-22",
    "2000-10-03", "2000-11-15", "2000-12-19",
    # 2001
    "2001-01-31", "2001-03-20", "2001-05-15", "2001-06-27", "2001-08-21",
    "2001-10-02", "2001-11-06", "2001-12-11",
    # 2002
    "2002-01-30", "2002-03-19", "2002-05-07", "2002-06-26", "2002-08-13",
    "2002-09-24", "2002-11-06", "2002-12-10",
    # 2003
    "2003-01-29", "2003-03-18", "2003-05-06", "2003-06-25", "2003-08-12",
    "2003-09-16", "2003-10-28", "2003-12-09",
    # 2004
    "2004-01-28", "2004-03-16", "2004-05-04", "2004-06-30", "2004-08-10",
    "2004-09-21", "2004-11-10", "2004-12-14",
    # 2005
    "2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30", "2005-08-09",
    "2005-09-20", "2005-11-01", "2005-12-13",
    # 2006
    "2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29", "2006-08-08",
    "2006-09-20", "2006-10-25", "2006-12-12",
    # 2007
    "2007-01-31", "2007-03-21", "2007-05-09", "2007-06-28", "2007-08-07",
    "2007-09-18", "2007-10-31", "2007-12-11",
    # 2008
    "2008-01-30", "2008-03-18", "2008-04-30", "2008-06-25", "2008-08-05",
    "2008-09-16", "2008-10-29", "2008-12-16",
    # 2009
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24", "2009-08-12",
    "2009-09-23", "2009-11-04", "2009-12-16",
    # 2010
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23", "2010-08-10",
    "2010-09-21", "2010-11-03", "2010-12-14",
    # 2011
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22", "2011-08-09",
    "2011-09-21", "2011-11-02", "2011-12-13",
    # 2012
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20", "2012-08-01",
    "2012-09-13", "2012-10-24", "2012-12-12",
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19", "2013-07-31",
    "2013-09-18", "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30",
    "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29",
    "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27",
    "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26",
    "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01",
    "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31",
    "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-18", "2020-04-29", "2020-06-10", "2020-07-29",
    "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (scheduled, through mid-year — within our as-of)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
]


def fomc_calendar() -> pd.DatetimeIndex:
    """The hardcoded scheduled-FOMC-announcement calendar as a sorted DatetimeIndex."""
    return pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in FOMC_DATES))


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "1993-01-29", end: str | None = None,
                prices_path: str = PRICES_CACHE, ohlc_path: str = OHLC_CACHE,
                retries: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download SPY + basket adjusted closes and SPY open/close, and cache them.

    Network-only; used once to build the cache. Writes a wide adjusted-close CSV (SPY + the
    survivor basket) and a SPY open/close CSV used for the overnight-vs-intraday split.
    """
    import time

    import yfinance as yf

    tickers = ["SPY"] + BASKET
    last_err = None
    for attempt in range(retries):
        try:
            adj = yf.download(tickers, start=start, end=end, auto_adjust=True,
                              progress=False)["Close"]
            raw = yf.download("SPY", start=start, end=end, auto_adjust=False,
                              progress=False)
            break
        except Exception as e:  # pragma: no cover - network flakiness
            last_err = e
            time.sleep(2 * (attempt + 1))
    else:  # pragma: no cover
        raise RuntimeError(f"yfinance failed after {retries} tries: {last_err}")

    adj = adj.dropna(how="all")
    keep = [c for c in adj.columns if adj[c].notna().mean() >= 0.50]
    prices = adj[keep].copy()
    os.makedirs(os.path.dirname(prices_path), exist_ok=True)
    prices.to_csv(prices_path)

    # SPY raw Open/Close for the overnight (prev close -> open) vs intraday (open -> close) split
    ohlc = pd.DataFrame({
        "Open": raw["Open"].squeeze(),
        "Close": raw["Close"].squeeze(),
        "AdjClose": raw["Adj Close"].squeeze() if "Adj Close" in raw else raw["Close"].squeeze(),
    }).dropna(how="all")
    ohlc.to_csv(ohlc_path)
    return prices, ohlc


def have_real(prices_path: str = PRICES_CACHE, ohlc_path: str = OHLC_CACHE) -> bool:
    return os.path.exists(prices_path) and os.path.exists(ohlc_path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide adjusted-close frame (index = date, columns = SPY + basket tickers)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def load_ohlc(path: str = OHLC_CACHE) -> pd.DataFrame:
    """SPY Open/Close frame for the overnight-vs-intraday split."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def load_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: cached (adjusted-close panel, SPY OHLC) in one call."""
    return load_prices(), load_ohlc()


def tag_pre_fomc(index: pd.DatetimeIndex, shift: int = 0) -> np.ndarray:
    """Boolean mask over ``index`` of the pre-FOMC trading session for each meeting.

    With ``shift = 0`` we tag the **announcement-day** session itself (the modern Lucca-Moench
    convention: the statement lands intraday, so the pre-FOMC window is essentially that
    session's run-up). With ``shift = -1`` we tag the trading session *strictly before* the
    announcement (the conservative no-look-ahead variant). Either way the meeting date is
    pre-known months ahead, so there is no information leakage.
    """
    cal = fomc_calendar()
    mask = np.zeros(len(index), dtype=bool)
    for d in cal:
        pos = index.searchsorted(d, side="left")
        # first trading day on/after the (calendar) announcement date
        if pos >= len(index):
            continue
        # if the announcement date is itself a trading day use it, else the next session
        tgt = pos + shift
        if 0 <= tgt < len(index):
            mask[tgt] = True
    return mask


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_world(n_days: int = 4000, edge: float = 0.0, seed: int = 517,
                    sig_daily: float = 0.010, meeting_every: int = 33
                    ) -> tuple[pd.Series, np.ndarray]:
    """Deterministic daily SPY-like return series + a planted pre-meeting drift mask.

    A random-walk return series; every ``meeting_every`` trading days carries a (synthetic)
    FOMC meeting, and the session *before* it gets an extra drift of ``edge``. With
    ``edge = 0`` the pre-meeting days are statistically identical to other days; with a planted
    ``edge`` the pre-vs-other test must light up. Returns (daily return Series, pre-meeting
    boolean mask aligned to it).
    """
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0003, sig_daily, size=n_days)
    # decorative business-day index (label only; positions carry the logic)
    idx = pd.bdate_range("2000-01-03", periods=n_days)
    pre = np.zeros(n_days, dtype=bool)
    m = meeting_every
    while m < n_days:
        pre[m - 1] = True            # the session BEFORE the meeting
        m += meeting_every
    ret[pre] += edge
    return pd.Series(ret, index=idx, name="ret"), pre
