"""Data for the pre-FOMC drift study — an offline synthetic world, and the cached SPY daily returns
plus the scheduled FOMC announcement calendar.

FOMC_DATES are the **statement-release days** (the last day of each scheduled meeting, when the policy
statement is published ~14:00 ET). Source: Federal Reserve FOMC meeting calendars. A few early dates may
be off by a day; the strategy measures a small window around each to be robust to that.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
TRADING_DAYS = 252

# Scheduled FOMC announcement (statement-release) dates, 1994–2026. Eight per year.
FOMC_DATES = pd.to_datetime([
    "1994-02-04", "1994-03-22", "1994-04-18", "1994-05-17", "1994-07-06", "1994-08-16", "1994-09-27", "1994-11-15", "1994-12-20",
    "1995-02-01", "1995-03-28", "1995-05-23", "1995-07-06", "1995-08-22", "1995-09-26", "1995-11-15", "1995-12-19",
    "1996-01-31", "1996-03-26", "1996-05-21", "1996-07-03", "1996-08-20", "1996-09-24", "1996-11-13", "1996-12-17",
    "1997-02-05", "1997-03-25", "1997-05-20", "1997-07-02", "1997-08-19", "1997-09-30", "1997-11-12", "1997-12-16",
    "1998-02-04", "1998-03-31", "1998-05-19", "1998-07-01", "1998-08-18", "1998-09-29", "1998-10-15", "1998-11-17", "1998-12-22",
    "1999-02-03", "1999-03-30", "1999-05-18", "1999-06-30", "1999-08-24", "1999-10-05", "1999-11-16", "1999-12-21",
    "2000-02-02", "2000-03-21", "2000-05-16", "2000-06-28", "2000-08-22", "2000-10-03", "2000-11-15", "2000-12-19",
    "2001-01-31", "2001-03-20", "2001-04-18", "2001-05-15", "2001-06-27", "2001-08-21", "2001-09-17", "2001-10-02", "2001-11-06", "2001-12-11",
    "2002-01-30", "2002-03-19", "2002-05-07", "2002-06-26", "2002-08-13", "2002-09-24", "2002-11-06", "2002-12-10",
    "2003-01-29", "2003-03-18", "2003-05-06", "2003-06-25", "2003-08-12", "2003-09-16", "2003-10-28", "2003-12-09",
    "2004-01-28", "2004-03-16", "2004-05-04", "2004-06-30", "2004-08-10", "2004-09-21", "2004-11-10", "2004-12-14",
    "2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30", "2005-08-09", "2005-09-20", "2005-11-01", "2005-12-13",
    "2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29", "2006-08-08", "2006-09-20", "2006-10-25", "2006-12-12",
    "2007-01-31", "2007-03-21", "2007-05-09", "2007-06-28", "2007-08-07", "2007-09-18", "2007-10-31", "2007-12-11",
    "2008-01-30", "2008-03-18", "2008-04-30", "2008-06-25", "2008-08-05", "2008-09-16", "2008-10-29", "2008-12-16",
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24", "2009-08-12", "2009-09-23", "2009-11-04", "2009-12-16",
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23", "2010-08-10", "2010-09-21", "2010-11-03", "2010-12-14",
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22", "2011-08-09", "2011-09-21", "2011-11-02", "2011-12-13",
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20", "2012-08-01", "2012-09-13", "2012-10-24", "2012-12-12",
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19", "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18",
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-03-18", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
])


@dataclass(frozen=True)
class WorldTruth:
    drift: float

    @property
    def has_drift(self) -> bool:
        return self.drift != 0.0


def synthetic_world(n_days=4000, drift=0.004, seed=67, base=0.0002, vol=0.011):
    """A daily-return world with FOMC days every ~31 trading days; on the trading day *before* each
    announcement the return is lifted by ``drift`` (the pre-FOMC drift). 0 = the null. Returns a
    (ret Series, fomc_dates) pair plus the truth."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n_days, name="date")
    ret = base + vol * rng.standard_normal(n_days)
    fomc_pos = np.arange(20, n_days, 31)               # an "announcement" every ~31 bdays
    pre_pos = fomc_pos - 1                              # the pre-FOMC day
    ret[pre_pos] += drift
    fomc_dates = pd.DatetimeIndex([idx[p] for p in fomc_pos])
    return pd.Series(ret, index=idx, name="ret"), fomc_dates, WorldTruth(drift)


def fetch_panel(cache_dir=DEFAULT_CACHE, fetch=False):
    """Daily SPY total returns (cache-first, from the shared ``last_call_spy`` pull) and the FOMC
    calendar. Returns (ret Series, fomc_dates)."""
    cache = os.path.join(cache_dir, "last_call_spy.parquet")
    if os.path.exists(cache):
        ret = pd.read_parquet(cache)["ret"]
        ret.index = pd.DatetimeIndex(ret.index).tz_localize(None)
        return ret, FOMC_DATES
    if not fetch:
        return pd.Series(dtype=float), FOMC_DATES
    import yfinance as yf
    px = yf.download("SPY", period="max", auto_adjust=True, progress=False)["Close"]
    ret = px.pct_change().dropna(); ret.name = "ret"
    ret.index = pd.DatetimeIndex(ret.index).tz_localize(None)
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_frame().to_parquet(cache)
    return ret, FOMC_DATES
