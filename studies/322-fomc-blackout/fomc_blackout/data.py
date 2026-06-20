"""Data layer for Study 322 (FOMC-Blackout).

Two tapes, one logical shape (a daily-return Series with a boolean ``blackout`` flag):

- ``synthetic_blackout`` — a *deterministic, offline* generator. A ``blackout_excess``
  knob injects an extra daily return on every blackout-window trading day, and a
  ``blackout_vol_mult`` knob scales the vol inside the window (to model — or refute —
  the "calm before the storm" idea). Set both to their null values
  (``blackout_excess=0``, ``blackout_vol_mult=1``) for the null: pure random walk, so
  the blackout-vs-rest test scores zero. This is the study's null in a bottle.
- ``load_real`` — cache-first SPY **total-return** daily tape from the shared repo
  ``_cache/SPY_total_return.parquet`` (graceful: raises a clear FileNotFoundError when
  the cache is absent, e.g. offline CI), tagged with the blackout flag derived from the
  hardcoded FOMC schedule. ``fetch=True`` hits Yahoo Finance and repopulates a local
  parquet.

The FOMC communications **blackout** ("quiet period") runs, by Board rule, from the
*second Saturday before* a meeting through the *Thursday after* it; Fed officials may not
speak publicly on monetary policy during that window. The *pre-meeting* portion is the
"calm before the storm" the folklore points at, so we define the blackout flag as the
trading days within ``BLACKOUT_DAYS`` calendar days *before* each meeting (the storm is
the decision itself; we are testing the run-up). No look-ahead: a day is flagged using
only the *scheduled* meeting date, which is known months in advance.

Reference: Federal Reserve, *Policy on External Communications of Committee Participants*
(the "blackout"/quiet-period rule). Distinct from the post-meeting drift of Study 135.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
SHARED_CACHE = os.path.join(REPO_ROOT, "_cache")
LOCAL_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

TRADING_DAYS = 252
# Calendar days before a meeting counted as the pre-meeting blackout window.
# The rule's quiet period is ~10 days (second Saturday before -> meeting).
BLACKOUT_DAYS = 10

# ---------------------------------------------------------------------------
# FOMC scheduled decision dates, 1994-2026 (statement-release days).
# Source: Federal Reserve FOMC historical calendars (reused from Study 135).
# ---------------------------------------------------------------------------
FOMC_DATES = pd.to_datetime([
    # 1994
    "1994-02-04", "1994-03-22", "1994-04-18", "1994-05-17",
    "1994-07-06", "1994-08-16", "1994-09-27", "1994-11-15", "1994-12-20",
    # 1995
    "1995-02-01", "1995-03-28", "1995-05-23", "1995-07-06",
    "1995-08-22", "1995-09-26", "1995-11-15", "1995-12-19",
    # 1996
    "1996-01-31", "1996-03-26", "1996-05-21", "1996-07-03",
    "1996-08-20", "1996-09-24", "1996-11-13", "1996-12-17",
    # 1997
    "1997-02-05", "1997-03-25", "1997-05-20", "1997-07-02",
    "1997-08-19", "1997-09-30", "1997-11-12", "1997-12-16",
    # 1998
    "1998-02-04", "1998-03-31", "1998-05-19", "1998-07-01",
    "1998-08-18", "1998-09-29", "1998-10-15", "1998-11-17", "1998-12-22",
    # 1999
    "1999-02-03", "1999-03-30", "1999-05-18", "1999-06-30",
    "1999-08-24", "1999-10-05", "1999-11-16", "1999-12-21",
    # 2000
    "2000-02-02", "2000-03-21", "2000-05-16", "2000-06-28",
    "2000-08-22", "2000-10-03", "2000-11-15", "2000-12-19",
    # 2001
    "2001-01-31", "2001-03-20", "2001-04-18", "2001-05-15",
    "2001-06-27", "2001-08-21", "2001-09-17", "2001-10-02",
    "2001-11-06", "2001-12-11",
    # 2002
    "2002-01-30", "2002-03-19", "2002-05-07", "2002-06-26",
    "2002-08-13", "2002-09-24", "2002-11-06", "2002-12-10",
    # 2003
    "2003-01-29", "2003-03-18", "2003-05-06", "2003-06-25",
    "2003-08-12", "2003-09-16", "2003-10-28", "2003-12-09",
    # 2004
    "2004-01-28", "2004-03-16", "2004-05-04", "2004-06-30",
    "2004-08-10", "2004-09-21", "2004-11-10", "2004-12-14",
    # 2005
    "2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30",
    "2005-08-09", "2005-09-20", "2005-11-01", "2005-12-13",
    # 2006
    "2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29",
    "2006-08-08", "2006-09-20", "2006-10-25", "2006-12-12",
    # 2007
    "2007-01-31", "2007-03-21", "2007-05-09", "2007-06-28",
    "2007-08-07", "2007-09-18", "2007-10-31", "2007-12-11",
    # 2008
    "2008-01-30", "2008-03-18", "2008-04-30", "2008-06-25",
    "2008-08-05", "2008-09-16", "2008-10-29", "2008-12-16",
    # 2009
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24",
    "2009-08-12", "2009-09-23", "2009-11-04", "2009-12-16",
    # 2010
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23",
    "2010-08-10", "2010-09-21", "2010-11-03", "2010-12-14",
    # 2011
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22",
    "2011-08-09", "2011-09-21", "2011-11-02", "2011-12-13",
    # 2012
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20",
    "2012-08-01", "2012-09-13", "2012-10-24", "2012-12-12",
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19",
    "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18",
    "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17",
    "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-18", "2020-04-29", "2020-06-10",
    "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (through our as-of date)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
])


# ---------------------------------------------------------------------------
# Blackout-window flag
# ---------------------------------------------------------------------------
def in_blackout(
    index: pd.DatetimeIndex,
    fomc_dates: pd.DatetimeIndex | None = None,
    window: int = BLACKOUT_DAYS,
    offset: int = 0,
) -> pd.Series:
    """Boolean blackout flag for each trading day in ``index``.

    A day is in the (pre-meeting) blackout window if it falls within ``window``
    calendar days *before* a scheduled FOMC decision, i.e. in
    ``[meeting - window - offset, meeting - 1 - offset]`` calendar days. ``offset``
    shifts the whole window forward/back for the placebo (offset=0 is the real rule;
    the decision day itself and post-meeting days are *not* in this window — we are
    testing the calm run-up, not the post-statement drift of Study 135).

    No look-ahead: the schedule is known months in advance.
    """
    if fomc_dates is None:
        fomc_dates = FOMC_DATES
    idx_d = pd.DatetimeIndex(index).normalize().values.astype("datetime64[D]")
    fa = pd.DatetimeIndex(fomc_dates).normalize().values.astype("datetime64[D]")
    flag = np.zeros(len(idx_d), dtype=bool)
    for m in fa:
        lo = m - np.timedelta64(window + offset, "D")
        hi = m - np.timedelta64(1 + offset, "D")
        flag |= (idx_d >= lo) & (idx_d <= hi)
    return pd.Series(flag, index=pd.DatetimeIndex(index), name="blackout")


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_blackout(
    n_days: int = 4000,
    blackout_excess: float = 0.0,
    blackout_vol_mult: float = 1.0,
    base_ret: float = 4e-4,
    daily_vol: float = 0.011,
    seed: int = 322,
    fomc_dates: pd.DatetimeIndex | None = None,
    start: str = "1994-02-01",
    window: int = BLACKOUT_DAYS,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily-return tape with a tunable blackout excess and vol.

    Returns ``r_t = base_ret + daily_vol * eps_t`` i.i.d., except on blackout-window
    days where the mean is lifted by ``blackout_excess`` and the noise is scaled by
    ``blackout_vol_mult`` (so ``<1`` plants the "calm" hypothesis, ``>1`` the opposite).
    With ``blackout_excess=0`` and ``blackout_vol_mult=1`` the tape is a pure random
    walk — the null — so the blackout-vs-rest test scores zero except by chance.

    Returns ``(frame, truth)`` where ``frame`` has columns ``ret`` and ``blackout``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days, name="date")
    if fomc_dates is None:
        fomc_dates = FOMC_DATES
    flag = in_blackout(idx, fomc_dates, window=window)
    mask = flag.to_numpy()

    eps = rng.standard_normal(n_days)
    vol = np.where(mask, daily_vol * blackout_vol_mult, daily_vol)
    rets = base_ret + vol * eps
    rets[mask] += blackout_excess

    frame = pd.DataFrame({"ret": rets, "blackout": mask}, index=idx)
    truth = {
        "blackout_excess": blackout_excess,
        "blackout_vol_mult": blackout_vol_mult,
        "base_ret": base_ret,
        "daily_vol": daily_vol,
        "n_days": n_days,
        "seed": seed,
        "window": window,
        "n_blackout": int(mask.sum()),
        "n_other": int((~mask).sum()),
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — SPY total returns from the shared repo cache, cache-first
# ---------------------------------------------------------------------------
def _real_cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "SPY_total_return.parquet")


def load_real(
    cache_dir: str = SHARED_CACHE,
    fetch: bool = False,
    start: str = "1994-01-01",
    window: int = BLACKOUT_DAYS,
) -> pd.DataFrame:
    """SPY daily *total-return* with the blackout flag attached; cache-first.

    Reads ``_cache/SPY_total_return.parquet`` from the shared repo cache (the same tape
    several other studies pin). On ``fetch=True`` it downloads from Yahoo (auto-adjusted,
    a total-return proxy) and writes a local parquet under the study's ``_cache/``.
    Network is touched *only* on an explicit ``fetch=True`` — the reproducible core and
    the test-suite never hit it.

    Returns a frame with columns ``ret`` (daily total return) and ``blackout`` (bool),
    indexed by trading date from ``start`` onward.
    """
    if fetch:
        import yfinance as yf

        px = yf.download("SPY", start="1993-01-01", auto_adjust=True, progress=False)
        if px.empty:
            raise RuntimeError("yfinance returned no data for SPY")
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.get_level_values(0)
        close = px["Close"]
        os.makedirs(LOCAL_CACHE, exist_ok=True)
        px.to_parquet(_real_cache_path(LOCAL_CACHE))
    else:
        path = _real_cache_path(cache_dir)
        if not os.path.exists(path):
            # Fall back to a study-local cache if the shared one is absent.
            local = _real_cache_path(LOCAL_CACHE)
            if os.path.exists(local):
                path = local
            else:
                raise FileNotFoundError(
                    f"No cached SPY total-return tape at {path}. "
                    "Run examples/verify.py --fetch once to populate, or rely on the "
                    "synthetic tape (the offline core)."
                )
        raw = pd.read_parquet(path)
        close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]

    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    ret = close.pct_change().dropna()
    ret = ret[ret.index >= start]
    ret.name = "ret"
    flag = in_blackout(ret.index, FOMC_DATES, window=window)
    return pd.DataFrame({"ret": ret.values, "blackout": flag.values}, index=ret.index)


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of the return column, for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(frame["ret"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
