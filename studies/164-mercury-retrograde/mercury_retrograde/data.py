"""Data layer for Study 164 (Mercury-Retrograde).

Two tapes, one shape (a daily close-to-close return Series indexed by date):

- ``synthetic_daily`` — a *deterministic, offline* generator. A ``retro_penalty``
  knob adds a downward drift to retrograde days; ``retro_penalty=0`` is a pure
  random walk with no planetary seasonality. This is the study's null in a bottle.
- ``fetch_daily`` — the real S&P 500 (^GSPC) daily tape back to 2000, via
  Yahoo Finance, **cache-first by default**. The result is stored as parquet under
  ``_cache/`` so the test suite and the reproducible core never touch the network.

The Mercury-retrograde date table is HARDCODED here from the well-documented
astronomical ephemeris (NASA / astrology.com / TimeandDate.com sources).
Mercury goes retrograde roughly three times per year for about three weeks each
time (approximately 19% of calendar days). The exact dates are deterministic
astronomical facts, not contested data.

No look-ahead is baked in here — dates are assigned to the *actual* calendar day
of each observation, not a forward-looking window.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
STUDY_CACHE = os.path.join(_HERE, "..", "_cache")

_CACHE_FILE = "mercury_retrograde_gspc_daily.parquet"


# ---------------------------------------------------------------------------
# Hardcoded Mercury-retrograde periods 2000-2026
# Source: NASA JPL Horizons ephemeris; cross-checked with astrology.com and
# TimeandDate.com/astronomy/mercury-retrograde-dates
# Format: (start_inclusive, end_inclusive) — both dates are retrograde.
# ---------------------------------------------------------------------------
RETROGRADE_PERIODS: list[tuple[str, str]] = [
    # 2000
    ("2000-07-17", "2000-08-11"),
    ("2000-11-15", "2000-12-04"),
    # 2001
    ("2001-03-11", "2001-04-04"),
    ("2001-07-17", "2001-08-11"),
    ("2001-10-01", "2001-10-22"),
    # 2002
    ("2002-01-18", "2002-02-08"),
    ("2002-05-15", "2002-06-08"),
    ("2002-09-14", "2002-10-06"),
    # 2003
    ("2003-01-02", "2003-01-22"),
    ("2003-04-26", "2003-05-20"),
    ("2003-08-28", "2003-09-20"),
    ("2003-12-17", "2003-12-31"),
    # 2004
    ("2004-01-01", "2004-01-06"),
    ("2004-04-06", "2004-04-30"),
    ("2004-08-09", "2004-09-02"),
    ("2004-11-30", "2004-12-20"),
    # 2005
    ("2005-03-19", "2005-04-12"),
    ("2005-07-22", "2005-08-15"),
    ("2005-11-13", "2005-12-03"),
    # 2006
    ("2006-03-02", "2006-03-25"),
    ("2006-07-04", "2006-07-28"),
    ("2006-10-28", "2006-11-17"),
    # 2007
    ("2007-02-13", "2007-03-07"),
    ("2007-06-15", "2007-07-09"),
    ("2007-10-11", "2007-11-01"),
    # 2008
    ("2008-01-28", "2008-02-18"),
    ("2008-05-26", "2008-06-19"),
    ("2008-09-24", "2008-10-15"),
    # 2009
    ("2009-01-11", "2009-02-01"),
    ("2009-05-06", "2009-05-30"),
    ("2009-09-06", "2009-09-29"),
    ("2009-12-26", "2009-12-31"),
    # 2010
    ("2010-01-01", "2010-01-15"),
    ("2010-04-17", "2010-05-11"),
    ("2010-08-20", "2010-09-12"),
    ("2010-12-10", "2010-12-29"),
    # 2011
    ("2011-03-30", "2011-04-23"),
    ("2011-08-02", "2011-08-26"),
    ("2011-11-23", "2011-12-13"),
    # 2012
    ("2012-03-11", "2012-04-04"),
    ("2012-07-14", "2012-08-07"),
    ("2012-11-06", "2012-11-26"),
    # 2013
    ("2013-02-23", "2013-03-17"),
    ("2013-06-26", "2013-07-20"),
    ("2013-10-21", "2013-11-10"),
    # 2014
    ("2014-02-06", "2014-02-28"),
    ("2014-06-07", "2014-07-01"),
    ("2014-10-04", "2014-10-25"),
    # 2015
    ("2015-01-21", "2015-02-11"),
    ("2015-05-18", "2015-06-11"),
    ("2015-09-17", "2015-10-09"),
    # 2016
    ("2016-01-05", "2016-01-25"),
    ("2016-04-28", "2016-05-22"),
    ("2016-08-30", "2016-09-21"),
    ("2016-12-19", "2016-12-31"),
    # 2017
    ("2017-01-01", "2017-01-08"),
    ("2017-04-09", "2017-05-03"),
    ("2017-08-12", "2017-09-05"),
    ("2017-12-02", "2017-12-22"),
    # 2018
    ("2018-03-22", "2018-04-15"),
    ("2018-07-26", "2018-08-18"),
    ("2018-11-16", "2018-12-06"),
    # 2019
    ("2019-03-05", "2019-03-28"),
    ("2019-07-07", "2019-07-31"),
    ("2019-10-31", "2019-11-20"),
    # 2020
    ("2020-02-16", "2020-03-09"),
    ("2020-06-17", "2020-07-12"),
    ("2020-10-13", "2020-11-03"),
    # 2021
    ("2021-01-30", "2021-02-20"),
    ("2021-05-29", "2021-06-22"),
    ("2021-09-26", "2021-10-18"),
    # 2022
    ("2022-01-14", "2022-02-03"),
    ("2022-05-10", "2022-06-02"),
    ("2022-09-09", "2022-10-02"),
    ("2022-12-29", "2022-12-31"),
    # 2023
    ("2023-01-01", "2023-01-18"),
    ("2023-04-21", "2023-05-14"),
    ("2023-08-23", "2023-09-15"),
    ("2023-12-13", "2023-12-31"),
    # 2024
    ("2024-01-01", "2024-01-01"),
    ("2024-04-01", "2024-04-24"),
    ("2024-08-04", "2024-08-28"),
    ("2024-11-25", "2024-12-15"),
    # 2025
    ("2025-03-14", "2025-04-06"),
    ("2025-07-17", "2025-08-11"),
    ("2025-11-09", "2025-11-29"),
    # 2026
    ("2026-02-25", "2026-03-19"),
    ("2026-06-29", "2026-07-23"),
]


def retrograde_dates(
    start: str = "2000-01-01",
    end: str = "2026-06-15",
) -> pd.DatetimeIndex:
    """Return a DatetimeIndex of all calendar days that fall inside a retrograde period.

    Days are date-level (no time component). Both the start and end of each
    period are inclusive. The list is drawn from the hardcoded ``RETROGRADE_PERIODS``
    table above — a purely deterministic, offline computation.
    """
    days: list[pd.Timestamp] = []
    window_start = pd.Timestamp(start)
    window_end = pd.Timestamp(end)
    for s, e in RETROGRADE_PERIODS:
        ts, te = pd.Timestamp(s), pd.Timestamp(e)
        # Clip to the requested window
        ts = max(ts, window_start)
        te = min(te, window_end)
        if ts > te:
            continue
        days.extend(pd.date_range(ts, te, freq="D"))
    return pd.DatetimeIndex(sorted(set(days)))


def is_retrograde(dates: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask: True if the date falls inside any Mercury-retrograde period.

    Parameters
    ----------
    dates :
        DatetimeIndex to label (calendar dates, no time component needed).

    Returns a boolean Series aligned to ``dates``.
    """
    retro_set = set(retrograde_dates(
        start=str(dates.min().date()),
        end=str(dates.max().date()),
    ).normalize())
    mask = pd.Series(
        [d.normalize() in retro_set for d in dates],
        index=dates,
        name="retrograde",
    )
    return mask


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 25,
    annual_drift_bp: float = 700.0,
    vol_ann: float = 0.16,
    retro_penalty_bp: float = 0.0,
    seed: int = 164,
) -> tuple[pd.Series, pd.Series, dict]:
    """A reproducible daily return series with a planted retrograde drag (or none).

    Log returns are i.i.d. normal with an annualised drift of ``annual_drift_bp``
    basis points and annual vol ``vol_ann``, except Mercury-retrograde days which
    receive an additional ``retro_penalty_bp`` per-day bias (negative = extra
    downside during retrograde). The only forecastable structure is the planetary
    calendar label: when ``retro_penalty_bp = 0`` (the null) all days are identical.

    Returns ``(returns, retrograde_mask, truth)`` where ``truth`` records the
    planted parameters and ``retrograde_mask`` is a boolean Series aligned to
    ``returns``.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2000-01-03")
    all_days = pd.bdate_range(start=start, periods=n_years * 252)
    n = len(all_days)
    daily_drift = annual_drift_bp * 1e-4 / 252
    daily_vol = vol_ann / np.sqrt(252)
    r = rng.normal(daily_drift, daily_vol, n)

    # Plant the retrograde penalty on retrograde trading days
    retro = is_retrograde(pd.DatetimeIndex(all_days))
    r[retro.values] += retro_penalty_bp * 1e-4

    ret = pd.Series(r, index=all_days, name="ret")
    truth = {
        "n_years": n_years,
        "annual_drift_bp": annual_drift_bp,
        "vol_ann": vol_ann,
        "retro_penalty_bp": retro_penalty_bp,
        "seed": seed,
        "n_obs": int(n),
        "n_retro_days": int(retro.sum()),
        "frac_retro": float(retro.mean()),
    }
    return ret, retro, truth


# ---------------------------------------------------------------------------
# Real tape — ^GSPC daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, _CACHE_FILE)


def fetch_daily(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "2000-01-01",
) -> tuple[pd.Series, pd.Series]:
    """Daily close-to-close return for ^GSPC since ``start``; cache-first unless ``fetch=True``.

    Returns ``(returns, retrograde_mask)``:

    - ``returns``   — a simple-return Series aligned to business days.
    - ``retrograde_mask`` — a boolean Series, True on Mercury-retrograde trading days.

    Network is touched only when ``fetch=True``; the cached parquet persists so every
    downstream consumer — tests, notebooks, verify.py — is offline by default.
    """
    # First try study-local cache, then repo-wide _cache
    for cdir in (STUDY_CACHE, cache_dir):
        path = _cache_path(cdir)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            ret = df["ret"] if "ret" in df.columns else df.iloc[:, 0]
            ret.index = pd.DatetimeIndex(ret.index)
            if ret.index.tz is not None:
                ret.index = ret.index.tz_localize(None)
            ret = ret.sort_index()
            mask = is_retrograde(pd.DatetimeIndex(ret.index))
            return ret, mask

    if not fetch:
        raise FileNotFoundError(
            f"No cached daily tape for ^GSPC. "
            f"Call fetch_daily(fetch=True) once to populate the cache."
        )

    import yfinance as yf  # lazy: only when we actually go to the network

    raw = yf.download("^GSPC", start=start, interval="1d", auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("yfinance returned no data for ^GSPC")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    px = raw["Close"].dropna()
    px.index = pd.DatetimeIndex(px.index)
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    px = px.sort_index()
    ret = px.pct_change().dropna()
    ret.name = "ret"
    ret.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_frame().to_parquet(_cache_path(cache_dir))
    mask = is_retrograde(pd.DatetimeIndex(ret.index))
    return ret, mask


def fingerprint(ret: pd.Series) -> str:
    """A short content fingerprint of the return series, for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(ret.to_numpy()).tobytes())
    return h.hexdigest()[:12]
