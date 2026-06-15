"""Data layer for Study 165 (Chinese-Zodiac).

Two tapes, one shape (a daily close-to-close return Series indexed by date):

- ``synthetic_daily`` -- a *deterministic, offline* generator. A ``dragon_bonus``
  knob adds an upward drift to Year-of-the-Dragon returns; ``dragon_bonus=0`` is a
  pure random walk with no zodiac seasonality. This is the study's null in a bottle.
- ``fetch_daily`` -- the real FXI (iShares FTSE/Xinhua China 25 ETF) and ^GSPC
  daily tapes from Yahoo Finance, **cache-first by default**.

The Chinese New Year dates and zodiac animals for 1990-2026 are HARDCODED as a
deterministic constant table below. These are well-known public dates from official
Chinese calendars; see the source comment for the reference. The table assigns:
  - The CNY date (first day of the lunar year)
  - The zodiac animal for that year
  - A 10-trading-day pre-CNY window for the rally test

No look-ahead is baked in here -- calendar labels are assigned to the actual calendar
date of each observation, not a forward-looking window.
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

_CACHE_FILE_FXI = "chinese_zodiac_fxi_daily.parquet"
_CACHE_FILE_GSPC = "chinese_zodiac_gspc_daily.parquet"

# ---------------------------------------------------------------------------
# Hardcoded Chinese New Year dates + zodiac animals, 1990-2026
#
# Source: Hong Kong Observatory / official Chinese lunisolar calendar.
# Each row: (CNY date, zodiac animal, western year of that lunar new year)
# The animal cycle is: Rat, Ox, Tiger, Rabbit, Dragon, Snake, Horse, Goat,
#                      Monkey, Rooster, Dog, Pig  (12-year cycle)
# ---------------------------------------------------------------------------
CNY_TABLE: list[tuple[str, str, int]] = [
    # (date, animal, gregorian_year)
    ("1990-01-27", "Horse",   1990),
    ("1991-02-15", "Goat",    1991),
    ("1992-02-04", "Monkey",  1992),
    ("1993-01-23", "Rooster", 1993),
    ("1994-02-10", "Dog",     1994),
    ("1995-01-31", "Pig",     1995),
    ("1996-02-19", "Rat",     1996),
    ("1997-02-07", "Ox",      1997),
    ("1998-01-28", "Tiger",   1998),
    ("1999-02-16", "Rabbit",  1999),
    ("2000-02-05", "Dragon",  2000),
    ("2001-01-24", "Snake",   2001),
    ("2002-02-12", "Horse",   2002),
    ("2003-02-01", "Goat",    2003),
    ("2004-01-22", "Monkey",  2004),
    ("2005-02-09", "Rooster", 2005),
    ("2006-01-29", "Dog",     2006),
    ("2007-02-18", "Pig",     2007),
    ("2008-02-07", "Rat",     2008),
    ("2009-01-26", "Ox",      2009),
    ("2010-02-14", "Tiger",   2010),
    ("2011-02-03", "Rabbit",  2011),
    ("2012-01-23", "Dragon",  2012),
    ("2013-02-10", "Snake",   2013),
    ("2014-01-31", "Horse",   2014),
    ("2015-02-19", "Goat",    2015),
    ("2016-02-08", "Monkey",  2016),
    ("2017-01-28", "Rooster", 2017),
    ("2018-02-16", "Dog",     2018),
    ("2019-02-05", "Pig",     2019),
    ("2020-01-25", "Rat",     2020),
    ("2021-02-12", "Ox",      2021),
    ("2022-02-01", "Tiger",   2022),
    ("2023-01-22", "Rabbit",  2023),
    ("2024-02-10", "Dragon",  2024),
    ("2025-01-29", "Snake",   2025),
    ("2026-02-17", "Horse",   2026),
]

# Canonical 12-animal ordering (for indexing / cross-sectional plots)
ANIMALS = ["Rat", "Ox", "Tiger", "Rabbit", "Dragon", "Snake",
           "Horse", "Goat", "Monkey", "Rooster", "Dog", "Pig"]

# Build a tidy DataFrame from the hardcoded table
CNY_DF: pd.DataFrame = (
    pd.DataFrame(CNY_TABLE, columns=["cny_date", "animal", "year"])
    .assign(cny_date=lambda df: pd.to_datetime(df["cny_date"]))
    .set_index("year")
)

# Number of trading days BEFORE CNY to define the pre-CNY rally window
PRE_CNY_WINDOW = 10


# ---------------------------------------------------------------------------
# Synthetic tape -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 36,
    annual_drift_bp: float = 700.0,
    vol_ann: float = 0.20,
    dragon_bonus_bp: float = 0.0,
    seed: int = 165,
) -> tuple[pd.Series, dict]:
    """A reproducible daily return series with a planted Dragon-year / pre-CNY bonus.

    Log returns are i.i.d. normal with an annualised drift of ``annual_drift_bp``
    basis points and annual vol ``vol_ann``, except trading days in a Dragon year
    (years 2000, 2012, 2024 in the covered period) which receive an additional
    ``dragon_bonus_bp`` per-day bias. The only forecastable structure is the zodiac
    label: when ``dragon_bonus_bp = 0`` (the null) all years are identical.

    Returns ``(returns, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("1990-01-01")
    all_days = pd.bdate_range(start=start, periods=n_years * 252)
    n = len(all_days)
    daily_drift = annual_drift_bp * 1e-4 / 252
    daily_vol = vol_ann / np.sqrt(252)
    r = rng.normal(daily_drift, daily_vol, n)

    # Identify Dragon years in the table (every 12 years)
    dragon_years = {row.cny_date.year for _, row in CNY_DF.iterrows()
                    if row.animal == "Dragon"}
    # Also the calendar year AFTER each CNY (since the dragon year spans two Gregorian years)
    dragon_gy = set()
    for yr, row in CNY_DF.iterrows():
        if row.animal == "Dragon":
            dragon_gy.add(row.cny_date.year)
            dragon_gy.add(row.cny_date.year + 1)

    # Map each trading day to its zodiac year
    year_map = _build_year_map()
    for i, day in enumerate(all_days):
        animal = year_map.get(day.year, {}).get(day.month, {}).get(day.day)
        # Approximate: if the day falls in a Dragon zodiac year label
        gregorian_yr = day.year
        if gregorian_yr in dragon_gy:
            r[i] += dragon_bonus_bp * 1e-4

    ret = pd.Series(r, index=all_days, name="ret")
    truth = {
        "n_years": n_years,
        "annual_drift_bp": annual_drift_bp,
        "vol_ann": vol_ann,
        "dragon_bonus_bp": dragon_bonus_bp,
        "seed": seed,
        "n_obs": int(n),
    }
    return ret, truth


def _build_year_map() -> dict:
    """Not used directly; helper for a simpler Dragon-year identification approach."""
    return {}


def _dragon_days_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask: True when the trading day falls in a Dragon zodiac year.

    A 'Dragon zodiac year' runs from the CNY date of that year until the day
    before the next CNY date.
    """
    mask = np.zeros(len(index), dtype=bool)
    dragon_rows = CNY_DF[CNY_DF["animal"] == "Dragon"]
    for yr, row in dragon_rows.iterrows():
        start = row.cny_date
        # Find the next CNY
        next_yr = yr + 1
        if next_yr in CNY_DF.index:
            end = CNY_DF.loc[next_yr, "cny_date"] - pd.Timedelta(days=1)
        else:
            end = start + pd.Timedelta(days=385)
        mask |= (index >= start) & (index <= end)
    return mask


# ---------------------------------------------------------------------------
# Real tape -- Yahoo daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("^", "").replace("=", "").replace("/", "")
    return os.path.join(cache_dir, f"chinese_zodiac_{safe}_daily.parquet")


def fetch_daily(
    ticker: str = "FXI",
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "1994-01-01",
) -> pd.Series:
    """Daily close-to-close return for ``ticker``; cache-first unless ``fetch=True``.

    FXI (iShares China Large-Cap ETF) is the primary proxy for Chinese equity returns
    since its 2004 inception. For longer history use ^GSPC. The result is a
    simple-return Series aligned to business days.

    Network is touched only when ``fetch=True``; the cached parquet persists so every
    downstream consumer -- tests, notebooks, verify.py -- is offline by default.
    """
    # Try study-local cache first, then repo-wide _cache
    for cdir in (STUDY_CACHE, cache_dir):
        path = _cache_path(ticker, cdir)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            ret = df["ret"] if "ret" in df.columns else df.iloc[:, 0]
            ret.index = pd.DatetimeIndex(ret.index)
            if ret.index.tz is not None:
                ret.index = ret.index.tz_localize(None)
            return ret.sort_index()

    if not fetch:
        raise FileNotFoundError(
            f"No cached daily tape for {ticker}. "
            f"Call fetch_daily({ticker!r}, fetch=True) once to populate the cache."
        )

    import yfinance as yf  # lazy: only when we actually go to the network

    raw = yf.download(ticker, start=start, interval="1d", auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
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
    ret.to_frame().to_parquet(_cache_path(ticker, cache_dir))
    return ret


def fingerprint(ret: pd.Series) -> str:
    """A short content fingerprint of the return series, for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(ret.to_numpy()).tobytes())
    return h.hexdigest()[:12]
