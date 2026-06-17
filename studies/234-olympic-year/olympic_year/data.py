"""Data layer for Study 234 (Olympic-Year).

Two tapes, one shape (an annual return series for the S&P 500 index):

- ``synthetic_annual`` — a *deterministic, offline* generator.  An ``olympic_boost``
  knob injects an extra return in Summer-Olympic years, so tests can confirm
  the analysis correctly recovers a planted effect and finds nothing when the boost
  is zero.  ``olympic_boost = 0`` is the null — Olympic year and annual return are
  independent.

- ``load_gspc`` — real calendar-year returns from yfinance (^GSPC).  We cover
  the years 1928–2024 (the span of modern Summer Olympics in the financial era).
  The data is fetched once and cached in the study's own ``_cache/`` directory.

Hardcoded Summer-Olympic-year table (the treatment group):
    Modern Summer Olympics have been held every 4 years starting 1896, interrupted
    by WWII (1916, 1940, 1944 cancelled).  We use the full post-1928 window for which
    GSPC data is available.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GSPC_CACHE = os.path.join(HERE, "..", "_cache", "gspc_annual.parquet")

# ---------------------------------------------------------------------------
# Hardcoded Summer Olympic year table (post-1928, years with GSPC data)
# ---------------------------------------------------------------------------
# Summer Olympics held in these years (no games in 1916, 1940, 1944 due to WWI/WWII).
# Source: International Olympic Committee official records.
OLYMPIC_YEARS = frozenset([
    1928, 1932, 1936,            # pre-WWII gap (1940, 1944 cancelled)
    1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984,
    1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020,  # Tokyo held 2021
    2024,
])
# Note: the 2020 Tokyo Olympics were held in 2021 due to COVID-19 postponement.
# We use the scheduled year (2020) to preserve the pre-announced calendar.
# A robustness check using 2021 instead is included in the notebooks.


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_annual(
    n_years: int = 100,
    annual_vol: float = 0.18,
    annual_drift: float = 0.07,
    olympic_boost: float = 0.0,
    start_year: int = 1928,
    seed: int = 234,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible annual return series with an optional Olympic-year boost.

    Log returns are drawn i.i.d. from N(annual_drift, annual_vol**2).  In years
    that are Summer Olympic years (using the hardcoded OLYMPIC_YEARS table),
    an extra ``olympic_boost`` is added to the log return.
    ``olympic_boost = 0`` is the pure null: Olympic years carry zero information.
    ``olympic_boost > 0`` plants the effect the folklore claims.

    Returns ``(df, truth)`` where ``df`` has columns ``year``, ``log_ret``,
    ``ret_pct``, and ``is_olympic``.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(start_year, start_year + n_years)
    log_ret = rng.normal(annual_drift, annual_vol, n_years)

    # Inject the Olympic boost for Olympic years.
    is_olympic = np.array([y in OLYMPIC_YEARS for y in years], dtype=bool)
    log_ret[is_olympic] += olympic_boost

    ret_pct = (np.exp(log_ret) - 1.0) * 100.0
    df = pd.DataFrame(
        {"year": years, "log_ret": log_ret, "ret_pct": ret_pct, "is_olympic": is_olympic}
    ).set_index("year")

    truth = {
        "n_years": n_years,
        "annual_vol": annual_vol,
        "annual_drift": annual_drift,
        "olympic_boost": olympic_boost,
        "start_year": start_year,
        "seed": seed,
        "n_olympic_years": int(is_olympic.sum()),
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — yfinance ^GSPC annual returns, cached locally
# ---------------------------------------------------------------------------
def load_gspc(
    cache_path: str = GSPC_CACHE,
    start_year: int = 1928,
    end_year: int = 2024,
) -> pd.DataFrame:
    """Calendar-year simple returns for the S&P 500 (^GSPC) from yfinance.

    Fetches if cache is absent, then reads from cache.  Annual returns are
    computed from December-close to December-close (year-end prices).

    Returns a DataFrame indexed by year with columns:
    ``log_ret, ret_pct, is_olympic``.
    """
    if not os.path.exists(cache_path):
        _fetch_and_cache(cache_path, start_year=start_year - 1, end_year=end_year)
    raw = pd.read_parquet(cache_path)
    raw.index = pd.to_datetime(raw.index)
    raw["year"] = raw.index.year

    # December close for each year — end-of-year price.
    dec = raw.resample("YE").last()["Close"]
    dec.index = dec.index.year

    log_ret = np.log(dec / dec.shift(1)).dropna()
    log_ret = log_ret[(log_ret.index >= start_year) & (log_ret.index <= end_year)]

    ret_pct = (np.exp(log_ret) - 1.0) * 100.0
    df = pd.DataFrame({"log_ret": log_ret, "ret_pct": ret_pct})
    df.index.name = "year"
    df["is_olympic"] = df.index.map(lambda y: y in OLYMPIC_YEARS)
    return df.sort_index()


def _fetch_and_cache(cache_path: str, start_year: int = 1927, end_year: int = 2024) -> None:
    """Fetch ^GSPC daily data from yfinance and save to cache_path."""
    import yfinance as yf  # deferred import so tests never need network

    ticker = yf.Ticker("^GSPC")
    raw = ticker.history(
        start=f"{start_year}-01-01",
        end=f"{end_year + 1}-01-01",
        interval="1d",
        auto_adjust=True,
    )
    raw = raw[["Close"]].copy()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    raw.to_parquet(cache_path)


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the annual return series (ret_pct column)."""
    arr = df["ret_pct"].to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
