"""Data layer for Study 266 (Misery-Index).

The *misery index* (Arthur Okun, 1970s) is the sum of the headline inflation
rate and the unemployment rate.  Folk-macro wisdom comes in two flavours:

  - **Pessimist read**: high misery means a sick economy, so equities should do
    badly next year (misery is a leading bear signal).
  - **Contrarian read**: misery peaks at the bottom -- maximum pessimism is the
    moment to *buy*, so high misery should precede *high* forward returns.

Both are testable.  This study tests whether the misery index *level* (or its
year-on-year *change*) has any out-of-sample predictive content for the
subsequent calendar-year S&P 500 return.

Three components, the first two fully offline:

- ``MISERY_TABLE`` -- a hardcoded table of December CPI year-on-year inflation
  (BLS CPI-U, NSA, December/December) and the December U-3 unemployment rate
  (BLS, seasonally adjusted), 1948-2025.  ``misery = cpi_yoy + unemployment``.
  These are small, well-known, public macro series; hardcoded here to keep the
  core fully offline and reproducible.  Source: U.S. Bureau of Labor Statistics
  (series CUUR0000SA0 and LNS14000000); FRED (CPIAUCNS, UNRATE).

- ``synthetic_panel`` -- a deterministic, offline generator of an annual
  (misery, next-year return) series with an optional planted ``beta`` linking
  the misery level to the forward return.  ``beta = 0`` is the null (misery
  carries no information): tests confirm the machinery is truthful before we
  look at real data.

- ``fetch_sp500_annual`` -- real S&P 500 calendar-year *price* returns from the
  repo-level cache ``_cache/^GSPC_split_only.parquet`` (December/December).
  Cache-only by default (``fetch=False``); the network is touched only on an
  explicit ``fetch=True`` (lazy ``yfinance`` import).

No look-ahead: the misery index for December of year Y is observable at the end
of year Y, so it is matched to the S&P 500 return *for year Y+1*.  We never use
year-Y+1 macro data to predict the year-Y+1 return.  One full year of execution
lag (the position is opened at the December close of year Y).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))
GSPC_CACHE = os.path.join(REPO_CACHE, "^GSPC_split_only.parquet")

# ---------------------------------------------------------------------------
# The hardcoded misery-index table -- the study's deterministic, offline core
# ---------------------------------------------------------------------------
# Columns:
#   year       : calendar year of the December observation
#   cpi_yoy    : CPI-U December/December year-on-year inflation, % (BLS CUUR0000SA0)
#   unemp      : December U-3 unemployment rate, % seasonally adjusted (BLS LNS14000000)
#
# misery = cpi_yoy + unemp  (Okun misery index).
#
# Sources:
#   - U.S. Bureau of Labor Statistics, CPI-U (CUUR0000SA0): December YoY inflation.
#   - U.S. Bureau of Labor Statistics, civilian unemployment rate (LNS14000000),
#     December value, seasonally adjusted.
#   - Mirrored on FRED as CPIAUCNS (12-month % change) and UNRATE.
#
# As-of 2026-06-17.  Values rounded to one decimal as published by BLS.
# The 1948 unemployment series starts in 1948 (the official BLS series origin),
# so the table begins in 1948; CPI-U YoY for December 1948 is included.

MISERY_RECORDS: list[tuple[int, float, float]] = [
    # (year, cpi_yoy_dec, unemployment_dec)
    (1948,  3.0, 4.0),
    (1949, -2.1, 6.6),
    (1950,  5.9, 4.3),
    (1951,  6.0, 3.1),
    (1952,  0.8, 2.7),
    (1953,  0.7, 4.5),
    (1954, -0.7, 5.0),
    (1955,  0.4, 4.2),
    (1956,  3.0, 4.2),
    (1957,  2.9, 5.2),
    (1958,  1.8, 6.2),
    (1959,  1.7, 5.3),
    (1960,  1.4, 6.6),
    (1961,  0.7, 6.0),
    (1962,  1.3, 5.5),
    (1963,  1.6, 5.5),
    (1964,  1.0, 5.0),
    (1965,  1.9, 4.0),
    (1966,  3.5, 3.8),
    (1967,  3.0, 3.8),
    (1968,  4.7, 3.4),
    (1969,  6.2, 3.5),
    (1970,  5.6, 6.1),
    (1971,  3.3, 6.0),
    (1972,  3.4, 5.2),
    (1973,  8.7, 4.9),
    (1974, 12.3, 7.2),
    (1975,  6.9, 8.2),
    (1976,  4.9, 7.8),
    (1977,  6.7, 6.4),
    (1978,  9.0, 6.0),
    (1979, 13.3, 6.0),
    (1980, 12.5, 7.2),
    (1981,  8.9, 8.5),
    (1982,  3.8, 10.8),
    (1983,  3.8, 8.3),
    (1984,  3.9, 7.3),
    (1985,  3.8, 7.0),
    (1986,  1.1, 6.6),
    (1987,  4.4, 5.7),
    (1988,  4.4, 5.3),
    (1989,  4.6, 5.4),
    (1990,  6.1, 6.3),
    (1991,  3.1, 7.3),
    (1992,  2.9, 7.4),
    (1993,  2.7, 6.5),
    (1994,  2.7, 5.5),
    (1995,  2.5, 5.6),
    (1996,  3.3, 5.4),
    (1997,  1.7, 4.7),
    (1998,  1.6, 4.4),
    (1999,  2.7, 4.0),
    (2000,  3.4, 3.9),
    (2001,  1.6, 5.7),
    (2002,  2.4, 6.0),
    (2003,  1.9, 5.7),
    (2004,  3.3, 5.4),
    (2005,  3.4, 4.9),
    (2006,  2.5, 4.4),
    (2007,  4.1, 5.0),
    (2008,  0.1, 7.3),
    (2009,  2.7, 9.9),
    (2010,  1.5, 9.3),
    (2011,  3.0, 8.5),
    (2012,  1.7, 7.9),
    (2013,  1.5, 6.7),
    (2014,  0.8, 5.6),
    (2015,  0.7, 5.0),
    (2016,  2.1, 4.7),
    (2017,  2.1, 4.1),
    (2018,  1.9, 3.9),
    (2019,  2.3, 3.6),
    (2020,  1.4, 6.7),
    (2021,  7.0, 3.9),
    (2022,  6.5, 3.5),
    (2023,  3.4, 3.7),
    (2024,  2.9, 4.1),
    (2025,  3.0, 4.2),
]

MISERY_DF: pd.DataFrame = pd.DataFrame(
    MISERY_RECORDS, columns=["year", "cpi_yoy", "unemp"]
)
MISERY_DF["misery"] = MISERY_DF["cpi_yoy"] + MISERY_DF["unemp"]


def misery_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded misery-index table.

    Columns: ``year`` (int), ``cpi_yoy`` (float, % Dec/Dec CPI-U), ``unemp``
    (float, % December U-3), ``misery`` (float = cpi_yoy + unemp).  Spans
    1948-2025.
    """
    return MISERY_DF.copy()


# ---------------------------------------------------------------------------
# Synthetic generator -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_years: int = 77,
    beta: float = 0.0,
    base_return: float = 0.07,
    vol_ann: float = 0.16,
    misery_mean: float = 9.0,
    misery_sd: float = 3.5,
    seed: int = 266,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible (misery, next-year return) series with an optional planted link.

    For each year we draw a misery level ``m_t ~ N(misery_mean, misery_sd^2)``
    (clipped to be positive), and the *next* year's return::

        ret_{t+1} = base_return + beta * (m_t - misery_mean) / misery_sd + noise

    so ``beta`` is the per-1-SD-of-misery effect on the forward annual return.
    ``beta = 0`` is the null: misery carries no information about forward
    returns.  This is the study's null in a bottle.

    Returns ``(df, truth)`` where ``df`` has columns ``year`` (int), ``misery``
    (float), ``fwd_return`` (next-year simple return), and ``truth`` records the
    planted parameters.
    """
    rng = np.random.default_rng(seed)
    years = list(range(1948, 1948 + n_years))
    misery = rng.normal(misery_mean, misery_sd, n_years)
    misery = np.clip(misery, 0.5, None)
    z = (misery - misery_mean) / misery_sd
    fwd = base_return + beta * z + rng.normal(0.0, vol_ann, n_years)

    df = pd.DataFrame({
        "year": years,
        "misery": misery,
        "fwd_return": fwd,
    })
    truth = {
        "n_years": n_years,
        "beta": beta,
        "base_return": base_return,
        "vol_ann": vol_ann,
        "misery_mean": misery_mean,
        "misery_sd": misery_sd,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data -- S&P 500 annual price returns (repo-level cache, read-only)
# ---------------------------------------------------------------------------
def _sp500_annual_returns(cache_path: str = GSPC_CACHE) -> pd.Series:
    """December/December S&P 500 calendar-year *price* returns from the cache.

    Returns a Series indexed by calendar year (int), values = simple annual
    price return.  Price-only (no dividends), consistent with a headline
    macro-vs-index test.  Raises ``FileNotFoundError`` if the cache is absent.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"S&P 500 cache not found at {cache_path}. "
            "The repo-level _cache/^GSPC_split_only.parquet must be present."
        )
    px = pd.read_parquet(cache_path)
    close = px["Close"].copy()
    close.index = pd.to_datetime(close.index)
    dec = close[close.index.month == 12]
    dec_year = pd.Series(dec.values, index=dec.index.year)
    year_end = dec_year.groupby(level=0).last()
    ret = year_end.pct_change().dropna()
    ret.index = ret.index.astype(int)
    ret.name = "sp500_return"
    return ret


def fetch_sp500_annual(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
    start_year: int = 1949,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Join the misery table to next-year S&P 500 returns (no look-ahead).

    The misery index measured at December of year Y is matched to the S&P 500
    calendar-year return *for year Y+1* (``fwd_return``).  This is the honest
    no-look-ahead alignment: the macro print is known at year-end Y, the trade
    is opened at the December close of Y, and earns the return of year Y+1.

    Cache-only by default.  ``fetch=True`` triggers a one-time ``yfinance``
    download of ^GSPC daily prices, written to ``cache_path``; the network is
    touched only then (lazy import).

    Returns a DataFrame with columns:
      ``year`` (the misery-measurement year Y), ``cpi_yoy``, ``unemp``,
      ``misery``, ``misery_chg`` (Y-on-Y change in misery), ``fwd_return``
      (S&P return in year Y+1), ``mkt_up`` (bool: fwd_return > 0).

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False``.
    """
    if fetch and not os.path.exists(cache_path):
        _fetch_gspc_to_cache(cache_path)

    ret = _sp500_annual_returns(cache_path)  # indexed by the return's own year

    mis = misery_table()
    mis["misery_chg"] = mis["misery"].diff()
    # The misery of year Y predicts the return of year Y+1.
    mis["ret_year"] = mis["year"] + 1
    ret_df = ret.reset_index()
    ret_df.columns = ["ret_year", "fwd_return"]

    merged = mis.merge(ret_df, on="ret_year", how="inner")
    merged = merged[(merged["ret_year"] >= start_year) &
                    (merged["ret_year"] <= end_year)].copy()
    merged["mkt_up"] = merged["fwd_return"] > 0
    cols = ["year", "cpi_yoy", "unemp", "misery", "misery_chg",
            "fwd_return", "mkt_up"]
    return merged[cols].dropna(subset=["misery_chg"]).reset_index(drop=True)


def _fetch_gspc_to_cache(cache_path: str) -> None:
    """One-time live fetch of ^GSPC daily closes into the repo cache (network)."""
    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download("^GSPC", start="1927-01-01", auto_adjust=False,
                      progress=False)
    if raw.empty:
        raise RuntimeError("yfinance returned no ^GSPC data")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    raw.to_parquet(cache_path)


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the fwd_return column, for the as-of stamp."""
    vals = df["fwd_return"].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
