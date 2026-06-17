"""Data layer for Study 264 (Buffett-Indicator).

The "Buffett Indicator" is the ratio of total US stock-market capitalisation to
US GDP.  Warren Buffett called it (Fortune, 2001) "probably the best single
measure of where valuations stand at any given moment."  When the ratio is high
the market is said to be expensive (time to leave the party); when low, cheap.

Three components, the core fully offline:

- ``BUFFETT_INDICATOR`` -- a hardcoded, year-end table of the indicator from
  1971 through 2025.  The numerator is the Wilshire 5000 total US market value
  (the FRED ``WILL5000PR``/``NCBEILQ027S`` market-value lineage that the popular
  "Buffett Indicator" trackers use); the denominator is nominal annual GDP
  (FRED ``GDP``).  We store the ratio directly (in percent of GDP), because the
  level is what every commentator quotes ("the Buffett Indicator is at 200% of
  GDP!").  This series is small, well-known and slow-moving -- hardcoded here to
  keep the reproducible core fully offline.  Sources: FRED (Wilshire 5000 market
  value, nominal GDP); GuruFocus / CurrentMarketValuation Buffett-Indicator
  trackers.  As-of 2026-06-17.

- ``synthetic_panel`` -- a deterministic, offline generator that plants (or not)
  a *mean-reversion* relationship between a valuation level and the subsequent
  year's return.  ``beta = 0`` is the null: the valuation level carries no
  forward information.  This is the study's positive control in a bottle.

- ``fetch_sp500_annual`` -- real S&P 500 (``^GSPC``) calendar-year **price**
  returns from the repo-level split-only cache (``_cache/^GSPC_split_only``).
  Cache-only by default (``fetch=False``); network is touched only on an
  explicit ``fetch=True`` (lazy yfinance import).

**Price-only caveat**: the forward-return tape is the S&P 500 *price* index
(no dividends).  Total-return numbers would be ~1.8 pp/yr higher across the
board; this does not change the *relative* high-vs-low comparison, which is what
the indicator claims, but it is named on the Signal axis.

**No look-ahead**: the indicator is observed at year-end Y (it is built from
data that is fully settled by 31-Dec-Y -- GDP for Y is a vintage that is revised,
a real-world wrinkle we note but do not exploit), and it is matched to the S&P
500 *price* return earned during year Y+1.  One full year of execution lag.
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
# The hardcoded Buffett Indicator table -- the study's deterministic core
# ---------------------------------------------------------------------------
# Columns:
#   year  : calendar year (the indicator is measured at year-end, 31-Dec).
#   ratio : total US market cap / nominal GDP, expressed in PERCENT of GDP
#           (so 100.0 means market cap == GDP; 200.0 means market cap == 2x GDP).
#
# The numerator is the Wilshire 5000 total US market value; the denominator is
# nominal annual GDP.  The levels below are the widely-quoted year-end Buffett
# Indicator readings (rounded to the nearest whole percent).  They are slow
# moving and well documented; we hardcode them so the reproducible core never
# touches a macro API.
#
# Sources:
#   - FRED: Wilshire 5000 Total Market (full-cap) value; Gross Domestic Product (GDP).
#   - GuruFocus "Buffett Indicator: Market Cap to GDP".
#   - CurrentMarketValuation.com "Buffett Indicator" history.
# As-of 2026-06-17.

BUFFETT_INDICATOR: list[dict] = [
    {"year": 1971, "ratio": 71.0},
    {"year": 1972, "ratio": 78.0},
    {"year": 1973, "ratio": 60.0},
    {"year": 1974, "ratio": 42.0},
    {"year": 1975, "ratio": 48.0},
    {"year": 1976, "ratio": 53.0},
    {"year": 1977, "ratio": 46.0},
    {"year": 1978, "ratio": 45.0},
    {"year": 1979, "ratio": 46.0},
    {"year": 1980, "ratio": 50.0},
    {"year": 1981, "ratio": 44.0},
    {"year": 1982, "ratio": 47.0},
    {"year": 1983, "ratio": 53.0},
    {"year": 1984, "ratio": 48.0},
    {"year": 1985, "ratio": 56.0},
    {"year": 1986, "ratio": 60.0},
    {"year": 1987, "ratio": 56.0},
    {"year": 1988, "ratio": 57.0},
    {"year": 1989, "ratio": 64.0},
    {"year": 1990, "ratio": 59.0},
    {"year": 1991, "ratio": 73.0},
    {"year": 1992, "ratio": 76.0},
    {"year": 1993, "ratio": 81.0},
    {"year": 1994, "ratio": 79.0},
    {"year": 1995, "ratio": 96.0},
    {"year": 1996, "ratio": 110.0},
    {"year": 1997, "ratio": 130.0},
    {"year": 1998, "ratio": 152.0},
    {"year": 1999, "ratio": 172.0},
    {"year": 2000, "ratio": 153.0},
    {"year": 2001, "ratio": 134.0},
    {"year": 2002, "ratio": 104.0},
    {"year": 2003, "ratio": 124.0},
    {"year": 2004, "ratio": 128.0},
    {"year": 2005, "ratio": 126.0},
    {"year": 2006, "ratio": 134.0},
    {"year": 2007, "ratio": 134.0},
    {"year": 2008, "ratio": 80.0},
    {"year": 2009, "ratio": 96.0},
    {"year": 2010, "ratio": 105.0},
    {"year": 2011, "ratio": 100.0},
    {"year": 2012, "ratio": 110.0},
    {"year": 2013, "ratio": 134.0},
    {"year": 2014, "ratio": 141.0},
    {"year": 2015, "ratio": 137.0},
    {"year": 2016, "ratio": 144.0},
    {"year": 2017, "ratio": 156.0},
    {"year": 2018, "ratio": 134.0},
    {"year": 2019, "ratio": 158.0},
    {"year": 2020, "ratio": 188.0},
    {"year": 2021, "ratio": 207.0},
    {"year": 2022, "ratio": 156.0},
    {"year": 2023, "ratio": 175.0},
    {"year": 2024, "ratio": 198.0},
    {"year": 2025, "ratio": 209.0},
]

BUFFETT_DF: pd.DataFrame = pd.DataFrame(BUFFETT_INDICATOR)


def buffett_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded Buffett-Indicator table.

    Columns: ``year`` (int) and ``ratio`` (float, market-cap/GDP in percent).
    The ratio is observed at year-end; see ``fetch_sp500_annual`` for the
    matched forward return.
    """
    return BUFFETT_DF.copy()


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline positive control
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_years: int = 55,
    beta: float = -0.05,
    base_ret: float = 0.09,
    ret_vol: float = 0.16,
    val_mean: float = 110.0,
    val_vol: float = 45.0,
    val_ar: float = 0.85,
    seed: int = 264,
) -> tuple[pd.DataFrame, dict]:
    """A year-by-year (valuation, forward-return) tape with a known relationship.

    A persistent valuation level ``v_t`` follows a mean-reverting AR(1) around
    ``val_mean``.  The *next* year's return is::

        ret_{t+1} = base_ret + beta * (v_t - val_mean) / 100 + noise

    so a positive ``beta`` would mean "high valuation -> high return" and a
    *negative* ``beta`` is the Buffett claim ("high valuation -> low forward
    return = mean reversion").  ``beta = 0`` is the null: the valuation level
    carries no forward information.

    Returns ``(df, truth)`` where ``df`` has columns ``year`` (int),
    ``ratio`` (the valuation level, in percent units, comparable to the real
    Buffett Indicator), and ``fwd_ret`` (the simple return earned the FOLLOWING
    year); ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    years = list(range(1971, 1971 + n_years))

    # AR(1) valuation level, mean-reverting around val_mean.
    v = np.empty(n_years)
    v[0] = val_mean
    innov_sd = val_vol * np.sqrt(1.0 - val_ar ** 2)
    for t in range(1, n_years):
        v[t] = val_mean + val_ar * (v[t - 1] - val_mean) + rng.normal(0.0, innov_sd)
    v = np.clip(v, 20.0, 320.0)

    noise = rng.normal(0.0, ret_vol, n_years)
    fwd = base_ret + beta * (v - val_mean) / 100.0 + noise

    df = pd.DataFrame({"year": years, "ratio": v, "fwd_ret": fwd})
    truth = {
        "n_years": n_years,
        "beta": beta,
        "base_ret": base_ret,
        "ret_vol": ret_vol,
        "val_mean": val_mean,
        "val_vol": val_vol,
        "val_ar": val_ar,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape -- S&P 500 (^GSPC) annual price returns, cache-only by default
# ---------------------------------------------------------------------------
def fetch_sp500_annual(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
    start_year: int = 1971,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Calendar-year S&P 500 *price* returns joined to the Buffett-Indicator table.

    Reads ``^GSPC`` daily closes from the repo-level split-only cache (price
    index, no dividends), resamples to year-end, and computes the Dec/Dec
    calendar-year simple return.  The indicator observed at year-end Y is paired
    with the return earned during year Y+1 (``fwd_ret``) -- one full year of
    execution lag, no look-ahead.

    Cache-only unless ``fetch=True`` (then a fresh ``^GSPC`` download is cached
    via a lazy yfinance import).  Raises ``FileNotFoundError`` if the cache is
    absent and ``fetch=False``.

    Returns a DataFrame with columns:
      ``year`` (the year the indicator is measured),
      ``ratio`` (Buffett Indicator at year-end Y, percent of GDP),
      ``fwd_ret`` (S&P 500 price return during year Y+1),
      ``fwd_up`` (bool: fwd_ret > 0).
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached ^GSPC series at {cache_path}. "
                "Call fetch_sp500_annual(fetch=True) once to populate the cache."
            )
        px = pd.read_parquet(cache_path)
    else:
        import yfinance as yf  # lazy import: network only on fetch=True

        raw = yf.download("^GSPC", start="1927-01-01", auto_adjust=False, progress=False)
        if raw.empty:
            raise RuntimeError("yfinance returned no ^GSPC data")
        px = raw
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        px.to_parquet(cache_path)

    # Extract the close column (handles MultiIndex from a fresh download).
    if isinstance(px.columns, pd.MultiIndex):
        close = px["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        close = px["Close"]
    close.index = pd.to_datetime(close.index)

    # Year-end close -> calendar-year price return.
    dec = close.resample("YE").last()
    ann_ret = dec.pct_change()
    ann = pd.DataFrame({"ret_year": ann_ret.index.year, "ret": ann_ret.values}).dropna()

    # Match indicator at year-end Y to the return earned in Y+1.
    bi = buffett_table()
    bi = bi[(bi["year"] >= start_year) & (bi["year"] <= end_year)].copy()
    bi["next_year"] = bi["year"] + 1

    merged = bi.merge(ann, left_on="next_year", right_on="ret_year", how="inner")
    merged = merged.rename(columns={"ret": "fwd_ret"})
    merged["fwd_up"] = merged["fwd_ret"] > 0
    merged = merged[["year", "ratio", "fwd_ret", "fwd_up"]].sort_values("year")
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the fwd_ret column, for the as-of stamp."""
    vals = df["fwd_ret"].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
