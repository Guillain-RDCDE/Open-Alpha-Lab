"""Data layer for Study 81 (Four-Year-Itch).

Two tapes, both daily:

- ``synthetic_daily`` — a *deterministic, offline* generator. Returns follow i.i.d. normal
  noise plus an optional per-year-of-term drift vector ``year_premia`` (a length-4 array of
  extra bps). ``year_premia = [0, 0, 0, 0]`` is the null (no cycle) — the machinery test's
  control arm. This is the study's null in a bottle.
- ``fetch_prices`` — real daily total-return S&P 500 proxies (``^GSPC`` for the long price
  history back to 1928, ``SPY`` for the post-1993 total-return version), cache-only by
  default so the test-suite and the offline core never touch the network.

The election-cycle label (year 1–4 of a presidential term) is assigned by
:func:`term_year_label`; it is a pure function of the calendar date and known *before* the
year begins — there is no look-ahead risk in the labelling.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

# US presidential terms since 1929 (year 1 = January of an inauguration year).
# Covers cycles starting 1929 (Hoover) through the projected 2025 term.
# Each element is the calendar year in which year-1 of the term begins.
_TERM_START_YEARS = [
    1929, 1933, 1937, 1941, 1945, 1949, 1953, 1957,
    1961, 1965, 1969, 1973, 1977, 1981, 1985, 1989,
    1993, 1997, 2001, 2005, 2009, 2013, 2017, 2021, 2025,
]


# ---------------------------------------------------------------------------
# Calendar: assign year-of-term (1–4) to a DatetimeIndex
# ---------------------------------------------------------------------------
def term_year_label(index: pd.DatetimeIndex) -> pd.Series:
    """Assign each date a Presidential-cycle label (1, 2, 3, or 4).

    Term year 1 = the inauguration year (January of the 1st year of a 4-year term). The label
    is a pure function of the calendar year and the list of known term-start years, so it is
    available *before* the year begins — no look-ahead risk.

    Dates outside any defined term window get ``NaN`` (they are dropped by the strategy).
    """
    years = pd.Series(index.year, index=index, name="yr")
    labels = pd.Series(index=index, dtype="float64", name="term_year")
    for start in _TERM_START_YEARS:
        for offset in range(4):
            mask = years == (start + offset)
            labels.loc[mask] = offset + 1
    return labels


# ---------------------------------------------------------------------------
# Synthetic tape — deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_cycles: int = 24,
    year_premia: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    base_bp: float = 3.5,
    vol_ann: float = 0.16,
    seed: int = 81,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily return series with a planted presidential-cycle premium.

    Each trading year gets ``base_bp`` of drift. The four year-of-term slots receive
    additional ``year_premia[i-1]`` bps on top. ``year_premia = (0, 0, 0, 0)`` is the
    null — the cycle is absent and every year-of-term should look the same.

    The canonical literature claim is roughly ``year_premia ≈ (-6, -3, +10, +4)`` (bps/day
    vs the base), but we default to the null so tests confirm the engine is truthful when
    nothing is planted.

    Returns ``(daily_df, truth)`` where ``daily_df`` has columns ``ret`` (simple daily
    return) and ``term_year`` (1–4), and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    # n_cycles full 4-year terms = n_cycles * 4 calendar years ≈ n_cycles * 4 * 252 days
    n_years = n_cycles * 4
    idx = pd.bdate_range("1929-01-01", periods=n_years * 252, name="date")

    ty = term_year_label(idx)
    sig_d = vol_ann / np.sqrt(252)

    drift = np.full(len(idx), base_bp * 1e-4, dtype=float)
    for i, prem in enumerate(year_premia):
        drift[ty.values == (i + 1)] += prem * 1e-4

    ret = drift + sig_d * rng.standard_normal(len(idx))
    df = pd.DataFrame({"ret": ret, "term_year": ty.astype("Int64")}, index=idx)
    truth = {
        "n_cycles": n_cycles,
        "year_premia": list(year_premia),
        "base_bp": base_bp,
        "vol_ann": vol_ann,
        "seed": seed,
        "n_days": len(idx),
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("^", "").replace("=", "").replace("/", "").lower()
    return os.path.join(cache_dir, f"four_year_itch_{safe}_daily.parquet")


def fetch_prices(
    ticker: str = "^GSPC",
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start_year: int = 1928,
) -> pd.DataFrame:
    """Daily close and simple-return for ``ticker``, cache-first.

    ``^GSPC`` is the S&P 500 **price index** (no dividends, but available back to 1928 on
    Yahoo — ~24 complete presidential cycles). ``SPY`` (auto-adjusted, so dividends
    re-invested) gives a shorter but total-return record. Returns a DataFrame with columns
    ``close`` and ``ret``, plus a ``term_year`` column (1–4, NaN outside defined terms).

    Network is touched only with ``fetch=True``; otherwise raises ``FileNotFoundError`` if
    the cache does not exist.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_prices({ticker!r}, fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only on explicit fetch

        raw = yf.download(ticker, period="max", interval="1d", auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close = raw["Close"].dropna().squeeze()
        close.index = pd.DatetimeIndex(close.index).tz_localize(None)
        close = close[close.index < pd.Timestamp.now().normalize()]
        close.index.name = "date"
        ret = close.pct_change().dropna()
        df = pd.DataFrame({"close": close, "ret": ret})
        df.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(path)

    df = df[df.index.year >= start_year].copy()
    df["term_year"] = term_year_label(pd.DatetimeIndex(df.index)).values
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the close or return column, for the as-of stamp."""
    col = "close" if "close" in df.columns else "ret"
    h = hashlib.sha1(np.ascontiguousarray(df[col].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]
