"""Data layer for Study 290 (September-Effect).

The "September effect" is the folklore that September is the stock market's
worst calendar month -- the only month with a (slightly) negative long-run
average return for the S&P 500.  This study isolates *September specifically*
(it is the sibling of the broader October-effect / Halloween-indicator studies,
which look at autumn crash-risk and the May-October seasonal).

Two tapes, one shape (a tidy month x return frame):

- ``synthetic_monthly`` -- a *deterministic, offline* monthly-return generator
  with an optional planted "September drag" knob (``sep_bps``).  ``sep_bps = 0``
  is the null: the calendar month carries no information.  A negative
  ``sep_bps`` plants exactly the September seasonality the folklore claims, so
  the positive control can confirm the machinery is honest before we look at
  the real tape.

- ``fetch_sp500_monthly`` -- real S&P 500 *price-only* monthly returns from the
  Shiller dataset (``_cache/shiller_sp500.parquet``, staged in the repo-level
  cache).  Cache-only by default (``fetch=False``); the network is only ever
  touched on an explicit ``fetch=True`` (a lazy yfinance fall-back for a live
  ^GSPC pull).  The core study never needs the network.

No look-ahead: the September dummy is contemporaneous with the return it
labels (the month *is* the observation).  A trading rule that "avoids
September" forms its position at the August month-end close and re-enters at
the September month-end close -- one month-end of execution lag, applied to
calendar information that is known years in advance.

Price-only vs total return: the Shiller monthly SP500 column is a *price*
index (dividends excluded).  The September drag is a price-return statistic;
since September pays roughly the same dividend yield as any other month, the
total-return September gap is slightly *less* negative than the price-only gap
reported here.  This is labelled wherever it matters.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))

SHILLER_PARQUET = os.path.join(REPO_CACHE, "shiller_sp500.parquet")
GSPC_CACHE = os.path.join(DEFAULT_CACHE, "september_gspc_monthly.parquet")

# Study window: the modern, liquid S&P 500 era.  1950 is the conventional
# start for "modern" S&P seasonality studies (the index was reconstituted to
# 500 names in 1957, but the 90-stock composite back to 1950 is the usual
# reference window in the September-effect literature).
START_YEAR = 1950
END_YEAR = 2025

SEPTEMBER = 9


# ---------------------------------------------------------------------------
# Synthetic monthly generator -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_monthly(
    n_years: int = 76,
    sep_bps: float = 0.0,
    base_ret: float = 0.006,
    vol_mo: float = 0.04,
    start_year: int = START_YEAR,
    seed: int = 290,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible monthly-return series with an optional planted September drag.

    Each month is drawn i.i.d. from ``N(base_ret, vol_mo**2)``.  In September
    (calendar month 9) an extra drift of ``sep_bps`` basis points is added
    (use a *negative* value to plant the folklore's September drag).  When
    ``sep_bps = 0`` the calendar month is uninformative -- this is the null in
    a bottle, used by the tests and the positive control.

    Returns ``(df, truth)`` where ``df`` has columns:
      ``date`` (month-end Timestamp), ``year`` (int), ``month`` (1..12),
      ``ret`` (monthly simple return), ``is_sep`` (bool).
    """
    rng = np.random.default_rng(seed)
    n_months = n_years * 12
    dates = pd.date_range(f"{start_year}-01-31", periods=n_months, freq="ME")
    months = dates.month.to_numpy()
    is_sep = months == SEPTEMBER

    rets = rng.normal(base_ret, vol_mo, n_months)
    rets = rets + np.where(is_sep, sep_bps * 1e-4, 0.0)

    df = pd.DataFrame(
        {
            "date": dates,
            "year": dates.year,
            "month": months,
            "ret": rets,
            "is_sep": is_sep,
        }
    )
    truth = {
        "n_years": n_years,
        "sep_bps": sep_bps,
        "base_ret": base_ret,
        "vol_mo": vol_mo,
        "start_year": start_year,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape -- Shiller S&P 500 monthly price returns (repo-level cache)
# ---------------------------------------------------------------------------
def fetch_sp500_monthly(
    fetch: bool = False,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    cache_dir: str = REPO_CACHE,
) -> pd.DataFrame:
    """Monthly S&P 500 *price* returns from the Shiller dataset; cache-only by default.

    By default (``fetch=False``) this reads the repo-level Shiller parquet at
    ``_cache/shiller_sp500.parquet`` (a monthly nominal SP500 price column back
    to 1871) and computes month-over-month simple returns in the study window.
    No network is touched.

    On an explicit ``fetch=True`` a lazy yfinance fall-back downloads ^GSPC
    monthly closes and caches them under
    ``_cache/september_gspc_monthly.parquet`` -- used only to refresh the tape,
    never in CI.

    Returns a DataFrame with columns:
      ``date`` (month-end Timestamp), ``year`` (int), ``month`` (1..12),
      ``ret`` (monthly price return), ``is_sep`` (bool), ``up`` (bool).

    Raises ``FileNotFoundError`` if ``fetch=False`` and the Shiller cache is
    absent.
    """
    if fetch:
        return _fetch_gspc_live(start_year, end_year)

    shiller_path = os.path.join(cache_dir, "shiller_sp500.parquet")
    if not os.path.exists(shiller_path):
        raise FileNotFoundError(
            f"Shiller cache not found at {shiller_path}. "
            "The repo-level _cache/shiller_sp500.parquet must be present, "
            "or call fetch_sp500_monthly(fetch=True) once to populate a ^GSPC cache."
        )
    sh = pd.read_parquet(shiller_path)

    if "Date" in sh.columns:
        idx = pd.to_datetime(sh["Date"], errors="coerce")
    elif isinstance(sh.index, pd.DatetimeIndex):
        idx = sh.index
    else:
        idx = pd.to_datetime(sh.index, errors="coerce")

    sp = pd.Series(sh["SP500"].to_numpy(dtype=float), index=idx)
    sp = sp[sp.gt(0) & sp.notna()].sort_index()

    return _frame_from_price_series(sp, start_year, end_year)


def _frame_from_price_series(sp: pd.Series, start_year: int, end_year: int) -> pd.DataFrame:
    """Turn a month-indexed price series into the tidy returns frame."""
    ret = sp.pct_change()
    df = pd.DataFrame(
        {
            "date": ret.index,
            "year": ret.index.year,
            "month": ret.index.month,
            "ret": ret.to_numpy(dtype=float),
        }
    ).dropna(subset=["ret"])
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
    df["is_sep"] = df["month"] == SEPTEMBER
    df["up"] = df["ret"] > 0
    return df.reset_index(drop=True)


def _fetch_gspc_live(start_year: int, end_year: int) -> pd.DataFrame:
    """Lazy yfinance fall-back: download ^GSPC monthly closes and cache them."""
    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download(
        "^GSPC",
        start=f"{start_year}-01-01",
        interval="1mo",
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no ^GSPC monthly data")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index) + pd.offsets.MonthEnd(0)
    close = close[close.gt(0) & close.notna()].sort_index()

    os.makedirs(os.path.dirname(GSPC_CACHE), exist_ok=True)
    close.to_frame("Close").to_parquet(GSPC_CACHE)
    print(f"Cached ^GSPC monthly closes -> {GSPC_CACHE}")

    return _frame_from_price_series(close, start_year, end_year)


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the ``ret`` column, for the as-of stamp."""
    vals = df["ret"].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
