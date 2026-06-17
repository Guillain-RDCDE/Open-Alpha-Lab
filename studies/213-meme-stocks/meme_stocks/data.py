"""Data layer for Study 213 (Meme Stocks).

The meme-stock universe is a fixed basket of names that became famous in the
January-2021 WSB (WallStreetBets) short-squeeze mania: GME, AMC, BB, BBBY,
KOSS, NOK.  This file handles:

- ``MEME_BASKET`` — the six names, their entry eligibility, and delisting notes.
  BBBY (Bed Bath & Beyond) filed for bankruptcy April 2023 and was delisted;
  we keep it in the basket *as a disclosed gap*, reflecting real survivor bias
  that a buy-and-hold investor would have faced.

- ``load_panel`` — cache-first yfinance download; each ticker cached to the
  study's own ``_cache/`` subdirectory (study-local, never the shared root).

- ``synthetic_panel`` — a deterministic offline control with a tunable
  ``premium`` knob: ``premium > 0`` plants a genuine return edge (the harness
  must detect it); ``premium == 0`` is the null / placebo.

No look-ahead: the basket is fixed at the *documented public listing* date and
each strategy enters at the next trading close after the relevant signal date.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# ---------------------------------------------------------------------------
# The meme basket (WSB / GameStop mania, Jan-2021)
# ---------------------------------------------------------------------------
# Each name + the date it first became "meme" and the note on its fate.
# BBBY delisted 2023-04-26 (bankruptcy).  We include it so the backtest
# carries the full survivorship cost — a basket that silently drops delisted
# names inflates the return.
MEME_TICKERS = ["GME", "AMC", "BB", "BBBY", "KOSS", "NOK"]

# BBBY delisted; yfinance may or may not have a tail; we handle NaN gracefully.
DELISTED = {"BBBY": "2023-04-26"}

# The benchmark is SPY (S&P 500 total return, auto-adjusted).
BENCH_TICKER = "SPY"

# Study window: from 2020-01-01 to capture the run-up and aftermath.
STUDY_START = "2020-01-01"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_path(name: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, name)


def _load_one(ticker: str, cache_dir: str, use_cache: bool) -> pd.Series:
    """Total-return (auto-adjusted) daily closes for one ticker."""
    path = _cache_path(f"{ticker}_px.parquet", cache_dir)
    if use_cache and os.path.exists(path):
        return pd.read_parquet(path)["close"]
    import yfinance as yf

    raw = yf.Ticker(ticker).history(period="max", auto_adjust=True)
    if raw is None or raw.empty:
        return pd.Series(dtype=float, name="close")
    px = raw["Close"].rename("close")
    px.index = pd.to_datetime(px.index).tz_localize(None)
    if use_cache:
        px.to_frame().to_parquet(path)
    return px


def _load_bench(cache_dir: str, use_cache: bool) -> pd.Series:
    """SPY total-return closes."""
    path = _cache_path("SPY_bench.parquet", cache_dir)
    if use_cache and os.path.exists(path):
        return pd.read_parquet(path)["close"]
    import yfinance as yf

    raw = yf.Ticker(BENCH_TICKER).history(period="max", auto_adjust=True)
    px = raw["Close"].rename("close")
    px.index = pd.to_datetime(px.index).tz_localize(None)
    if use_cache:
        px.to_frame().to_parquet(path)
    return px


# ---------------------------------------------------------------------------
# Real panel loader
# ---------------------------------------------------------------------------
def load_panel(
    start: str = STUDY_START,
    end: str | None = None,
    cache_dir: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Load the meme-stock panel and the SPY benchmark.

    Returns a dict with:
    - ``prices``  — DataFrame (dates x tickers), NaN after delisting for BBBY.
    - ``bench``   — Series of SPY total-return closes.
    - ``tickers`` — the basket list (including BBBY with delisting note).
    - ``delisted``— the DELISTED mapping for transparency.
    """
    cache_dir = cache_dir or DEFAULT_CACHE
    os.makedirs(cache_dir, exist_ok=True)

    prices = {}
    for t in MEME_TICKERS:
        s = _load_one(t, cache_dir, use_cache)
        if len(s):
            prices[t] = s

    price_df = pd.DataFrame(prices).sort_index()
    price_df = price_df.loc[start:end] if end else price_df.loc[start:]
    bench = _load_bench(cache_dir, use_cache)
    bench = bench.loc[start:end] if end else bench.loc[start:]

    return {
        "prices": price_df,
        "bench": bench,
        "tickers": MEME_TICKERS,
        "delisted": DELISTED,
    }


# ---------------------------------------------------------------------------
# Synthetic panel — deterministic offline control
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_names: int = 6,
    n_days: int = 1260,          # ~5 trading years
    premium: float = 0.20,       # planted excess CAGR for the meme basket
    vol: float = 1.50,           # annualised daily vol (meme-stock realistic: 150%)
    seed: int = 213,
) -> dict:
    """A deterministic daily panel for the positive/negative control.

    When ``premium > 0`` the meme basket earns a genuine edge over the benchmark
    (the harness MUST detect it).  When ``premium == 0`` the basket and benchmark
    share the same drift; any detected edge is spurious.

    Returns a dict with:
    - ``prices``  — DataFrame (trading days x tickers), auto-adjusted-style.
    - ``bench``   — Series of benchmark prices.
    - ``premium`` — the planted excess CAGR (0.0 = null / placebo).
    - ``seed``    — for reproducibility.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days)

    bench_daily = 0.10 / 252          # ~10% CAGR benchmark
    bench_vol = 0.20 / np.sqrt(252)
    bench_log_ret = rng.normal(bench_daily - 0.5 * bench_vol ** 2, bench_vol, n_days)
    bench_px = 100.0 * np.exp(np.cumsum(bench_log_ret))
    bench = pd.Series(bench_px, index=dates, name="bench")

    meme_daily = (bench_daily + premium / 252) if premium > 0 else bench_daily
    meme_vol = vol / np.sqrt(252)
    prices = {}
    for i, t in enumerate([f"M{i}" for i in range(n_names)]):
        log_ret = rng.normal(meme_daily - 0.5 * meme_vol ** 2, meme_vol, n_days)
        prices[t] = 10.0 * np.exp(np.cumsum(log_ret))
    price_df = pd.DataFrame(prices, index=dates)

    return {
        "prices": price_df,
        "bench": bench,
        "premium": premium,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
def fingerprint(obj) -> str:
    """Short content hash of a tape."""
    h = hashlib.sha1()
    if isinstance(obj, dict):
        for key in ("bench", "prices"):
            if key in obj and obj[key] is not None:
                arr = np.ascontiguousarray(
                    np.nan_to_num(
                        obj[key].to_numpy(dtype=float)
                        if isinstance(obj[key], pd.DataFrame)
                        else np.asarray(obj[key], dtype=float)
                    )
                )
                h.update(arr.tobytes())
    elif isinstance(obj, pd.DataFrame):
        h.update(
            np.ascontiguousarray(np.nan_to_num(obj.to_numpy(dtype=float))).tobytes()
        )
    else:
        h.update(
            np.ascontiguousarray(np.nan_to_num(np.asarray(obj, dtype=float))).tobytes()
        )
    return h.hexdigest()[:12]
