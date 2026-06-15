"""Data layer for Study 206 (Dividend-Aristocrats).

Two tapes, one shape (a date-indexed daily adjusted-close price frame):

- ``synthetic_daily`` — a *deterministic, offline* generator.  Two correlated
  equity series are generated from a two-regime drift process.  The ``alpha_bps``
  knob plants an annualised daily-return advantage for the "aristocrat" series over
  its benchmark.  At ``alpha_bps=0`` both series share the same drift — the null
  hypothesis that quality/dividend-growth adds nothing net of shared market beta.
  This is the study's null in a bottle: any test that asserts a return advantage
  must fail here and succeed only when alpha is planted.

- ``fetch_prices`` — daily adjusted closes from Yahoo! Finance (``yfinance``), cache-
  only by default.  ``fetch=False`` raises ``FileNotFoundError`` if the parquet is
  absent (CI-safe).  ``fetch=True`` downloads and caches as parquet.  Uses
  ``auto_adjust=True`` for split/dividend-adjusted total-return closes — critical for
  a dividend-focused study.

Tickers used:
  NOBL  — ProShares S&P 500 Dividend Aristocrats ETF (tracks the index since Oct 2013)
  SPY   — SPDR S&P 500 ETF (market benchmark, total return)
  QQQ   — Invesco QQQ (growth benchmark; expected to widen the underperformance gap)

No look-ahead is baked in here — that discipline lives in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# Study tickers — NOBL as the aristocrat proxy, SPY as the benchmark.
TICKERS = ["NOBL", "SPY"]
NOBL_START = "2013-10-09"   # NOBL inception date


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 12,
    market_drift_ann: float = 0.10,   # benchmark annualised return
    market_vol_ann: float = 0.16,     # shared market vol
    alpha_bps: float = 0.0,           # planted daily alpha for aristocrat (bps/day)
    correlation: float = 0.97,         # NOBL/SPY are highly correlated (quality factor)
    start: str = "2013-10-09",
    seed: int = 206,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible two-asset daily price series: a benchmark and an aristocrat proxy.

    Both assets share a common market factor drawn from a normal distribution with
    ``market_drift_ann`` and ``market_vol_ann``.  The aristocrat series gets an
    additional planted daily alpha of ``alpha_bps`` basis points.  Residual noise is
    added so the correlation matches ``correlation``.

    ``alpha_bps = 0`` is the null: the aristocrat and the benchmark have identical
    risk-adjusted returns.  Tests that claim a quality/dividend-growth premium should
    fail here and pass only when alpha is planted.

    Returns ``(prices, truth)`` where ``prices`` has columns ``NOBL`` (aristocrat) and
    ``SPY`` (benchmark), indexed by date, and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    n_days = n_years * TRADING_DAYS_PER_YEAR
    dates = pd.bdate_range(start=start, periods=n_days)

    d_market = market_drift_ann / TRADING_DAYS_PER_YEAR
    s_market = market_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)

    # Shared market factor
    market_ret = rng.normal(d_market, s_market, n_days)

    # Residual vol to achieve target correlation:
    # corr(NOBL, SPY) ≈ 1/sqrt(1 + (s_resid/s_market)^2)  (simplified)
    # => s_resid = s_market * sqrt((1/corr^2) - 1)
    s_resid = s_market * np.sqrt(max(1.0 / correlation**2 - 1.0, 1e-10))

    alpha_daily = alpha_bps * 1e-4

    # SPY tracks the market factor
    spy_log_ret = market_ret + rng.normal(0.0, s_resid * 0.5, n_days)
    # NOBL = market + planted alpha + idiosyncratic noise
    nobl_log_ret = market_ret + alpha_daily + rng.normal(0.0, s_resid, n_days)

    spy_close = 100.0 * np.exp(np.cumsum(spy_log_ret))
    nobl_close = 100.0 * np.exp(np.cumsum(nobl_log_ret))

    prices = pd.DataFrame(
        {"NOBL": nobl_close, "SPY": spy_close},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "n_years": n_years,
        "n_days": n_days,
        "market_drift_ann": market_drift_ann,
        "market_vol_ann": market_vol_ann,
        "alpha_bps": alpha_bps,
        "alpha_ann_pct": alpha_daily * TRADING_DAYS_PER_YEAR * 100,
        "correlation": correlation,
        "seed": seed,
    }
    return prices, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo! Finance daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch_prices(
    ticker: str,
    start: str = NOBL_START,
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Daily adjusted close prices for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``).  Returns a DataFrame with a single ``close`` column
    indexed by date.  Uses ``auto_adjust=True`` — split/dividend-adjusted total-return
    close — essential for a dividend-focused study (unadjusted prices would understate
    total return for NOBL, which passes through its held dividends).
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily prices for {ticker} at {path}. "
                f"Call fetch_prices({ticker!r}, fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, start=start, end=end, interval="1d",
            auto_adjust=True, progress=False,
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        df = raw[["close"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(path)

    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.dropna(subset=["close"])
    return df


def load_aligned(
    tickers: list[str] | None = None,
    start: str = NOBL_START,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Load and align multiple tickers into one price DataFrame (inner join on dates).

    Returns a date-indexed DataFrame with one column per ticker, starting from
    ``start``.  All prices are adjusted total-return closes.
    """
    if tickers is None:
        tickers = TICKERS
    frames = {}
    for t in tickers:
        df = fetch_prices(t, start=start, fetch=fetch, cache_dir=cache_dir)
        frames[t] = df["close"].rename(t)
    prices = pd.concat(frames.values(), axis=1, join="inner")
    prices = prices[prices.index >= pd.Timestamp(start)]
    return prices.dropna()


def fingerprint(prices: pd.DataFrame) -> str:
    """A short content fingerprint of a price frame (all columns), for the as-of stamp."""
    arr = np.ascontiguousarray(prices.to_numpy().ravel())
    h = hashlib.sha1(arr.tobytes())
    return h.hexdigest()[:12]
