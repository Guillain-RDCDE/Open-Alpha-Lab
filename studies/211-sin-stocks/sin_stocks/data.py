"""Data layer for Study 211 (Sin-Stocks).

Two tapes, one shape (a date-indexed daily adjusted-close price frame):

- ``synthetic_daily`` — a *deterministic, offline* generator.  Three correlated
  equity series are generated from a common market factor plus idiosyncratic noise.
  The ``alpha_bps`` knob plants an annualised daily-return advantage for the sin basket
  over its benchmark.  At ``alpha_bps=0`` all series share the same risk-adjusted
  drift — the null hypothesis that vice stocks add nothing beyond shared market beta.

- ``fetch_prices`` — daily adjusted closes from Yahoo! Finance (``yfinance``), cache-
  only by default.  ``fetch=False`` raises ``FileNotFoundError`` if the parquet is
  absent (CI-safe).  ``fetch=True`` downloads and caches as parquet.  Uses
  ``auto_adjust=True`` for split/dividend-adjusted total-return closes.

Tickers used:
  SIN basket (equally weighted):
    MO    — Altria Group (tobacco, US)
    PM    — Philip Morris International (tobacco, international)
    STZ   — Constellation Brands (alcohol/beer)
    LVS   — Las Vegas Sands (gambling/casinos)
    LMT   — Lockheed Martin (defense)
    RTX   — RTX Corp (defense, formerly Raytheon Technologies)

  SPY   — SPDR S&P 500 ETF (market benchmark)
  DSI   — iShares MSCI KLD 400 Social ETF (ESG/socially responsible benchmark)

No look-ahead is baked in here — that discipline lives in ``strategy.py``.

NOTE on the start date: DSI (iShares MSCI KLD 400 Social ETF) launched 2006-11-14.
We use 2007-01-03 as the common start for a clean calendar year.
PM (Philip Morris spun out in 2008-03-17) sets the effective inner-join start date.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# Sin basket tickers — a broad cross-sector vice portfolio
SIN_BASKET = ["MO", "PM", "STZ", "LVS", "LMT", "RTX"]

# Benchmark tickers
BENCHMARKS = ["SPY", "DSI"]

ALL_TICKERS = SIN_BASKET + BENCHMARKS

# Earliest reliable date: DSI launched 2006-11-14; we start at the first
# full calendar year after that for a clean sample.
# The inner join with PM (spun out 2008-03-17) means the aligned sample
# starts 2008-03-17 in practice.
START_DATE = "2007-01-03"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 15,
    market_drift_ann: float = 0.09,   # benchmark annualised return
    market_vol_ann: float = 0.17,     # shared market vol
    alpha_bps: float = 0.0,           # planted daily alpha for sin basket (bps/day)
    correlation: float = 0.75,         # sin stocks vs SPY — materially lower than pure equity
    start: str = "2007-01-03",
    seed: int = 211,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible three-asset daily price series: sin basket, SPY, and ESG proxy.

    All three assets share a common market factor drawn from a normal distribution with
    ``market_drift_ann`` and ``market_vol_ann``.  The sin basket gets an additional
    planted daily alpha of ``alpha_bps`` basis points and lower market loading (beta ~ 0.75).
    Residual noise is added so the sin-vs-SPY correlation matches ``correlation``.

    The ESG proxy is treated as nearly identical to SPY (correlation ~0.97) to reflect
    how ESG ETFs in practice track the market closely while excluding sin stocks.

    ``alpha_bps = 0`` is the null: the sin basket and the benchmark have identical
    risk-adjusted returns.  Tests that claim a neglect premium should fail here and
    pass only when alpha is planted.

    Returns ``(prices, truth)`` where ``prices`` has columns ``SIN``, ``SPY``, and ``DSI``,
    indexed by date, and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    n_days = n_years * TRADING_DAYS_PER_YEAR
    dates = pd.bdate_range(start=start, periods=n_days)

    d_market = market_drift_ann / TRADING_DAYS_PER_YEAR
    s_market = market_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)

    # Shared market factor
    market_ret = rng.normal(d_market, s_market, n_days)

    alpha_daily = alpha_bps * 1e-4

    # SPY tracks the market factor closely
    s_resid_spy = s_market * 0.1   # very small idiosyncratic noise for SPY
    spy_log_ret = market_ret + rng.normal(0.0, s_resid_spy, n_days)

    # Sin basket: lower market loading (beta ~ 0.75) + planted alpha + idiosyncratic noise
    s_resid_sin = s_market * np.sqrt(max(1.0 / correlation**2 - 1.0, 1e-10))
    sin_log_ret = 0.75 * market_ret + alpha_daily + rng.normal(0.0, s_resid_sin, n_days)

    # DSI: similar to SPY but slightly different sector mix
    s_resid_esg = s_market * 0.08
    dsi_log_ret = 0.98 * market_ret + rng.normal(0.0, s_resid_esg, n_days)

    spy_close = 100.0 * np.exp(np.cumsum(spy_log_ret))
    sin_close = 100.0 * np.exp(np.cumsum(sin_log_ret))
    dsi_close = 100.0 * np.exp(np.cumsum(dsi_log_ret))

    prices = pd.DataFrame(
        {"SIN": sin_close, "SPY": spy_close, "DSI": dsi_close},
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
    start: str = START_DATE,
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Daily adjusted close prices for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``).  Returns a DataFrame with a single ``close`` column
    indexed by date.  Uses ``auto_adjust=True`` — split/dividend-adjusted total-return
    close — essential for a dividend-heavy sin-stocks comparison (tobacco and defense
    stocks are high-dividend payers; unadjusted prices would understate their total return).
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


def load_sin_basket(
    basket: list[str] | None = None,
    start: str = START_DATE,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Load and align the sin basket tickers into one price DataFrame (inner join on dates).

    Returns a date-indexed DataFrame with one column per ticker, starting from
    ``start``.  All prices are adjusted total-return closes.
    """
    if basket is None:
        basket = SIN_BASKET
    frames = {}
    for t in basket:
        df = fetch_prices(t, start=start, fetch=fetch, cache_dir=cache_dir)
        frames[t] = df["close"].rename(t)
    prices = pd.concat(frames.values(), axis=1, join="inner")
    prices = prices[prices.index >= pd.Timestamp(start)]
    return prices.dropna()


def load_aligned(
    tickers: list[str] | None = None,
    start: str = START_DATE,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Load all tickers (sin basket + benchmarks) aligned on common dates.

    Returns a DataFrame with columns: MO, PM, STZ, LVS, LMT, RTX, SPY, DSI.
    The inner join effectively starts at PM's inception date (2008-03-17).
    """
    if tickers is None:
        tickers = ALL_TICKERS
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
