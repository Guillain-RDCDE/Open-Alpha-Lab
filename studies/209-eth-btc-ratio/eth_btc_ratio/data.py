"""Data layer for Study 209 (ETH-BTC-Ratio).

Two tapes, one shape (a tz-naive daily price frame for ETH-USD and BTC-USD):

- ``synthetic_daily`` — a *deterministic, offline* generator. A ``ratio_momentum``
  knob injects autocorrelation into the ETH/BTC log-ratio's daily changes so the
  momentum signal has something real to harvest. ``ratio_momentum=0`` is a pure
  random walk — the null hypothesis — so a test can assert the rotation strategy
  beats buy-and-hold only when we plant trend, and not otherwise.
- ``fetch_daily`` — the real Yahoo! daily tapes for ETH-USD and BTC-USD,
  cache-only by default so the test-suite and reproducible core never touch the
  network. ETH-USD history on Yahoo starts 2017-11-09; this gives ~7.5 years of
  data at the as-of date.

No look-ahead: the ratio signal is computed on close prices up to day *t*; the
rotation trade is entered at *t+1*'s open.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

ETH_TICKER = "ETH-USD"
BTC_TICKER = "BTC-USD"

# ETH-USD first available date on Yahoo Finance
ETH_START = "2017-11-09"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 1000,
    ratio_momentum: float = 0.0,
    daily_vol_btc: float = 0.04,
    daily_vol_eth: float = 0.06,
    corr: float = 0.80,
    annual_drift_btc: float = 0.30,
    annual_drift_eth: float = 0.40,
    start: str = "2018-01-01",
    seed: int = 209,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape for ETH-USD and BTC-USD with a tunable ratio trend.

    Both assets follow correlated geometric Brownian motions. The ETH/BTC log-ratio
    has an additional AR(1) momentum layer: if ``ratio_momentum > 0`` the ratio
    trend persists from day to day, giving a momentum signal something to harvest.
    ``ratio_momentum=0`` is the null: the ratio is a martingale and a momentum
    strategy on it is no better than buy-and-hold or 50/50 rebalancing.

    ``corr`` sets the Cholesky-decomposed correlation between BTC and ETH log-returns
    (before the ratio layer), reflecting the empirical ~0.80 intra-crypto correlation.

    Returns ``(bars, truth)`` where ``bars`` has a two-level column (asset × OHLCV) and
    ``truth`` records the planted parameters. The index is a ``pd.DatetimeIndex`` of
    business days.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days, name="date")

    mu_btc = annual_drift_btc / 252.0
    mu_eth = annual_drift_eth / 252.0

    # Correlated normal shocks via Cholesky
    cov = np.array([
        [daily_vol_btc**2, corr * daily_vol_btc * daily_vol_eth],
        [corr * daily_vol_btc * daily_vol_eth, daily_vol_eth**2],
    ])
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_days, 2))
    shocks = z @ L.T  # shape (n_days, 2): col0=BTC, col1=ETH

    btc_ret = mu_btc + shocks[:, 0]
    eth_ret = mu_eth + shocks[:, 1]

    # Inject ratio momentum: the ETH/BTC log-ratio change has an AR(1) component.
    # We replace the ETH idiosyncratic component with an AR(1) process so that
    # ratio changes (eth_ret - btc_ret) exhibit positive autocorrelation.
    if ratio_momentum != 0.0:
        ratio_noise = rng.normal(0.0, daily_vol_eth * 0.5, n_days)
        ratio_persistent = np.empty(n_days)
        prev = 0.0
        for i in range(n_days):
            ratio_persistent[i] = ratio_momentum * prev + ratio_noise[i]
            prev = ratio_persistent[i]
        # Replace the ETH idiosyncratic shock with the AR(1) process (keeping BTC unchanged)
        eth_ret = btc_ret + ratio_persistent

    btc_close = 30_000.0 * np.exp(np.cumsum(btc_ret))
    eth_close = 2_000.0 * np.exp(np.cumsum(eth_ret))

    def _ohlcv(close: np.ndarray, daily_vol: float, rng_obj) -> dict:
        open_ = np.empty_like(close)
        open_[0] = close[0] * np.exp(-btc_ret[0] if close[0] > 1000 else -eth_ret[0])
        open_[1:] = close[:-1]
        wick = np.abs(rng_obj.normal(0.0, daily_vol * 0.4, close.size)) * close
        hi = np.maximum(open_, close) + wick
        lo = np.minimum(open_, close) - wick
        vol = rng_obj.integers(1_000, 100_000, close.size).astype(float)
        return {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol}

    btc_d = _ohlcv(btc_close, daily_vol_btc, rng)
    eth_d = _ohlcv(eth_close, daily_vol_eth, rng)

    btc_df = pd.DataFrame(btc_d, index=idx)
    eth_df = pd.DataFrame(eth_d, index=idx)

    bars = pd.concat({"BTC-USD": btc_df, "ETH-USD": eth_df}, axis=1)
    bars.columns.names = ["ticker", "field"]

    truth = {
        "ratio_momentum": ratio_momentum,
        "n_days": n_days,
        "daily_vol_btc": daily_vol_btc,
        "daily_vol_eth": daily_vol_eth,
        "corr": corr,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("-", "_").replace("=", "").replace("^", "")
    return os.path.join(cache_dir, f"crypto_{safe}_daily.parquet")


def fetch_daily(
    ticker: str,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached
    as a parquet under ``_cache/``). For ETH-USD, Yahoo history starts 2017-11-09;
    for BTC-USD, from 2014-09-17. We align both to ETH's start date in strategy.py.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_daily({ticker!r}, fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, period="max", interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars.sort_index()


def load_pair(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    start: str = ETH_START,
) -> pd.DataFrame:
    """Load and align BTC-USD and ETH-USD into a dual-column frame.

    Both tapes are clipped to ``start`` (ETH's first available date on Yahoo) so
    the sample is balanced. Returns a DataFrame with a two-level column index
    (ticker × field), tz-naive daily index, sorted ascending.
    """
    btc = fetch_daily(BTC_TICKER, fetch=fetch, cache_dir=cache_dir)
    eth = fetch_daily(ETH_TICKER, fetch=fetch, cache_dir=cache_dir)

    btc = btc[btc.index >= pd.Timestamp(start)]
    eth = eth[eth.index >= pd.Timestamp(start)]

    # Align on the intersection of trading days (crypto trades 7 days but
    # Yahoo sometimes has gaps — inner join keeps only shared dates)
    common = btc.index.intersection(eth.index)
    btc = btc.loc[common]
    eth = eth.loc[common]

    bars = pd.concat({"BTC-USD": btc, "ETH-USD": eth}, axis=1)
    bars.columns.names = ["ticker", "field"]
    return bars


def fingerprint(bars: pd.DataFrame, ticker: str = "BTC-USD") -> str:
    """A short content fingerprint of a tape's close column, for the as-of stamp."""
    arr = bars[ticker]["close"].to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
