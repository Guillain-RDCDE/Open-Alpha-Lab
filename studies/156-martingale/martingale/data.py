"""Data layer for Study 156 (Martingale).

Two tapes, one shape (a daily price series):

- ``synthetic_daily`` — a *deterministic, offline* generator. A drift knob (``mu``) and
  a mean-reversion knob (``mean_reversion``) let us dial the market regime:

  - ``mean_reversion=0, mu=0`` — a pure random walk / driftless martingale (the null).
  - ``mean_reversion>0``       — a mean-reverting process where averaging-down might look
    good short-term, making the strategy look compelling before the tail strikes.
  - ``mu<0``                   — a downtrending market that triggers a ruin cascade.

  This is the study's null in a bottle: only a truly mean-reverting market (negative
  short-lag autocorrelation) could help an averaging-down scheme; random walks and trending
  markets expose its gambler's-ruin tail.

- ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only by default so
  the test-suite and reproducible core never touch the network. A long daily history
  (~30 years of SPY) gives adequate power. The same shapes are also pulled for a small set
  of individual stocks so we can showcase the ruin scenario.

No look-ahead: the signal is constructed on the close at day *t*; the position change is
executed at day *t+1*'s open (approximated as that day's close for simplicity, with a
note that at daily resolution this is a minor approximation).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 1000,
    mu: float = 0.0,
    annual_vol: float = 0.18,
    mean_reversion: float = 0.0,
    start: str = "2010-01-04",
    seed: int = 156,
    start_price: float = 100.0,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily price series with a known market regime.

    Log returns follow a discrete-time Ornstein-Uhlenbeck-flavoured AR(1):
    ``r_t = mu_daily + mean_reversion*(log_mean - log_price_{t-1}) + sigma * eps_t``
    where ``eps_t`` i.i.d. normal. Setting ``mean_reversion=0`` gives a
    geometric random walk (the martingale benchmark). Positive ``mean_reversion``
    creates a pull-back effect that makes averaging-down *look* better before the
    tail eventually arrives. Negative ``mu`` creates a trend that triggers ruin.

    Returns ``(prices, truth)`` where ``prices`` is a DataFrame with columns
    ``open, high, low, close, volume``, and ``truth`` records planted parameters.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)

    mu_daily = mu / TRADING_DAYS_PER_YEAR
    sigma_daily = annual_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    log_mean = np.log(start_price)

    log_price = np.empty(n_days)
    log_price[0] = np.log(start_price)
    eps = rng.standard_normal(n_days)

    for t in range(1, n_days):
        pull = mean_reversion * (log_mean - log_price[t - 1])
        log_price[t] = log_price[t - 1] + mu_daily + pull + sigma_daily * eps[t]

    close = np.exp(log_price)
    # Intraday range: a small uniform wick around the close.
    wick = np.abs(rng.normal(0.0, sigma_daily * 0.5, n_days)) * close
    open_ = close * np.exp(rng.normal(0.0, sigma_daily * 0.3, n_days))
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - np.minimum(wick, close * 0.9)
    lo = np.maximum(lo, close * 0.01)  # floor to avoid negatives
    volume = rng.integers(1_000_000, 50_000_000, n_days).astype(float)

    prices = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": volume},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "mu": mu,
        "annual_vol": annual_vol,
        "mean_reversion": mean_reversion,
        "n_days": n_days,
        "seed": seed,
        "start_price": start_price,
    }
    return prices, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"martingale_{safe}_1d.parquet")


def fetch_daily(
    ticker: str,
    start: str = "1993-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``). Returns a DataFrame with lowercase columns
    ``open, high, low, close, volume`` and a ``date`` DatetimeIndex.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_daily({ticker!r}, fetch=True) once to populate the cache."
            )
        prices = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when actually fetching

        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        prices = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        prices.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        prices.to_parquet(path)

    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    return prices


def fingerprint(prices: pd.DataFrame) -> str:
    """A short content fingerprint of the price series (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(prices["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
