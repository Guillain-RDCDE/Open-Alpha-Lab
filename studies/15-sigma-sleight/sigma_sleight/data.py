"""Data access for the RSI study — the price tape and where it comes from.

The whole study runs on one thing: a single column of **closing prices** on a daily index. RSI
and every AdaptiveRSI transform are pure functions of that series, so the data layer is
deliberately thin — the desk's standing split between an offline synthetic tape and a cache-only
real reader:

    * :func:`synthetic_prices` — fully **offline**. A log-price Ornstein–Uhlenbeck process with a
      *baked-in* mean-reversion strength ``kappa`` (the real, modest edge an RSI mean-reversion
      strategy is supposed to harvest) and **one** characteristic horizon (~``1/kappa`` bars).
      Two facts are baked in on purpose: a genuine oversold-bounce edge *exists* (so the strategy
      is not testing noise), and because there is a *single* reversion horizon, a longer-length
      RSI rescaled onto the short scale carries **no information the native short RSI lacks** —
      the null the Rescaled-RSI leg exists to expose. Deterministic given ``seed``; no network.
    * :func:`fetch_prices` — pull real daily closes from Yahoo! Finance, cache each ticker to
      parquet, and return the single ``close`` column. **Cache-only** unless ``fetch=True``: a
      missing cache is skipped, never a silent stall, so the offline core (and CI) never imports
      ``yfinance``.

Daily fidelity is the decision here (named up front, per the house rule): the AdaptiveRSI pitch
is written for and illustrated on daily/weekly charts, the σ↔RSI identity it rests on is
timeframe-agnostic, and daily closes give ~10 years of clean, split-adjusted history for the
horse race without the rolling-window cap that intraday Yahoo imposes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")


# --------------------------------------------------------------------------- #
# Synthetic tape — offline, with a baked-in single-horizon mean reversion
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GroundTruth:
    """What the synthetic generator baked in, so a test can check the decomposition."""
    kappa: float          # OU mean-reversion strength per bar (the real, modest edge)
    horizon: float        # the single characteristic reversion horizon, ~1/kappa bars
    sigma_step: float     # per-bar shock vol of the log-price
    n_bars: int


def synthetic_prices(
    n_bars: int = 2520,
    kappa: float = 0.06,
    sigma_step: float = 0.011,
    start_price: float = 100.0,
    seed: int = 0,
) -> tuple[pd.Series, GroundTruth]:
    """A toy daily close series that mean-reverts at a **known** rate around a slow-drifting level.

    Log-price follows an Ornstein–Uhlenbeck process around a level ``mu`` that itself takes a tiny
    random walk (so the series wanders like a real index instead of pinning to a constant)::

        x[t] = x[t-1] + kappa * (mu[t] - x[t-1]) + sigma_step * eps[t]

    The pull-back term ``kappa * (mu - x)`` is the baked-in edge: after a down-swing (RSI oversold)
    the next moves tilt *up*, which is exactly what an RSI mean-reversion rule tries to monetise.
    Crucially there is a **single** reversion horizon ``~1/kappa`` — no second, slower cycle — so a
    long-length RSI rescaled onto a short scale adds nothing the short RSI didn't already see. That
    is the null the Rescaled-RSI leg is built to catch.

    Returns ``(close, truth)`` — ``close`` is a strictly-positive price ``pd.Series`` on a
    business-day index named ``date``; ``truth`` carries the baked-in parameters. Deterministic
    given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2016-01-04", periods=n_bars, name="date")

    x = np.empty(n_bars)
    x[0] = np.log(start_price)
    mu = np.log(start_price)
    mu_step = sigma_step * 0.15                      # the level drifts much slower than the noise
    for t in range(1, n_bars):
        mu += mu_step * rng.standard_normal()
        x[t] = x[t - 1] + kappa * (mu - x[t - 1]) + sigma_step * rng.standard_normal()

    close = pd.Series(np.exp(x), index=idx, name="close")
    truth = GroundTruth(kappa=kappa, horizon=1.0 / kappa, sigma_step=sigma_step, n_bars=n_bars)
    return close, truth


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! daily closes, cached per ticker
# --------------------------------------------------------------------------- #

def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "_").replace("^", "_")
    return os.path.join(cache_dir, f"close_{safe}.parquet")


def fetch_prices(
    ticker: str,
    period: str = "10y",
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
) -> pd.Series:
    """Return (and cache) the daily **close** series for ``ticker``, on a ``date`` index.

    **Cache-only by default**: if a parquet exists it is returned; otherwise an empty series is
    returned *unless* ``fetch=True``, in which case Yahoo is hit once (``auto_adjust=True``, so the
    closes are split/dividend-adjusted — a data-choice the house rules ask us to state) and the
    result cached. The network import is lazy, so the offline core never imports ``yfinance``.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path) and not fetch:
        return pd.read_parquet(path)["close"]

    if not fetch:
        return pd.Series(dtype="float64", name="close")

    import yfinance as yf  # lazy: offline core never imports it

    raw = yf.download(ticker, period=period, interval="1d",
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.Series(dtype="float64", name="close")

    col = raw["Close"]
    if isinstance(col, pd.DataFrame):                # yfinance MultiIndex on a single ticker
        col = col.iloc[:, 0]
    close = col.astype("float64")
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    close.index.name = "date"
    close.name = "close"
    close = close[~close.index.duplicated(keep="last")].sort_index()

    os.makedirs(cache_dir, exist_ok=True)
    close.to_frame().to_parquet(path)
    return close
