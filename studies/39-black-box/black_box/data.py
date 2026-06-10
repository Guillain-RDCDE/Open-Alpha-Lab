"""Data for the neural-net crypto study — an offline synthetic price series with a KNOWN weak nonlinear
predictable component (the positive control) and a pure-random-walk null, plus the real crypto tape.

The desk's offline/cache split:

  * :func:`synthetic_predictable` — fully **offline, deterministic**. A daily log-price built from a
    geometric random walk whose *next-day* drift carries a small, deliberately **nonlinear** function of
    the last two daily returns (``μ_t = strength · tanh(8·r_{t-1}·r_{t-2})``). The dependence is weak,
    nonlinear and lagged — exactly the kind of thing an MLP *can* learn on the control but a linear rule
    largely misses. ``predictable_strength = 0`` is the **null**: a pure geometric Brownian random walk
    with nothing to predict. Returns ``(prices DataFrame, NextDayTruth)``.
  * :func:`fetch_crypto` — real daily closes for a few liquid coins (BTC-USD, ETH-USD, LTC-USD, XRP-USD)
    from Yahoo! Finance, **cache-first** into ``_cache/black_box_crypto.parquet``. Network only with
    ``fetch=True``; the yfinance import stays lazy so the offline core never needs it.

Two data choices, stated. **Daily closes** (crypto trades 24/7, so a "day" is a clean UTC daily bar) and
**no survivorship filtering** — we name the coins explicitly, so what you see is what was tested.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
CACHE_DIR = os.path.join(_HERE, "..", "_cache")
CACHE_FILE = os.path.join(CACHE_DIR, "black_box_crypto.parquet")
TRADING_DAYS = 365  # crypto trades every calendar day

DEFAULT_COINS = ("BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD")


@dataclass(frozen=True)
class NextDayTruth:
    n_bars: int
    predictable_strength: float

    @property
    def has_signal(self) -> bool:
        return self.predictable_strength != 0.0


def synthetic_predictable(n_bars: int = 365 * 6, predictable_strength: float = 0.0009,
                          vol: float = 0.03, seed: int = 39
                          ) -> tuple[pd.DataFrame, NextDayTruth]:
    """A daily OHLCV-like price frame whose next-day drift is a **weak nonlinear** function of the last
    two returns — learnable by an MLP, near-invisible to a linear rule.

    The generative process for daily log-return ``r_t``::

        μ_t = predictable_strength · tanh( r_{t-1}·r_{t-2} / vol² )    # nonlinear, lagged signal
        r_t = μ_t + vol · ε_t ,   ε_t ~ N(0,1)

    The predictable piece is tiny relative to ``vol`` (signal-to-noise is deliberately low), so the *only*
    way to harvest it is to actually learn the interaction — which is the whole point of the control.
    ``predictable_strength = 0`` removes ``μ_t`` entirely (a pure geometric random walk: the **null**).
    A synthetic OHLCV frame is returned (open/high/low/close/volume) so the feature builder has the same
    columns it gets from the real tape. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2014-01-02", periods=n_bars, name="date")
    eps = rng.standard_normal(n_bars)
    r = np.zeros(n_bars)
    inv_v2 = 1.0 / (vol * vol)
    for t in range(2, n_bars):
        # Scale the lagged product by 1/vol^2 so the tanh argument is O(1): the nonlinear interaction
        # actually swings across [-1, 1] day to day, instead of sitting pinned near zero.
        mu = predictable_strength * np.tanh(r[t - 1] * r[t - 2] * inv_v2)
        r[t] = mu + vol * eps[t]
    close = 100.0 * np.exp(np.cumsum(r))
    # Build a plausible OHLCV frame around the close path (intraday range scaled by daily vol).
    rng2 = np.random.default_rng(seed + 1)
    span = np.abs(close) * vol * 0.5
    high = close + np.abs(rng2.standard_normal(n_bars)) * span
    low = close - np.abs(rng2.standard_normal(n_bars)) * span
    open_ = np.r_[close[0], close[:-1]]
    volume = 1e6 * np.exp(0.3 * rng2.standard_normal(n_bars))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume},
                      index=idx)
    return df, NextDayTruth(n_bars, predictable_strength)


def fetch_crypto(cache_dir: str | None = None, coins: tuple[str, ...] = DEFAULT_COINS,
                 start: str = "2017-01-01", fetch: bool = False) -> pd.DataFrame:
    """Return a ``dates × coin`` daily-close frame for a few liquid coins, **cache-first**.

    Reads ``_cache/black_box_crypto.parquet`` if present. With ``fetch=True`` it downloads daily closes
    for ``coins`` from Yahoo! Finance (lazy yfinance import), writes the parquet cache, and returns it.
    **Cache-only by default** — returns an empty frame on a cache miss so the offline core never touches
    the network.
    """
    cache_file = (os.path.join(cache_dir, "black_box_crypto.parquet") if cache_dir else CACHE_FILE)
    if os.path.exists(cache_file):
        df = pd.read_parquet(cache_file)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy: only on an explicit fetch

    raw = yf.download(list(coins), start=start, auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.dropna(how="all").sort_index()
    close.index = pd.to_datetime(close.index)
    close.index.name = "date"
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    close.to_parquet(cache_file)
    return close
