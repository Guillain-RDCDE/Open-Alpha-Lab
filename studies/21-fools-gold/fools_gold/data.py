"""Data access for the moving-average-crossover study — the price tape and where it comes from.

The "golden cross" is a single-series timing rule: when a fast moving average crosses above a slow one,
go long; when it crosses below (the "death cross"), step aside. So the tape is one daily close series,
and the data layer keeps the desk's standing split between an offline synthetic generator and a
cache-only real reader:

    * :func:`synthetic_prices` — fully **offline**. A daily close whose drift is a slow, *persistent*
      AR(1) process, so the series *trends* (and a crossover can in principle catch those trends). With
      ``trend_strength = 0`` the drift is zero: the close is a driftless random walk where a crossover
      is pure noise — the **null**, against which any "golden cross works" claim must be measured.
      Deterministic given ``seed``.
    * :func:`fetch_closes` — read cached daily closes for a few liquid instruments from the shared
      ``_cache/<TICKER>_split_only.parquet`` files. **Cache-only** unless ``fetch=True``: a missing
      cache is skipped, never a silent download, so the offline core (and CI) never imports ``yfinance``.

Data choice, named up front: **split-only closes**. The crossover acts on the *shape* of the price
path, not its total return; using split-only keeps the moving averages on the traded price, and the
benchmark (buy-and-hold) is charged the same series, so the comparison is apples-to-apples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class TrendTruth:
    """What the synthetic generator baked in, so a test can check the diagnostics recover it."""
    n_bars: int
    trend_strength: float     # amplitude of the persistent drift; 0 == driftless random-walk null
    phi: float                # AR(1) persistence of the drift
    noise_daily: float

    @property
    def has_trend(self) -> bool:
        return self.trend_strength != 0.0


def synthetic_prices(
    n_bars: int = 6048,
    trend_strength: float = 0.0006,
    phi: float = 0.99,
    noise_daily: float = 0.010,
    start_price: float = 100.0,
    seed: int = 0,
) -> tuple[pd.Series, TrendTruth]:
    """A daily close that **trends** (persistent drift) — the regime a crossover could ride.

    The return process is ``r_t = mu_t + noise*eps``, with ``mu_t`` a stationary AR(1)
    (``mu_t = phi*mu_{t-1} + sqrt(1-phi^2)*trend_strength*nu``). With ``phi`` near 1 the drift wanders
    slowly, so the price path has runs (trends) a crossover can in principle catch. ``trend_strength =
    0`` gives a driftless random walk — the null, where a crossover catches nothing. Deterministic
    given ``seed``; returns ``(close, truth)``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2002-01-02", periods=n_bars, name="date")
    mu = np.zeros(n_bars)
    innov = trend_strength * np.sqrt(1.0 - phi**2)
    nu = rng.standard_normal(n_bars)
    for t in range(1, n_bars):
        mu[t] = phi * mu[t - 1] + innov * nu[t]
    rets = mu + noise_daily * rng.standard_normal(n_bars)
    close = pd.Series(start_price * np.cumprod(1.0 + rets), index=idx, name="close")
    truth = TrendTruth(n_bars=n_bars, trend_strength=trend_strength, phi=phi, noise_daily=noise_daily)
    return close, truth


DEFAULT_TICKERS = ["SPY", "QQQ", "TLT", "GLD", "EWJ", "EWZ", "FXI", "USO"]


def fetch_closes(
    tickers: list[str] | None = None,
    cache_dir: str = DEFAULT_CACHE,
    min_days: int = 1500,
    fetch: bool = False,
) -> dict[str, pd.Series]:
    """Return ``{ticker: close-series}`` for a few liquid instruments, cache-first.

    Reads each ``<TICKER>_split_only.parquet`` close, keeps names with at least ``min_days`` of
    history. **Cache-only by default**: a missing ticker is skipped unless ``fetch=True`` (one Yahoo
    pull). The network import stays lazy, so the offline core never imports ``yfinance``.
    """
    tickers = tickers or DEFAULT_TICKERS
    out: dict[str, pd.Series] = {}
    for tk in tickers:
        path = os.path.join(cache_dir, f"{tk}_split_only.parquet")
        if os.path.exists(path):
            c = pd.read_parquet(path)["Close"].dropna()
            if len(c) >= min_days:
                c.index = pd.DatetimeIndex(c.index).tz_localize(None)
                c.index.name = "date"
                out[tk] = c.rename("close")
            continue
        if not fetch:
            continue
        import yfinance as yf  # lazy

        raw = yf.download(tk, period="max", interval="1d", auto_adjust=False, progress=False)
        if raw is None or raw.empty:
            continue
        raw[["Open", "High", "Low", "Close", "Volume"]].to_parquet(path)
        c = raw["Close"].dropna()
        c.index = pd.DatetimeIndex(c.index).tz_localize(None)
        if len(c) >= min_days:
            out[tk] = c.rename("close")
    return out
