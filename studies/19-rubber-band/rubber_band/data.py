"""Data access for the IBS mean-reversion study — the price bars and where they come from.

This study turns on one intraday number: **Internal Bar Strength**, ``IBS = (Close − Low) / (High −
Low)``, where a close near the day's low (IBS ≈ 0) marks a bar that sold off into the close and a close
near the high (IBS ≈ 1) one that ran up. The folk claim is that IBS *mean-reverts*: a low-IBS day
tends to bounce the next day. So the tape we need is daily **OHLC**, and the data layer keeps the
desk's standing split between an offline synthetic generator and a cache-only real reader:

    * :func:`synthetic_ohlc` — fully **offline**. Daily bars with a *baked-in* next-day reversal tied
      to IBS: we draw a latent close-position ``IBS_t``, build a consistent ``(Open, High, Low, Close)``
      bar around it, and set the *next* day's return to ``drift − kappa·(IBS_t − ½)`` plus noise — so a
      close near the low (IBS low) is followed by a positive day, exactly the reversal the strategy
      hunts. ``kappa`` is the baked strength; pass ``kappa = 0`` for the **null** — a random walk where
      IBS carries no information and the strategy must add nothing. Deterministic given ``seed``.
    * :func:`fetch_basket` — read cached daily OHLC for a basket of liquid ETFs from the shared
      ``_cache/<TICKER>_split_only.parquet`` files. **Cache-only** unless ``fetch=True``: a missing
      cache is skipped, never a silent download, so the offline core (and CI) never imports
      ``yfinance``.

Two data choices, named up front. **Split-only (not total-return) closes**: IBS is a *daily,
intraday-shape* signal and the strategy holds overnight at most one day, so dividends — a slow,
total-return effect — are immaterial here, while split-only keeps the OHLC bar internally consistent
(a dividend adjustment shifts close but not the same-day high/low cleanly). **A small, liquid ETF
basket**: IBS reversal is a *microstructure* effect, so the relevant universe is instruments you could
actually trade at a tight spread — which is also exactly where the cost teardown bites.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Synthetic tape — offline OHLC bars with a baked-in IBS reversal
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BarTruth:
    """What the synthetic generator baked in, so a test can check the diagnostics recover it."""
    n_bars: int
    kappa: float              # reversal strength: next return = drift - kappa*(IBS - 0.5) + noise
    drift_daily: float
    noise_daily: float

    @property
    def has_reversal(self) -> bool:
        """True when IBS actually predicts the next day (kappa != 0) — something to find."""
        return self.kappa != 0.0


def synthetic_ohlc(
    n_bars: int = 5040,
    kappa: float = 0.0035,
    drift_daily: float = 0.0003,
    noise_daily: float = 0.009,
    range_frac: float = 0.012,
    start_price: float = 100.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, BarTruth]:
    """Daily OHLC bars whose **low-IBS days bounce** the next session, by construction.

    The generative model, one bar at a time::

        IBS_t        ~ Uniform(0, 1)                          (where the close sits in the day's range)
        r_{t+1}      = drift − kappa·(IBS_t − ½) + noise·eps  (low IBS  ⇒  positive next-day return)
        range_t      = range_frac · price                    (the day's High−Low)
        Low_t, High_t set so (Close_t − Low_t)/(High_t − Low_t) == IBS_t exactly

    The reversal is therefore *causal and lagged*: today's close-position drives tomorrow's return, so a
    strategy that buys low-IBS names and sells high-IBS ones harvests a real, baked edge. With
    ``kappa = 0`` the next return is a pure random walk (drift + noise), IBS is uninformative, and the
    strategy must earn nothing — the null. Returns ``(ohlc, truth)``; ``ohlc`` has columns
    ``Open, High, Low, Close`` on a business-day index named ``date``. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2005-01-03", periods=n_bars, name="date")

    ibs = rng.uniform(0.0, 1.0, size=n_bars)
    eps = rng.standard_normal(n_bars)
    # next-day return driven by today's IBS (shift so r[t] uses ibs[t-1])
    rets = np.empty(n_bars)
    rets[0] = drift_daily + noise_daily * eps[0]
    rets[1:] = drift_daily - kappa * (ibs[:-1] - 0.5) + noise_daily * eps[1:]

    close = start_price * np.cumprod(1.0 + rets)
    rng_range = range_frac * close
    low = close - ibs * rng_range
    high = low + rng_range
    # an Open drawn inside the bar (uninformative — present so the tape looks like a real OHLC frame)
    open_ = low + rng.uniform(0.0, 1.0, size=n_bars) * rng_range

    ohlc = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close}, index=idx
    )
    truth = BarTruth(n_bars=n_bars, kappa=kappa, drift_daily=drift_daily, noise_daily=noise_daily)
    return ohlc, truth


# --------------------------------------------------------------------------- #
# Real tape — cached ETF OHLC, one parquet per ticker
# --------------------------------------------------------------------------- #

DEFAULT_BASKET = [
    "SPY", "QQQ", "TLT", "GLD", "USO", "UUP",
    "EWG", "EWH", "EWJ", "EWQ", "EWU", "EWZ", "FXI", "INDA",
]


def _cache_path(ticker: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"{ticker}_split_only.parquet")


def fetch_basket(
    tickers: list[str] | None = None,
    cache_dir: str = DEFAULT_CACHE,
    min_days: int = 1000,
    fetch: bool = False,
) -> dict[str, pd.DataFrame]:
    """Return ``{ticker: OHLC DataFrame}`` for a basket of liquid ETFs, cache-first.

    Reads each ``<TICKER>_split_only.parquet`` (columns ``Open, High, Low, Close``) and keeps names
    with at least ``min_days`` of history. **Cache-only by default**: a ticker with no cached parquet
    is skipped unless ``fetch=True``, which pulls it from Yahoo once (split-adjusted OHLC) and caches
    it. The network import is lazy, so the offline core never imports ``yfinance``.
    """
    tickers = tickers or DEFAULT_BASKET
    out: dict[str, pd.DataFrame] = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if os.path.exists(path):
            df = pd.read_parquet(path)[["Open", "High", "Low", "Close"]].dropna()
            if len(df) >= min_days:
                df.index = pd.DatetimeIndex(df.index).tz_localize(None)
                df.index.name = "date"
                out[tk] = df
            continue
        if not fetch:
            continue
        import yfinance as yf  # lazy: offline core never imports it

        raw = yf.download(tk, period="max", interval="1d", auto_adjust=False, progress=False)
        if raw is None or raw.empty:
            continue
        df = raw[["Open", "High", "Low", "Close"]].dropna()
        df.columns = ["Open", "High", "Low", "Close"]
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        df.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(path)
        if len(df) >= min_days:
            out[tk] = df
    return out
