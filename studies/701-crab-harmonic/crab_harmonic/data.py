"""Data layer for Study 701 — Crab-Harmonic.

Two ingredients:

* **Real tape.** Daily OHLC for the same six-name basket used by the harmonic-pattern
  siblings (SPY, QQQ, AAPL, MSFT, TSLA, NVDA — see ``docs/references.md`` for the dedup
  map), from yfinance (no key), cached as CSV under the study's own ``_cache/``.

* **Synthetic world.** A deterministic, seeded daily price path with a TUNABLE
  mean-reversion knob (``mean_rev``): at ``mean_rev = 0`` price is a pure random walk
  (the null the Crab detector must NOT manufacture significance from); at
  ``mean_rev > 0`` price is nudged back toward a slow moving average, giving *any*
  well-formed extension-and-reversal structure (Crab-labelled or not) genuine
  forecastable reversion. This is the desk's "faithful engine / power check" —
  never cited in support of the real-tape stamp.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once per
ticker to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

BASKET = ("SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA")
FETCH_PERIOD = "25y"
AS_OF = "2026-06-30"        # last complete month at publication (2026-07-10)


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"crab_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(basket: tuple[str, ...] = BASKET, period: str = FETCH_PERIOD) -> None:
    """Download daily OHLC (total-return adjusted) for every ticker in ``basket``;
    cache each as CSV. Network; run once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in basket:
        raw = yf.download(t, period=period, interval="1d", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
        bars.to_csv(_cache_path(t))


def have_real(basket: tuple[str, ...] = BASKET) -> bool:
    return all(os.path.exists(_cache_path(t)) for t in basket)


def load_real(ticker: str, asof: str = AS_OF) -> pd.DataFrame:
    """Cached OHLC frame for ``ticker``, sliced to <= ``asof``, lower-cased columns."""
    df = pd.read_csv(_cache_path(ticker), index_col=0, parse_dates=True).sort_index()
    df = df.loc[df.index <= asof].copy()
    df.columns = [c.lower() for c in df.columns]
    return df


def load_basket(basket: tuple[str, ...] = BASKET, asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    return {t: load_real(t, asof=asof) for t in basket}


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape's close column, for the data stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic world — the faithful-engine / power check
# --------------------------------------------------------------------------- #
def synthetic_world(mean_rev: float = 0.0, n_days: int = 2500, annual_vol: float = 0.22,
                    start: str = "2010-01-04", seed: int = 701) -> pd.DataFrame:
    """A deterministic daily OHLC(ish) tape with a TUNABLE mean-reversion tendency.

    Log returns follow ``r_t = -mean_rev * (price_t / ema_t - 1) + eps_t`` with a slow
    (span-50) EMA anchor — the same generic "does price reliably pull back toward an
    anchor" mechanism that would make ANY extension-and-reversal projection
    (Crab-labelled or not) show a genuine edge once ``mean_rev > 0``.
    ``mean_rev = 0`` is the null: the Crab detector's split must NOT reach
    significance on it, across many seeds.

    Business-day index, span well under the ~250-year pandas ns-timestamp trap.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    bar_vol = annual_vol / np.sqrt(252)
    eps = rng.normal(0.0, bar_vol, n_days)

    log_price = np.empty(n_days)
    log_price[0] = np.log(100.0)
    alpha_ema = 2.0 / (50 + 1)
    ema = np.empty(n_days)
    ema[0] = 100.0
    for t in range(1, n_days):
        px = np.exp(log_price[t - 1])
        ema[t] = alpha_ema * px + (1 - alpha_ema) * ema[t - 1]
        reversion = -mean_rev * (px / ema[t] - 1.0) if ema[t] > 0 else 0.0
        log_price[t] = log_price[t - 1] + reversion + eps[t]

    close = np.exp(log_price)
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, bar_vol * 0.5, n_days)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=pd.DatetimeIndex(idx, name="date"),
    )
