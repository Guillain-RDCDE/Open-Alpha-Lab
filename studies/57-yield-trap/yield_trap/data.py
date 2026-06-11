"""Data for the dividend study — an offline synthetic world, and a cached ETF pair.

  * :func:`synthetic_world` — **offline, deterministic**. Two correlated total-return series (the
    "high-dividend" sleeve and the "market"); the dividend sleeve carries an annual ``premium`` (set to
    0 or negative to reproduce the real result that it lags). Pins the machinery offline.
  * :func:`fetch_pairs` — monthly total returns for **VYM** (high dividend) and **SPY** (market),
    **cache-first**. Fingerprinted run in ``docs/results.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
TICKERS = ["VYM", "SPY"]


@dataclass(frozen=True)
class WorldTruth:
    premium: float

    @property
    def has_premium(self) -> bool:
        return self.premium != 0.0


def synthetic_world(
    n_years: int = 20, premium: float = 0.0, mkt_vol: float = 0.15, idio_vol: float = 0.06, seed: int = 57
) -> tuple[pd.DataFrame, WorldTruth]:
    """A monthly two-ETF world — deterministic given ``seed``.

    A market factor drives both; the high-dividend sleeve adds an annual ``premium`` (0 = the null, the
    real-world result). Returns a frame with columns ``VYM`` and ``SPY``.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("2007-01-31", periods=n, freq="ME", name="date")
    mkt = (mkt_vol / np.sqrt(12)) * rng.standard_normal(n) + 0.006
    vym = mkt + premium / 12.0 + (idio_vol / np.sqrt(12)) * rng.standard_normal(n)
    spy = mkt + (idio_vol / np.sqrt(12)) * rng.standard_normal(n)
    return pd.DataFrame({"VYM": vym, "SPY": spy}, index=idx), WorldTruth(premium)


def fetch_pairs(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Monthly total returns for VYM and SPY, cache-first."""
    cache = os.path.join(cache_dir, "yield_trap_pairs.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf  # lazy

    px = yf.download(TICKERS, period="max", interval="1mo", auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    ret = px.resample("ME").last().pct_change().dropna(how="all")
    ret.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_parquet(cache)
    return ret
