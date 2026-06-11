"""Data for the turn-of-the-month study — an offline synthetic world, and a cached real tape.

  * :func:`synthetic_daily` — **offline, deterministic**. A daily index whose returns carry an extra
    drift on turn-of-the-month days controlled by ``premium`` (0 = the null, no seasonal). Pins the
    machinery offline.
  * :func:`fetch_prices` — real daily total return for an index/ETF (default ``^GSPC`` for the effect's
    long history, ``SPY`` for the tradable/cost run), **cache-first**. Fingerprinted run in
    ``docs/results.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategy import tom_mask

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")


@dataclass(frozen=True)
class WorldTruth:
    premium_bp: float

    @property
    def has_premium(self) -> bool:
        return self.premium_bp != 0.0


def synthetic_daily(
    n_years: int = 30, premium_bp: float = 9.0, base_bp: float = 2.0, vol_ann: float = 0.16, seed: int = 42
) -> tuple[pd.Series, WorldTruth]:
    """A daily return series with a turn-of-the-month bump — deterministic given ``seed``.

    Every day gets ``base_bp`` of drift plus i.i.d. noise; turn-of-the-month days (the [-1,+3] window)
    get an extra ``premium_bp``. ``premium_bp = 0`` is the null (a flat seasonal). Returns a daily
    simple-return Series over ``n_years`` business years.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1990-01-01", periods=n_years * 252, name="date")
    m = tom_mask(idx)
    sig_d = vol_ann / np.sqrt(252)
    drift = base_bp * 1e-4 + (premium_bp * 1e-4) * m.values
    r = drift + sig_d * rng.standard_normal(len(idx))
    return pd.Series(r, index=idx, name="ret"), WorldTruth(premium_bp)


def fetch_prices(ticker: str = "^GSPC", cache_dir: str = DEFAULT_CACHE, fetch: bool = False,
                 start_year: int = 1950) -> pd.Series:
    """Daily total-return series for ``ticker`` (^GSPC or SPY), cache-first.

    **Cache-only** unless ``fetch=True``. Returns a daily simple-return Series from ``start_year`` (or
    the ticker's inception), empty on a cache miss with ``fetch=False``.
    """
    cache = os.path.join(cache_dir, f"last_call_{ticker.lstrip('^').lower()}.parquet")
    if os.path.exists(cache):
        s = pd.read_parquet(cache)["ret"]
        return s[s.index.year >= start_year]
    if not fetch:
        return pd.Series(dtype=float)
    import yfinance as yf  # lazy

    px = yf.download(ticker, period="max", interval="1d", auto_adjust=True, progress=False)["Close"].dropna()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    ret = px.pct_change().dropna().squeeze()
    ret.name = "ret"
    ret.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_frame().to_parquet(cache)
    return ret[ret.index.year >= start_year]
