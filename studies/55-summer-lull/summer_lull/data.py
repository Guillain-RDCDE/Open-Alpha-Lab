"""Data for the Sell-in-May study — an offline synthetic world, and the cached S&P tape.

  * :func:`synthetic_monthly` — **offline, deterministic**. Monthly returns with a base drift plus a
    ``winter_premium`` added to Nov–Apr months; 0 = the null (no seasonal). Pins the machinery offline.
  * :func:`fetch_index` — the real **S&P 500** (^GSPC), daily from Yahoo resampled to month-end total
    return, back to 1928, **cache-first**. Fingerprinted run in ``docs/results.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
WINTER = {11, 12, 1, 2, 3, 4}


@dataclass(frozen=True)
class WorldTruth:
    winter_premium: float

    @property
    def has_seasonal(self) -> bool:
        return self.winter_premium != 0.0


def synthetic_monthly(
    n_years: int = 80, base_bp: float = 50.0, winter_premium_bp: float = 40.0, vol_ann: float = 0.15,
    seed: int = 55
) -> tuple[pd.Series, WorldTruth]:
    """A monthly return series with a winter (Nov–Apr) bump — deterministic given ``seed``.

    Each month gets ``base_bp`` of drift; Nov–Apr months get an extra ``winter_premium_bp``.
    ``winter_premium_bp = 0`` is the null. Returns a monthly simple-return Series.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("1946-01-31", periods=n, freq="ME", name="date")
    winter = pd.DatetimeIndex(idx).month.isin(WINTER).astype(float)
    drift = base_bp * 1e-4 + (winter_premium_bp * 1e-4) * winter
    r = drift + (vol_ann / np.sqrt(12)) * rng.standard_normal(n)
    return pd.Series(r, index=idx, name="ret"), WorldTruth(winter_premium_bp)


def fetch_index(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.Series:
    """Monthly total-return of the S&P 500 (^GSPC), cache-first (daily → month-end, back to 1928)."""
    cache = os.path.join(cache_dir, "summer_lull_sp500_monthly.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)["ret"]
    if not fetch:
        return pd.Series(dtype=float)
    import yfinance as yf  # lazy

    px = yf.download("^GSPC", period="max", interval="1d", auto_adjust=True, progress=False)["Close"].dropna()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    ret = px.resample("ME").last().pct_change().dropna().squeeze()
    ret.name = "ret"
    ret.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_frame().to_parquet(cache)
    return ret
