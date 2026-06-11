"""Data for the oil→equities study — an offline synthetic world, and a cached oil/equity pair.

  * :func:`synthetic_world` — **offline, deterministic**. An oil return series and an equity series
    whose next-month return loads on last month's oil by ``oil_loads`` (negative reproduces Driesprong);
    0 = the null. Pins the regression/timing machinery offline.
  * :func:`fetch_pair` — monthly returns for WTI crude (``CL=F``) and the S&P 500 (``^GSPC``),
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
OIL, EQUITY = "CL=F", "^GSPC"


@dataclass(frozen=True)
class WorldTruth:
    oil_loads: float

    @property
    def oil_predicts(self) -> bool:
        return self.oil_loads != 0.0


def synthetic_world(
    n_years: int = 25, oil_loads: float = -0.10, oil_vol: float = 0.30, eq_vol: float = 0.15, seed: int = 49
) -> tuple[pd.DataFrame, WorldTruth]:
    """A monthly oil/equity world — deterministic given ``seed``.

    Oil returns are i.i.d. with annual vol ``oil_vol``; this month's equity return =
    ``oil_loads × last-month oil`` + noise. With ``oil_loads < 0`` (the Driesprong sign), a falling
    oil month precedes a rising equity month, so the timing rule should work. ``oil_loads = 0`` is the
    null. Returns a frame with columns ``oil`` and ``eq``.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    oil = (oil_vol / np.sqrt(12)) * rng.standard_normal(n)
    eq = oil_loads * np.concatenate([[0.0], oil[:-1]]) + (eq_vol / np.sqrt(12)) * rng.standard_normal(n)
    return pd.DataFrame({"oil": oil, "eq": eq}, index=idx), WorldTruth(oil_loads)


def fetch_pair(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Monthly returns for WTI crude (CL=F) and the S&P 500 (^GSPC), cache-first.

    **Cache-only** unless ``fetch=True``. Columns ``oil, eq`` (empty on a cache miss with
    ``fetch=False``).
    """
    cache = os.path.join(cache_dir, "black_gold_pair.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf  # lazy

    px = yf.download([OIL, EQUITY], period="max", interval="1mo", auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    ret = px.resample("ME").last().pct_change()
    out = pd.DataFrame({"oil": ret[OIL], "eq": ret[EQUITY]}).dropna()
    out.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(cache)
    return out
