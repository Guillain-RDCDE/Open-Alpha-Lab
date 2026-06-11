"""Data for the value study — an offline synthetic world, and a cached value/growth pairs panel.

  * :func:`synthetic_hml` — **offline, deterministic**. A market factor drives a value and a growth
    leg; the value leg carries a regime-switching ``premium`` (positive early, negative in a middle
    "lost decade", recovering after) — or a flat ``premium`` if ``regimes=False``; 0 = the null.
  * :func:`fetch_pairs` — monthly total returns for tradable value/growth ETF pairs (**IVE/IVW**,
    **VTV/VUG**, **RPV/RPG**), **cache-first**. Fingerprinted run in ``docs/results.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
MONTHS = 12

TICKERS = ["IVE", "IVW", "VTV", "VUG", "RPV", "RPG"]
PAIRS = {"ive_ivw": ("IVE", "IVW"), "vtv_vug": ("VTV", "VUG"), "rpv_rpg": ("RPV", "RPG")}


@dataclass(frozen=True)
class WorldTruth:
    premium: float
    regimes: bool

    @property
    def has_premium(self) -> bool:
        return self.premium != 0.0


def synthetic_hml(
    n_years: int = 26, premium: float = 0.05, regimes: bool = True,
    mkt_vol: float = 0.16, idio_vol: float = 0.10, seed: int = 46
) -> tuple[pd.DataFrame, WorldTruth]:
    """A value/growth monthly world — deterministic given ``seed``.

    A market factor drives both legs (beta 1). The value leg adds an annual ``premium``; if ``regimes``
    is set, the premium follows +1 / −1 / +0.3 over three equal thirds of the sample (worked, lost
    decade, partial recovery). ``premium = 0`` is the null. Columns ``value`` and ``growth``.
    """
    rng = np.random.default_rng(seed)
    n = n_years * MONTHS
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    mkt = (mkt_vol / np.sqrt(MONTHS)) * rng.standard_normal(n)
    if regimes:
        third = n // 3
        ramp = np.concatenate([np.full(third, 1.0), np.full(third, -1.0), np.full(n - 2 * third, 0.3)])
    else:
        ramp = np.ones(n)
    val_alpha = (premium / MONTHS) * ramp
    value = mkt + val_alpha + (idio_vol / np.sqrt(MONTHS)) * rng.standard_normal(n)
    growth = mkt + (idio_vol / np.sqrt(MONTHS)) * rng.standard_normal(n)
    return pd.DataFrame({"value": value, "growth": growth}, index=idx), WorldTruth(premium, regimes)


def fetch_pairs(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Monthly total returns for the value/growth ETF pairs, cache-first.

    **Cache-only** unless ``fetch=True``. Columns ``IVE, IVW, VTV, VUG, RPV, RPG`` (empty on a cache
    miss with ``fetch=False``).
    """
    cache = os.path.join(cache_dir, "bargain_bin_pairs.parquet")
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
