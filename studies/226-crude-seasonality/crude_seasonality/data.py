"""Data for the crude-seasonality study — an offline synthetic world and a cached crude/energy pair.

  * :func:`synthetic_world` — **offline, deterministic**. Monthly crude and energy-equity returns
    with a tunable spring premium (``spring_premium``) injected into April/May/June and a symmetric
    autumn discount in August–November. ``spring_premium = 0`` is the null. Pins the seasonality
    machinery offline.
  * :func:`fetch_data` — monthly returns for WTI crude (``CL=F``), energy equities (``XLE``) and
    the 13-week T-bill (``^IRX``, the cash leg), **cache-first**, built from *daily* closes
    resampled to month-end. Fingerprinted run in ``docs/results.md``.

Cache lives in the study-local ``_cache/`` directory (gitignored) to avoid races with other agents.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

CRUDE, ENERGY, TBILL = "CL=F", "XLE", "^IRX"
MONTHS_PER_YEAR = 12

# Spring = Apr–Jun (driving-season ramp-up); Autumn = Aug–Nov (post-summer, refinery maintenance)
SPRING_MONTHS = [4, 5, 6]
AUTUMN_MONTHS = [8, 9, 10, 11]


@dataclass(frozen=True)
class WorldTruth:
    spring_premium: float

    @property
    def has_seasonality(self) -> bool:
        return self.spring_premium != 0.0


def synthetic_world(
    n_years: int = 26,
    spring_premium: float = 0.03,
    crude_vol: float = 0.30,
    energy_vol: float = 0.18,
    seed: int = 226,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A monthly crude/energy world — deterministic given ``seed``.

    Monthly crude returns are i.i.d. with annual vol ``crude_vol`` plus a ``spring_premium``
    added to April/May/June months and subtracted from August–November. Energy equities (XLE)
    load on crude with beta ~0.7 and independent noise. ``spring_premium = 0`` is the null.
    Returns a frame with columns ``crude``, ``energy``, ``tbill``.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    months = idx.month

    crude_base = (crude_vol / np.sqrt(12)) * rng.standard_normal(n)
    # Inject spring premium and autumn discount
    seasonal = np.where(
        np.isin(months, SPRING_MONTHS),
        spring_premium / len(SPRING_MONTHS),
        np.where(np.isin(months, AUTUMN_MONTHS), -spring_premium / len(AUTUMN_MONTHS), 0.0),
    )
    crude = crude_base + seasonal

    # Energy equities: beta ~0.7 on crude + idiosyncratic
    energy = 0.7 * crude + (energy_vol / np.sqrt(12)) * rng.standard_normal(n)

    # Flat 2%/yr T-bill cash rate
    tbill = pd.Series(0.02 / 12, index=idx)

    return pd.DataFrame({"crude": crude, "energy": energy, "tbill": tbill.values}, index=idx), WorldTruth(
        spring_premium
    )


def _assert_continuous_monthly(df: pd.DataFrame) -> None:
    """Fail loudly if the monthly grid has interior holes."""
    months = pd.PeriodIndex(df.index, freq="M")
    expected = pd.period_range(months[0], months[-1], freq="M")
    missing = expected.difference(months)
    if len(missing):
        raise AssertionError(
            f"monthly grid has {len(missing)} interior hole(s) out of {len(expected)} months "
            f"(first few: {list(missing[:5])}) — refetch with fetch_data(fetch=True)"
        )


def fetch_data(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Monthly returns for WTI crude (CL=F), energy equities (XLE) and the T-bill (^IRX), cache-first.

    **Cache-only** unless ``fetch=True``. Columns ``crude, energy, tbill`` (empty DataFrame on a
    cache miss with ``fetch=False``). Built from **daily** closes resampled to month-end (last close
    of each calendar month) — not Yahoo's native monthly feed, which has holes. The in-progress
    month is dropped so the last bar is always complete. Grid is asserted hole-free on every read.
    """
    cache = os.path.join(cache_dir, "crude_seasonality.parquet")
    if os.path.exists(cache):
        out = pd.read_parquet(cache)
        _assert_continuous_monthly(out)
        return out
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf  # lazy

    px = yf.download(
        [CRUDE, ENERGY, TBILL], period="max", interval="1d", auto_adjust=True, progress=False
    )["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    me = px.resample("ME").last()

    out = pd.DataFrame(
        {
            "crude": me[CRUDE].pct_change(),
            "energy": me[ENERGY].pct_change(),
            "tbill": (me[TBILL].ffill() / 100.0) / MONTHS_PER_YEAR,
        }
    ).dropna(subset=["crude", "energy"])

    # Drop in-progress month
    last_complete = pd.Timestamp.today().to_period("M") - 1
    out = out[out.index.to_period("M") <= last_complete]
    out.index.name = "date"
    _assert_continuous_monthly(out)
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(cache)
    return out
