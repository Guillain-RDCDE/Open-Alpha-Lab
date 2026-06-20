"""Data for the coffee-seasonality study — an offline synthetic world and a cached KC=F series.

  * :func:`synthetic_world` — **offline, deterministic**. Monthly Arabica-coffee returns with a tunable
    ``frost_premium`` injected into the frost-risk window (Jun–Aug, the Brazilian winter) and a symmetric
    harvest discount in the harvest-pressure window (May–Sep). ``frost_premium = 0`` is the null. Pins the
    seasonality machinery offline. NEVER backs a Signal stamp.
  * :func:`fetch_data` — monthly returns for Arabica coffee (``KC=F``) and the 13-week T-bill (``^IRX``,
    the cash leg), **cache-first**, built from *daily* closes resampled to month-end. Fingerprinted run in
    ``docs/results.md``.

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

COFFEE, TBILL = "KC=F", "^IRX"
MONTHS_PER_YEAR = 12

# Frost-risk window: Jun–Aug (Brazilian winter; a single overnight frost can spike Arabica prices).
# Harvest-pressure window: May–Sep (new Brazilian crop floods the market, weighs on price).
# The two windows OVERLAP on purpose (Jun–Aug sit inside May–Sep) — frost risk and harvest supply
# pull in opposite directions during the same calendar stretch, which is exactly why coffee
# seasonality is folklore rather than a clean signal. We test them as separately-defined groups.
FROST_MONTHS = [6, 7, 8]
HARVEST_MONTHS = [5, 9]  # the harvest-pressure shoulders not contaminated by the frost window


@dataclass(frozen=True)
class WorldTruth:
    frost_premium: float

    @property
    def has_seasonality(self) -> bool:
        return self.frost_premium != 0.0


def synthetic_world(
    n_years: int = 30,
    frost_premium: float = 0.03,
    coffee_vol: float = 0.36,
    seed: int = 307,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A monthly Arabica-coffee world — deterministic given ``seed``.

    Monthly coffee returns are i.i.d. with annual vol ``coffee_vol`` (coffee is a notoriously volatile
    softs market, hence the high default) plus a ``frost_premium`` added to the frost-risk months
    (Jun–Aug) and subtracted from the harvest-pressure shoulder months (May, Sep). ``frost_premium = 0``
    is the null. Returns a frame with columns ``coffee``, ``tbill``.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("1996-01-31", periods=n, freq="ME", name="date")
    months = idx.month

    base = (coffee_vol / np.sqrt(12)) * rng.standard_normal(n)
    seasonal = np.where(
        np.isin(months, FROST_MONTHS),
        frost_premium / len(FROST_MONTHS),
        np.where(np.isin(months, HARVEST_MONTHS), -frost_premium / len(HARVEST_MONTHS), 0.0),
    )
    coffee = base + seasonal

    tbill = pd.Series(0.02 / 12, index=idx)  # flat 2%/yr cash leg
    return pd.DataFrame({"coffee": coffee, "tbill": tbill.values}, index=idx), WorldTruth(frost_premium)


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
    """Monthly returns for Arabica coffee (KC=F) and the T-bill (^IRX), cache-first.

    **Cache-only** unless ``fetch=True``. Columns ``coffee, tbill`` (empty DataFrame on a cache miss
    with ``fetch=False``). Built from **daily** closes resampled to month-end (last close of each
    calendar month) — not Yahoo's native monthly feed, which has holes. The in-progress month is
    dropped so the last bar is always complete. Grid is asserted hole-free on every read.
    """
    cache = os.path.join(cache_dir, "coffee_seasonality.parquet")
    if os.path.exists(cache):
        out = pd.read_parquet(cache)
        _assert_continuous_monthly(out)
        return out
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf  # lazy

    px = yf.download([COFFEE, TBILL], period="max", interval="1d", auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    me = px.resample("ME").last()

    out = pd.DataFrame(
        {
            "coffee": me[COFFEE].pct_change(),
            "tbill": (me[TBILL].ffill() / 100.0) / MONTHS_PER_YEAR,
        }
    ).dropna(subset=["coffee"])

    last_complete = pd.Timestamp.today().to_period("M") - 1
    out = out[out.index.to_period("M") <= last_complete]
    out.index.name = "date"
    _assert_continuous_monthly(out)
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(cache)
    return out
