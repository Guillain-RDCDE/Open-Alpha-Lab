"""Data for the natgas-winter study — an offline synthetic world, and cached UNG monthly returns.

  * :func:`synthetic_world` — **offline, deterministic**. A monthly UNG return series with a tunable
    winter premium (``winter_premium``) injected in Oct-Mar; 0 = the null. Pins the seasonality/timing
    machinery offline.
  * :func:`fetch_ung` — monthly UNG returns, **cache-first**, built from daily closes resampled to
    month-end. The cache lives inside THIS study's _cache/ dir (study-local, gitignored) to avoid races
    with other agents. Fingerprinted run in ``docs/results.md``.

Data note: UNG (United States Natural Gas Fund LP) started trading in April 2007. We pull from inception
through the latest complete calendar month. Monthly returns are built from daily closes (last trading day
of each month) — not Yahoo's native monthly feed, which can have gaps. The in-progress calendar month is
always dropped so the last bar is a complete month.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")
CACHE_FILE = "natgas_winter_ung.parquet"

UNG_TICKER = "UNG"
MONTHS_PER_YEAR = 12
WINTER_MONTHS = {10, 11, 12, 1, 2, 3}


@dataclass(frozen=True)
class WorldTruth:
    winter_premium: float

    @property
    def has_premium(self) -> bool:
        return self.winter_premium != 0.0


def synthetic_world(
    n_years: int = 20,
    winter_premium: float = 0.02,
    base_vol: float = 0.12,
    seed: int = 227,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A monthly UNG-like world — deterministic given ``seed``.

    Monthly returns have annual vol ``base_vol`` (natgas is ~100% annualised but we use a moderate
    synthetic); winter months (Oct-Mar) get an additional ``winter_premium`` each month. With
    ``winter_premium > 0``, the seasonal strategy should outperform. ``winter_premium = 0`` is the null.
    Returns a DataFrame with column ``ung``.
    """
    rng = np.random.default_rng(seed)
    n = n_years * MONTHS_PER_YEAR
    idx = pd.date_range("2005-01-31", periods=n, freq="ME", name="date")
    noise = (base_vol / np.sqrt(MONTHS_PER_YEAR)) * rng.standard_normal(n)
    premium = np.array([winter_premium if m in WINTER_MONTHS else 0.0 for m in idx.month])
    ung = noise + premium
    return pd.DataFrame({"ung": ung}, index=idx), WorldTruth(winter_premium)


def _assert_continuous_monthly(df: pd.DataFrame) -> None:
    """Fail loudly if the monthly grid has interior holes."""
    months = pd.PeriodIndex(df.index, freq="M")
    expected = pd.period_range(months[0], months[-1], freq="M")
    missing = expected.difference(months)
    if len(missing):
        raise AssertionError(
            f"monthly grid has {len(missing)} interior hole(s) out of {len(expected)} months "
            f"(first few: {list(missing[:5])}) — refetch with fetch_ung(fetch=True)"
        )


def fetch_ung(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Monthly UNG returns, cache-first. Column ``ung``.

    **Cache-only** unless ``fetch=True``. Built from **daily** closes, month-end resampled (last close
    of each calendar month). The in-progress calendar month is dropped so every bar is a complete month.
    The grid is asserted hole-free on every read — cache included.

    The cache lives at ``_cache/natgas_winter_ung.parquet`` inside THIS study's directory
    (studies/227-natgas-winter/_cache/) — study-local so it doesn't collide with other agents.
    """
    cache = os.path.join(cache_dir, CACHE_FILE)
    if os.path.exists(cache):
        out = pd.read_parquet(cache)
        _assert_continuous_monthly(out)
        return out
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf  # lazy

    px = yf.download(UNG_TICKER, period="max", interval="1d", auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    me = px.resample("ME").last()
    out = pd.DataFrame({"ung": me.pct_change()}).dropna()
    # drop the in-progress calendar month
    last_complete = pd.Timestamp.today().to_period("M") - 1
    out = out[out.index.to_period("M") <= last_complete]
    out.index.name = "date"
    _assert_continuous_monthly(out)
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(cache)
    return out
