"""Data for the oil→equities study — an offline synthetic world, and a cached oil/equity pair.

  * :func:`synthetic_world` — **offline, deterministic**. An oil return series and an equity series
    whose next-month return loads on last month's oil by ``oil_loads`` (negative reproduces Driesprong);
    0 = the null. Pins the regression/timing machinery offline.
  * :func:`fetch_pair` — monthly returns for WTI crude (``CL=F``), the S&P 500 (``^GSPC``) and the
    13-week T-bill (``^IRX``, the cash leg), **cache-first**, built from *daily* closes resampled to
    month-end on a verified hole-free monthly grid. Fingerprinted run in ``docs/results.md``.

A data choice, stated up front: we fetch **daily** bars and resample to month-end ourselves. Yahoo's
``interval="1mo"`` feed for CL=F is full of holes (89 of 310 months missing on a 2026 pull), and dropping
those rows before a *positional* one-step lag silently turns "last month's oil return" into oil from two
or three months back — while the buy-and-hold benchmark skips the same 89 real S&P months. The daily
route has no missing months; :func:`_assert_continuous_monthly` makes that a hard guarantee on every
read, cache included.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
OIL, EQUITY, TBILL = "CL=F", "^GSPC", "^IRX"
MONTHS_PER_YEAR = 12


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


def _assert_continuous_monthly(df: pd.DataFrame) -> None:
    """Fail loudly if the monthly grid has interior holes.

    A hole would put the calendar lag and the buy-and-hold benchmark on different clocks — the exact
    bug the daily-resample fetch exists to prevent — so a gapped frame is an error, never a warning.
    """
    months = pd.PeriodIndex(df.index, freq="M")
    expected = pd.period_range(months[0], months[-1], freq="M")
    missing = expected.difference(months)
    if len(missing):
        raise AssertionError(
            f"monthly grid has {len(missing)} interior hole(s) out of {len(expected)} months "
            f"(first few: {list(missing[:5])}) — refetch with fetch_pair(fetch=True)"
        )


def fetch_pair(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Monthly returns for WTI crude (CL=F), the S&P 500 (^GSPC) and the T-bill (^IRX), cache-first.

    **Cache-only** unless ``fetch=True``. Columns ``oil, eq, tbill`` (empty on a cache miss with
    ``fetch=False``); ``tbill`` is the ^IRX 13-week discount yield turned into a monthly cash return
    (``yield% / 100 / 12``), the rate the timing rule earns when it is out of the market.

    Built from **daily** closes, month-end resampled (last close of each calendar month) — *not* the
    naive ``interval="1mo"`` feed, whose CL=F bars are full of holes (see the module docstring). The
    in-progress calendar month is dropped so the last bar is always a complete month, and the grid is
    asserted hole-free on every read — cache included, so a stale gapped cache fails loudly instead of
    silently mis-lagging.
    """
    cache = os.path.join(cache_dir, "black_gold_pair.parquet")
    if os.path.exists(cache):
        out = pd.read_parquet(cache)
        _assert_continuous_monthly(out)
        return out
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf  # lazy

    px = yf.download([OIL, EQUITY, TBILL], period="max", interval="1d", auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    me = px.resample("ME").last()  # last available daily close in each calendar month
    out = pd.DataFrame(
        {
            "oil": me[OIL].pct_change(),
            "eq": me[EQUITY].pct_change(),
            "tbill": (me[TBILL].ffill() / 100.0) / MONTHS_PER_YEAR,
        }
    ).dropna(subset=["oil", "eq"])
    # never publish a "monthly" return built from a few days — drop the in-progress month
    last_complete = pd.Timestamp.today().to_period("M") - 1
    out = out[out.index.to_period("M") <= last_complete]
    out.index.name = "date"
    _assert_continuous_monthly(out)
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(cache)
    return out
