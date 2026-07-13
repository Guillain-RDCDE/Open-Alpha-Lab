"""Data for the chicken-wing-index study — an offline synthetic world, a cached WING tape,
and a hardcoded, cited wholesale-wing-price series.

  * :func:`synthetic_world` — **offline, deterministic**. Monthly Wingstop-style returns with a tunable
    ``superbowl_premium`` injected into the run-up-to-the-game window (January, the month before the
    early-February Super Bowl). ``superbowl_premium = 0`` is the null. Pins the seasonality machinery
    offline. NEVER backs a Signal stamp.
  * :func:`fetch_data` — monthly returns for **Wingstop (``WING``)**, the S&P 500 ETF (``SPY``, the
    market leg for the alpha regression) and the 13-week T-bill (``^IRX``, the cash leg), **cache-first**,
    built from *daily* closes resampled to month-end. Fingerprinted run in ``docs/results.md``.
  * :func:`load_wing_price` — a small **hardcoded, cited, APPROXIMATE** annual series of the pre-Super-Bowl
    national wholesale whole-chicken-wing price (US$/lb), reconstructed from public USDA/Urner-Barry
    reporting. A LABELLED PROXY for the wing-demand story, never presented as a live feed — and there is
    no wing futures market, so it is not directly tradable; the tradable leg is WING equity.

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

WING, SPY, TBILL = "WING", "SPY", "^IRX"
MONTHS_PER_YEAR = 12

# The Super Bowl is the first Sunday of February; wing demand is realised on game day. The tradable
# steelman is the ANTICIPATION window — January, the month running into the game, when the wing story
# is loudest. February is the GAME MONTH itself (the "sell the news" check). Everything else is the
# baseline the window must beat.
SUPERBOWL_MONTHS = [1]   # January: the run-up-to-the-game window → the bullish leg
GAME_MONTH = [2]         # February: the Super Bowl itself → the sell-the-news check


@dataclass(frozen=True)
class WorldTruth:
    superbowl_premium: float

    @property
    def has_seasonality(self) -> bool:
        return self.superbowl_premium != 0.0


def synthetic_world(
    n_years: int = 30,
    superbowl_premium: float = 0.05,
    wing_vol: float = 0.42,
    seed: int = 726,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A monthly Wingstop-style world — deterministic given ``seed``.

    Monthly returns are i.i.d. with annual vol ``wing_vol`` (WING is a high-vol single-stock growth name)
    plus a ``superbowl_premium`` added to the run-up window (January). ``superbowl_premium = 0`` is the
    null. Returns a frame with columns ``wing``, ``spy``, ``tbill``. The ``spy`` leg is a plain market
    factor so the offline control can exercise the alpha regression too.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("1996-01-31", periods=n, freq="ME", name="date")
    months = idx.month

    spy = (0.15 / np.sqrt(12)) * rng.standard_normal(n) + 0.08 / 12
    base = (wing_vol / np.sqrt(12)) * rng.standard_normal(n) + 1.0 * spy  # beta ~1 to the market
    seasonal = np.where(np.isin(months, SUPERBOWL_MONTHS), superbowl_premium, 0.0)
    wing = base + seasonal

    tbill = np.full(n, 0.02 / 12)  # flat 2%/yr cash leg
    return (
        pd.DataFrame({"wing": wing, "spy": spy, "tbill": tbill}, index=idx),
        WorldTruth(superbowl_premium),
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
    """Monthly returns for WING, SPY and the T-bill (^IRX), cache-first.

    **Cache-only** unless ``fetch=True``. Columns ``wing, spy, tbill`` (empty DataFrame on a cache miss
    with ``fetch=False``). Built from **daily** closes resampled to month-end (last close of each calendar
    month) — not Yahoo's native monthly feed, which has holes. The in-progress month is dropped so the
    last bar is always complete. Grid is asserted hole-free on every read.

    WING and SPY are auto-adjusted (dividends reinvested → total-return-ish); WING pays no dividend so it
    is effectively price-only. Labeled accordingly in ``docs/references.md``.
    """
    cache = os.path.join(cache_dir, "chicken_wing_index.parquet")
    if os.path.exists(cache):
        out = pd.read_parquet(cache)
        _assert_continuous_monthly(out)
        return out
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf  # lazy

    px = yf.download([WING, SPY, TBILL], period="max", interval="1d", auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    me = px.resample("ME").last()

    out = pd.DataFrame(
        {
            "wing": me[WING].pct_change(),
            "spy": me[SPY].pct_change(),
            "tbill": (me[TBILL].ffill() / 100.0) / MONTHS_PER_YEAR,
        }
    ).dropna(subset=["wing"])

    last_complete = pd.Timestamp.today().to_period("M") - 1
    out = out[out.index.to_period("M") <= last_complete]
    out.index.name = "date"
    _assert_continuous_monthly(out)
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(cache)
    return out


# --------------------------------------------------------------------------- #
# Wholesale wing price — hardcoded, cited, APPROXIMATE (a labelled proxy)
# --------------------------------------------------------------------------- #
# National wholesale WHOLE (jumbo) chicken-wing price, US$/lb, at the pre-Super-Bowl turn of the year
# (roughly Dec–Jan of each Super-Bowl season). Reconstructed from *public* reporting — USDA AMS national
# poultry price sheets, Urner Barry quotes echoed in trade/press coverage, and the National Chicken
# Council's annual Super-Bowl wing notes. The load-bearing fact is the SHAPE, not any single decimal:
# a pre-2020 range around $1.5–1.9, a **record spike into early 2021** (the pandemic wing shortage — wings
# briefly the priciest cut of the bird), a **2022–2023 collapse** as supply normalised, then a mild
# recovery. This is a PROXY, never a live feed — and wings have no futures market, so it is not directly
# tradable. Sources listed in docs/references.md.
_WING_PRICE_JAN = {
    2016: 1.62,
    2017: 1.86,
    2018: 1.53,
    2019: 1.44,
    2020: 1.66,
    2021: 2.86,  # the record pre-Super-Bowl spike (pandemic wing shortage)
    2022: 2.71,  # still elevated at the turn of the year, then crashed through mid-2022
    2023: 1.28,  # the collapse
    2024: 1.49,
    2025: 1.71,
    2026: 1.64,
}


def load_wing_price() -> pd.Series:
    """Pre-Super-Bowl national wholesale WHOLE-wing price (US$/lb), hardcoded, cited, APPROXIMATE.

    Returns a ``pd.Series`` indexed by January ``Timestamp`` of each Super-Bowl season. LABELLED A PROXY:
    reconstructed from public USDA/Urner-Barry/NCC reporting, not a live data feed, and NOT directly
    tradable (there is no wing futures contract). Used only to show the price is driven by supply shocks
    (avian flu, the 2021 shortage), not by a clean, harvestable Super-Bowl calendar.
    """
    idx = pd.to_datetime([f"{y}-01-31" for y in _WING_PRICE_JAN])
    return pd.Series(list(_WING_PRICE_JAN.values()), index=idx, name="wholesale_wing_usd_lb")


def wing_price_changes() -> pd.Series:
    """Year-over-year % change of the hardcoded wholesale-wing-price proxy (the public anchors)."""
    s = load_wing_price()
    return (s / s.shift(1) - 1.0).dropna() * 100.0
