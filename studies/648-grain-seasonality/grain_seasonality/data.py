"""Data layer for Study 648 — Grain Seasonality.

Three ingredients, all offline-friendly once cached:

* **Real tape — the tradable ETFs.** Daily adjusted closes of the three Teucrium single-commodity
  grain ETFs: **CORN** (corn, inception 2010-06-09), **WEAT** (wheat, inception 2011-09-19) and
  **SOYB** (soybeans, inception 2011-01-13). Each ETF holds a *weighted roll* across near, second
  and third contract months — i.e. it is the honest, already-costed, tradable expression of the
  "old-crop/new-crop" calendar claim. All three fetched from yfinance (no key), cached as CSV.

* **Real tape — the roll-naive futures cross-check.** Daily front-month continuous futures
  (``ZC=F`` corn, ``ZW=F`` wheat, ``ZS=F`` soybeans) from yfinance: a roll-naive splice of
  whichever contract Yahoo currently quotes as front-month. This is **not** a tradable
  total-return series (no one can hold "the front month forever" for free) — it is a *spot-price
  proxy* used only to size how much of any seasonal the ETF's own roll mechanics give back. Named
  explicitly wherever it appears; never presented as something you could have earned.

* **The agronomic calendar, hardcoded.** USDA NASS crop-progress reporting windows for the three
  crops — planting, the weather-risk stretch that feeds the "spring planting-scare high", and
  harvest, the "harvest-time low". Facts, not a network fetch, exactly like study 637's
  ``FOMC_DATES``: a table anyone can check against USDA's published crop calendars
  (https://usda.library.cornell.edu/concern/publications/8336h188j).

* **Synthetic world.** A deterministic, seeded monthly-return series with a TUNABLE planted
  spring-premium / harvest-discount pair (knob ``seasonal``). ``seasonal = 0`` is the null world;
  the Welch/HAC machinery must NOT manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

# --------------------------------------------------------------------------- #
# Tickers
# --------------------------------------------------------------------------- #
GRAINS = ("corn", "wheat", "soy")
ETF_TICKERS = {"corn": "CORN", "wheat": "WEAT", "soy": "SOYB"}
FUT_TICKERS = {"corn": "ZC=F", "wheat": "ZW=F", "soy": "ZS=F"}

# WEAT (2011-09-19) is the youngest of the three ETFs — the common-inception start for a fair
# cross-grain panel. AS_OF is the last COMPLETE calendar month at the time this study was built.
START = "2011-10-01"
AS_OF = "2026-06-30"


def _etf_cache(grain: str) -> str:
    return os.path.join(CACHE_DIR, f"gs_etf_{grain}.csv")


def _fut_cache(grain: str) -> str:
    return os.path.join(CACHE_DIR, f"gs_fut_{grain}.csv")


# --------------------------------------------------------------------------- #
# Hardcoded USDA crop-progress calendar (facts, no network).
# Source: USDA NASS Field Crops usual planting/harvesting dates
# (https://usda.library.cornell.edu/concern/publications/8336h188j) — the same generic windows
# taught in every grain-desk seasonality writeup. "weather_scare" is the pollination / dormancy
# -break stretch believers point to for the spring high; "harvest" is the new-crop-supply stretch
# believers point to for the autumn low. Windows are inclusive month numbers.
# --------------------------------------------------------------------------- #
GRAIN_CALENDAR = {
    "corn": {
        "planting": (4, 5),           # Apr-May
        "weather_scare": (6, 7),      # Jun-Jul: pollination/silking — the classic "weather market"
        "harvest": (9, 10, 11),       # Sep-Nov: new-crop pressure
    },
    "soy": {
        "planting": (5, 6),           # May-Jun
        "weather_scare": (7, 8),      # Jul-Aug: pod-fill — the other "weather market"
        "harvest": (9, 10, 11),       # Sep-Nov: new-crop pressure
    },
    "wheat": {
        # WEAT tracks CBOT (soft red winter) wheat. Winter wheat is planted in the fall, breaks
        # dormancy in spring, and is harvested early relative to corn/soy.
        "planting": (9, 10),          # Sep-Oct
        "weather_scare": (3, 4, 5),   # Mar-May: green-up / freeze-and-drought risk before harvest
        "harvest": (6, 7),            # Jun-Jul
    },
}

# The pooled "old-crop/new-crop" claim tested across all three grains at once: SPRING = the union
# of each grain's own weather-scare window; HARVEST = the union of each grain's own harvest window.
SPRING_MONTHS = {"corn": (6, 7), "soy": (7, 8), "wheat": (3, 4, 5)}
HARVEST_MONTHS = {"corn": (9, 10, 11), "soy": (9, 10, 11), "wheat": (6, 7)}


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2010-01-01", end: str = "2026-07-01") -> None:
    """Download CORN/WEAT/SOYB adjusted closes and ZC=F/ZW=F/ZS=F raw closes; cache. Network; once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for grain, ticker in ETF_TICKERS.items():
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df[["Close"]].dropna().to_csv(_etf_cache(grain))

    for grain, ticker in FUT_TICKERS.items():
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df[["Close"]].dropna().to_csv(_fut_cache(grain))


def have_real() -> bool:
    return all(os.path.exists(_etf_cache(g)) for g in GRAINS) and all(
        os.path.exists(_fut_cache(g)) for g in GRAINS
    )


def _load_one(path: str, start: str, asof: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    s = df["Close"].astype(float)
    return s.loc[(s.index >= start) & (s.index <= asof)]


def load_real(start: str = START, asof: str = AS_OF
              ) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Cached ({grain: etf close}, {grain: futures close}) daily series, sliced to [start, asof]."""
    etf = {g: _load_one(_etf_cache(g), start, asof) for g in GRAINS}
    fut = {g: _load_one(_fut_cache(g), start, asof) for g in GRAINS}
    return etf, fut


# --------------------------------------------------------------------------- #
# Synthetic world — planted spring-premium / harvest-discount (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(seasonal: float = 0.0, seed: int = 648, n_years: int = 30,
                    vol: float = 0.24,
                    spring: tuple[int, ...] = (6, 7), harvest: tuple[int, ...] = (9, 10, 11),
                    ) -> pd.DataFrame:
    """Deterministic monthly grain-return world with a TUNABLE planted spring/harvest calendar.

    i.i.d. monthly log returns at annualised vol ``vol`` (grains run hot, ~20-30%/yr), plus a
    ``seasonal`` premium spread evenly over ``spring`` months and an equal-and-opposite discount
    spread over ``harvest`` months. ``seasonal = 0`` is the null world: spring and harvest months
    are statistically identical, and the Welch/HAC machinery must NOT reach significance.

    Monthly PeriodIndex grid (n_years * 12 points, far below the pandas ns-timestamp Timestamp
    trap even before conversion) -> converted to a month-end DatetimeIndex for downstream reuse.
    Returns a one-column ("ret") frame.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    pidx = pd.period_range("1996-01", periods=n, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()
    months = idx.month

    base = (vol / np.sqrt(12)) * rng.standard_normal(n)
    add = np.where(np.isin(months, spring), seasonal / len(spring),
                   np.where(np.isin(months, harvest), -seasonal / len(harvest), 0.0))
    return pd.DataFrame({"ret": base + add}, index=idx)
