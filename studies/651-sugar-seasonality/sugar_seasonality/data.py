"""Data layer for Study 651 — Sugar-Seasonality.

Three ingredients, all offline-friendly once cached:

* **Real tape — the tradable ETF.** Daily adjusted closes of **CANE**, the Teucrium Sugar Fund
  (inception 2011-09-19), which holds a weighted roll across three ICE No.11 raw-sugar contract
  months. This is the honest, already-costed, tradable expression of the "Brazil crush calendar"
  claim — nobody can hold spot raw sugar for free.

* **Real tape — the roll-naive futures cross-check.** Daily front-month continuous ICE No.11 raw
  sugar futures (``SB=F``) from yfinance: a roll-naive splice of whichever contract Yahoo currently
  quotes as front-month. **Not** a tradable total-return series (no one can hold "the front month
  forever" for free) — a *spot-price proxy* used only to size how much of any seasonal the ETF's own
  roll mechanics give back. Named explicitly wherever it appears; never presented as something you
  could have earned.

* **The crush calendar, hardcoded.** Sugar's seasonal folklore has two supply engines pulling in the
  same direction: Brazil's Center-South cane crush (the world's largest raw-sugar supply, roughly
  **April → November**) and India's cane-crushing season (roughly **October → April**). The claimed
  mechanism: raw-sugar stocks run *tightest* in the Northern-Hemisphere winter, before Brazil's new
  crush gets into full swing — a "pre-harvest tight" window — and prices are supposed to give that
  back once the Brazilian crush floods the market with new-crop supply every spring. Facts, not a
  network fetch, exactly like study 637's ``FOMC_DATES`` / study 648's ``GRAIN_CALENDAR``: a table
  anyone can check against USDA FAS and Brazil's UNICA crush-progress reports.

* **Synthetic world.** A deterministic, seeded monthly-return series with a TUNABLE planted
  pre-harvest-premium / crush-discount pair (knob ``seasonal``). ``seasonal = 0`` is the null world;
  the Welch/HAC machinery must NOT manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build the cache
and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
ETF_CACHE = os.path.join(CACHE_DIR, "ss_etf_cane.csv")
FUT_CACHE = os.path.join(CACHE_DIR, "ss_fut_sb.csv")

ETF_TICKER = "CANE"       # Teucrium Sugar Fund — the tradable, roll-costed instrument
FUT_TICKER = "SB=F"       # ICE No.11 raw sugar, roll-naive continuous front-month splice

# CANE's 2011-09-19 inception sets the common start (same launch date as Teucrium's WEAT/CORN).
# AS_OF is the last COMPLETE calendar month at the time this study was built.
START = "2011-10-01"
AS_OF = "2026-06-30"

# --------------------------------------------------------------------------- #
# Hardcoded crush calendar (facts, no network). Source: USDA FAS Sugar: World Markets and Trade
# (https://www.fas.usda.gov/data/sugar-world-markets-and-trade), UNICA Brazil crush-progress
# reports (https://unicadata.com.br) and ISO (International Sugar Organization) crop calendars.
# "pre_harvest_tight" is the Northern-Hemisphere-winter stretch believers point to for the
# seasonal high (Brazilian old-crop stocks at their scarcest, ahead of the new crush); "crush" is
# the stretch believers point to for the seasonal low (Brazil's Center-South crush ramps to full
# volume and new-crop supply floods the market). Windows are inclusive month numbers.
# --------------------------------------------------------------------------- #
SUGAR_CALENDAR = {
    "brazil_crush": (4, 5, 6, 7, 8, 9, 10, 11),      # Apr-Nov: Center-South cane crush
    "india_crush": (10, 11, 12, 1, 2, 3, 4),         # Oct-Apr: Indian cane-crushing season
    "pre_harvest_tight": (1, 2, 3),                  # Jan-Mar: old-crop stocks scarcest
}

# The pooled "pre-harvest tight / crush glut" claim tested against the real tape: TIGHT is the
# stretch before the Brazilian crush is in full swing; CRUSH is the early bulk of that crush, when
# the largest single supply source floods the market fastest (the shoulder Aug-Nov months overlap
# with India's own harvest coming online too, so the cleanest single-direction test window is the
# early Brazilian ramp-up).
TIGHT_MONTHS = (1, 2, 3)          # Jan-Mar: the claimed seasonal high
CRUSH_MONTHS = (4, 5, 6, 7)       # Apr-Jul: the claimed seasonal low (Brazil crush ramp-up)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2010-01-01", end: str = "2026-07-01") -> None:
    """Download CANE adjusted closes and SB=F raw closes; cache them. Network; once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    etf = yf.download(ETF_TICKER, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(etf.columns, pd.MultiIndex):
        etf.columns = etf.columns.get_level_values(0)
    etf[["Close"]].dropna().to_csv(ETF_CACHE)

    fut = yf.download(FUT_TICKER, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(fut.columns, pd.MultiIndex):
        fut.columns = fut.columns.get_level_values(0)
    fut[["Close"]].dropna().to_csv(FUT_CACHE)


def have_real() -> bool:
    return os.path.exists(ETF_CACHE) and os.path.exists(FUT_CACHE)


def _load_one(path: str, start: str, asof: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    s = df["Close"].astype(float)
    return s.loc[(s.index >= start) & (s.index <= asof)]


def load_real(start: str = START, asof: str = AS_OF) -> tuple[pd.Series, pd.Series]:
    """Cached (CANE close, SB=F close) daily series, sliced to [start, asof]."""
    etf = _load_one(ETF_CACHE, start, asof)
    fut = _load_one(FUT_CACHE, start, asof)
    return etf, fut


# --------------------------------------------------------------------------- #
# Synthetic world — planted pre-harvest-tight premium / crush discount (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(seasonal: float = 0.0, seed: int = 651, n_years: int = 30,
                    vol: float = 0.28,
                    tight: tuple[int, ...] = TIGHT_MONTHS, crush: tuple[int, ...] = CRUSH_MONTHS,
                    ) -> pd.DataFrame:
    """Deterministic monthly sugar-return world with a TUNABLE planted tight/crush calendar.

    i.i.d. monthly log returns at annualised vol ``vol`` (raw sugar runs hot, ~25-30%/yr), plus a
    ``seasonal`` premium spread evenly over ``tight`` months and an equal-and-opposite discount
    spread over ``crush`` months. ``seasonal = 0`` is the null world: tight and crush months are
    statistically identical, and the Welch/HAC machinery must NOT reach significance.

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
    add = np.where(np.isin(months, tight), seasonal / len(tight),
                   np.where(np.isin(months, crush), -seasonal / len(crush), 0.0))
    return pd.DataFrame({"ret": base + add}, index=idx)
