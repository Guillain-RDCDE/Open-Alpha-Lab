"""Data layer for Study 787 — Heatwave-Utilities.

The claim under test: **the utilities sector (XLU) rallies through peak-summer heat.**
The folklore is a fundamental "cooling-demand" story — the hottest weeks of the year
drive record air-conditioning load, electricity demand spikes, and (so the tale goes)
utility revenues and utility stocks with them, so you should be long the sector *through*
the dog-days of summer. It is a pure seasonality claim anchored on a **calendar-known**
window, so a "hold XLU across the peak-heat weeks" rule is zero-look-ahead by construction.

Four ingredients:

* **The peak-heat calendar, hardcoded — a real climatological fact, not a fabricated
  event.** Because of *seasonal temperature lag*, the hottest average temperatures across
  the contiguous United States do **not** fall on the June-21 solstice but ~4-5 weeks
  later, in mid-to-late July (NOAA / National Weather Service climatology). We anchor each
  year on a fixed **July 22** — the climatological centre of peak-summer heat — and test
  the window *approaching* the peak (the "into the heat" run-up) and the window *through
  and past* it (late July into August, the core cooling-demand weeks). The anchor is a
  published calendar/climatology convention, identical every year, so nothing here is a
  data-mined or look-ahead "hottest day."

* **The tradable instrument (yfinance).** ``XLU`` — the Utilities Select Sector SPDR
  (inception Dec 1998). Benchmarked against ``SPY`` so the test measures the sector's
  *abnormal* return, not the market's. Utilities are a low-beta (~0.3-0.5) defensive
  sector, so a raw XLU move over any window is only weakly market-driven — but we still
  quote XLU − SPY so a summer that was simply a good tape for *everything* is netted out.

* **No fundamental proxy needed.** "Rally through the heat" is a pure price-path claim on a
  calendar-known window; there is nothing to reconstruct from filings or weather stations.
  The peak-heat calendar anchor *is* the event.

* **Synthetic world.** A deterministic, seeded paired (asset, benchmark) log-return world
  with a TUNABLE planted "into-the-heat run-up bump" and an optional "post-peak fade" on a
  synthetic calendar. ``bump = 0`` is the null world; the one-sample-t machinery must not
  manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"     # last complete month at publication
INSTRUMENT = "XLU"       # Utilities Select Sector SPDR
BENCHMARK = "SPY"        # S&P 500 total-return proxy

# --------------------------------------------------------------------------- #
# The peak-summer-heat calendar, hardcoded: for each year, the fixed climatological centre
# of peak heat across the contiguous US. Because of the ~4-5 week SEASONAL TEMPERATURE LAG,
# the hottest average temperatures land in mid-to-late July, not on the solstice (NOAA /
# NWS climatology). We anchor on a fixed **July 22** every year — a published calendar
# convention, not a data-mined "hottest day of that year" — so "hold XLU across the
# peak-heat weeks" is calendar-known and zero-look-ahead. Coverage starts 1999, the first
# full summer after XLU's Dec-1998 inception. Source: NOAA/NWS seasonal-lag climatology.
# --------------------------------------------------------------------------- #
EVENTS = [(year, f"{year}-07-22") for year in range(1999, 2026)]   # 1999..2025


def all_tickers() -> list[str]:
    return [INSTRUMENT, BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"heatwave_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "1998-01-01", end: str = "2026-07-01", retries: int = 4) -> None:
    """Download adjusted (total-return) daily closes for XLU + SPY; cache them.

    Retries with linear backoff — Yahoo rate-limits transient bursts, so a first empty
    frame is usually cured by a short wait rather than a real "no such ticker".
    """
    import time

    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in all_tickers():
        last_err = None
        for attempt in range(retries):
            try:
                d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                d = d[["Close"]].dropna()
                if len(d) > 0:
                    d.to_csv(_cache_path(t))
                    break
                last_err = f"empty frame for {t}"
            except Exception as e:  # noqa: BLE001 -- transient network/rate-limit
                last_err = str(e)
            time.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError(f"fetch failed for {t} after {retries} tries: {last_err}")


def have_real() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in all_tickers())


def load_real(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached {ticker: adjusted-close Series}, each sliced to <= asof."""
    out = {}
    for t in all_tickers():
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        s = df["Close"]
        out[t] = s[s.index <= pd.Timestamp(asof)]
    return out


# --------------------------------------------------------------------------- #
# Synthetic world -- planted into-the-heat run-up + optional post-peak fade
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, fade: float = 0.0, seed: int = 814,
                    n_events: int = 27, n_days: int = 6800, spacing: int = 252,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (asset, benchmark) log-return world with a planted run-up
    into the peak-heat anchor and an optional post-peak fade.

    Both series are correlated (rho ~ 0.45, like a low-beta defensive sector vs SPY)
    zero-mean noise; on the trading day just before each synthetic "peak-heat day" (every
    ``spacing``-th business day) the asset gets an EXTRA ``bump`` log-return -- a planted
    into-the-heat run-up -- and on the day just after, an EXTRA ``-fade`` -- a planted
    post-peak fade. ``bump = fade = 0`` is the null world.

    Business-day integer index (positions 0..n_days). Returns
    (asset_logret, bench_logret, peak_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.45
    common = rng.normal(0.0, 0.010, n_days)
    idio_a = rng.normal(0.0, 0.010, n_days)
    idio_b = rng.normal(0.0, 0.009, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    peak_pos = list(range(spacing, n_days - 130, spacing))[:n_events]
    for p in peak_pos:
        a[p - 1] += bump      # planted run-up: shows up in the into-the-heat window
        if p + 1 < n_days:
            a[p + 1] -= fade  # planted fade: shows up in the past-peak window

    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), peak_pos
