"""Data layer for Study 786 — Flu-Season.

The claim under test: **the pharmacy tape (CVS Health, ``CVS``) rallies *into* flu
season** — the folklore that the big drugstore chains are bid up ahead of their peak
revenue window (flu shots, cold-and-flu OTC, cough/cold scripts) as the market front-runs
the autumn/winter cold season. Pharmacies begin their flu-shot marketing push in
September and the season's clinical burden ramps through October. So a "buy K sessions
*before* the season anchor, hold into it" rule is anchored on a **fixed, calendar-known
convention**, not on any realised-in-the-data event.

Four ingredients:

* **The flu-season calendar, hardcoded — a REAL, publicly-known convention.** The U.S.
  influenza surveillance season, by CDC/WHO definition, runs from **MMWR epidemiological
  week 40 to week 20**, i.e. it *begins in early October*. The CDC's standard phrasing is
  "flu activity often begins to increase in October." We anchor each year's event on
  **October 1** — the canonical flu-season start date — which is a fixed calendar
  convention known years ahead and therefore zero-look-ahead by construction. (This is the
  season *start*, not a peak — peaks land Dec-Feb and vary; we deliberately test the
  *anticipation* window into the known start, which is the tradable, forward-known part of
  the folklore.) Source: CDC "The Flu Season" / FluView surveillance definition (MMWR
  weeks 40-20).

* **The tradable instrument (yfinance).** ``CVS`` — CVS Health Corp., the largest U.S.
  pharmacy chain (retail drugstores + Caremark PBM + Aetna). Benchmarked against ``SPY``
  (S&P 500, total return) so the test measures CVS's *abnormal* return, not the market's —
  CVS is a defensive healthcare/retail name with a below-market beta (~0.6-0.8), so a raw
  CVS move over a drifting window is partly just its (muted) beta.

* **No fundamental proxy needed.** "Rally into flu season" is a pure price-path claim
  anchored on the fixed October-1 season start: there is nothing to reconstruct from
  filings. The calendar convention *is* the event.

* **Synthetic world.** A deterministic, seeded paired (asset, benchmark) log-return world
  with a TUNABLE planted "pre-season run-up bump" and an optional "in-season fade" on a
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
INSTRUMENT = "CVS"       # CVS Health Corp.
BENCHMARK = "SPY"        # S&P 500 total-return proxy

# --------------------------------------------------------------------------- #
# The flu-season calendar, hardcoded: year and the fixed October-1 season start. The U.S.
# influenza surveillance season is defined (CDC/WHO) as MMWR weeks 40-20 — it *begins in
# early October* every year, a fixed calendar convention known years ahead. So "buy K
# sessions before the October-1 season start, hold into it" is calendar-known and
# zero-look-ahead. Unlike a company event there is no year-to-year slippage: the
# convention is a constant. Source: CDC "The Flu Season" / FluView (MMWR weeks 40-20).
# --------------------------------------------------------------------------- #
EVENTS = [
    # year, flu-season start (fixed CDC convention: early October / MMWR week 40)
    (2007, "2007-10-01"),
    (2008, "2008-10-01"),
    (2009, "2009-10-01"),   # 2009 H1N1 pandemic autumn wave
    (2010, "2010-10-01"),
    (2011, "2011-10-01"),
    (2012, "2012-10-01"),
    (2013, "2013-10-01"),
    (2014, "2014-10-01"),
    (2015, "2015-10-01"),
    (2016, "2016-10-01"),
    (2017, "2017-10-01"),   # severe H3N2 season
    (2018, "2018-10-01"),
    (2019, "2019-10-01"),
    (2020, "2020-10-01"),   # COVID era — near-absent flu season
    (2021, "2021-10-01"),
    (2022, "2022-10-01"),   # early, hard "tripledemic" season
    (2023, "2023-10-01"),
    (2024, "2024-10-01"),
    (2025, "2025-10-01"),
]


def all_tickers() -> list[str]:
    return [INSTRUMENT, BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"flu_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2006-01-01", end: str = "2026-07-01", retries: int = 4) -> None:
    """Download adjusted (total-return) daily closes for CVS + SPY; cache them.

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
# Synthetic world -- planted pre-season run-up + optional in-season fade
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, fade: float = 0.0, seed: int = 813,
                    n_events: int = 19, n_days: int = 5000, spacing: int = 252,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (asset, benchmark) log-return world with a planted run-up
    and an optional in-season fade.

    Both series are correlated (rho ~ 0.6, like a single defensive name vs SPY) zero-mean
    noise; on the trading day just before each synthetic "season-start day" (every
    ``spacing``-th business day) the asset gets an EXTRA ``bump`` log-return -- a planted
    pre-season run-up -- and on the day just after, an EXTRA ``-fade`` -- a planted
    in-season give-back. ``bump = fade = 0`` is the null world.

    Business-day integer index (positions 0..n_days). Returns
    (asset_logret, bench_logret, season_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.6
    common = rng.normal(0.0, 0.012, n_days)
    idio_a = rng.normal(0.0, 0.014, n_days)
    idio_b = rng.normal(0.0, 0.008, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    key_pos = list(range(spacing, n_days - 130, spacing))[:n_events]
    for p in key_pos:
        a[p - 1] += bump      # planted run-up: shows up in the pre-season window
        if p + 1 < n_days:
            a[p + 1] -= fade  # planted give-back: shows up in the in-season window

    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), key_pos
