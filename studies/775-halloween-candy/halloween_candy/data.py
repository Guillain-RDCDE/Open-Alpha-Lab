"""Data layer for Study 775 — Halloween-Candy.

The claim under test: **Hershey (HSY) enjoys a Halloween seasonal run-up** — the stock
rallies *into* October 31 because Halloween is the confectioner's single biggest selling
season (Hershey ships a huge share of its annual candy volume for the trick-or-treat
window). The folklore, as retail/financial-media tell it, has the familiar two halves of a
seasonal-catalyst trade: (1) the stock *rallies into* Halloween as the market front-runs
the candy season, and (2) it *fades after* once the season's news is out — a calendar
"buy the rumour, sell the news." Both halves are testable against a calendar that is
**fixed and known decades ahead**: Halloween is **always October 31**, so a "buy K sessions
before, sell on the day" rule is calendar-known and zero-look-ahead by construction.

Four ingredients:

* **The Halloween calendar, hardcoded.** One event per year — October 31 — from 1994->2025.
  Unlike a shifting product-launch date, this is a *fixed public-holiday* calendar: there
  is nothing to "verify" beyond the fact that Halloween is October 31 every year. We anchor
  on the last trading close on/before Oct 31, so a weekend/holiday Halloween just rolls back
  to the prior session.

* **The tradable instrument (yfinance).** ``HSY`` — The Hershey Company. Benchmarked against
  ``SPY`` (S&P 500, total return) so the test measures HSY's *abnormal* return, not the
  market's — a slow-moving consumer-staple still has a market beta, so a raw HSY move over a
  drifting window is partly just beta.

* **No fundamental proxy needed.** "Seasonal run-up" is a pure price-path claim anchored on
  a fixed calendar date: there is nothing to reconstruct from filings. Oct 31 *is* the event.

* **Synthetic world.** A deterministic, seeded paired (asset, benchmark) log-return world
  with a TUNABLE planted "pre-Halloween run-up bump" and an optional "post-holiday fade" on
  a synthetic calendar. ``bump = 0`` is the null world; the one-sample-t machinery must not
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
INSTRUMENT = "HSY"       # The Hershey Company
BENCHMARK = "SPY"        # S&P 500 total-return proxy

# --------------------------------------------------------------------------- #
# The Halloween calendar, hardcoded: one event per year on October 31. Halloween is a
# FIXED public holiday (always Oct 31), so "buy K sessions before the holiday, sell on the
# day" is calendar-known decades ahead and zero-look-ahead. We anchor on the last trading
# session on/before Oct 31, so a weekend/holiday Halloween rolls back to the prior close.
# The calendar starts in 1994 (SPY IPO'd Jan-1993, so 1994 is the first year with a full
# 1-month run-up window inside SPY's coverage) and runs to 2025.
# --------------------------------------------------------------------------- #
EVENTS = [(y, f"{y}-10-31") for y in range(1994, 2026)]   # 32 Halloweens, 1994->2025


def all_tickers() -> list[str]:
    return [INSTRUMENT, BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"halloween_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "1992-06-01", end: str = "2026-07-01", retries: int = 4) -> None:
    """Download adjusted (total-return) daily closes for HSY + SPY; cache them.

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
# Synthetic world -- planted pre-Halloween run-up + optional post-holiday fade
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, fade: float = 0.0, seed: int = 781,
                    n_events: int = 32, n_days: int = 8200, spacing: int = 252,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (asset, benchmark) log-return world with a planted run-up
    and an optional post-holiday fade.

    Both series are correlated (rho ~ 0.5, like a consumer-staple name vs SPY) zero-mean
    noise; on the trading day just before each synthetic "Halloween day" (every
    ``spacing``-th business day) the asset gets an EXTRA ``bump`` log-return -- a planted
    pre-Halloween run-up -- and on the day just after, an EXTRA ``-fade`` -- a planted
    sell-the-season. ``bump = fade = 0`` is the null world.

    Business-day integer index (positions 0..n_days). Returns
    (asset_logret, bench_logret, halloween_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.5
    common = rng.normal(0.0, 0.010, n_days)
    idio_a = rng.normal(0.0, 0.011, n_days)
    idio_b = rng.normal(0.0, 0.008, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    key_pos = list(range(spacing, n_days - 130, spacing))[:n_events]
    for p in key_pos:
        a[p - 1] += bump      # planted run-up: shows up in the pre-Halloween window
        if p + 1 < n_days:
            a[p + 1] -= fade  # planted fade: shows up in the post-Halloween window

    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), key_pos
