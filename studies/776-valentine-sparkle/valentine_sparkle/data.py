"""Data layer for Study 776 — Valentine-Sparkle.

The claim under test: **Signet Jewelers (SIG) "rallies into" Valentine's Day** — the
single biggest gifting catalyst on a jeweller's calendar. Valentine's Day is the year's
first big diamond/engagement-gift peak (Signet owns Kay, Zales, Jared, and historically
the bulk of US mall jewellery), so the folklore says the stock is bid up in the weeks
*before* February 14 as the market front-runs the gifting quarter, and — the "sell the
news" half — fades once the holiday is past. Both halves are testable against a calendar
that is **fixed and known years ahead** (February 14, every year), so a "buy K sessions
before, sell on/after the date" rule is calendar-known and zero-look-ahead by
construction.

Four ingredients:

* **The Valentine's-Day calendar, hardcoded.** February 14 of every year 2009->2026.
  Unlike a corporate event this needs no source beyond the Gregorian calendar — Valentine's
  Day is February 14, full stop. When the 14th lands on a weekend the event logic anchors
  on the last trading close on/before it (handled in ``strategy.py``), so the actual date
  is preserved rather than idealised to "the second Friday of February."

* **The tradable instrument (yfinance).** ``SIG`` — Signet Jewelers Limited (NYSE, relisted
  September 2008). Benchmarked against ``SPY`` (S&P 500, total return) so the test measures
  SIG's *abnormal* return, not the market's — Signet is a volatile mid-cap specialty
  retailer with a market beta well above 1, so a raw SIG move over a drifting window is
  partly just beta.

* **No fundamental proxy needed.** "Rally into Valentine's" is a pure price-path claim;
  there is nothing to reconstruct from filings. The February 14 date *is* the event.

* **Synthetic world.** A deterministic, seeded paired (asset, benchmark) log-return world
  with a TUNABLE planted "pre-Valentine's run-up bump" and an optional "post-holiday fade"
  on a synthetic calendar. ``bump = 0`` is the null world; the one-sample-t machinery must
  not manufacture significance from it.

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
INSTRUMENT = "SIG"       # Signet Jewelers Limited (Kay / Zales / Jared)
BENCHMARK = "SPY"        # S&P 500 total-return proxy

# --------------------------------------------------------------------------- #
# The Valentine's-Day calendar, hardcoded: year and February 14 of that year. This is
# fixed by the Gregorian calendar and known years ahead, so "buy K sessions before the
# 14th, sell on/after the date" is calendar-known and zero-look-ahead. When the 14th lands
# on a weekend, the event logic (strategy.py) anchors on the last trading close on/before
# it — the true date is kept, not idealised. Window 2009 is the first with SIG coverage
# after Signet's September-2008 NYSE relisting.
# --------------------------------------------------------------------------- #
EVENTS = [
    # year, valentines_date
    (2009, "2009-02-14"),
    (2010, "2010-02-14"),
    (2011, "2011-02-14"),
    (2012, "2012-02-14"),
    (2013, "2013-02-14"),
    (2014, "2014-02-14"),
    (2015, "2015-02-14"),
    (2016, "2016-02-14"),
    (2017, "2017-02-14"),
    (2018, "2018-02-14"),
    (2019, "2019-02-14"),
    (2020, "2020-02-14"),
    (2021, "2021-02-14"),
    (2022, "2022-02-14"),
    (2023, "2023-02-14"),
    (2024, "2024-02-14"),
    (2025, "2025-02-14"),
    (2026, "2026-02-14"),
]


def all_tickers() -> list[str]:
    return [INSTRUMENT, BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"valentine_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2008-01-01", end: str = "2026-07-01", retries: int = 4) -> None:
    """Download adjusted (total-return) daily closes for SIG + SPY; cache them.

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
# Synthetic world -- planted pre-Valentine's run-up + optional post-holiday fade
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, fade: float = 0.0, seed: int = 782,
                    n_events: int = 18, n_days: int = 5000, spacing: int = 252,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (asset, benchmark) log-return world with a planted run-up
    and an optional post-holiday fade.

    Both series are correlated (rho ~ 0.6, like a single high-beta name vs SPY) zero-mean
    noise; on the trading day just before each synthetic "Valentine's day" (every
    ``spacing``-th business day) the asset gets an EXTRA ``bump`` log-return -- a planted
    pre-Valentine's run-up -- and on the day just after, an EXTRA ``-fade`` -- a planted
    sell-the-holiday. ``bump = fade = 0`` is the null world.

    Business-day integer index (positions 0..n_days). Returns
    (asset_logret, bench_logret, valentine_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.6
    common = rng.normal(0.0, 0.012, n_days)
    idio_a = rng.normal(0.0, 0.014, n_days)
    idio_b = rng.normal(0.0, 0.008, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    val_pos = list(range(spacing, n_days - 130, spacing))[:n_events]
    for p in val_pos:
        a[p - 1] += bump      # planted run-up: shows up in the pre-Valentine's window
        if p + 1 < n_days:
            a[p + 1] -= fade  # planted fade: shows up in the post-holiday window

    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), val_pos
