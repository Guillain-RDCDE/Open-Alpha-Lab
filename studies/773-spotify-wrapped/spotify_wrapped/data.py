"""Data layer for Study 773 — Spotify-Wrapped.

The claim under test: **Spotify (SPOT) rallies into its December "Wrapped" launch** — the
viral, personalised year-in-review campaign that dominates social feeds every late
November / early December. The retail story is that Wrapped is a free, self-marketing
engagement blitz (millions of shareable cards) that reminds the market how sticky the
product is, so the stock is supposed to *rally into* the launch as attention builds, and
maybe *fade after* once the buzz is spent — a "buy the buzz, sell the news" on a date that
is **known weeks in advance** (Wrapped ships in the same late-Nov/early-Dec slot every
year). A "buy K sessions before the launch, sell on the day" rule is therefore
calendar-known and zero-look-ahead by construction.

Four ingredients:

* **The Wrapped calendar, hardcoded.** Every year's Spotify Wrapped launch date from the
  first "Wrapped"-branded edition (2016) through 2025, with its real date. The rollout
  clusters between **Nov 29 and Dec 6**; we keep the *actual* launch day each year, not an
  idealised "first Tuesday of December." Source: Spotify Newsroom press releases /
  Wikipedia "Spotify Wrapped", cross-checked against contemporaneous coverage.

* **The tradable instrument (yfinance).** ``SPOT`` — Spotify Technology S.A., which began
  trading via a NYSE **direct listing on 2018-04-03**. So only Wrapped launches from 2018
  onward have a tradable tape; 2016 and 2017 are kept in the calendar but excluded with an
  auditable reason (no SPOT coverage). Benchmarked against ``SPY`` (S&P 500, total return)
  so the test measures SPOT's *abnormal* return, not the market's — SPOT is a high-beta
  single name, so a raw move over a drifting window is partly just beta.

* **No fundamental proxy needed.** "Rally into the launch" is a pure price-path claim: there
  is nothing to reconstruct from filings. The Wrapped launch date *is* the event.

* **Synthetic world.** A deterministic, seeded paired (asset, benchmark) log-return world
  with a TUNABLE planted "pre-launch run-up bump" and an optional "post-event fade" on a
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
INSTRUMENT = "SPOT"      # Spotify Technology S.A. (NYSE direct listing 2018-04-03)
BENCHMARK = "SPY"        # S&P 500 total-return proxy

# --------------------------------------------------------------------------- #
# The Spotify Wrapped calendar, hardcoded: year and the real launch date of that year's
# Wrapped campaign. Wrapped ships in the same late-Nov/early-Dec slot every year (rollout
# clusters Nov 29 -> Dec 6), announced/anticipated weeks ahead, so "buy K sessions before
# the launch, sell on the day" is calendar-known and zero-look-ahead. SPOT only trades from
# its 2018-04-03 direct listing, so 2016 and 2017 are carried in the calendar but excluded
# downstream (no SPOT coverage). Source: Spotify Newsroom / Wikipedia "Spotify Wrapped".
# --------------------------------------------------------------------------- #
EVENTS = [
    # year, wrapped_launch_date
    (2016, "2016-12-06"),   # first "Wrapped"-branded edition (no SPOT tape yet)
    (2017, "2017-12-05"),   # (no SPOT tape yet — SPOT lists 2018-04-03)
    (2018, "2018-12-06"),   # first Wrapped as a public company
    (2019, "2019-12-05"),
    (2020, "2020-12-01"),
    (2021, "2021-12-01"),
    (2022, "2022-11-30"),
    (2023, "2023-11-29"),
    (2024, "2024-12-04"),
    (2025, "2025-12-03"),
]


def all_tickers() -> list[str]:
    return [INSTRUMENT, BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"wrapped_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2018-01-01", end: str = "2026-07-01", retries: int = 4) -> None:
    """Download adjusted (total-return) daily closes for SPOT + SPY; cache them.

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
# Synthetic world -- planted pre-launch run-up + optional post-event fade
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, fade: float = 0.0, seed: int = 777,
                    n_events: int = 18, n_days: int = 5000, spacing: int = 252,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (asset, benchmark) log-return world with a planted run-up
    and an optional post-event fade.

    Both series are correlated (rho ~ 0.6, like a single high-beta name vs SPY) zero-mean
    noise; on the trading day just before each synthetic "launch day" (every
    ``spacing``-th business day) the asset gets an EXTRA ``bump`` log-return -- a planted
    pre-launch run-up -- and on the day just after, an EXTRA ``-fade`` -- a planted
    sell-the-news. ``bump = fade = 0`` is the null world.

    The synthetic world is a generic detector proof (18 events for a clean control); the
    real study resolves only ~8 events (SPOT lists in 2018), which is why the observed
    numbers carry a much larger small-sample band than this control.

    Business-day integer index (positions 0..n_days). Returns
    (asset_logret, bench_logret, launch_positions).
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
        a[p - 1] += bump      # planted run-up: shows up in the pre-launch window
        if p + 1 < n_days:
            a[p + 1] -= fade  # planted fade: shows up in the post-launch window

    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), key_pos
