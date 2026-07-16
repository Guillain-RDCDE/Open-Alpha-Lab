"""Data layer for Study 772 — Album-Drop.

The claim under test: **Spotify (SPOT) moves on a blockbuster album release.** The
folklore is intuitive — a chart-smashing "album drop" (a Taylor Swift, Drake, Adele or
Bad Bunny record that shatters single-day / single-week streaming records) sends millions
of listeners onto the platform, so surely the *stock* should twitch: a "buy SPOT into the
drop" retail idea. Two halves, both testable against a **publicly-known-in-advance**
calendar (major albums are announced weeks ahead — pre-order pages, teaser singles, launch
dates), so a "buy K sessions before, sell on the day" rule is calendar-known and
zero-look-ahead by construction:

* **The run-up (the "buy the rumour").** Does SPOT rally in the days *into* a blockbuster
  drop as the hype builds?
* **The reaction (the "sell the news").** Does SPOT pop (or fade) in the days *after* the
  album lands and the streaming records get announced?

The economics say it *shouldn't*: Spotify's revenue is subscription (ARPU × MAU), not a
per-stream cut that a single mega-album meaningfully moves. One record-breaking week is a
rounding error against a ~600M-user base, and the release date is common knowledge. The
efficient-markets prior is a flat zero. But the folklore is loud, so we test it honestly.

Four ingredients:

* **The album calendar, hardcoded.** 27 genuinely blockbuster, record-adjacent album
  releases from 2018-06 (right after Spotify's April-2018 direct listing) through 2024-11,
  each with its real, publicly-verifiable release date. Source: artist / label press
  releases and contemporaneous coverage of the Spotify streaming records these set.

* **The tradable instrument (yfinance).** ``SPOT`` — Spotify Technology S.A. Benchmarked
  against ``SPY`` (S&P 500, total return) so the test measures SPOT's *abnormal* return,
  not the market's — SPOT is a high-beta growth name, so a raw SPOT move over a drifting
  window is partly just beta.

* **No fundamental proxy needed.** "Does the stock move on the drop" is a pure price-path
  claim anchored on the real release date; there is nothing to reconstruct from filings.

* **Synthetic world.** A deterministic, seeded paired (asset, benchmark) log-return world
  with a TUNABLE planted "pre-drop run-up bump" and an optional "post-drop fade" on a
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
INSTRUMENT = "SPOT"      # Spotify Technology S.A.
BENCHMARK = "SPY"        # S&P 500 total-return proxy

# --------------------------------------------------------------------------- #
# The blockbuster-album calendar, hardcoded: (artist — album, real release date). These
# are all genuinely record-adjacent drops (single-day / single-week Spotify streaming
# records or near-misses) from just after Spotify's 2018-04-03 direct listing onward, so
# every event has SPOT price history. Release dates are announced weeks ahead (pre-orders,
# teaser singles), so "buy K sessions before the drop, sell on the day" is calendar-known
# and zero-look-ahead. Source: artist/label press releases + contemporaneous coverage.
# --------------------------------------------------------------------------- #
EVENTS = [
    # label, release_date
    ("Drake — Scorpion", "2018-06-29"),
    ("Travis Scott — Astroworld", "2018-08-03"),
    ("Ariana Grande — thank u, next", "2019-02-08"),
    ("Taylor Swift — Lover", "2019-08-23"),
    ("Post Malone — Hollywood's Bleeding", "2019-09-06"),
    ("BTS — Map of the Soul: 7", "2020-02-21"),
    ("The Weeknd — After Hours", "2020-03-20"),
    ("Taylor Swift — folklore", "2020-07-24"),
    ("Ariana Grande — Positions", "2020-10-30"),
    ("Taylor Swift — evermore", "2020-12-11"),
    ("Olivia Rodrigo — SOUR", "2021-05-21"),
    ("Billie Eilish — Happier Than Ever", "2021-07-30"),
    ("Drake — Certified Lover Boy", "2021-09-03"),
    ("Taylor Swift — Red (Taylor's Version)", "2021-11-12"),
    ("Adele — 30", "2021-11-19"),
    ("Bad Bunny — Un Verano Sin Ti", "2022-05-06"),
    ("Harry Styles — Harry's House", "2022-05-20"),
    ("Beyoncé — Renaissance", "2022-07-29"),
    ("Taylor Swift — Midnights", "2022-10-21"),
    ("Drake & 21 Savage — Her Loss", "2022-11-04"),
    ("Olivia Rodrigo — GUTS", "2023-09-08"),
    ("Drake — For All the Dogs", "2023-10-06"),
    ("Taylor Swift — 1989 (Taylor's Version)", "2023-10-27"),
    ("Beyoncé — Cowboy Carter", "2024-03-29"),
    ("Taylor Swift — The Tortured Poets Department", "2024-04-19"),
    ("Billie Eilish — Hit Me Hard and Soft", "2024-05-17"),
    ("Kendrick Lamar — GNX", "2024-11-22"),
]


def all_tickers() -> list[str]:
    return [INSTRUMENT, BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"album_{ticker.lower()}.csv")


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
# Synthetic world -- planted pre-drop run-up + optional post-drop fade
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, fade: float = 0.0, seed: int = 776,
                    n_events: int = 27, n_days: int = 5000, spacing: int = 160,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (asset, benchmark) log-return world with a planted run-up
    and an optional post-drop fade.

    Both series are correlated (rho ~ 0.6, like a single high-beta name vs SPY) zero-mean
    noise; on the trading day just before each synthetic "drop day" (every ``spacing``-th
    business day) the asset gets an EXTRA ``bump`` log-return -- a planted pre-drop run-up
    -- and on the day just after, an EXTRA ``-fade`` -- a planted sell-the-news.
    ``bump = fade = 0`` is the null world.

    Business-day integer index (positions 0..n_days). Returns
    (asset_logret, bench_logret, drop_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.6
    common = rng.normal(0.0, 0.012, n_days)
    idio_a = rng.normal(0.0, 0.018, n_days)   # SPOT is noisier than AAPL
    idio_b = rng.normal(0.0, 0.008, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    key_pos = list(range(spacing, n_days - 130, spacing))[:n_events]
    for p in key_pos:
        a[p - 1] += bump      # planted run-up: shows up in the pre-drop window
        if p + 1 < n_days:
            a[p + 1] -= fade  # planted fade: shows up in the post-drop window

    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), key_pos
