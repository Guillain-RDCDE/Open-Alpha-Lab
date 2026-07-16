"""Data layer for Study 774 — Nintendo-Direct.

The claim under test: **"Buy the hype into a Nintendo Direct."** A Nintendo Direct is
Nintendo's flagship self-produced video presentation — the ~40-minute broadcast where new
games, hardware and release dates are unveiled. Gaming/finance folklore says the stock
*rallies into* a Direct as the hype builds, and either keeps ripping on a great showing or
*sells the news* once it airs — the textbook "buy the rumour, sell the news," transplanted
onto Nintendo's biggest owned-media catalyst.

There is a REAL wrinkle that makes this a harder (and more honest) calendar test than the
Apple keynote: **a Nintendo Direct is usually announced only ~1-3 days ahead**, not 7-10
like an Apple press invite. The big *general* Directs cluster seasonally (February, an
E3/June slot, September), so the season is guessable — but the exact date is not known two
weeks out. A literal "buy 10 sessions before the Direct" rule therefore carries a
look-ahead flavour (you had to know the date to place the trade). We test the run-up anyway
because that is exactly the folklore, but we flag the caveat loudly in the docs and treat it
as one more reason the trade is untradeable, not a hidden edge.

Four ingredients:

* **The Direct calendar, hardcoded.** The flagship *general* Nintendo Direct / Nintendo E3
  Direct broadcasts from 2013->2025, with their real air dates. These are the big
  multi-title showcases the "buy the hype" lore is about — NOT Direct Mini, Indie World,
  Partner Showcase, or single-game Directs. Dates cross-checked against Wikipedia's Nintendo
  Direct presentation tables and Nintendo Life's full broadcast-history list. 2020 is absent
  on purpose: Nintendo ran no traditional general Direct that year (COVID; only Minis and
  themed showcases).

* **The tradable instrument (yfinance).** ``NTDOY`` — Nintendo Co., Ltd. US ADR (OTC).
  Benchmarked against ``SPY`` (S&P 500 total return) so the test measures Nintendo's
  *abnormal* return, not the market's drift. NTDOY is a thinly-traded OTC ADR whose US close
  can lag the Tokyo tape — itself part of the honest tradability story.

* **No fundamental proxy needed.** "Buy the hype" is a pure price-path claim: the Direct air
  date *is* the event. Nothing to reconstruct from filings.

* **Synthetic world.** A deterministic, seeded paired (asset, benchmark) log-return world
  with a TUNABLE planted "pre-Direct run-up bump" and an optional "post-broadcast fade" on a
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
INSTRUMENT = "NTDOY"     # Nintendo Co., Ltd. US ADR (OTC)
BENCHMARK = "SPY"        # S&P 500 total-return proxy

# --------------------------------------------------------------------------- #
# The flagship Nintendo Direct calendar, hardcoded: air date and a short label for each
# big *general* Nintendo Direct / Nintendo E3 Direct broadcast, 2013->2025. These are the
# multi-title showcases the "buy the hype" folklore is about (not Direct Mini / Indie World
# / Partner Showcase / single-game Directs). Directs are typically announced only ~1-3 days
# ahead, so the exact date is NOT calendar-known two weeks out — a look-ahead caveat we flag
# in the docs. 2020 is intentionally absent (no traditional general Direct that year).
# Sources: Wikipedia "Nintendo Direct" presentation tables; Nintendo Life full
# broadcast-history list.
# --------------------------------------------------------------------------- #
EVENTS = [
    # air_date, label
    ("2013-06-11", "E3 2013"),
    ("2013-12-18", "Dec 2013 (Wii U / 3DS)"),
    ("2014-02-13", "Feb 2014 (Wii U / 3DS)"),
    ("2014-06-10", "E3 2014"),
    ("2015-06-16", "E3 2015"),
    ("2016-03-03", "Mar 2016 (Wii U / 3DS)"),
    ("2017-06-13", "E3 2017"),
    ("2017-09-13", "Sep 2017 (Switch / 3DS)"),
    ("2018-03-08", "Mar 2018"),
    ("2018-06-12", "E3 2018"),
    ("2018-09-13", "Sep 2018"),
    ("2019-02-13", "Feb 2019"),
    ("2019-06-11", "E3 2019"),
    ("2019-09-04", "Sep 2019"),
    ("2021-02-17", "Feb 2021"),
    ("2021-06-15", "E3 2021"),
    ("2021-09-23", "Sep 2021"),
    ("2022-02-09", "Feb 2022"),
    ("2022-09-13", "Sep 2022"),
    ("2023-02-08", "Feb 2023"),
    ("2023-06-21", "Jun 2023"),
    ("2023-09-14", "Sep 2023"),
    ("2024-06-18", "Jun 2024"),
    ("2025-03-27", "Mar 2025"),
    ("2025-09-12", "Sep 2025 (Switch 2)"),
]


def all_tickers() -> list[str]:
    return [INSTRUMENT, BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"nintendo_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2012-01-01", end: str = "2026-07-01", retries: int = 4) -> None:
    """Download adjusted (total-return) daily closes for NTDOY + SPY; cache them.

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
# Synthetic world -- planted pre-Direct run-up + optional post-broadcast fade
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, fade: float = 0.0, seed: int = 778,
                    n_events: int = 25, n_days: int = 5000, spacing: int = 180,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (asset, benchmark) log-return world with a planted run-up
    and an optional post-broadcast fade.

    Both series are correlated (rho ~ 0.55, like a single mid-beta ADR vs SPY) zero-mean
    noise; on the trading day just before each synthetic "Direct day" (every
    ``spacing``-th business day) the asset gets an EXTRA ``bump`` log-return -- a planted
    pre-Direct run-up -- and on the day just after, an EXTRA ``-fade`` -- a planted
    sell-the-news. ``bump = fade = 0`` is the null world.

    Business-day integer index (positions 0..n_days). Returns
    (asset_logret, bench_logret, direct_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.55
    common = rng.normal(0.0, 0.011, n_days)
    idio_a = rng.normal(0.0, 0.016, n_days)
    idio_b = rng.normal(0.0, 0.008, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    key_pos = list(range(spacing, n_days - 130, spacing))[:n_events]
    for p in key_pos:
        a[p - 1] += bump      # planted run-up: shows up in the pre-Direct window
        if p + 1 < n_days:
            a[p + 1] -= fade  # planted fade: shows up in the post-broadcast window

    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), key_pos
