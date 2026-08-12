"""Data layer for Study 844 — Madden-Cover-Curse.

The claim under test. The gaming folklore "**Madden curse**" says the athlete on the
cover of that year's *Madden NFL* gets injured or slumps the following season (the same
superstition rides along with the *NBA 2K* cover). We are not a sports-medicine desk, so
we test the **finance transplant** of the curse: does the *publisher* — **Electronic
Arts** (``EA``, who makes *Madden NFL*) or **Take-Two Interactive** (``TTWO``, whose 2K
Sports label makes *NBA 2K*) — earn an **abnormal return** around the cover-athlete
reveal and the annual game launch? "Buy the hype into the new Madden / 2K" is the
tradable reflex; the desk's prior is that a *scheduled, pre-announced, once-a-year
product ship* is exactly the kind of catalyst a semi-strong-efficient market has already
priced, so we expect **None**.

Four ingredients:

* **The launch calendar, hardcoded.** The annual release dates of the last ten *Madden
  NFL* editions (Madden 16 → Madden 25, shipping 2015→2024) and the last ten *NBA 2K*
  editions (2K16 → 2K25, shipping 2015→2024), each tagged with its publisher and its
  cover athlete(s). Twenty events in all. The **launch date** is the rock-solid public
  fact we anchor on (release dates are unambiguous and widely documented); the cover
  athlete is carried for colour and to name the "cursed" player. Cross-checked against
  each title's Wikipedia release box and the publishers' press releases.

* **The tradable instruments (yfinance).** ``EA`` (Electronic Arts) and ``TTWO``
  (Take-Two Interactive), each benchmarked against ``SPY`` (S&P 500 total return) so we
  measure the publisher's *abnormal* return, not the market's drift. Each event is
  anchored to *its own* publisher — a Madden launch to EA, a 2K launch to TTWO.

* **No fundamental proxy needed.** "Buy the hype around the launch" is a pure price-path
  claim: the ship date *is* the event. A one-title annual release is a small slice of a
  large multi-franchise publisher's revenue, which is itself a reason to expect nothing.

* **Synthetic world.** A deterministic, seeded paired (publisher, benchmark) log-return
  world with a TUNABLE planted "launch-week drift" on a synthetic calendar. ``drift = 0``
  is the null world; the one-sample-t / Newey-West machinery must not manufacture
  significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"        # last complete month at publication
PUBLISHERS = ["EA", "TTWO"]  # Electronic Arts (Madden), Take-Two (NBA 2K)
BENCHMARK = "SPY"           # S&P 500 total-return proxy

# --------------------------------------------------------------------------- #
# The launch calendar, hardcoded: (launch_date, publisher, franchise, cover_athlete).
# The launch date is the anchor (an unambiguous public fact); the cover athlete is the
# "cursed" player, carried for colour. Ten Madden editions (Madden 16->25, EA) and ten
# NBA 2K editions (2K16->2K25, TTWO), all shipping 2015->2024.
# Sources: each title's Wikipedia release/"cover athlete" box; EA and Take-Two/2K press
# releases. Dates are US launch dates (standard-edition street date).
# --------------------------------------------------------------------------- #
EVENTS = [
    # launch_date, publisher, franchise+edition, cover athlete(s)
    ("2015-08-25", "EA",   "Madden NFL 16",  "Odell Beckham Jr."),
    ("2016-08-23", "EA",   "Madden NFL 17",  "Rob Gronkowski"),
    ("2017-08-25", "EA",   "Madden NFL 18",  "Tom Brady"),
    ("2018-08-10", "EA",   "Madden NFL 19",  "Antonio Brown"),
    ("2019-08-02", "EA",   "Madden NFL 20",  "Patrick Mahomes"),
    ("2020-08-28", "EA",   "Madden NFL 21",  "Lamar Jackson"),
    ("2021-08-20", "EA",   "Madden NFL 22",  "Tom Brady & Patrick Mahomes"),
    ("2022-08-19", "EA",   "Madden NFL 23",  "John Madden"),
    ("2023-08-18", "EA",   "Madden NFL 24",  "Josh Allen"),
    ("2024-08-16", "EA",   "Madden NFL 25",  "Christian McCaffrey"),
    ("2015-09-29", "TTWO", "NBA 2K16",       "Curry / Harden / Davis"),
    ("2016-09-20", "TTWO", "NBA 2K17",       "Paul George"),
    ("2017-09-19", "TTWO", "NBA 2K18",       "Kyrie Irving"),
    ("2018-09-11", "TTWO", "NBA 2K19",       "Giannis Antetokounmpo"),
    ("2019-09-06", "TTWO", "NBA 2K20",       "Anthony Davis"),
    ("2020-09-04", "TTWO", "NBA 2K21",       "Damian Lillard / Zion Williamson"),
    ("2021-09-10", "TTWO", "NBA 2K22",       "Luka Doncic"),
    ("2022-09-09", "TTWO", "NBA 2K23",       "Devin Booker"),
    ("2023-09-08", "TTWO", "NBA 2K24",       "Kobe Bryant"),
    ("2024-09-06", "TTWO", "NBA 2K25",       "Jayson Tatum / A'ja Wilson"),
]


def all_tickers() -> list[str]:
    return PUBLISHERS + [BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"madden_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2014-06-01", end: str = "2026-07-01", retries: int = 4) -> None:
    """Download adjusted (total-return) daily closes for EA + TTWO + SPY; cache them.

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
# Synthetic world -- planted launch-week drift (null at drift = 0)
# --------------------------------------------------------------------------- #
def synthetic_world(drift: float = 0.0, seed: int = 844,
                    n_events: int = 20, n_days: int = 3000, spacing: int = 130,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (publisher, benchmark) log-return world with a planted
    launch drift.

    Both series are correlated (rho ~ 0.5, like a single mid-beta software name vs SPY)
    zero-mean noise; on each synthetic "launch day" (every ``spacing``-th business day)
    the publisher gets an EXTRA ``drift`` log-return spread across the launch day and the
    two sessions after it -- a planted post-launch pop. ``drift = 0`` is the null world;
    the detector must not manufacture significance from it.

    Business-day integer index (positions 0..n_days). Returns
    (pub_logret, bench_logret, launch_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.5
    common = rng.normal(0.0, 0.010, n_days)
    idio_a = rng.normal(0.0, 0.015, n_days)
    idio_b = rng.normal(0.0, 0.008, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    key_pos = list(range(spacing, n_days - 40, spacing))[:n_events]
    for p in key_pos:
        # spread the planted drift over the launch day and the two sessions after it
        for off in (0, 1, 2):
            if p + off < n_days:
                a[p + off] += drift / 3.0
    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), key_pos
