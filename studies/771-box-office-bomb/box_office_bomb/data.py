"""Data layer for Study 771 — Box-Office-Bomb.

The claim under test: **you should sell The Walt Disney Company (DIS) right after one of
its movies flops** — a notorious "box-office bomb" opening weekend is a public,
market-moving humiliation (a nine-figure write-down waiting to be reported), so the stock
should sag in the weeks after the disappointment is known. This is the folklore version of
"a bad opening weekend is a sell signal on the studio."

Why this is a clean, zero-look-ahead calendar test:

* **The reveal is scheduled and public.** A wide theatrical release opens on a known
  Friday; the *weekend* box-office estimate is reported publicly on the following Sunday
  and the actuals on Monday. So the first trading session at which "it flopped" is common
  knowledge is the **Monday after the opening weekend** — and a "short/sell DIS on that
  Monday, cover K sessions later" rule is calendar-known and look-ahead-free by
  construction. We hardcode the real opening Friday of each film (a verifiable fact) and
  anchor the trade on the first trading session on/after the following Monday.

Four ingredients:

* **The flop calendar, hardcoded.** Sixteen genuinely notorious Disney-distributed
  box-office bombs, 2012→2025, each with its real wide-release opening date. Every one of
  these was widely reported as a major flop or drove an actual studio write-down (e.g.
  *John Carter*'s ~$200M write-down in 2012, *Dark Phoenix*'s ~$170M hit reported in Disney's
  Q4-2019, *The Marvels*' record-low MCU opening in 2023). Dates are cross-checked against
  Box Office Mojo / studio releases. NOTE: *Dark Phoenix* and *West Side Story* are 20th
  Century (ex-Fox) titles, but both were released **after** Disney's March-2019 Fox
  acquisition and their losses hit Disney's own P&L — they are Disney flops.

* **The tradable instrument (yfinance).** ``DIS`` — The Walt Disney Company. Benchmarked
  against ``SPY`` (S&P 500 total return) so the test measures DIS's *abnormal* return, not
  the market's. A single flop is a tiny fraction of Disney's revenue (parks, ESPN,
  streaming, consumer products dwarf any one film), so the efficient-markets prior is that a
  known flop is a non-event for a ~$200bn conglomerate.

* **No fundamental proxy needed.** "Sell after the flop" is a pure price-path claim anchored
  on the release calendar; there is nothing to reconstruct from filings. The opening weekend
  *is* the event.

* **Synthetic world.** A deterministic, seeded paired (asset, benchmark) log-return world
  with a TUNABLE planted "post-flop sell-off" (and an optional pre-release drift) on a
  synthetic calendar. ``drop = 0`` is the null world; the one-sample-t machinery must not
  manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"     # last complete month at publication
INSTRUMENT = "DIS"       # The Walt Disney Company
BENCHMARK = "SPY"        # S&P 500 total-return proxy

# --------------------------------------------------------------------------- #
# The Disney box-office-bomb calendar, hardcoded: title, real wide-release opening date,
# and a one-line note on why it is a notorious flop. The weekend box-office estimate is
# public by the following Sunday/Monday, so "sell DIS on the first session on/after the
# Monday after opening" is calendar-known and zero-look-ahead. Sources: Box Office Mojo,
# studio releases, and contemporaneous coverage of the write-downs. Wednesday/Thanksgiving
# openings (Lone Ranger, Strange World, Wish) still resolve to the Monday-after via the
# anchor rule below.
# --------------------------------------------------------------------------- #
EVENTS = [
    # title, opening_date (wide release, real)
    ("John Carter",                  "2012-03-09"),  # ~$200M write-down
    ("The Lone Ranger",              "2013-07-03"),  # ~$160-190M loss
    ("Tomorrowland",                 "2015-05-22"),  # ~$140M loss
    ("The BFG",                      "2016-07-01"),  # ~$18M open vs $140M budget
    ("A Wrinkle in Time",            "2018-03-09"),  # underperformed $100M+ budget
    ("Nutcracker and Four Realms",   "2018-11-02"),  # flop, weak opening
    ("Dark Phoenix",                 "2019-06-07"),  # ~$170M Disney write-down (Fox)
    ("West Side Story",              "2021-12-10"),  # ~$10M open, box-office bomb (Fox)
    ("Lightyear",                    "2022-06-17"),  # underperformed Pixar expectations
    ("Strange World",                "2022-11-23"),  # ~$100M+ loss, Thanksgiving flop
    ("Indiana Jones Dial of Destiny","2023-06-30"),  # big loss, weak $60M open
    ("Haunted Mansion",              "2023-07-28"),  # ~$24M open, flop
    ("The Marvels",                  "2023-11-10"),  # record-low MCU open, $200M+ loss
    ("Wish",                         "2023-11-22"),  # Thanksgiving flop
    ("Snow White",                   "2025-03-21"),  # ~$43M open vs $250M budget
    ("Elio",                         "2025-06-20"),  # lowest Pixar opening ever
]


def all_tickers() -> list[str]:
    return [INSTRUMENT, BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"bomb_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2010-01-01", end: str = "2026-07-01", retries: int = 4) -> None:
    """Download adjusted (total-return) daily closes for DIS + SPY; cache them.

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
# Synthetic world -- planted post-flop sell-off (+ optional pre-release drift)
# --------------------------------------------------------------------------- #
def synthetic_world(drop: float = 0.0, runup: float = 0.0, seed: int = 773,
                    n_events: int = 16, n_days: int = 5000, spacing: int = 252,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (asset, benchmark) log-return world with a planted post-flop
    sell-off and an optional pre-release drift.

    Both series are correlated (rho ~ 0.55, like a single large-cap name vs SPY) zero-mean
    noise; on the trading day just AFTER each synthetic "flop day" (every ``spacing``-th
    business day) the asset gets an EXTRA ``-drop`` log-return -- the planted post-flop
    sell-off -- and on the day just before, an EXTRA ``runup`` -- an optional pre-release
    drift. ``drop = runup = 0`` is the null world.

    Business-day integer index (positions 0..n_days). Returns
    (asset_logret, bench_logret, flop_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.55
    common = rng.normal(0.0, 0.011, n_days)
    idio_a = rng.normal(0.0, 0.013, n_days)
    idio_b = rng.normal(0.0, 0.008, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    flop_pos = list(range(spacing, n_days - 130, spacing))[:n_events]
    for p in flop_pos:
        a[p - 1] += runup       # optional pre-release drift (pre window)
        if p + 1 < n_days:
            a[p + 1] -= drop    # planted post-flop sell-off (post window)

    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), flop_pos
