"""Data layer for Study 778 — Chipotle-Scare.

The claim under test: **"buy the dip" on a Chipotle (CMG) food-safety scare.** The
folklore is a value/contrarian reflex — when a headline breaks that a Chipotle location
has poisoned its customers (E. coli, norovirus, Salmonella, Clostridium), the stock gaps
down on the fear, so a brave contrarian buys the panic and rides the recovery. This is a
**single-name event study** (shape D): the "event" is not a scheduled calendar date but
the day a specific food-safety scare became *public, market-moving news*, and the trade is
to buy CMG at that day's close and hold K sessions, measuring the **abnormal** return
(CMG − SPY) so we are not just reading the market.

Unlike a keynote, these events are **unscheduled** — you cannot front-run them. That is
the whole point of the "buy the dip" framing: the news is already out and public when you
act, so a "buy on the announcement close, hold K sessions" rule is *executable and
zero-look-ahead by construction* (you only ever trade on public information).

Four ingredients:

* **The scare calendar, hardcoded — REAL, publicly-verifiable events.** Six Chipotle
  food-safety scares 2015->2018, each anchored on the trading day the scare became
  public, market-moving news (cross-checked against CDC/state-health-department releases
  and contemporaneous financial coverage; sources in ``docs/references.md``). The Aug-2015
  Simi Valley, CA norovirus outbreak (234 sick) is **deliberately excluded** — Chipotle
  did not disclose it at the time (it surfaced only in a Dec-2015 / Jan-2016 lawsuit), so
  there was no contemporaneous market-moving date to anchor on. Honesty over sample size.

* **The tradable instrument (yfinance).** ``CMG`` — Chipotle Mexican Grill. Benchmarked
  against ``SPY`` (S&P 500, total return) so the test measures CMG's *abnormal* return, not
  the market's drift over the window. (``auto_adjust=True`` also folds in CMG's 50-for-1
  June-2024 stock split, so the tape is continuous.)

* **No fundamental proxy needed.** "Buy the dip" is a pure price-path claim: there is
  nothing to reconstruct from filings. The scare-announcement date *is* the event.

* **Synthetic world.** A deterministic, seeded paired (asset, benchmark) log-return world
  with a TUNABLE planted acute "dip" into the event and an optional planted "rebound"
  after it. ``dip = rebound = 0`` is the null world; the one-sample-t machinery must not
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
INSTRUMENT = "CMG"       # Chipotle Mexican Grill
BENCHMARK = "SPY"        # S&P 500 total-return proxy

# --------------------------------------------------------------------------- #
# The Chipotle food-safety scare calendar, hardcoded. Each row is (year, anchor_date,
# label): the anchor is the trading day the scare became PUBLIC, market-moving news, so
# "buy CMG at this close, hold K sessions" trades only on public information. These are
# REAL, verifiable events (CDC / state health departments / contemporaneous coverage —
# see docs/references.md). Note the Dec-2015 cluster (rows 3 & 4) is only a few sessions
# apart, so those two post-windows OVERLAP — a dependence we flag loudly in the results;
# they are nonetheless two distinct, separately-reported scares.
# --------------------------------------------------------------------------- #
EVENTS = [
    # year, anchor_date, what broke
    (2015, "2015-09-17", "Minnesota Salmonella Newport (tomatoes, 64 sick) — MDH publicly links it to Chipotle"),
    (2015, "2015-11-02", "E. coli O26, WA/OR — Chipotle closes 43 restaurants (news broke over the weekend of Oct 31)"),
    (2015, "2015-12-04", "CDC announces a second, genetically distinct E. coli outbreak across 5 more states"),
    (2015, "2015-12-08", "Boston College norovirus (~140 sickened, incl. BC basketball players)"),
    (2017, "2017-07-18", "Sterling, VA norovirus (~135 sickened); CMG fell ~4% intraday"),
    (2018, "2018-08-01", "Powell, OH Clostridium perfringens (~647 sickened); CMG fell ~4.5%"),
]


def all_tickers() -> list[str]:
    return [INSTRUMENT, BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"chipotle_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2013-01-01", end: str = "2026-07-01", retries: int = 4) -> None:
    """Download adjusted (total-return, split-adjusted) daily closes for CMG + SPY; cache.

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
# Synthetic world -- planted acute dip into the event + optional rebound after it
# --------------------------------------------------------------------------- #
def synthetic_world(dip: float = 0.0, rebound: float = 0.0, seed: int = 784,
                    n_events: int = 6, n_days: int = 3000, spacing: int = 252,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (asset, benchmark) log-return world with a planted acute dip
    and an optional post-event rebound.

    Both series are correlated (rho ~ 0.55, like a single high-vol single name vs SPY)
    zero-mean noise; on the trading day just before each synthetic "scare day" (every
    ``spacing``-th business day) the asset gets an EXTRA ``-dip`` log-return -- a planted
    acute drop into the scare -- and on the day just after, an EXTRA ``+rebound`` -- a
    planted buy-the-dip recovery. ``dip = rebound = 0`` is the null world.

    Business-day integer index (positions 0..n_days). Returns
    (asset_logret, bench_logret, event_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.55
    common = rng.normal(0.0, 0.011, n_days)
    idio_a = rng.normal(0.0, 0.016, n_days)   # CMG is a higher-vol single name than SPY
    idio_b = rng.normal(0.0, 0.008, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    ev_pos = list(range(spacing, n_days - 130, spacing))[:n_events]
    for p in ev_pos:
        a[p - 1] -= dip            # planted acute drop: shows up in the pre-event window
        if p + 1 < n_days:
            a[p + 1] += rebound    # planted rebound: shows up in the post-event window

    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), ev_pos
