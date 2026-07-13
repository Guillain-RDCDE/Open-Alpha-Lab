"""Data layer for Study 770 — Concert-Economy.

The claim under test: **Live Nation (LYV) rallies INTO festival season** — the annual
run-up to Coachella and the summer touring circuit. The steelman is genuinely strong:
Live Nation's business *is* seasonal (Q3, the summer touring quarter, is far and away
its biggest), so a rational market might *front-run* that seasonality, bidding the
stock up in the weeks before the festival calendar opens. The question is whether that
front-running is (a) real in the tape and (b) anything you could have banked — or
whether the seasonality is already fully priced and the "rally into festival season"
is just financial-media folklore.

Four ingredients:

* **The festival calendar, hardcoded.** Coachella (Indio, CA) is the traditional
  kickoff of the US festival/touring season — its weekend-1 opening date is announced
  months ahead (typically each January), which is what makes a "buy K sessions before
  Coachella" rule **calendar-known and zero-look-ahead by construction**. Every edition
  2006->2025 is hardcoded with its weekend-1 Friday; 2020 and 2021 were COVID-cancelled
  (no festival season kickoff those years). Source: Wikipedia "Coachella Valley Music
  and Arts Festival" per-year pages, cross-checked against Live Nation / Goldenvoice
  announcements.

* **The tradable instrument (yfinance).** ``LYV`` — Live Nation Entertainment, the
  world's largest live-events promoter (Live Nation + Ticketmaster). IPO'd Dec-2005, so
  the sample begins with the 2006 festival season. Benchmarked against ``SPY``
  (S&P 500, total return) so the test measures LYV's *abnormal* return, not the market's.

* **A labelled, cited touring-revenue PROXY.** Live Nation's real segment-level revenue
  is in its 10-K/10-Q filings but not on any free price API. So we hardcode a small,
  clearly-labelled **annual total-revenue** series and an approximate **quarterly
  seasonality share**, both reconstructed from the public filings (see
  ``docs/references.md``). This is a PROXY used only to establish the *fundamental*
  shape — festival season is a real revenue event — never presented as a live tape.

* **Synthetic world.** A deterministic, seeded paired (asset, benchmark) log-return
  world with a TUNABLE planted "pre-festival run-up bump" on a synthetic calendar.
  ``bump = 0`` is the null world; the one-sample-t machinery must not manufacture
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

AS_OF = "2026-06-30"     # last complete month at publication (2026-07-13)
INSTRUMENT = "LYV"       # Live Nation Entertainment
BENCHMARK = "SPY"        # S&P 500 total-return proxy

# --------------------------------------------------------------------------- #
# The Coachella calendar, hardcoded: year, weekend-1 FRIDAY (the festival-season
# kickoff), and a cancellation reason (None = held). 2020 and 2021 were both
# COVID-cancelled (Coachella did not run at all those years -> no festival-season
# kickoff -> no event). The weekend-1 Friday is the anchor: the "rally into festival
# season" window ENDS on the last close on/before this date. Coachella's dates are
# announced months ahead (each January), so a "buy K sessions before this Friday" rule
# is calendar-known and zero-look-ahead. Source: Wikipedia "Coachella Valley Music and
# Arts Festival" per-year pages.
# --------------------------------------------------------------------------- #
EVENTS = [
    # year, weekend1_friday, cancelled_reason
    (2006, "2006-04-28", None),
    (2007, "2007-04-27", None),
    (2008, "2008-04-25", None),
    (2009, "2009-04-17", None),
    (2010, "2010-04-16", None),
    (2011, "2011-04-15", None),
    (2012, "2012-04-13", None),   # first two-weekend edition; weekend 1
    (2013, "2013-04-12", None),
    (2014, "2014-04-11", None),
    (2015, "2015-04-10", None),
    (2016, "2016-04-15", None),
    (2017, "2017-04-14", None),
    (2018, "2018-04-13", None),
    (2019, "2019-04-12", None),
    (2020, None,         "COVID-19 (cancelled)"),
    (2021, None,         "COVID-19 (cancelled)"),
    (2022, "2022-04-15", None),   # festival returns
    (2023, "2023-04-14", None),
    (2024, "2024-04-12", None),
    (2025, "2025-04-11", None),
]

# --------------------------------------------------------------------------- #
# LABELLED PROXY — Live Nation total revenue ($bn), reconstructed from the 10-K
# filings (see docs/references.md). NOT a live feed; used only to draw the "festival
# season is a real revenue event" backdrop. The Concerts segment (touring/festivals)
# is roughly ~80% of these totals in a normal year.
# --------------------------------------------------------------------------- #
_ANNUAL_REVENUE_USD_B = {
    2010: 5.06, 2011: 5.38, 2012: 5.82, 2013: 6.48, 2014: 6.87,
    2015: 7.24, 2016: 8.35, 2017: 10.34, 2018: 10.79, 2019: 11.55,
    2020: 1.86,   # COVID -- live events shut down
    2021: 6.07, 2022: 16.68, 2023: 22.75, 2024: 23.16,
}

# LABELLED PROXY — approximate share of annual revenue by calendar quarter in a normal
# (pre-COVID) touring year, reconstructed from the 10-Q filings. Q3 (the summer touring
# quarter) dominates: this is the *fundamental* seasonality the folklore says the stock
# front-runs. Shares sum to 1.00.
_QUARTERLY_SHARE = {"Q1": 0.16, "Q2": 0.28, "Q3": 0.37, "Q4": 0.19}


def revenue_series() -> pd.Series:
    """Annual Live Nation total revenue ($bn), a LABELLED PROXY from the 10-Ks."""
    idx = pd.to_datetime([f"{y}-12-31" for y in _ANNUAL_REVENUE_USD_B])
    return pd.Series(list(_ANNUAL_REVENUE_USD_B.values()), index=idx, name="revenue_usd_b")


def quarterly_share() -> pd.Series:
    """Approximate share of annual revenue by quarter (LABELLED PROXY, pre-COVID)."""
    return pd.Series(_QUARTERLY_SHARE, name="rev_share")


def all_tickers() -> list[str]:
    return [INSTRUMENT, BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"concert_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2005-01-01", end: str = "2026-07-01") -> None:
    """Download adjusted (total-return) daily closes for LYV + SPY; cache them.

    ``auto_adjust=True`` -- both are dividend-paying listed equities/ETFs, so
    total-return (dividends reinvested) is the honest comparison.
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in all_tickers():
        d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d[["Close"]].dropna()
        d.to_csv(_cache_path(t))


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
# Synthetic world -- planted pre-festival run-up bump (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, seed: int = 770, n_events: int = 18,
                    n_days: int = 5000, spacing: int = 252,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (asset, benchmark) log-return world with a planted run-up.

    Both series are correlated (rho ~ 0.6, like a single high-beta name vs SPY) zero-mean
    noise; on the trading day just before each synthetic "festival day" (every
    ``spacing``-th business day) the asset gets an EXTRA ``bump`` log-return -- a planted
    pre-festival run-up -- while the benchmark does not. ``bump = 0`` is the null world:
    the run-up window is statistically identical to the rest of the tape.

    Business-day integer index (positions 0..n_days). Returns
    (asset_logret, bench_logret, festival_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.6
    common = rng.normal(0.0, 0.012, n_days)
    idio_a = rng.normal(0.0, 0.014, n_days)
    idio_b = rng.normal(0.0, 0.008, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    fest_pos = list(range(spacing, n_days - 130, spacing))[:n_events]
    for p in fest_pos:
        a[p - 1] += bump      # planted run-up: shows up in the pre-festival window

    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), fest_pos
