"""Data layer for Study 777 — Black-Friday-Retail.

The claim under test: **retail stocks rally into Black-Friday week** — the SPDR S&P
Retail ETF (``XRT``) is supposed to be bid up ahead of the biggest shopping day of the
US calendar (the Friday after Thanksgiving, the traditional kick-off of the holiday
season) as the market front-runs strong holiday-sales expectations. Two halves usually
get told together: (1) the sector *rallies into* Black Friday as the "holiday-sales
trade" is put on, and (2) it *fades after* once the Cyber-week numbers are out — the
retail flavour of "buy the rumour, sell the news."

Both halves are testable against a calendar that is **fixed by statute weeks ahead**:
Black Friday is always the day after the fourth Thursday of November (US Thanksgiving),
so a "buy K sessions before, sell on the day" rule is calendar-known and zero-look-ahead
by construction — there is no announcement to wait for.

Four ingredients:

* **The Black-Friday calendar, hardcoded.** Every year's Black Friday (the trading day
  after US Thanksgiving) from 2006->2025, with its real date. Black Friday itself is a
  half trading session (US equity markets close at 1pm ET), but it *is* a session, so we
  anchor on the real Black-Friday close (or the last close on/before it). The rule "day
  after the fourth Thursday of November" is a fixed statute, so every date is
  publicly-verifiable and known years ahead. Cross-checked against the NYSE holiday
  calendar / a standard 4th-Thursday-of-November computation.

* **The tradable instrument (yfinance).** ``XRT`` — SPDR S&P Retail ETF (an
  equal-weighted basket of US retailers), which began trading 2006-06-19. Benchmarked
  against ``SPY`` (S&P 500, total return) so the test measures retail's *abnormal* return
  vs the broad market, not the market's own November drift — XRT's market beta is near 1,
  so a raw XRT move over a drifting window is partly just beta.

* **No fundamental proxy needed.** "Rally into Black Friday" is a pure price-path claim:
  there is nothing to reconstruct from filings. The Black-Friday date *is* the event.

* **Synthetic world.** A deterministic, seeded paired (asset, benchmark) log-return world
  with a TUNABLE planted "pre-Black-Friday run-up bump" and an optional "post-event fade"
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
INSTRUMENT = "XRT"       # SPDR S&P Retail ETF
BENCHMARK = "SPY"        # S&P 500 total-return proxy

# --------------------------------------------------------------------------- #
# The Black-Friday calendar, hardcoded: year and the real date of that year's Black
# Friday (the trading day after US Thanksgiving = day after the fourth Thursday of
# November). This is fixed by statute, so "buy K sessions before Black Friday, sell on
# the day" is calendar-known and zero-look-ahead. XRT began trading 2006-06-19, so 2006
# is the first year with coverage. Black Friday is a half session (1pm ET close) but a
# real trading day. Source: 4th-Thursday-of-November rule, cross-checked vs NYSE holidays.
# --------------------------------------------------------------------------- #
EVENTS = [
    # year, black_friday_date (day after US Thanksgiving)
    (2006, "2006-11-24"),
    (2007, "2007-11-23"),
    (2008, "2008-11-28"),
    (2009, "2009-11-27"),
    (2010, "2010-11-26"),
    (2011, "2011-11-25"),
    (2012, "2012-11-23"),
    (2013, "2013-11-29"),
    (2014, "2014-11-28"),
    (2015, "2015-11-27"),
    (2016, "2016-11-25"),
    (2017, "2017-11-24"),
    (2018, "2018-11-23"),
    (2019, "2019-11-29"),
    (2020, "2020-11-27"),
    (2021, "2021-11-26"),
    (2022, "2022-11-25"),
    (2023, "2023-11-24"),
    (2024, "2024-11-29"),
    (2025, "2025-11-28"),
]


def all_tickers() -> list[str]:
    return [INSTRUMENT, BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"blackfriday_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2006-01-01", end: str = "2026-07-01", retries: int = 4) -> None:
    """Download adjusted (total-return) daily closes for XRT + SPY; cache them.

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
# Synthetic world -- planted pre-Black-Friday run-up + optional post-event fade
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, fade: float = 0.0, seed: int = 783,
                    n_events: int = 20, n_days: int = 5000, spacing: int = 252,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (asset, benchmark) log-return world with a planted run-up
    and an optional post-event fade.

    Both series are correlated (rho ~ 0.65, like a sector ETF vs SPY) zero-mean noise; on
    the trading day just before each synthetic "Black-Friday day" (every ``spacing``-th
    business day) the asset gets an EXTRA ``bump`` log-return -- a planted pre-Black-Friday
    run-up -- and on the day just after, an EXTRA ``-fade`` -- a planted sell-the-news.
    ``bump = fade = 0`` is the null world.

    Business-day integer index (positions 0..n_days). Returns
    (asset_logret, bench_logret, blackfriday_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.65
    common = rng.normal(0.0, 0.010, n_days)
    idio_a = rng.normal(0.0, 0.011, n_days)
    idio_b = rng.normal(0.0, 0.007, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    key_pos = list(range(spacing, n_days - 130, spacing))[:n_events]
    for p in key_pos:
        a[p - 1] += bump      # planted run-up: shows up in the pre-Black-Friday window
        if p + 1 < n_days:
            a[p + 1] -= fade  # planted fade: shows up in the post-Black-Friday window

    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), key_pos
