"""Data layer for Study 731 — Wimbledon-Effect.

The claim under test: the **Wimbledon fortnight** (late June → mid-July) is the UK
market's "summer lull" — a quiet tennis-and-strawberries window where the City empties,
volume thins, and the FTSE goes sleepy. City folklore holds that this is a distinctive,
low-energy calendar window worth stepping aside for (or, in the trader-bar version,
worth *fading*). We test it on ``EWU`` (iShares MSCI United Kingdom, the tradable UK
equity vehicle), against a broad Europe benchmark, over 2005→2025.

Three ingredients:

* **The calendar, hardcoded.** Every Championships fortnight 2005→2025, its exact start
  (first Monday) and end (second Sunday — the gentlemen's singles final). **2020 was
  cancelled** (COVID-19 — the first cancellation since WWII). The fortnight is a
  *pre-scheduled, calendar-known* window (the All England Club publishes dates years in
  advance), so — unlike an announcement study — there is **no look-ahead problem at
  all**: you know the exact window before the year begins. Source: Wikipedia "Wimbledon
  Championships" and the per-year "<YEAR> Wimbledon Championships" pages; each pair is
  asserted to be a Monday→Sunday span exactly 13 days apart on load.
* **Real tape.** Daily adjusted (total-return) closes for ``EWU`` (the UK ETF) and
  ``VGK`` (Vanguard FTSE Europe, the benchmark that removes the Europe-wide summer
  drift so the test isolates a *UK-specific* window effect), from yfinance, cached as
  CSV under the study's own ``_cache/``. ``VGK``'s inception (2005-03-10) is the hard
  floor on how far back the abnormal-return test can reach — which is why the sample
  starts at the 2005 Championships, not EWU's 1996 inception.
* **Synthetic world.** A deterministic, seeded pair of (asset, benchmark) log-return
  series with a TUNABLE planted "fortnight seasonal" on a synthetic calendar window.
  ``bump = 0`` is the null world; the one-sample-*t* machinery must not manufacture
  significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import datetime as _dt
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

UK_ETF = "EWU"              # iShares MSCI United Kingdom -- the tradable UK vehicle
EUROPE_BENCHMARK = "VGK"    # Vanguard FTSE Europe -- removes the Europe-wide summer drift
TICKERS = (UK_ETF, EUROPE_BENCHMARK)

START = "2004-06-01"        # a year of run-up before the first (2005) event window
AS_OF = "2026-06-30"        # last complete calendar month at publication

# --------------------------------------------------------------------------- #
# The Wimbledon calendar, hardcoded: year, fortnight START (first Monday),
# fortnight END (second Sunday = gentlemen's singles final day). 2020 cancelled
# (COVID-19). Since 2015 the Championships begin one week later than before
# (three weeks after the French Open final) -- visible in the drift of the start
# dates from ~20-25 June (2005-2014) to ~30 Jun-3 Jul (2015-2025).
# Source: Wikipedia "Wimbledon Championships" + the per-year contest pages; every
# pair is asserted Monday->Sunday, 13 days apart, at import time (see _validate).
# --------------------------------------------------------------------------- #
WIMBLEDON: list[tuple[int, str | None, str | None]] = [
    (2005, "2005-06-20", "2005-07-03"),
    (2006, "2006-06-26", "2006-07-09"),
    (2007, "2007-06-25", "2007-07-08"),
    (2008, "2008-06-23", "2008-07-06"),
    (2009, "2009-06-22", "2009-07-05"),
    (2010, "2010-06-21", "2010-07-04"),
    (2011, "2011-06-20", "2011-07-03"),
    (2012, "2012-06-25", "2012-07-08"),
    (2013, "2013-06-24", "2013-07-07"),
    (2014, "2014-06-23", "2014-07-06"),
    (2015, "2015-06-29", "2015-07-12"),
    (2016, "2016-06-27", "2016-07-10"),
    (2017, "2017-07-03", "2017-07-16"),
    (2018, "2018-07-02", "2018-07-15"),
    (2019, "2019-07-01", "2019-07-14"),
    (2020, None,          None),          # cancelled, COVID-19
    (2021, "2021-06-28", "2021-07-11"),
    (2022, "2022-06-27", "2022-07-10"),
    (2023, "2023-07-03", "2023-07-16"),
    (2024, "2024-07-01", "2024-07-14"),
    (2025, "2025-06-30", "2025-07-13"),
]


def _validate() -> None:
    """Assert every non-cancelled fortnight is a Monday->Sunday, 13-day span.

    This is the calendar's own integrity check: a fat-fingered date would flip a
    weekday or the span and fire here rather than silently poisoning the event study.
    """
    for year, s, e in WIMBLEDON:
        if s is None:
            continue
        ds, de = _dt.date.fromisoformat(s), _dt.date.fromisoformat(e)
        assert ds.weekday() == 0, f"{year}: start {s} is not a Monday"
        assert de.weekday() == 6, f"{year}: end {e} is not a Sunday"
        assert (de - ds).days == 13, f"{year}: span {s}..{e} is not 13 days"


_validate()


def wimbledon_table() -> pd.DataFrame:
    """The calendar as a frame: ``year``, ``start``, ``end`` (Timestamps; NaT if cancelled)."""
    rows = []
    for year, s, e in WIMBLEDON:
        rows.append({"year": year,
                     "start": pd.Timestamp(s) if s else pd.NaT,
                     "end": pd.Timestamp(e) if e else pd.NaT})
    return pd.DataFrame(rows)


def contested_years() -> list[int]:
    return [y for y, s, _ in WIMBLEDON if s is not None]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"wimbledon_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01") -> None:
    """Download adjusted (total-return) daily closes for EWU + VGK; cache them.

    ``auto_adjust=True`` -- these are equity ETFs, so total-return (dividends
    reinvested) is the honest series; the window returns below are then plain ratios
    of the cached close. Network; run once to build the cache.
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in TICKERS:
        d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d[["Close"]].dropna()
        d.to_csv(_cache_path(t))


def have_real() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in TICKERS)


def load_real(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached {ticker: adjusted-close Series}, each sliced to <= asof."""
    out = {}
    for t in TICKERS:
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        s = df["Close"]
        out[t] = s[s.index <= pd.Timestamp(asof)]
    return out


# --------------------------------------------------------------------------- #
# Synthetic world -- planted fortnight seasonal (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, seed: int = 731, n_years: int = 20,
                    win_len: int = 10, year_len: int = 252,
                    ) -> tuple[pd.Series, pd.Series, list[tuple[int, int]]]:
    """Deterministic paired (asset, benchmark) log-return world with a planted seasonal.

    Both series are correlated (rho ~ 0.85, like a UK ETF vs a Europe benchmark)
    zero-mean noise. Once per synthetic "year" a fixed ``win_len``-session window (the
    stand-in for the Wimbledon fortnight, anchored at the same offset every year) gets
    an EXTRA ``bump`` of total abnormal log-return spread evenly across its sessions,
    on the ASSET only (not the benchmark). ``bump = 0`` is the null world -- the
    fortnight window is statistically identical to every other window.

    Integer business-day index (positions 0..n_years*year_len), far below the
    250-year ns-timestamp trap -- no calendar dates are generated at all here.
    Returns (asset_logret, bench_logret, list of (win_start, win_end) positions).
    """
    rng = np.random.default_rng(seed)
    n = n_years * year_len
    rho = 0.85
    common = rng.normal(0.0, 0.009, n)
    idio_a = rng.normal(0.0, 0.009, n)
    idio_b = rng.normal(0.0, 0.009, n)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    offset = 120                       # the fortnight sits ~mid-year, same slot each year
    per_day = bump / max(win_len, 1)
    wins = []
    for y in range(n_years):
        w0 = y * year_len + offset
        w1 = w0 + win_len
        if w1 >= n:
            break
        a[w0:w1] += per_day            # the planted fortnight seasonal, asset only
        wins.append((w0, w1))

    idx = pd.RangeIndex(n)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), wins
