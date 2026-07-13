"""Data layer for Study 730 — Ferrari-F1.

The claim under test: **Ferrari's stock (NYSE: RACE) gets a "brand-halo" / fan-sentiment
pop on the Monday after Scuderia Ferrari wins a Formula 1 Grand Prix** — the Ferrari-
specific cousin of the Edmans-Garcia-Norli sports-sentiment effect, aimed at the one
listed company whose brand *is* a race team.

Three ingredients:

* **The calendar, hardcoded.** Every Ferrari Formula 1 Grand Prix **victory** from the
  October 2015 NYSE IPO through the frozen as-of date — season, exact race date (always
  a Sunday), Grand Prix, and winning driver. Ferrari were **winless in 2016, 2020, 2021
  and 2025**, so the win calendar spans 2017->2024 (24 victories). Source: STATS F1
  "Ferrari — Wins" and Wikipedia "Ferrari Grand Prix results", cross-checked per race
  against the official Formula1.com season results for the exact date. (Lewis Hamilton's
  2025 China *Sprint* win is deliberately NOT a Grand Prix win and is excluded.)
* **The instrument.** ``RACE`` — Ferrari N.V., listed on the NYSE on 2015-10-21 at
  USD 52 (the full FCA separation completed 2016-01-03; Milan listing 2016-01-04). It is
  a USD, US-listed line, so a US market benchmark is the fair counterfactual.
* **A market benchmark**, ``SPY`` (SPDR S&P 500) — the abnormal return is RACE minus SPY,
  both total-return (dividends reinvested). RACE is priced as a global *luxury* stock on
  margins and unit economics, not as a race team; SPY strips out the market-wide part of
  any given Monday move so what is left is Ferrari-specific. (An auto/luxury-peer
  benchmark is a natural sequel — see docs/references.md.)
* **Synthetic world.** A deterministic, seeded pair of (asset, benchmark) log-return
  series with a TUNABLE planted "fan-halo bump" on a synthetic win calendar. ``bump = 0``
  is the null world; the one-sample-t machinery must not manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"        # last complete month at publication (2026-07-13)
TICKER = "RACE"            # Ferrari N.V., NYSE, USD
BENCHMARK = "SPY"          # SPDR S&P 500 -- the market counterfactual

# --------------------------------------------------------------------------- #
# The Ferrari F1 victory calendar, hardcoded: season, race date (Sunday), Grand
# Prix, winning driver, and an "era" tag. Ferrari were winless in 2016 (the first
# full post-IPO season), 2020, 2021 and 2025, so the win list runs 2017->2024.
#
# era tag:
#   "contender" = 2017-2018, when a Ferrari win plausibly signalled a live title
#                 campaign (Vettel led both championships deep into the year) -- i.e.
#                 a win with genuine competitive/fundamental content.
#   "sporadic"  = 2019, 2022-2024, opportunistic one-off wins in seasons Ferrari
#                 finished well behind the champion -- pure "we won a race" sentiment.
# The split is used only for a within-sample Welch contrast (beat 4), never as a
# separate verdict axis.
#
# Source: STATS F1 "Ferrari - Wins" (statsf1.com/en/ferrari/victoire.aspx) and
# Wikipedia "Ferrari Grand Prix results", each date cross-checked against the
# official Formula1.com season results page for that year.
# --------------------------------------------------------------------------- #
EVENTS = [
    # season, race_date (Sunday), grand_prix,            driver,        era
    (2017, "2017-03-26", "Australian GP",   "Vettel",     "contender"),
    (2017, "2017-04-16", "Bahrain GP",      "Vettel",     "contender"),
    (2017, "2017-05-28", "Monaco GP",       "Vettel",     "contender"),
    (2017, "2017-07-30", "Hungarian GP",    "Vettel",     "contender"),
    (2017, "2017-11-12", "Brazilian GP",    "Vettel",     "contender"),
    (2018, "2018-03-25", "Australian GP",   "Vettel",     "contender"),
    (2018, "2018-04-08", "Bahrain GP",      "Vettel",     "contender"),
    (2018, "2018-06-10", "Canadian GP",     "Vettel",     "contender"),
    (2018, "2018-07-08", "British GP",      "Vettel",     "contender"),
    (2018, "2018-08-26", "Belgian GP",      "Vettel",     "contender"),
    (2018, "2018-10-21", "United States GP","Raikkonen",  "contender"),
    (2019, "2019-09-01", "Belgian GP",      "Leclerc",    "sporadic"),
    (2019, "2019-09-08", "Italian GP",      "Leclerc",    "sporadic"),
    (2019, "2019-09-22", "Singapore GP",    "Vettel",     "sporadic"),
    (2022, "2022-03-20", "Bahrain GP",      "Leclerc",    "sporadic"),
    (2022, "2022-04-10", "Australian GP",   "Leclerc",    "sporadic"),
    (2022, "2022-07-03", "British GP",      "Sainz",      "sporadic"),
    (2022, "2022-07-10", "Austrian GP",     "Leclerc",    "sporadic"),
    (2023, "2023-09-17", "Singapore GP",    "Sainz",      "sporadic"),
    (2024, "2024-03-24", "Australian GP",   "Sainz",      "sporadic"),
    (2024, "2024-05-26", "Monaco GP",       "Leclerc",    "sporadic"),
    (2024, "2024-09-01", "Italian GP",      "Leclerc",    "sporadic"),
    (2024, "2024-10-20", "United States GP","Leclerc",    "sporadic"),
    (2024, "2024-10-27", "Mexico City GP",  "Sainz",      "sporadic"),
]

# Winless Ferrari seasons in the RACE-listed era (documented, not a data gap).
WINLESS_SEASONS = [2016, 2020, 2021, 2025]

# Back-to-back win pairs whose 1-week (k=5) windows OVERLAP, breaking the
# independence the one-sample t assumes at the weekly horizon. The DAY(0) reaction
# (the headline) is one session and never overlaps; these matter only for ar_week.
# Each tuple is (later_race_date,) -- the second race of a pair one week after the first.
WEEKLY_OVERLAP_DROP = {"2019-09-08", "2022-07-10", "2024-10-27"}


def all_tickers() -> list[str]:
    return [TICKER, BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"ferrari_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2015-10-01", end: str = "2026-07-01") -> None:
    """Download adjusted (total-return) daily closes for RACE and SPY; cache them.

    ``auto_adjust=True`` -- both are dividend-paying equity lines, so total-return is
    the honest comparison (RACE has paid a growing dividend since 2016).
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
# Synthetic world -- planted fan-halo bump (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, seed: int = 730, n_events: int = 24,
                    n_days: int = 2600, spacing: int = 90,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (asset, benchmark) log-return world with a planted bump.

    Both series share a common market factor (rho ~ 0.55, like RACE vs SPY) plus
    idiosyncratic noise; on a synthetic "win day" (every ``spacing``-th business day)
    the asset gets an EXTRA ``bump`` log-return while the benchmark does not.
    ``bump = 0`` is the null world -- win days statistically identical to the rest.

    Business-day integer index (positions 0..n_days), far below the ns-timestamp trap
    (no calendar dates are generated). Returns (asset_logret, bench_logret, win_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.55
    common = rng.normal(0.0, 0.010, n_days)
    idio_a = rng.normal(0.0, 0.014, n_days)   # RACE single-name vol > index vol
    idio_b = rng.normal(0.0, 0.006, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    win_pos = list(range(spacing, n_days - 30, spacing))[:n_events]
    for p in win_pos:
        a[p] += bump

    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), win_pos
