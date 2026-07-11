"""Data layer for Study 709 — World-Series-Effect.

Two ingredients, both offline-friendly once cached:

* **The hardcoded World Series champion table, 1950 -> 2025.** Winner, league (AL/NL) and
  host city for every World Series in the window (76 seasons; 1994 was cancelled by the
  players' strike and carries no champion — a named quirk, not a silent gap). Facts, no
  network. Source: MLB / Baseball-Reference official postseason results.

* **Real tape.** Daily ^GSPC (S&P 500 price index) OHLC from yfinance (no key), cached as
  CSV under the study's own ``_cache/``. We resample to December-close-to-December-close
  calendar-year returns and test them against the WS-champion's league (and, as the
  alternative "champion-city" variant the brief asks for, whether the champion plays in
  New York).

* **Synthetic world.** A deterministic, seeded annual-return generator with a TUNABLE
  planted "omen" effect (knob ``boost``, applied to whichever group the caller flags as
  bullish) — the faithful-engine / power positive control. ``boost = 0`` is the null world.

The claim under test, stated the way it circulates: *"the league of the World Series
champion (AL vs NL) — or, in its city-mythology cousin, whether a New York team wins —
predicts the direction of next year's stock market."* We mirror the football version's
NFC/AFC mnemonic and test **NL win -> bullish next year** (National League ~ "National
economy" folklore association, exactly as arbitrary as it sounds) and, separately, **a
New York champion -> bullish next year** (the "Wall Street's hometown team" story, since
NY franchises have won disproportionately often). Neither has ANY published mechanism —
that absence is itself part of the honest read.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
GSPC_CACHE = os.path.join(CACHE_DIR, "wse_gspc.csv")

START = "1949-11-01"        # need Dec-1949 close to compute the CY1950 return
AS_OF = "2026-06-30"        # last complete month at publication (2026-07-10)
FIRST_TARGET_YEAR = 1951    # first "next year" a WS champion (1950) can be scored against
LAST_TARGET_YEAR = 2025     # last COMPLETE calendar year on the tape

# --------------------------------------------------------------------------- #
# Hardcoded World Series champions, 1950 -> 2025 (76 seasons; 1994 cancelled).
# Columns: ws_year (season the Series was played/decided), champion, league
# (post-1901 AL/NL alignment; expansion teams keep their current league), city
# (the metro the franchise represented that year — relocations tracked explicitly),
# is_ny (True for a New York-borough franchise: Yankees/Giants/Mets/Brooklyn Dodgers —
# the "Wall Street's hometown team" city-omen variant named in the brief).
# Source: MLB official postseason history / Baseball-Reference
# (https://www.baseball-reference.com/postseason/world-series.shtml), cross-checked
# against Wikipedia's "List of World Series champions". The World Series is decided by
# early November of ws_year — the calendar is fully known before Jan 1 of ws_year + 1,
# so any "next year" position implied by the omen involves zero look-ahead.
# --------------------------------------------------------------------------- #
WORLD_SERIES_CHAMPIONS: list[dict] = [
    {"ws_year": 1950, "champion": "New York Yankees",       "league": "AL", "city": "New York",    "is_ny": True},
    {"ws_year": 1951, "champion": "New York Yankees",       "league": "AL", "city": "New York",    "is_ny": True},
    {"ws_year": 1952, "champion": "New York Yankees",       "league": "AL", "city": "New York",    "is_ny": True},
    {"ws_year": 1953, "champion": "New York Yankees",       "league": "AL", "city": "New York",    "is_ny": True},
    {"ws_year": 1954, "champion": "New York Giants",        "league": "NL", "city": "New York",    "is_ny": True},
    {"ws_year": 1955, "champion": "Brooklyn Dodgers",       "league": "NL", "city": "Brooklyn",    "is_ny": True},
    {"ws_year": 1956, "champion": "New York Yankees",       "league": "AL", "city": "New York",    "is_ny": True},
    {"ws_year": 1957, "champion": "Milwaukee Braves",       "league": "NL", "city": "Milwaukee",   "is_ny": False},
    {"ws_year": 1958, "champion": "New York Yankees",       "league": "AL", "city": "New York",    "is_ny": True},
    {"ws_year": 1959, "champion": "Los Angeles Dodgers",    "league": "NL", "city": "Los Angeles", "is_ny": False},
    {"ws_year": 1960, "champion": "Pittsburgh Pirates",     "league": "NL", "city": "Pittsburgh",  "is_ny": False},
    {"ws_year": 1961, "champion": "New York Yankees",       "league": "AL", "city": "New York",    "is_ny": True},
    {"ws_year": 1962, "champion": "New York Yankees",       "league": "AL", "city": "New York",    "is_ny": True},
    {"ws_year": 1963, "champion": "Los Angeles Dodgers",    "league": "NL", "city": "Los Angeles", "is_ny": False},
    {"ws_year": 1964, "champion": "St. Louis Cardinals",    "league": "NL", "city": "St. Louis",   "is_ny": False},
    {"ws_year": 1965, "champion": "Los Angeles Dodgers",    "league": "NL", "city": "Los Angeles", "is_ny": False},
    {"ws_year": 1966, "champion": "Baltimore Orioles",      "league": "AL", "city": "Baltimore",   "is_ny": False},
    {"ws_year": 1967, "champion": "St. Louis Cardinals",    "league": "NL", "city": "St. Louis",   "is_ny": False},
    {"ws_year": 1968, "champion": "Detroit Tigers",         "league": "AL", "city": "Detroit",     "is_ny": False},
    {"ws_year": 1969, "champion": "New York Mets",          "league": "NL", "city": "New York",    "is_ny": True},
    {"ws_year": 1970, "champion": "Baltimore Orioles",      "league": "AL", "city": "Baltimore",   "is_ny": False},
    {"ws_year": 1971, "champion": "Pittsburgh Pirates",     "league": "NL", "city": "Pittsburgh",  "is_ny": False},
    {"ws_year": 1972, "champion": "Oakland Athletics",      "league": "AL", "city": "Oakland",     "is_ny": False},
    {"ws_year": 1973, "champion": "Oakland Athletics",      "league": "AL", "city": "Oakland",     "is_ny": False},
    {"ws_year": 1974, "champion": "Oakland Athletics",      "league": "AL", "city": "Oakland",     "is_ny": False},
    {"ws_year": 1975, "champion": "Cincinnati Reds",        "league": "NL", "city": "Cincinnati",  "is_ny": False},
    {"ws_year": 1976, "champion": "Cincinnati Reds",        "league": "NL", "city": "Cincinnati",  "is_ny": False},
    {"ws_year": 1977, "champion": "New York Yankees",       "league": "AL", "city": "New York",    "is_ny": True},
    {"ws_year": 1978, "champion": "New York Yankees",       "league": "AL", "city": "New York",    "is_ny": True},
    {"ws_year": 1979, "champion": "Pittsburgh Pirates",     "league": "NL", "city": "Pittsburgh",  "is_ny": False},
    {"ws_year": 1980, "champion": "Philadelphia Phillies",  "league": "NL", "city": "Philadelphia", "is_ny": False},
    {"ws_year": 1981, "champion": "Los Angeles Dodgers",    "league": "NL", "city": "Los Angeles", "is_ny": False},
    {"ws_year": 1982, "champion": "St. Louis Cardinals",    "league": "NL", "city": "St. Louis",   "is_ny": False},
    {"ws_year": 1983, "champion": "Baltimore Orioles",      "league": "AL", "city": "Baltimore",   "is_ny": False},
    {"ws_year": 1984, "champion": "Detroit Tigers",         "league": "AL", "city": "Detroit",     "is_ny": False},
    {"ws_year": 1985, "champion": "Kansas City Royals",     "league": "AL", "city": "Kansas City", "is_ny": False},
    {"ws_year": 1986, "champion": "New York Mets",          "league": "NL", "city": "New York",    "is_ny": True},
    {"ws_year": 1987, "champion": "Minnesota Twins",        "league": "AL", "city": "Minneapolis", "is_ny": False},
    {"ws_year": 1988, "champion": "Los Angeles Dodgers",    "league": "NL", "city": "Los Angeles", "is_ny": False},
    {"ws_year": 1989, "champion": "Oakland Athletics",      "league": "AL", "city": "Oakland",     "is_ny": False},
    {"ws_year": 1990, "champion": "Cincinnati Reds",        "league": "NL", "city": "Cincinnati",  "is_ny": False},
    {"ws_year": 1991, "champion": "Minnesota Twins",        "league": "AL", "city": "Minneapolis", "is_ny": False},
    {"ws_year": 1992, "champion": "Toronto Blue Jays",      "league": "AL", "city": "Toronto",     "is_ny": False},
    {"ws_year": 1993, "champion": "Toronto Blue Jays",      "league": "AL", "city": "Toronto",     "is_ny": False},
    {"ws_year": 1994, "champion": None,                     "league": None, "city": None,          "is_ny": False},  # strike — no Series
    {"ws_year": 1995, "champion": "Atlanta Braves",         "league": "NL", "city": "Atlanta",     "is_ny": False},
    {"ws_year": 1996, "champion": "New York Yankees",       "league": "AL", "city": "New York",    "is_ny": True},
    {"ws_year": 1997, "champion": "Florida Marlins",        "league": "NL", "city": "Miami",       "is_ny": False},
    {"ws_year": 1998, "champion": "New York Yankees",       "league": "AL", "city": "New York",    "is_ny": True},
    {"ws_year": 1999, "champion": "New York Yankees",       "league": "AL", "city": "New York",    "is_ny": True},
    {"ws_year": 2000, "champion": "New York Yankees",       "league": "AL", "city": "New York",    "is_ny": True},
    {"ws_year": 2001, "champion": "Arizona Diamondbacks",   "league": "NL", "city": "Phoenix",     "is_ny": False},
    {"ws_year": 2002, "champion": "Anaheim Angels",         "league": "AL", "city": "Anaheim",     "is_ny": False},
    {"ws_year": 2003, "champion": "Florida Marlins",        "league": "NL", "city": "Miami",       "is_ny": False},
    {"ws_year": 2004, "champion": "Boston Red Sox",         "league": "AL", "city": "Boston",      "is_ny": False},
    {"ws_year": 2005, "champion": "Chicago White Sox",      "league": "AL", "city": "Chicago",     "is_ny": False},
    {"ws_year": 2006, "champion": "St. Louis Cardinals",    "league": "NL", "city": "St. Louis",   "is_ny": False},
    {"ws_year": 2007, "champion": "Boston Red Sox",         "league": "AL", "city": "Boston",      "is_ny": False},
    {"ws_year": 2008, "champion": "Philadelphia Phillies",  "league": "NL", "city": "Philadelphia", "is_ny": False},
    {"ws_year": 2009, "champion": "New York Yankees",       "league": "AL", "city": "New York",    "is_ny": True},
    {"ws_year": 2010, "champion": "San Francisco Giants",   "league": "NL", "city": "San Francisco", "is_ny": False},
    {"ws_year": 2011, "champion": "St. Louis Cardinals",    "league": "NL", "city": "St. Louis",   "is_ny": False},
    {"ws_year": 2012, "champion": "San Francisco Giants",   "league": "NL", "city": "San Francisco", "is_ny": False},
    {"ws_year": 2013, "champion": "Boston Red Sox",         "league": "AL", "city": "Boston",      "is_ny": False},
    {"ws_year": 2014, "champion": "San Francisco Giants",   "league": "NL", "city": "San Francisco", "is_ny": False},
    {"ws_year": 2015, "champion": "Kansas City Royals",     "league": "AL", "city": "Kansas City", "is_ny": False},
    {"ws_year": 2016, "champion": "Chicago Cubs",           "league": "NL", "city": "Chicago",     "is_ny": False},
    {"ws_year": 2017, "champion": "Houston Astros",         "league": "AL", "city": "Houston",     "is_ny": False},
    {"ws_year": 2018, "champion": "Boston Red Sox",         "league": "AL", "city": "Boston",      "is_ny": False},
    {"ws_year": 2019, "champion": "Washington Nationals",   "league": "NL", "city": "Washington",  "is_ny": False},
    {"ws_year": 2020, "champion": "Los Angeles Dodgers",    "league": "NL", "city": "Los Angeles", "is_ny": False},
    {"ws_year": 2021, "champion": "Atlanta Braves",         "league": "NL", "city": "Atlanta",     "is_ny": False},
    {"ws_year": 2022, "champion": "Houston Astros",         "league": "AL", "city": "Houston",     "is_ny": False},
    {"ws_year": 2023, "champion": "Texas Rangers",          "league": "AL", "city": "Arlington",   "is_ny": False},
    {"ws_year": 2024, "champion": "Los Angeles Dodgers",    "league": "NL", "city": "Los Angeles", "is_ny": False},
    {"ws_year": 2025, "champion": "Los Angeles Dodgers",    "league": "NL", "city": "Los Angeles", "is_ny": False},
]

WS_DF: pd.DataFrame = pd.DataFrame(WORLD_SERIES_CHAMPIONS)


def ws_table() -> pd.DataFrame:
    """The hardcoded champion table, seasons that were actually played (drops 1994)."""
    return WS_DF[WS_DF["champion"].notna()].reset_index(drop=True).copy()


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01") -> None:
    """Download ^GSPC daily OHLC; cache the Close column. Network; runs once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    raw = yf.download("^GSPC", start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw[["Close"]].dropna().to_csv(GSPC_CACHE)


def have_real() -> bool:
    return os.path.exists(GSPC_CACHE)


def load_real(asof: str = AS_OF) -> pd.DataFrame:
    """Cached daily ^GSPC close, sliced to [START, asof]."""
    df = pd.read_csv(GSPC_CACHE, index_col=0, parse_dates=True).sort_index()
    return df.loc[df.index <= asof].copy()


def annual_returns(gspc: pd.DataFrame, start_year: int = 1950,
                   end_year: int = LAST_TARGET_YEAR) -> pd.Series:
    """December-close-to-December-close calendar-year simple returns (%), price-only.

    ``^GSPC`` carries no dividends (it's a price index, not total return) — labeled
    price-only throughout. Only COMPLETE calendar years are ever returned: a year whose
    last cached trading day isn't in December is dropped (guards a partial current year).
    """
    close = gspc["Close"]
    yearly = close.resample("YE").last()
    complete = yearly.index.map(lambda ts: ts.month == 12)
    yearly = yearly[complete]
    yearly.index = yearly.index.year
    ret = yearly.pct_change().dropna() * 100.0
    return ret.loc[(ret.index >= start_year) & (ret.index <= end_year)]


# --------------------------------------------------------------------------- #
# Synthetic world — planted "omen" effect (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(boost: float = 0.0, seed: int = 709, n_years: int = 74,
                    base_return: float = 8.0, vol_ann: float = 17.0,
                    bull_fraction: float = 0.53) -> tuple[pd.DataFrame, dict]:
    """A reproducible next-year-return series with an optional planted omen effect.

    Each synthetic "season" draws a random bull/bear league flag (bull with probability
    ``bull_fraction`` — matching the real NL 35/74 ~ 47% share) and an i.i.d. annual
    return N(base_return, vol_ann**2) in percent. In bull-flagged seasons an extra
    ``boost`` percentage points is added to the *following* year's return.
    ``boost = 0`` is the null: the flag carries zero information, and the Welch/binomial/
    permutation machinery must not manufacture significance from it.

    Returns ``(df, truth)`` where ``df`` has columns ``ws_year``, ``league`` ('NL'/'AL'),
    ``next_year_return`` and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(1950, 1950 + n_years)
    is_bull = rng.random(n_years) < bull_fraction
    rets = rng.normal(base_return, vol_ann, n_years)
    rets = rets + np.where(is_bull, boost, 0.0)
    df = pd.DataFrame({
        "ws_year": years,
        "league": np.where(is_bull, "NL", "AL"),
        "next_year_return": rets,
    })
    truth = {"n_years": n_years, "boost": boost, "base_return": base_return,
             "vol_ann": vol_ann, "bull_fraction": bull_fraction, "seed": seed}
    return df, truth
