"""Data layer for Study 734 — NBA-Finals-Effect.

The claim under test: the **Edmans-Garcia-Norli (2007) sports-sentiment effect**, applied
to the NBA Finals. EGN found a robust *next-day* decline in a national stock market after
that country is **eliminated** from the soccer World Cup — a real mood-to-market channel,
identified off a *loss* shock. The folklore corollary here: when a team **loses** the NBA
Finals, its home city's local market should dip (the deflated-fanbase channel), and when a
team **wins**, its city should get a feel-good pop.

The catch is structural and is the whole story: EGN's mechanism needs the two competitors
to belong to **different stock markets** (different countries). The NBA Finals is almost
always USA-vs-USA — one shared tape (`SPY`) — so any net national-mood shock cancels (one
US city elated, one deflated, same market). The single 2000->2025 exception is **2019**
(Toronto Raptors, Canada, beat Golden State, USA), the one Finals that spans two national
markets. That collapse of the cross-country design is named, not papered over.

Four ingredients:

* **The calendar, hardcoded.** Every NBA Finals 2000->2025 (26 straight, none cancelled —
  2020 and 2021 were only COVID-*delayed*, into Oct-2020 and Jul-2021), its champion team,
  its runner-up team, and the exact date of the series-clinching game. Source: NBA.com /
  Basketball-Reference official Finals results, cross-checked against Wikipedia per year.
* **A team -> home-metro "civic proxy" map** (``TEAM_PROXY``). No US city has a stock
  index, so each metro is mapped to a single, **real, tradable, clearly-LABELLED-as-coarse**
  hometown large-cap headquartered in that metro (a regional bank / iconic local employer /
  local utility). This is a genuine instrument you could trade — the caveat, named loudly,
  is only that a single company's return is a *noisy* stand-in for civic mood, dominated by
  its own business. Toronto maps to the `EWC` Canada ETF (its market really is a separate
  national one). That noisiness is itself part of why the effect is undetectable.
* **A benchmark**, `SPY` (S&P 500 total-return) — the shared US tape every US metro trades
  inside. Abnormal return = metro proxy minus `SPY`, both total-return.
* **Synthetic world.** A deterministic, seeded paired (proxy, benchmark) log-return series
  with a TUNABLE planted "loser dip" on a synthetic event calendar. ``bump = 0`` is the
  null world; the one-sample-t machinery must not manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"     # last complete calendar month at publication (2026-07-13)
BENCHMARK = "SPY"        # S&P 500 total-return -- the shared US tape

# --------------------------------------------------------------------------- #
# The NBA Finals calendar, hardcoded: year, series-clinching game date, champion
# team, runner-up team. All 26 seasons 2000->2025 were contested and decided (2020
# was played in the Orlando "bubble" and clinched in OCTOBER; 2021 was COVID-delayed
# into JULY -- both named quirks, neither a cancellation). The clinching game tips
# ~9pm local and ends ~11:30pm, i.e. AFTER the market close on the game date, so that
# day's close does not yet know the result -- this gives the study its free, unavoidable
# execution lag (see strategy.py). Source: Basketball-Reference "NBA Finals" index
# (https://www.basketball-reference.com/playoffs/) and NBA.com, cross-checked against
# each season's Wikipedia "20xx NBA Finals" page for the exact clinching-game date.
# --------------------------------------------------------------------------- #
EVENTS = [
    # year, clinch_date,  champion,                 runner_up
    (2000, "2000-06-19", "Los Angeles Lakers",     "Indiana Pacers"),
    (2001, "2001-06-15", "Los Angeles Lakers",     "Philadelphia 76ers"),
    (2002, "2002-06-12", "Los Angeles Lakers",     "New Jersey Nets"),
    (2003, "2003-06-15", "San Antonio Spurs",      "New Jersey Nets"),
    (2004, "2004-06-15", "Detroit Pistons",        "Los Angeles Lakers"),
    (2005, "2005-06-23", "San Antonio Spurs",      "Detroit Pistons"),
    (2006, "2006-06-20", "Miami Heat",             "Dallas Mavericks"),
    (2007, "2007-06-14", "San Antonio Spurs",      "Cleveland Cavaliers"),
    (2008, "2008-06-17", "Boston Celtics",         "Los Angeles Lakers"),
    (2009, "2009-06-14", "Los Angeles Lakers",     "Orlando Magic"),
    (2010, "2010-06-17", "Los Angeles Lakers",     "Boston Celtics"),
    (2011, "2011-06-12", "Dallas Mavericks",       "Miami Heat"),
    (2012, "2012-06-21", "Miami Heat",             "Oklahoma City Thunder"),
    (2013, "2013-06-20", "Miami Heat",             "San Antonio Spurs"),
    (2014, "2014-06-15", "San Antonio Spurs",      "Miami Heat"),
    (2015, "2015-06-16", "Golden State Warriors",  "Cleveland Cavaliers"),
    (2016, "2016-06-19", "Cleveland Cavaliers",    "Golden State Warriors"),
    (2017, "2017-06-12", "Golden State Warriors",  "Cleveland Cavaliers"),
    (2018, "2018-06-08", "Golden State Warriors",  "Cleveland Cavaliers"),
    (2019, "2019-06-13", "Toronto Raptors",        "Golden State Warriors"),  # cross-border
    (2020, "2020-10-11", "Los Angeles Lakers",     "Miami Heat"),             # Orlando bubble, Oct
    (2021, "2021-07-20", "Milwaukee Bucks",        "Phoenix Suns"),           # COVID-delayed, Jul
    (2022, "2022-06-16", "Golden State Warriors",  "Boston Celtics"),
    (2023, "2023-06-12", "Denver Nuggets",         "Miami Heat"),
    (2024, "2024-06-17", "Boston Celtics",         "Dallas Mavericks"),
    (2025, "2025-06-22", "Oklahoma City Thunder",  "Indiana Pacers"),
]

# --------------------------------------------------------------------------- #
# Team -> home-metro "civic proxy" ticker. A LABELLED, COARSE proxy: a single real,
# tradable large-cap HQ'd in (or iconically tied to) that metro. NOT a city index (none
# exists). Chosen for a long pre-2000 (or early) price history and a genuine local footprint
# -- a regional bank, a dominant local employer, or the local utility. The caveat travels
# with every result: a single stock's return is a noisy proxy for civic mood, dominated by
# company-specific business, which is exactly why a "home-market sentiment" signal is hard
# to see even if it exists. Toronto -> EWC (its market genuinely is a separate national one).
# Inception dates are NOT hardcoded; a proxy whose history postdates its event falls out of
# the cached data itself (funnel in strategy.build_event_table).
# --------------------------------------------------------------------------- #
TEAM_PROXY: dict[str, str] = {
    "Los Angeles Lakers":    "DIS",   # Walt Disney -- Burbank/LA metro
    "Indiana Pacers":        "LLY",   # Eli Lilly -- Indianapolis HQ
    "Philadelphia 76ers":    "CMCSA", # Comcast -- Philadelphia HQ
    "New Jersey Nets":       "PEG",   # Public Service Enterprise Group -- Newark NJ utility
    "San Antonio Spurs":     "CFR",   # Cullen/Frost Bankers -- San Antonio regional bank
    "Detroit Pistons":       "F",     # Ford -- Dearborn/Detroit
    "Miami Heat":            "CCL",   # Carnival -- Miami HQ (and Heat owner Micky Arison's firm)
    "Dallas Mavericks":      "T",     # AT&T -- Dallas HQ
    "Cleveland Cavaliers":   "KEY",   # KeyCorp -- Cleveland regional bank
    "Boston Celtics":        "STT",   # State Street -- Boston HQ
    "Orlando Magic":         "DRI",   # Darden Restaurants -- Orlando HQ
    "Oklahoma City Thunder": "DVN",   # Devon Energy -- Oklahoma City HQ
    "Golden State Warriors": "WFC",   # Wells Fargo -- San Francisco HQ
    "Toronto Raptors":       "EWC",   # iShares MSCI Canada -- the one separate national market
    "Milwaukee Bucks":       "ROK",   # Rockwell Automation -- Milwaukee HQ
    "Phoenix Suns":          "RSG",   # Republic Services -- Phoenix HQ
    "Denver Nuggets":        "DVA",   # DaVita -- Denver HQ
}


def all_tickers() -> list[str]:
    """Every distinct ticker this study ever needs: the proxy map plus the benchmark."""
    ts = sorted(set(TEAM_PROXY.values()))
    return ts + [BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"nba_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "1998-01-01", end: str = "2026-07-01") -> None:
    """Download adjusted (total-return) daily closes for every ticker; cache them.

    ``auto_adjust=True`` -- these are equities/ETFs, so total-return (dividends
    reinvested) is the honest comparison for a metro-proxy-minus-SPY abnormal return.
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
# Synthetic world -- planted "loser dip" (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, seed: int = 734, n_events: int = 26,
                    n_days: int = 6000, spacing: int = 220,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (proxy, benchmark) log-return world with a planted bump.

    Both series are correlated (rho ~ 0.55, like a single large-cap vs the S&P 500)
    zero-mean noise; on a synthetic "event day" (every ``spacing``-th business day) the
    proxy gets an EXTRA ``bump`` log-return (in addition to its normal correlated draw)
    while the benchmark does not. ``bump = 0`` is the null world -- event days
    statistically identical to the rest. A NEGATIVE bump plants the EGN "loser dip".

    Business-day integer index (positions 0..n_days), far below the 250-year
    ns-timestamp trap (no calendar dates are generated here). Returns
    (proxy_logret, bench_logret, event_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.55
    common = rng.normal(0.0, 0.010, n_days)
    idio_a = rng.normal(0.0, 0.013, n_days)   # single stock -> more idiosyncratic vol
    idio_b = rng.normal(0.0, 0.008, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    event_pos = list(range(spacing, n_days - 30, spacing))[:n_events]
    for p in event_pos:
        a[p + 1] += bump   # the shock lands the day AFTER the (non-trading-night) game
    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), event_pos
