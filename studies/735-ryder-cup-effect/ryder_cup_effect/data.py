"""Data layer for Study 735 — Ryder-Cup-Effect.

The claim under test: the biennial Ryder Cup — Team **USA** against Team **Europe** —
is a cross-Atlantic sports-sentiment shock, so the *losing continent's* stock market
underperforms the winner's the Monday after the Sunday result. It is a two-sided,
cross-market cousin of the Edmans-García-Norli elimination effect (real, one-directional,
for World Cup *losses*): here the loss shock is planted on whichever side of the Atlantic
lost, and the natural instrument is the **loser-minus-winner** return of one continent's
broad ETF against the other's.

Three ingredients, all offline-friendly once cached:

* **The Ryder Cup calendar, hardcoded.** Every edition of the modern USA-vs-Europe era
  (1979→2025), each with the concluding date, the winning side and the losing side.
  1989 was a **14-14 tie** (Europe retained the Cup) — there is no losing side, so it is
  a named, excluded quirk. 2001 was postponed to 2002 (9/11) and 2020 to 2021 (COVID);
  the calendar carries the year the match was actually *played*. Source: Wikipedia "Ryder
  Cup" and the per-edition pages, cross-checked for the concluding Sunday (2010 finished
  on the **Monday**, rain-delayed at Celtic Manor — handled by the engine, not fudged).
* **A side -> continent-ETF map.** ``SIDE_TICKER`` maps ``"USA" -> "SPY"`` (S&P 500 ETF)
  and ``"Europe" -> "VGK"`` (Vanguard FTSE Europe — spans euro AND non-euro Europe: the
  UK, Switzerland, Sweden, Norway, Denmark all sit in the European team's countries and
  are NOT in the euro-only FEZ, which is why VGK, not FEZ, is the European leg). VGK's
  inception (2005-03-10) is a hard floor: it is the reason only the **2006->2025**
  editions are testable, and every earlier edition is excluded *by the data itself*, not
  by a judgment call (the funnel is shown, not hidden).
* **Synthetic world.** A deterministic, seeded pair of (USA, Europe) log-return series
  with a TUNABLE planted "loser slump" on a synthetic Ryder-Cup calendar (the loser side
  takes an extra negative log-return on the event day). ``slump = 0`` is the null world;
  the paired one-sample-t machinery must not manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"          # last complete month at publication (2026-07-13)
USA_TICKER = "SPY"            # SPDR S&P 500 ETF  -> Team USA's market
EUROPE_TICKER = "VGK"         # Vanguard FTSE Europe -> Team Europe's market
SIDE_TICKER = {"USA": USA_TICKER, "Europe": EUROPE_TICKER}

# --------------------------------------------------------------------------- #
# The Ryder Cup, hardcoded: year (as PLAYED), concluding date, winning side,
# losing side. Modern USA-vs-Europe era only (Continental Europe joined Great
# Britain & Ireland in 1979). A 14-14 tie has NO losing side (loser = None) and
# the holder retains the Cup -- 1989 is the one such edition here.
#   * 2001 was postponed to 2002 after the September 11 attacks; the row is dated
#     2002 (the year it was actually contested).
#   * 2020 was postponed to 2021 (COVID-19); the row is dated 2021.
#   * 2010 (Celtic Manor, Wales) was rain-delayed and CONCLUDED ON THE MONDAY,
#     2010-10-04 -- the concluding date is stored as-is and the engine snaps the
#     first post-result close correctly (no special-casing needed).
# Concluding dates are the final Sunday (or the 2010 Monday), cross-checked against
# the per-edition Wikipedia pages.
# --------------------------------------------------------------------------- #
EVENTS: list[tuple[int, str, str, str | None]] = [
    # year, concluding_date, winner, loser
    (1979, "1979-09-16", "USA",    "Europe"),
    (1981, "1981-09-20", "USA",    "Europe"),
    (1983, "1983-10-16", "USA",    "Europe"),
    (1985, "1985-09-15", "Europe", "USA"),
    (1987, "1987-09-27", "Europe", "USA"),
    (1989, "1989-09-24", "Europe", None),      # 14-14 tie, Europe retained -> no loser
    (1991, "1991-09-29", "USA",    "Europe"),
    (1993, "1993-09-26", "USA",    "Europe"),
    (1995, "1995-09-24", "Europe", "USA"),
    (1997, "1997-09-28", "Europe", "USA"),
    (1999, "1999-09-26", "USA",    "Europe"),
    (2002, "2002-09-29", "Europe", "USA"),     # 2001 postponed (9/11)
    (2004, "2004-09-19", "Europe", "USA"),
    (2006, "2006-09-24", "Europe", "USA"),     # <- VGK coverage begins here
    (2008, "2008-09-21", "USA",    "Europe"),
    (2010, "2010-10-04", "Europe", "USA"),     # rain-delayed, concluded Monday
    (2012, "2012-09-30", "Europe", "USA"),     # "Miracle at Medinah"
    (2014, "2014-09-28", "Europe", "USA"),
    (2016, "2016-10-02", "USA",    "Europe"),
    (2018, "2018-09-30", "Europe", "USA"),
    (2021, "2021-09-26", "USA",    "Europe"),  # 2020 postponed (COVID)
    (2023, "2023-10-01", "Europe", "USA"),
    (2025, "2025-09-28", "Europe", "USA"),     # Bethpage Black, Europe won away
]


def all_tickers() -> list[str]:
    """The two continent ETFs this study needs."""
    return [USA_TICKER, EUROPE_TICKER]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"ryder_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "1993-01-01", end: str = "2026-07-01") -> None:
    """Download adjusted (total-return) daily closes for SPY and VGK; cache them.

    ``auto_adjust=True`` -- both are equity ETFs, so total-return (dividends
    reinvested) is the honest comparison; the closes already fold in splits and
    dividends and the returns below are a plain ratio on the cached close.
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
# Synthetic world -- planted "loser slump" (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(slump: float = 0.0, seed: int = 735, n_events: int = 10,
                    n_days: int = 5300, spacing: int = 500,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (USA, Europe) log-return world with a planted loser slump.

    Both series are correlated (rho ~ 0.7, like SPY vs a European ETF) zero-mean
    noise. On each synthetic "Ryder-Cup day" (every ``spacing``-th business day) a
    coin flip picks which side lost, and THAT side takes an extra ``-slump``
    log-return while the winner does not. ``slump = 0`` is the null world -- event
    days statistically identical to the rest, and the paired loser-minus-winner
    detector must NOT reach significance.

    Business-day integer index (positions 0..n_days), far below the 250-year
    ns-timestamp trap. Returns (usa_logret, europe_logret, event_positions, losers)
    so the detector knows which side lost each synthetic event.
    """
    rng = np.random.default_rng(seed)
    rho = 0.70
    common = rng.normal(0.0, 0.010, n_days)
    idio_u = rng.normal(0.0, 0.010, n_days)
    idio_e = rng.normal(0.0, 0.010, n_days)
    u = rho * common + np.sqrt(1 - rho**2) * idio_u
    e = rho * common + np.sqrt(1 - rho**2) * idio_e

    event_pos = list(range(spacing, n_days - 40, spacing))[:n_events]
    losers = []
    for p in event_pos:
        # deterministic "who lost": alternate USA/Europe so both legs get slumped
        if (p // spacing) % 2 == 0:
            u[p] -= slump          # USA lost
            losers.append("USA")
        else:
            e[p] -= slump          # Europe lost
            losers.append("Europe")
    idx = pd.RangeIndex(n_days)
    return pd.Series(u, index=idx), pd.Series(e, index=idx), event_pos, losers
