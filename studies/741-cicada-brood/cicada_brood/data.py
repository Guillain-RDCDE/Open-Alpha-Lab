"""Data layer for Study 741 — Cicada-Brood.

The (deliberately silly) claim under test: the springs when a **periodical cicada
brood emerges** — those famous 13- and 17-year *Magicicada* mass-emergences that carpet
the eastern US in bugs and headlines — are somehow special for the stock market. It is a
pure spurious-pattern demo, stated as such: a fixed-calendar "signal" whose only real
virtue is that (unlike a plane crash or an election) the emergence year has been on the
calendar, to the year, since the *previous* emergence 13 or 17 years earlier.

Three ingredients, all offline-friendly once cached:

* **The brood calendar, hardcoded.** ``BROODS`` is a curated table of the notable,
  mapped periodical-cicada brood emergences 1996 -> 2025 — every active 17-year brood
  (I-X, XIII, XIV) and the three active 13-year broods (XIX, XXII, XXIII), each row a
  ``(year, brood, cycle, region, famous)`` tuple. Source: the University of Connecticut
  / John Cooley *Magicicada* brood chart (magicicada.org, the canonical academic brood
  schedule), cross-checked against the US Forest Service "Periodical Cicada" page and
  contemporary national coverage of the marquee emergences (Brood X 2004 & 2021, the
  Northern-Illinois Brood XIII 2007, Brood II "Swarmageddon" 2013, and the rare
  2024 Brood XIII x XIX dual co-emergence). Distinct emergence YEARS are the study's
  event unit (``brood_years()``) — one S&P 500 spring per year, independent and
  non-overlapping. A punchline falls out of the table itself: a brood emerges *somewhere*
  in **24 of the 30** years 1996-2025, so "a cicada year" is very nearly the whole
  calendar — the base rate that makes the effect a mirage before we even price it.

* **Real tape.** Daily **SPY** (S&P 500 ETF) total-return adjusted closes from yfinance
  (no key), 1993-01-29 -> the as-of, cached as CSV under the study's own ``_cache/``.
  Total-return (``auto_adjust=True``) is the honest series for an equity ETF; labelled as
  such everywhere. SPY is a real, tradable instrument on yfinance, so **no proxy is
  used** — the tape is the tape.

* **Synthetic world.** A deterministic, seeded random-walk SPY-like tape with a TUNABLE
  planted extra-drift on the emergence-year spring windows (``bump`` in per-window
  return units). ``bump = 0`` is the null world — emergence springs statistically
  identical to every other spring; the event-study machinery must NOT manufacture
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
SPY_CACHE = os.path.join(CACHE_DIR, "cicada_spy.csv")

START = "1993-01-01"
AS_OF = "2026-06-30"        # last complete calendar month at publication (2026-07-13)

# The cicada spring window. Periodical cicadas surface when the soil ~20 cm down reaches
# ~18 C (~64 F) — early-to-mid May across the core mid-Atlantic/Midwest range — and are
# above ground, singing and swarming, for roughly six weeks (May through late June). So
# the event window is anchored to the first trading session on/after **May 1** of the
# emergence year and runs ``WINDOW_K`` sessions (~2 trading months) to cover the whole
# May-June emergence, with a shorter ``WINDOW_K_SHORT`` (~1 month) reported alongside.
ANCHOR_MONTH = 5
ANCHOR_DAY = 1
WINDOW_K = 42        # trading sessions ~ 2 months (May 1 -> ~end June)
WINDOW_K_SHORT = 21  # trading sessions ~ 1 month

# --------------------------------------------------------------------------- #
# Hardcoded periodical-cicada brood emergences, 1996 -> 2025.
# Each row: (year, brood, cycle_years, core_region, famous). ``famous`` flags the
# handful of emergences that were genuine national front-page news (the ones a folklore
# "cicada indicator" would actually be built on). ``cycle_years`` is 17 or 13.
# Source: UConn / Cooley Magicicada brood chart (magicicada.org), cross-checked against
# the US Forest Service periodical-cicada page and contemporary national reporting.
# --------------------------------------------------------------------------- #
BROODS: list[tuple[int, str, int, str, bool]] = [
    (1996, "II",    17, "US East Coast (CT-NC)",           False),
    (1997, "III",   17, "Iowa",                            False),
    (1998, "IV",    17, "Kansas / Great Plains",           False),
    (1998, "XIX",   13, "US Southeast (Great Southern)",   False),
    (1999, "V",     17, "Ohio / West Virginia",            False),
    (2000, "VI",    17, "Georgia / Carolinas",             False),
    (2001, "VII",   17, "Onondaga, upstate New York",      False),
    (2001, "XXII",  13, "Louisiana / Mississippi",         False),
    (2002, "VIII",  17, "Pennsylvania / Ohio / WV",        False),
    (2002, "XXIII", 13, "Mississippi River Valley",        False),
    (2003, "IX",    17, "Virginia / WV / NC",              False),
    (2004, "X",     17, "Great Eastern (Mid-Atlantic/MW)", True),   # marquee national event
    (2007, "XIII",  17, "Northern Illinois / Chicago",     True),   # marquee national event
    (2008, "XIV",   17, "Great Southeastern (KY-MA)",      False),
    (2011, "XIX",   13, "US Southeast (Great Southern)",   False),
    (2012, "I",     17, "Virginia / WV",                   False),
    (2013, "II",    17, "US East Coast 'Swarmageddon'",    True),   # marquee national event
    (2014, "III",   17, "Iowa",                            False),
    (2014, "XXII",  13, "Louisiana / Mississippi",         False),
    (2015, "IV",    17, "Kansas / Great Plains",           False),
    (2015, "XXIII", 13, "Mississippi River Valley",        False),
    (2016, "V",     17, "Ohio / Appalachia",               False),
    (2017, "VI",    17, "Georgia / South Carolina",        False),
    (2018, "VII",   17, "Onondaga, upstate New York",      False),
    (2019, "VIII",  17, "Western Pennsylvania",            False),
    (2020, "IX",    17, "Virginia / NC / WV",              False),
    (2021, "X",     17, "Great Eastern (Mid-Atlantic/MW)", True),   # marquee national event
    (2024, "XIII",  17, "Northern Illinois",               True),   # marquee 17x13 dual
    (2024, "XIX",   13, "US Midwest / Southeast",          True),   # marquee 17x13 dual
    (2025, "XIV",   17, "Great Southeastern (KY-MA)",      False),
]


def brood_table() -> pd.DataFrame:
    """The curated brood table as a frame: year, brood, cycle, region, famous."""
    df = pd.DataFrame(BROODS, columns=["year", "brood", "cycle", "region", "famous"])
    return df.sort_values(["year", "brood"]).reset_index(drop=True)


def brood_years(cycle: int | None = None, famous_only: bool = False) -> list[int]:
    """Distinct emergence YEARS (the event unit) — one SPY spring per year.

    ``cycle=17`` restricts to the iconic 17-year broods, ``cycle=13`` to the 13-year
    broods, ``None`` (default) takes any notable brood. ``famous_only`` keeps only the
    marquee, nationally-covered emergences (the ones a real "cicada indicator" would be
    built on). Years are de-duplicated: a year with two broods (e.g. 2024) is one event.
    """
    df = brood_table()
    if cycle is not None:
        df = df[df["cycle"] == cycle]
    if famous_only:
        df = df[df["famous"]]
    return sorted(df["year"].unique().tolist())


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01") -> None:
    """Download SPY total-return daily closes; cache as CSV. Network; run once.

    ``auto_adjust=True`` folds in splits and dividends (total-return, not price-only),
    so the event-study returns below are plain ``pct_change()`` on the cached close.
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    d = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d[["Close"]].dropna().to_csv(SPY_CACHE)


def have_real() -> bool:
    return os.path.exists(SPY_CACHE)


def load_real(asof: str = AS_OF) -> pd.Series:
    """Cached SPY total-return close Series, sliced to [START, asof]."""
    spy = pd.read_csv(SPY_CACHE, index_col=0, parse_dates=True).sort_index()["Close"]
    return spy.loc[(spy.index >= START) & (spy.index <= asof)]


def all_years(close: pd.Series) -> list[int]:
    """Every calendar year with a usable spring window on the tape (for the baseline /
    random-year placebo pool). A year qualifies if its May-1 anchor and the full
    ``WINDOW_K`` forward window sit inside the tape."""
    years = sorted({int(y) for y in close.index.year.unique()})
    ok = []
    for y in years:
        anchor = _anchor_pos(close, y)
        if anchor is not None and anchor - 1 >= 0 and anchor + WINDOW_K < len(close):
            ok.append(y)
    return ok


def _anchor_pos(close: pd.Series, year: int) -> int | None:
    """Index position of the first trading session on/after May 1 of ``year`` (or None
    if that date is off the tape)."""
    anchor_date = pd.Timestamp(year=year, month=ANCHOR_MONTH, day=ANCHOR_DAY)
    pos = close.index.searchsorted(anchor_date)
    if pos >= len(close):
        return None
    return int(pos)


# --------------------------------------------------------------------------- #
# Synthetic world — planted emergence-spring bump (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, seed: int = 741,
                    n_years: int = 46, daily_vol: float = 0.008,
                    start: str = "1980-01-02",
                    ) -> tuple[pd.Series, list[int]]:
    """A reproducible daily "SPY-like" random-walk tape with a TUNABLE planted spring bump.

    A random walk in log returns (i.i.d. normal, std ``daily_vol``). On the spring window
    (the first session on/after May 1, held ``WINDOW_K`` sessions) of each *emergence*
    year the tape earns an extra ``bump`` total return, spread evenly across the window.
    ``bump = 0`` is the null world: emergence springs are statistically identical to
    every other spring, and the event-study detector must NOT reach significance.

    Real bdate index (~30 years, far below the 250-year pandas ns-timestamp trap).
    Returns (close Series, emergence-year list). The synthetic emergence years are the
    first, third, fifth ... of the span (a fixed alternating schedule) so the control is
    fully deterministic and independent of the real brood calendar.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_years * 261)  # ~261 bdays/yr
    log_ret = rng.normal(0.0, daily_vol, len(idx))
    close = pd.Series(100.0 * np.exp(np.cumsum(log_ret)), index=idx)

    years = sorted({int(y) for y in idx.year.unique()})
    emergence = years[0::2]  # every other year is a synthetic "emergence" year
    if bump != 0.0:
        per_day = bump / WINDOW_K
        for y in emergence:
            anchor = _anchor_pos(close, y)
            if anchor is None or anchor + WINDOW_K >= len(close):
                continue
            for k in range(WINDOW_K):
                log_ret[anchor + k] += per_day
        close = pd.Series(100.0 * np.exp(np.cumsum(log_ret)), index=idx)
    return close, emergence
