"""Data layer for Study 733 — Kentucky-Derby-Effect.

The claim under test: the Kentucky Derby — the "Run for the Roses", run on the **first
Saturday in May** — is a market event. Two flavours of the folklore:

* **A market seasonal.** The first Saturday in May sits right on top of the "Sell in May
  and go away" calendar boundary; Derby-week almanac lore says the broad market does
  something special (up into the party, or the start of the summer swoon).
* **A gambling name.** ``CHDN`` (Churchill Downs Incorporated) literally *owns and
  operates* the Derby — it is the one US-listed stock with a direct, marquee exposure to
  the race. If any single equity should "pop" around the first Saturday in May, it's the
  company that runs the track. (The obvious honesty caveat, stated on the Signal axis:
  CHDN in 2000 was a near-pure racetrack operator; CHDN today is a diversified gaming
  company — regional casinos, the TwinSpires online-wagering platform, historical-racing
  machines — for which the Derby is a shrinking slice of revenue. The exposure the
  folklore assumes has *diluted* over exactly the window we test.)

Three ingredients:

* **The calendar, hardcoded.** Every Kentucky Derby 2000->2025. The Derby is run on the
  first Saturday in May every year, with **one named quirk**: the 2020 running was moved
  to **Saturday 5 September 2020** because of COVID-19 (it was not cancelled, only
  postponed). So the "first Saturday in May" *market seasonal* leg is a 25-event sample
  (2020 dropped — it did not fall in May), while the *CHDN gambling-name* leg is the full
  26-event sample (the marquee event still happened, just in September; that row is
  flagged ``ran_in_may=False``). Source: Wikipedia "Kentucky Derby" / "List of Kentucky
  Derby winners", cross-checked per-year for the exact Saturday date.

* **Real tape.** Daily ``SPY`` (S&P 500 ETF, the market-seasonal proxy) and ``CHDN``
  (Churchill Downs Inc.), total-return adjusted closes, yfinance (no key), cached as CSV
  under the study's own ``_cache/``. CHDN has traded continuously since the 1990s, so —
  unlike the Eurovision study's survivorship funnel — every event has full coverage; the
  limitation here is *exposure dilution*, not missing tape.

* **Synthetic world.** A deterministic, seeded paired (CHDN-like, SPY-like) log-return
  world with a TUNABLE planted "Derby bump" on a synthetic first-Saturday calendar.
  ``bump = 0`` is the null world; the one-sample-t / placebo machinery must not
  manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"        # last complete month at publication (2026-07-12)
START = "1998-01-01"        # a couple of years of CHDN/SPY lead-in before the 2000 events
MARKET = "SPY"              # the market-seasonal proxy (S&P 500 ETF)
GAMBLING = "CHDN"           # Churchill Downs Inc. -- the stock that runs the Derby

# --------------------------------------------------------------------------- #
# The Kentucky Derby calendar, hardcoded: year, running date (a Saturday), and whether
# it fell on the customary first Saturday in May. 2020 is the single quirk -- postponed
# from 2 May to 5 September 2020 by COVID-19 (run, not cancelled).
# Source: Wikipedia "Kentucky Derby" and "List of Kentucky Derby winners", each year's
# date cross-checked (the Derby is always the first Saturday in May bar 2020).
# --------------------------------------------------------------------------- #
EVENTS: list[tuple[int, str, bool]] = [
    # year, derby_date (Saturday),  ran_in_may (first Saturday in May?)
    (2000, "2000-05-06", True),
    (2001, "2001-05-05", True),
    (2002, "2002-05-04", True),
    (2003, "2003-05-03", True),
    (2004, "2004-05-01", True),
    (2005, "2005-05-07", True),
    (2006, "2006-05-06", True),
    (2007, "2007-05-05", True),
    (2008, "2008-05-03", True),
    (2009, "2009-05-02", True),
    (2010, "2010-05-01", True),
    (2011, "2011-05-07", True),
    (2012, "2012-05-05", True),
    (2013, "2013-05-04", True),
    (2014, "2014-05-03", True),
    (2015, "2015-05-02", True),
    (2016, "2016-05-07", True),
    (2017, "2017-05-06", True),
    (2018, "2018-05-05", True),
    (2019, "2019-05-04", True),
    (2020, "2020-09-05", False),   # COVID-19: postponed from 2 May to 5 September
    (2021, "2021-05-01", True),
    (2022, "2022-05-07", True),
    (2023, "2023-05-06", True),
    (2024, "2024-05-04", True),
    (2025, "2025-05-03", True),
]


def event_frame() -> pd.DataFrame:
    """The calendar as a frame: ``year``, ``date`` (Timestamp), ``ran_in_may`` (bool)."""
    df = pd.DataFrame(EVENTS, columns=["year", "date", "ran_in_may"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def all_tickers() -> list[str]:
    return [MARKET, GAMBLING]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"derby_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01") -> None:
    """Download adjusted (total-return) daily closes for SPY + CHDN; cache them.

    ``auto_adjust=True`` -- both are equities/ETFs, so total-return (splits & dividends
    folded in) is the honest series; the event-study returns below are then plain ratios
    on the cached close. Network; run once to build ``_cache/``.
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
    """Cached {ticker: adjusted-close Series}, each sliced to [START, asof]."""
    out = {}
    for t in all_tickers():
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        s = df["Close"]
        out[t] = s[(s.index >= pd.Timestamp(START)) & (s.index <= pd.Timestamp(asof))]
    return out


# --------------------------------------------------------------------------- #
# Synthetic world -- planted Derby bump (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, seed: int = 733, n_events: int = 26,
                    n_days: int = 7000, spacing: int = 252,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (CHDN-like, SPY-like) log-return world with a planted bump.

    Both series are correlated (rho ~ 0.55, like a single mid-cap gaming stock vs the
    S&P 500) zero-mean noise; on a synthetic "Derby day" (every ``spacing``-th business
    day, ~annual) the CHDN-like asset earns an EXTRA ``bump`` log-return while the market
    does not. ``bump = 0`` is the null world -- Derby days statistically identical to the
    rest. Integer business-day index (positions 0..n_days), far below the pandas
    ns-timestamp trap. Returns (chdn_logret, spy_logret, event_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.60
    common = rng.normal(0.0, 0.011, n_days)
    idio_a = rng.normal(0.0, 0.010, n_days)   # CHDN idio vol > market
    idio_b = rng.normal(0.0, 0.008, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    event_pos = list(range(spacing, n_days - 30, spacing))[:n_events]
    for p in event_pos:
        a[p + 1] += bump   # the bump lands on the first session AFTER the (Saturday) race

    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), event_pos
