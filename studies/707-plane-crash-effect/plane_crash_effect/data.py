"""Data layer for Study 707 — Plane-Crash-Effect.

Three ingredients, all offline-friendly once cached:

* **The disaster table, hardcoded.** ``DISASTERS`` is a curated table of **36 major
  commercial-aviation accidents, 2000 -> 2025** (fatal, globally reported crashes /
  shoot-downs / disappearances of scheduled or chartered commercial airliners). No
  free, machine-readable "major aviation disaster index" exists, so — exactly like the
  sibling studies that hand-build a shock calendar (``313-geopolitical-shock``'s
  ``SHOCK_TABLE``, ``637-fomc-vol-crush``'s ``FOMC_DATES``) — this is a hand-built table
  of the crashes any reasonable person would call "the market's front page that
  morning". **9/11 (2001-09-11) and MH17 (2014-07-17) are deliberately excluded**: both
  already sit in ``313-geopolitical-shock``'s shock table as *terrorism/war* events, and
  including them here would double-count the same market day under two different
  folklore claims. Kaplanski & Levy (2010) run the same exclusion for the same reason
  (a terror attack tests war/terror sentiment, not aviation-disaster mood).

* **Real tape.** Daily SPY (S&P 500 ETF, the market-wide sentiment proxy) and the four
  major US carriers used as the "airline basket" — American (AAL), Delta (DAL), United
  (UAL) and Southwest (LUV) — total-return adjusted closes, all from yfinance (no key),
  cached as CSV under the study's own ``_cache/``. AAL/DAL/UAL each trade under their
  *current* corporate entity only from a bankruptcy-emergence or merger date (United
  2006-02, Delta 2007-05, American 2013-12); LUV has traded continuously since 1977.
  Named honestly: crashes before a carrier's current-ticker start date simply have
  fewer than 4 names in the basket average that day (tracked via ``n_tickers``), and
  the earliest crashes in the table (2000-2002) may have **zero** airline-basket
  coverage — those events are dropped from the airline test, not silently zero-filled.

* **Synthetic world.** A deterministic, seeded random-walk tape with a TUNABLE planted
  event-day dip that mean-reverts over a few sessions (``dip`` in daily-return units).
  ``dip = 0`` is the null world — event days statistically identical to the rest; the
  Welch/event-study machinery must NOT manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

SPY_CACHE = os.path.join(CACHE_DIR, "pce_spy.csv")
AIRLINE_TICKERS = ("AAL", "DAL", "UAL", "LUV")
AIRLINE_CACHE = {t: os.path.join(CACHE_DIR, f"pce_{t.lower()}.csv") for t in AIRLINE_TICKERS}

START = "2000-01-01"
AS_OF = "2026-06-30"        # last complete month at publication (2026-07-10)

# --------------------------------------------------------------------------- #
# Hardcoded table of major commercial-aviation disasters, 2000 -> 2025.
# Each row: (crash_date, label, fatalities). ``crash_date`` is the calendar date the
# accident happened (as widely reported); the event-study code snaps this to the
# first NYSE session on/after that date via searchsorted — a weekend or overseas
# late-night crash rolls forward to the next open session, a weekday crash before the
# US close lands same-day. That snap is the study's single documented execution lag
# (see strategy.py). Deliberately excludes 2001-09-11 and 2014-07-17 (MH17) — both
# already live in 313-geopolitical-shock's shock table as terror/war events.
# --------------------------------------------------------------------------- #
DISASTERS: list[tuple[str, str, int]] = [
    ("2000-01-31", "Alaska Airlines 261 (Pacific, off California)", 88),
    ("2000-07-25", "Air France 4590 (Concorde, Paris)", 113),
    ("2001-11-12", "American Airlines 587 (Queens, NY)", 265),
    ("2002-05-25", "China Airlines 611 (Penghu)", 225),
    ("2005-08-14", "Helios Airways 522 (Grammatiko, Cyprus)", 121),
    ("2005-08-16", "West Caribbean Airways 708 (Venezuela)", 160),
    ("2006-08-27", "Comair 5191 (Lexington, KY)", 49),
    ("2007-01-01", "Adam Air 574 (Sulawesi, Indonesia)", 102),
    ("2007-03-07", "Garuda Indonesia 200 (Yogyakarta)", 21),
    ("2008-09-14", "Aeroflot-Nord 821 (Perm, Russia)", 88),
    ("2009-02-12", "Colgan Air 3407 (Buffalo, NY)", 50),
    ("2009-06-01", "Air France 447 (Atlantic)", 228),
    ("2009-06-30", "Yemenia 626 (Comoros)", 152),
    ("2010-01-25", "Ethiopian Airlines 409 (Beirut)", 90),
    ("2010-05-12", "Afriqiyah Airways 771 (Tripoli)", 103),
    ("2010-07-28", "Airblue 202 (Islamabad)", 152),
    ("2011-06-20", "RusAir 9605 (Karelia, Russia)", 47),
    ("2012-04-20", "Bhoja Air 213 (Islamabad)", 127),
    ("2013-07-06", "Asiana Airlines 214 (San Francisco)", 3),
    ("2014-03-08", "Malaysia Airlines 370 (disappearance)", 239),
    ("2014-07-23", "TransAsia Airways 222 (Penghu, Taiwan)", 48),
    ("2014-07-24", "Air Algerie 5017 (Mali)", 116),
    ("2014-12-28", "Indonesia AirAsia 8501 (Java Sea)", 162),
    ("2015-03-24", "Germanwings 9525 (French Alps)", 150),
    ("2015-10-31", "Metrojet 9268 (Sinai)", 224),
    ("2016-03-19", "FlyDubai 981 (Rostov-on-Don)", 62),
    ("2016-05-19", "EgyptAir 804 (Mediterranean)", 66),
    ("2018-10-29", "Lion Air 610 (Java Sea, 737 MAX)", 189),
    ("2019-03-10", "Ethiopian Airlines 302 (737 MAX, global grounding)", 157),
    ("2020-01-08", "Ukraine Int'l Airlines 752 (Tehran)", 176),
    ("2020-05-22", "Pakistan Int'l Airlines 8303 (Karachi)", 97),
    ("2021-01-09", "Sriwijaya Air 182 (Jakarta)", 62),
    ("2022-03-21", "China Eastern Airlines 5735 (Guangxi)", 132),
    ("2023-01-15", "Yeti Airlines 691 (Pokhara, Nepal)", 72),
    ("2024-12-29", "Jeju Air 2216 (Muan, South Korea)", 179),
    ("2025-06-12", "Air India 171 (Ahmedabad, Boeing 787)", 241),
]


def disaster_table() -> pd.DataFrame:
    """The curated table as a frame: ``date`` (Timestamp), ``label``, ``fatalities``."""
    df = pd.DataFrame(DISASTERS, columns=["date", "label", "fatalities"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01") -> None:
    """Download SPY + the 4-carrier airline basket total-return closes; cache them.

    Network; run once. ``auto_adjust=True`` so closes already fold in splits and
    dividends (total-return, not price-only) — the event-study returns below are then
    plain ``pct_change()`` on the cached close.
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)

    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy[["Close"]].dropna().to_csv(SPY_CACHE)

    for t in AIRLINE_TICKERS:
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df[["Close"]].dropna().to_csv(AIRLINE_CACHE[t])


def have_real() -> bool:
    return os.path.exists(SPY_CACHE) and all(os.path.exists(p) for p in AIRLINE_CACHE.values())


def load_real(asof: str = AS_OF) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Cached (spy_close, {ticker: close}) series, sliced to [START, asof]."""
    spy = pd.read_csv(SPY_CACHE, index_col=0, parse_dates=True).sort_index()["Close"]
    spy = spy.loc[(spy.index >= START) & (spy.index <= asof)]
    airlines = {}
    for t, path in AIRLINE_CACHE.items():
        s = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()["Close"]
        airlines[t] = s.loc[(s.index >= START) & (s.index <= asof)]
    return spy, airlines


# --------------------------------------------------------------------------- #
# Synthetic world — planted event-day dip that mean-reverts (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(dip: float = 0.0, seed: int = 707,
                     n_days: int = 6500, n_events: int = 36,
                     daily_vol: float = 0.011, revert_days: int = 5,
                     start: str = "2000-01-03",
                     ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """A reproducible daily "SPY-like" random-walk tape with a TUNABLE planted dip.

    A random walk in log returns (i.i.d. normal, std ``daily_vol``) on which
    ``n_events`` "crash" dates are sprinkled (well away from the edges). On each event
    date the close takes an extra ``dip`` log-return hit, then reverts it in equal
    installments over the next ``revert_days`` sessions — a clean, mechanical
    dip-and-recovery, the shape the folklore claims. ``dip = 0`` is the null world:
    event days are statistically identical to every other day, and the event-study
    detector must NOT reach significance.

    Business-day index, span ~26 years — far below the 250-year pandas ns-timestamp
    trap. Returns (frame with a ``Close`` column, event-date DatetimeIndex).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    log_ret = rng.normal(0.0, daily_vol, n_days)

    margin = 40
    pool = np.arange(margin, n_days - margin)
    locs = np.sort(rng.choice(pool, size=min(n_events, pool.size), replace=False))

    per_day = dip / max(revert_days, 1)
    for loc in locs:
        log_ret[loc] += dip                      # the planted event-day dip
        for k in range(1, revert_days + 1):       # reverts smoothly over the next days
            if loc + k < n_days:
                log_ret[loc + k] -= per_day

    close = pd.Series(100.0 * np.exp(np.cumsum(log_ret)), index=idx)
    return pd.DataFrame({"Close": close}), idx[locs]
