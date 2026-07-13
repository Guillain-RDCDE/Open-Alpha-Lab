"""Data layer for Study 739 — Wildfire-Season.

The claim under test, steelmanned: **California's fire season is a repeating,
tradable risk event for the state's utilities and property insurers.** When a major
wildfire breaks out, the market is supposed to hammer the exposed names — the
investor-owned utility whose lines might have sparked it (PG&E / ``PCG``, Edison /
``EIX``) and the property insurers on the hook for the claims (Allstate, Travelers,
Mercury General, Chubb) — and, more ambitiously, the whole late-summer-to-autumn
*fire window* should carry a systematically worse return for that basket than the rest
of the calendar. If both were true you could underweight the basket every July and
short it on the ignition headline.

Three ingredients, all offline-friendly once cached:

* **The fire calendar, hardcoded.** ``FIRES`` is a curated table of **14 major
  California wildfire events, 2003 -> 2025** (the front-page, billion-dollar-loss fires
  any Californian would name). No free, machine-readable "major California wildfire
  index" exists, so — exactly like the sibling event studies that hand-build a shock
  calendar (``707-plane-crash-effect``'s ``DISASTERS``, ``313-geopolitical-shock``'s
  ``SHOCK_TABLE``) — this is a hand-built table cross-referenced against Cal Fire /
  public reporting. Each row carries a ``utility_linked`` flag: whether an
  investor-owned utility's equipment was the confirmed or leading-suspected ignition
  cause (the fires that actually threaten the *utility*, not just the state). The
  ``fire_date`` is the widely-reported ignition / breakout date; the event-study code
  snaps it forward to the first NYSE session on/after it — the study's single
  documented execution lag (see strategy.py).

* **Real tape.** Daily total-return closes (``auto_adjust=True``) for the two California
  investor-owned utilities — Edison International (``EIX``) and PG&E (``PCG``) — the
  four large property/casualty insurers used as the "insurer leg" — Allstate (``ALL``),
  Travelers (``TRV``), Mercury General (``MCY``, the most California-concentrated
  homeowner insurer of the four) and Chubb (``CB``) — and ``SPY`` as the market
  benchmark. All from yfinance (no key), cached as CSV under the study's own
  ``_cache/``. Every ticker has continuous history back to 2000, so basket coverage is
  full for every event in the table (no survivorship gap to name on this axis — but see
  the honest note in ``docs/results.md`` on PG&E's 2019 bankruptcy, which is *in* the
  total-return tape, not survivored out).

* **Synthetic world.** A deterministic, seeded random-walk tape with a TUNABLE planted
  event-day dip that mean-reverts over a few sessions (``dip`` in daily-return units).
  ``dip = 0`` is the null world — event days statistically identical to the rest; the
  event-study machinery must NOT manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

UTIL_TICKERS = ("EIX", "PCG")
INS_TICKERS = ("ALL", "TRV", "MCY", "CB")
BASKET_TICKERS = UTIL_TICKERS + INS_TICKERS
BENCHMARK = "SPY"
ALL_TICKERS = BASKET_TICKERS + (BENCHMARK,)

CACHE = {t: os.path.join(CACHE_DIR, f"wfs_{t.lower()}.csv") for t in ALL_TICKERS}

START = "2003-01-01"
AS_OF = "2026-06-30"        # last complete month at publication (2026-07-12)

# Fire season = the late-summer -> early-winter window in which California's major
# wildfires cluster (peak burning + autumn Santa Ana / Diablo wind events). Used by the
# seasonal test. Named as a decision, not snooped: months 7..12 (Jul -> Dec). January
# 2025's Eaton/Palisades fires are the reminder the tail now spills past New Year, but
# the *seasonal* window is defined ex-ante on the historical July->December clustering.
FIRE_MONTHS = (7, 8, 9, 10, 11, 12)

# --------------------------------------------------------------------------- #
# Hardcoded table of major California wildfire events, 2003 -> 2025.
# Each row: (fire_date, label, acres_thousands, utility_linked). ``fire_date`` is the
# widely-reported ignition/breakout date; the event-study code snaps it to the first
# NYSE session on/after that date via searchsorted (a weekend ignition rolls forward to
# the next open). ``utility_linked`` = an investor-owned utility's equipment was the
# confirmed or leading-suspected cause (the fires that put *the utility itself* at
# risk, via inverse-condemnation liability), vs fires with a non-utility cause
# (lightning, arson, vehicle, private equipment). Same-market-day co-ignitions are
# merged into one row (2018-11-08 Camp + Woolsey; 2025-01-07 Eaton + Palisades).
# Sources: Cal Fire incident archive; CPUC / utility 8-K liability disclosures;
# contemporary reporting. acres_thousands is the final burned area (round thousands).
# --------------------------------------------------------------------------- #
FIRES: list[tuple[str, str, int, bool]] = [
    ("2003-10-25", "Cedar Fire (San Diego)",                        273, False),
    ("2007-10-21", "Witch/October-2007 firestorm (San Diego; SDG&E)", 198, True),
    ("2013-08-17", "Rim Fire (Yosemite; hunter's fire)",            257, False),
    ("2015-09-12", "Valley Fire (Lake County; private wiring)",      76, False),
    ("2017-10-08", "N. California Oct-2017 firestorm / Tubbs (PG&E)", 245, True),
    ("2017-12-04", "Thomas Fire (Ventura/Santa Barbara; SCE)",      282, True),
    ("2018-07-23", "Carr Fire (Redding; vehicle spark)",            229, False),
    ("2018-11-08", "Camp + Woolsey Fires (Paradise/Malibu; PG&E/SCE)", 250, True),
    ("2019-10-23", "Kincade Fire (Sonoma; PG&E)",                    78, True),
    ("2020-08-17", "LNU/SCU/CZU lightning siege (Aug-2020; lightning)", 1500, False),
    ("2020-09-04", "Creek Fire (Sierra NF; undetermined)",          380, False),
    ("2021-07-13", "Dixie Fire (N. Sierra; PG&E)",                   963, True),
    ("2022-09-06", "Mosquito Fire (Placer/El Dorado; undetermined)", 77, False),
    ("2025-01-07", "Eaton + Palisades Fires (Los Angeles; SCE)",     57, True),
]


def fire_table() -> pd.DataFrame:
    """The curated table as a frame: ``date`` (Timestamp), ``label``, ``acres_k``,
    ``utility_linked`` (bool)."""
    df = pd.DataFrame(FIRES, columns=["date", "label", "acres_k", "utility_linked"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2000-01-01", end: str = "2026-07-01") -> None:
    """Download total-return closes for the whole basket + SPY; cache them.

    Network; run once. ``auto_adjust=True`` so closes already fold in splits and
    dividends (total-return, not price-only) — the event-study returns below are then
    plain ``pct_change()`` on the cached close. NOTE: PG&E's 2019 Chapter 11 and the
    dilution that followed are *inside* this total-return series (a real shareholder
    would have lived through them); they are not survivored out.
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in ALL_TICKERS:
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df[["Close"]].dropna().to_csv(CACHE[t])


def have_real() -> bool:
    return all(os.path.exists(p) for p in CACHE.values())


def load_real(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached {ticker: total-return close Series}, each sliced to [START, asof]."""
    out = {}
    for t, path in CACHE.items():
        s = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()["Close"]
        out[t] = s.loc[(s.index >= START) & (s.index <= asof)]
    return out


# --------------------------------------------------------------------------- #
# Synthetic world — planted event-day dip that mean-reverts (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(dip: float = 0.0, seed: int = 739,
                    n_days: int = 5800, n_events: int = 14,
                    daily_vol: float = 0.014, revert_days: int = 5,
                    start: str = "2003-01-02",
                    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """A reproducible daily "basket-like" random-walk tape with a TUNABLE planted dip.

    A random walk in log returns (i.i.d. normal, std ``daily_vol`` — set near a
    utility+insurer basket's daily vol) on which ``n_events`` "fire" dates are sprinkled
    (well away from the edges). On each event date the close takes an extra ``dip``
    log-return hit, then reverts it in equal installments over the next ``revert_days``
    sessions — a clean, mechanical dip-and-recovery. ``dip = 0`` is the null world:
    event days are statistically identical to every other day, and the event-study
    detector must NOT reach significance.

    Business-day index, span ~23 years — far below the 250-year pandas ns-timestamp
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
