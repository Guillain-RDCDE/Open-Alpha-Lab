"""Data layer for Study 647 — PBoC RRR Effect.

Three ingredients, all offline-friendly once cached:

* **Real tape.** Daily FXI (iShares China Large-Cap ETF, the primary claim vehicle — largely
  H-shares/ADRs via a swap structure, USD-denominated) raw OHLC, and daily MCHI (iShares MSCI
  China ETF, a broader A/H/ADR blend, inception 2011-03-29) raw OHLC, both from yfinance (no
  key), cached as CSV under the study's own ``_cache/``. MCHI only exists from 2011 onward, so
  it runs as a **cross-check on the modern era**, not the full-sample instrument.

* **The PBoC Reserve Requirement Ratio (RRR) change calendar, hardcoded.** Every BROAD-BASED
  (economy/system-wide) change to the RRR for large financial institutions announced by the
  People's Bank of China, 2008 -> 2025: cuts AND hikes. "Broad-based" excludes purely
  *targeted/structural* RRR relief (e.g. carve-outs for rural credit cooperatives or specific
  inclusive-finance lending programmes) so the calendar tracks the same "big, market-wide
  stimulus/tightening lever" the folklore is actually about — the same filtering logic sibling
  study 637-fomc-vol-crush applies to scheduled-vs-emergency FOMC actions, just pointed at a
  different axis (broad vs targeted, rather than scheduled vs surprise).
  Source: PBoC official announcements (pbc.gov.cn "Required Reserves" archive), cross-checked
  against contemporaneous Reuters/Xinhua/Caixin coverage and the CEIC/Wikipedia RRR level
  series. **Named honestly:** the 2008-2012 multi-category cuts (large vs small/medium banks
  moved on overlapping but not always identical dates and magnitudes) carry more day-level
  uncertainty than the 2015-2025 era, where every date is independently corroborated by at
  least two sources; the event-study machinery below is robust to being off by a session or
  two on any single date, and no single date drives the headline verdict either way.

* **Synthetic world.** A deterministic, seeded i.i.d.-return world with a TUNABLE planted
  cut-day mean shift (knob ``effect``): on scheduled "cut days" (every 20th business day)
  daily log returns get an added drift. ``effect = 0`` is the null world — cut days
  statistically identical to the rest; the Welch machinery must NOT manufacture significance
  from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
FXI_CACHE = os.path.join(CACHE_DIR, "pboc_fxi.csv")
MCHI_CACHE = os.path.join(CACHE_DIR, "pboc_mchi.csv")

START = "2008-01-01"        # covers the full hardcoded RRR calendar
AS_OF = "2026-06-30"        # last complete month at publication (2026-07-10)
MCHI_INCEPTION = "2011-03-29"
ZERO_RATE_ERA_SPLIT = "2015-01-01"  # justified split: the RRR pivots from a HIKING to a
                                    # near-permanent CUTTING regime here (last hike 2011-06,
                                    # first modern cut wave 2015-02) — a genuine regime change,
                                    # not a snooped split

# --------------------------------------------------------------------------- #
# Hardcoded PBoC broad-based RRR change announcements, 2008 -> 2025.
# (date, direction, bps) — direction in {"cut", "hike"}; bps is the MAGNITUDE (always
# positive). Large-financial-institutions RRR, system-wide moves only (targeted/structural
# relief for rural/SME-focused lenders excluded). Announcement date (Beijing calendar date),
# not the later effective date — see day_frame() for the execution-lag convention.
# --------------------------------------------------------------------------- #
RRR_EVENTS = [
    # --- 2008: tail of the pre-GFC hiking cycle, then the GFC easing pivot ---
    ("2008-01-16", "hike", 50),
    ("2008-03-18", "hike", 50),
    ("2008-04-16", "hike", 50),
    ("2008-05-12", "hike", 50),
    ("2008-06-07", "hike", 100),
    ("2008-09-25", "cut", 100),
    ("2008-10-15", "cut", 100),
    ("2008-12-05", "cut", 100),
    ("2008-12-25", "cut", 50),
    # --- 2010-2011: the post-GFC re-tightening cycle (inflation fight) ---
    ("2010-01-18", "hike", 50),
    ("2010-02-25", "hike", 50),
    ("2010-05-02", "hike", 50),
    ("2010-11-10", "hike", 50),
    ("2010-11-19", "hike", 50),
    ("2010-12-10", "hike", 50),
    ("2011-01-14", "hike", 50),
    ("2011-02-18", "hike", 50),
    ("2011-03-25", "hike", 50),
    ("2011-04-21", "hike", 50),
    ("2011-05-12", "hike", 50),
    ("2011-06-14", "hike", 50),
    # --- 2011-2012: the Euro-crisis easing pivot (RRR peaks at 21.5%, June 2011) ---
    ("2011-11-30", "cut", 50),
    ("2012-02-18", "cut", 50),
    ("2012-05-12", "cut", 50),
    # --- 2015-2016: the "double cut" easing cycle (equity-market stress + growth slowdown) ---
    ("2015-02-04", "cut", 50),
    ("2015-04-19", "cut", 100),
    ("2015-06-27", "cut", 50),
    ("2015-08-25", "cut", 50),
    ("2015-09-06", "cut", 50),
    ("2015-10-23", "cut", 50),
    ("2016-03-01", "cut", 50),
    # --- 2018-2020: the trade-war / pre-COVID / COVID easing sequence ---
    ("2018-04-17", "cut", 100),
    ("2018-06-24", "cut", 50),
    ("2018-10-07", "cut", 100),
    ("2019-01-04", "cut", 100),
    ("2019-09-06", "cut", 50),
    ("2020-01-01", "cut", 50),
    ("2020-03-13", "cut", 50),
    ("2020-04-03", "cut", 100),
    # --- 2021-2025: the secular grind lower (property slowdown, growth support) ---
    ("2021-07-09", "cut", 50),
    ("2021-12-06", "cut", 50),
    ("2022-04-15", "cut", 25),
    ("2022-11-25", "cut", 25),
    ("2023-03-17", "cut", 25),
    ("2023-09-14", "cut", 25),
    ("2024-01-24", "cut", 50),
    ("2024-09-27", "cut", 50),
    ("2025-05-07", "cut", 50),
]


def rrr_frame(start: str = START, end: str = AS_OF) -> pd.DataFrame:
    """The full hardcoded RRR-event table as a DataFrame, sliced to [start, end], sorted."""
    df = pd.DataFrame(RRR_EVENTS, columns=["date", "direction", "bps"])
    df["date"] = pd.to_datetime(df["date"])
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    df = df[(df["date"] >= lo) & (df["date"] <= hi)].sort_values("date").reset_index(drop=True)
    return df


def rrr_calendar(direction: str | None = None, start: str = START,
                 end: str = AS_OF) -> pd.DatetimeIndex:
    """RRR announcement dates inside [start, end], optionally filtered to 'cut' or 'hike'."""
    df = rrr_frame(start, end)
    if direction is not None:
        df = df[df["direction"] == direction]
    return pd.DatetimeIndex(sorted(df["date"]))


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2007-06-01", end: str = "2026-07-01") -> None:
    """Download FXI and MCHI raw OHLC + adjusted closes; cache them. Network; once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for ticker, path in ((("FXI"), FXI_CACHE), (("MCHI"), MCHI_CACHE)):
        raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        out = pd.DataFrame({
            "Open": raw["Open"], "High": raw["High"], "Low": raw["Low"], "Close": raw["Close"],
            "AdjClose": raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"],
        }).dropna(how="all")
        out.to_csv(path)


def have_real() -> bool:
    return all(os.path.exists(p) for p in (FXI_CACHE, MCHI_CACHE))


def load_real(start: str = START, asof: str = AS_OF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached (fxi, mchi) frames, sliced to [start, asof]."""
    out = []
    for path in (FXI_CACHE, MCHI_CACHE):
        df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        out.append(df.loc[(df.index >= start) & (df.index <= asof)].copy())
    return out[0], out[1]


# --------------------------------------------------------------------------- #
# Synthetic world — planted cut-day mean shift (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(effect: float = 0.0, seed: int = 647,
                    start: str = "2008-01-01", end: str = "2026-06-30",
                    sigma: float = 0.014,
                    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Deterministic i.i.d.-return price world with a TUNABLE planted cut-day mean shift.

    Daily log returns are N(0, sigma) (sigma calibrated near FXI's own realized vol — Chinese
    large-cap equities are a noisy tape). Every 20th business day is a scheduled "cut day"
    (roughly the ~48-events-over-18-years average RRR cadence); on those days an EXTRA
    ``effect`` (log-return units) is added. ``effect = 0`` is the null world: cut days are
    statistically identical to every other day, and the Welch split must NOT reach
    significance.

    Business-day index, span ~18.5 years — far below the 250-year pandas ns-timestamp trap.
    Returns (frame with a Close column, cut-day DatetimeIndex).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    is_cut = np.zeros(n, dtype=bool)
    is_cut[19::20] = True                      # scheduled, evenly spaced pseudo-cut days

    ret = rng.normal(0.0, sigma, n)
    ret[is_cut] += effect                       # the planted cut-day mean shift
    ret[0] = 0.0
    close = 100.0 * np.exp(np.cumsum(ret))
    frame = pd.DataFrame({"Close": close}, index=idx)
    return frame, idx[is_cut]
