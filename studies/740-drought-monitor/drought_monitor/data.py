"""Data layer for Study 740 — Drought-Monitor.

The claim under test: **a worsening US Drought Monitor print is tradable news for the
agricultural complex** — when the weekly Drought Monitor shows severe/extreme drought
rapidly expanding across the US crop belt, grain gets scarcer and pricier, and the ag
names (Deere for equipment, Mosaic for fertilizer, ADM for processing, the ag ETFs)
and grain itself should rise. It is the alt-data cousin of a real supply-shock story:
drought → smaller harvest → higher crop prices → good for the people who sell into the
shortage.

Four ingredients, all offline-friendly once cached:

* **The drought-escalation calendar, hardcoded.** ``DROUGHT_EVENTS`` is a curated table
  of **21 major US drought-intensification episodes, 2000 → 2025** — the weeks when the
  US Drought Monitor showed severe-drought coverage (D2+) rapidly expanding across a
  major agricultural region (the 2011 Texas drought, the 2012 Corn Belt flash drought,
  the 2012–2015 California drought, the 2021–2022 Western megadrought, the 2023 Midwest
  flash drought, …). The **US Drought Monitor** (droughtmonitor.unl.edu) releases every
  **Thursday morning ~8:30 ET**, with data through the prior Tuesday — public *before*
  the US market close, which gives the study its single, unavoidable execution lag for
  free (enter at the release-Thursday close; see ``strategy.py``). No free,
  machine-readable "major drought-onset index" exists, so — exactly like the sibling
  event studies that hand-build a shock calendar (``707-plane-crash``'s ``DISASTERS``,
  ``313-geopolitical-shock``'s ``SHOCK_TABLE``) — this is a hand-built table of the
  drought escalations any reasonable observer would call front-page ag news, each dated
  to a representative Thursday release during the rapid-intensification phase and
  cross-referenced against USDM archive reporting and contemporary coverage.

* **A labelled drought-severity PROXY series.** ``DROUGHT_PROXY`` is a hardcoded,
  **clearly-approximate** monthly series of the share of the contiguous US in **D2+
  (severe drought or worse)**, 2000-01 → 2025-06, digitised by eye from the US Drought
  Monitor's public time-series / Drought Severity & Coverage Index charts. It is a
  **LABELLED PROXY** — never presented under a real-tape banner — used only for the
  context chart and the drought-*regime* test (high-drought months vs the rest); the
  event study proper keys off the hardcoded escalation dates, not this smoothed series.

* **Real tape (yfinance, no key).** Daily total-return closes for the **ag-equity
  basket** — Deere (``DE``), Mosaic (``MOS``), Archer-Daniels-Midland (``ADM``) and the
  VanEck Agribusiness ETF (``MOO``) — the **grain / soft-commodity basket** — the
  Invesco DB Agriculture ETF (``DBA``), Teucrium Corn (``CORN``) and Teucrium Wheat
  (``WEAT``) — and the **SPY** benchmark. Each ETF only trades from its inception (DBA
  2007-01, MOO 2007-09, CORN 2010-06, WEAT 2011-09; MOS from its 2004 IPO; DE/ADM/SPY
  span the whole sample). Named honestly: the earliest drought events (2000-2006) have
  thin basket coverage (DE/ADM/MOS only), the grain ETFs cover only events from ~2010
  onward, and events landing in a zero-coverage window for a basket are dropped from
  that basket's test, never silently zero-filled.

* **Synthetic world.** A deterministic, seeded random-walk tape with a TUNABLE planted
  event-day bump on a synthetic drought-print calendar. ``bump = 0`` is the null world —
  event days statistically identical to the rest; the event-study machinery must NOT
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

START = "2000-01-01"
AS_OF = "2026-06-30"        # last complete month at publication (2026-07-13)

BENCHMARK = "SPY"
AG_EQUITY_TICKERS = ("DE", "MOS", "ADM", "MOO")   # farm equipment, fertilizer, processing, agribiz ETF
GRAIN_TICKERS = ("DBA", "CORN", "WEAT")           # broad-ag ETF, corn ETF, wheat ETF
ALL_TICKERS = (BENCHMARK,) + AG_EQUITY_TICKERS + GRAIN_TICKERS

# --------------------------------------------------------------------------- #
# Hardcoded table of major US drought-intensification episodes, 2000 -> 2025.
# Each row: (release_date, label, peak_d2_pct) where release_date is a representative
# **Thursday** US Drought Monitor release during the rapid-escalation phase (the USDM
# publishes every Thursday ~8:30 ET, data through the prior Tuesday), label names the
# region/episode, and peak_d2_pct is the approximate share of the contiguous US in D2+
# (severe drought or worse) around that window (from the USDM public archive — used only
# for the table's colour, not for any statistic). The event-study code snaps each
# release date to the first NYSE session on/after it (a Thursday release lands same-day;
# a holiday rolls to the next open) -- the study's single documented execution lag: the
# print is public that morning, so entering at that session's close is zero look-ahead.
# Hand-curated from USDM archive reporting and contemporary drought coverage.
# --------------------------------------------------------------------------- #
DROUGHT_EVENTS: list[tuple[str, str, int]] = [
    ("2000-08-03", "Southern Plains / Southeast drought (2000)", 20),
    ("2002-08-15", "Widespread West + Plains drought (2002)", 35),
    ("2003-07-03", "Western US drought (2003)", 20),
    ("2005-06-16", "Midwest / Plains summer drought (2005)", 15),
    ("2006-08-17", "Southern Plains drought (2006)", 26),
    ("2007-10-18", "Southeast US drought (Atlanta water crisis, 2007)", 22),
    ("2008-07-17", "California / Southeast drought (2008)", 16),
    ("2011-07-14", "Texas / Southern Plains historic drought (2011)", 32),
    ("2012-07-19", "Corn Belt flash drought (2012)", 38),
    ("2013-08-01", "Midwest late-summer drought (2013)", 30),
    ("2014-01-16", "California drought emergency declared (2014)", 28),
    ("2014-07-31", "California exceptional drought peak (2014)", 31),
    ("2015-04-02", "California mandatory water restrictions (2015)", 25),
    ("2017-07-06", "Northern Plains flash drought, spring wheat (2017)", 12),
    ("2018-07-19", "Four Corners / Southwest drought (2018)", 22),
    ("2020-08-20", "Western US drought expansion (2020)", 26),
    ("2021-06-17", "Western megadrought + N. Plains wheat (2021)", 34),
    ("2022-08-18", "Widespread West + Plains + Mississippi low water (2022)", 40),
    ("2023-06-08", "Midwest / Corn Belt flash drought (2023)", 20),
    ("2024-09-19", "Southern / Midwest autumn drought (2024)", 22),
    ("2025-05-15", "Southwest / Southern Plains drought (2025)", 16),
]


def drought_events() -> pd.DataFrame:
    """The curated escalation table as a frame: ``date`` (Timestamp), ``label``,
    ``peak_d2_pct``."""
    df = pd.DataFrame(DROUGHT_EVENTS, columns=["date", "label", "peak_d2_pct"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# LABELLED PROXY -- monthly share of the contiguous US in D2+ (severe drought or worse),
# 2000-01 -> 2025-06. Digitised by eye from the US Drought Monitor public time-series /
# Drought Severity & Coverage Index charts (droughtmonitor.unl.edu). APPROXIMATE, cited,
# and used ONLY for the context chart and the drought-regime split -- NEVER under a
# real-tape banner. Encoded as YEAR -> 12 monthly values (2025 covers Jan-Jun only).
# --------------------------------------------------------------------------- #
_DROUGHT_PROXY_ANNUAL: dict[int, list[float]] = {
    2000: [12, 14, 15, 16, 18, 19, 20, 20, 19, 17, 15, 13],
    2001: [11, 10, 9, 9, 10, 12, 14, 14, 13, 12, 11, 10],
    2002: [12, 14, 17, 20, 24, 29, 33, 35, 34, 30, 26, 22],
    2003: [20, 19, 18, 18, 19, 20, 20, 19, 17, 15, 13, 12],
    2004: [11, 10, 10, 10, 11, 12, 12, 11, 10, 9, 9, 9],
    2005: [9, 10, 11, 12, 13, 15, 14, 13, 12, 12, 13, 13],
    2006: [13, 14, 16, 18, 21, 24, 26, 26, 24, 21, 18, 16],
    2007: [15, 15, 16, 17, 18, 19, 20, 21, 22, 22, 20, 18],
    2008: [16, 15, 14, 14, 15, 16, 16, 15, 13, 12, 11, 10],
    2009: [10, 9, 9, 8, 8, 9, 9, 8, 7, 7, 7, 7],
    2010: [6, 6, 5, 5, 5, 5, 6, 6, 6, 7, 8, 9],
    2011: [11, 14, 17, 21, 25, 28, 32, 31, 28, 24, 20, 17],
    2012: [15, 15, 16, 18, 22, 30, 38, 42, 45, 42, 38, 34],
    2013: [33, 32, 31, 30, 30, 30, 29, 27, 24, 21, 19, 18],
    2014: [18, 20, 23, 26, 28, 30, 31, 31, 30, 29, 28, 27],
    2015: [26, 25, 25, 25, 24, 23, 21, 20, 19, 18, 16, 14],
    2016: [13, 12, 11, 11, 12, 13, 14, 15, 16, 17, 18, 17],
    2017: [15, 13, 11, 10, 10, 11, 12, 13, 12, 11, 11, 12],
    2018: [14, 16, 18, 19, 20, 21, 22, 20, 17, 15, 13, 12],
    2019: [10, 8, 7, 6, 5, 5, 5, 6, 7, 8, 9, 9],
    2020: [9, 9, 9, 10, 12, 15, 19, 23, 26, 27, 26, 25],
    2021: [26, 27, 28, 29, 30, 33, 34, 33, 31, 29, 28, 27],
    2022: [28, 29, 31, 33, 35, 37, 39, 40, 38, 35, 32, 29],
    2023: [27, 25, 23, 21, 20, 20, 21, 22, 21, 19, 17, 15],
    2024: [14, 13, 12, 13, 15, 17, 19, 20, 22, 21, 19, 17],
    2025: [15, 15, 16, 16, 16, 15],
}


def drought_proxy() -> pd.Series:
    """Monthly D2+ (severe-drought-or-worse) US coverage %, as a **labelled proxy**
    Series indexed by month-end Timestamp. APPROXIMATE — see the module docstring."""
    idx, vals = [], []
    for year in sorted(_DROUGHT_PROXY_ANNUAL):
        for m, v in enumerate(_DROUGHT_PROXY_ANNUAL[year], start=1):
            idx.append(pd.Timestamp(year=year, month=m, day=1) + pd.offsets.MonthEnd(0))
            vals.append(float(v))
    return pd.Series(vals, index=pd.DatetimeIndex(idx), name="d2_pct")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"dm_{ticker.lower()}.csv")


def fetch(start: str = START, end: str = "2026-07-01") -> None:
    """Download SPY + the ag-equity and grain baskets' total-return closes; cache them.

    Network; run once. ``auto_adjust=True`` so closes already fold in splits and
    dividends (total-return, not price-only) — the event-study returns below are then
    plain ``pct_change()`` on the cached close.
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in ALL_TICKERS:
        d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d[["Close"]].dropna().to_csv(_cache_path(t))


def have_real() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in ALL_TICKERS)


def load_real(asof: str = AS_OF) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Cached (spy_close, {ticker: close}) series, sliced to [START, asof].

    The returned dict holds every ag-equity and grain ticker; SPY is returned
    separately as the benchmark.
    """
    def _load(t):
        s = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()["Close"]
        return s.loc[(s.index >= START) & (s.index <= asof)]

    spy = _load(BENCHMARK)
    names = {t: _load(t) for t in AG_EQUITY_TICKERS + GRAIN_TICKERS}
    return spy, names


# --------------------------------------------------------------------------- #
# Synthetic world -- planted event-day bump (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, seed: int = 740,
                    n_days: int = 6500, n_events: int = 21,
                    daily_vol: float = 0.012, revert_days: int = 5,
                    start: str = "2000-01-03",
                    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """A reproducible daily "ag-basket-like" random-walk tape with a TUNABLE planted bump.

    A random walk in log returns (i.i.d. normal, std ``daily_vol``) on which
    ``n_events`` synthetic "drought-print" dates are sprinkled (well away from the
    edges). On each event date the close takes an extra ``bump`` log-return, then unwinds
    it in equal installments over the next ``revert_days`` sessions — a clean, mechanical
    pop-and-fade, the shape a tradable news reaction would have. ``bump = 0`` is the null
    world: event days are statistically identical to every other day, and the
    event-study detector must NOT reach significance.

    Business-day index, span ~26 years — far below the 250-year pandas ns-timestamp
    trap. Returns (frame with a ``Close`` column, event-date DatetimeIndex).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    log_ret = rng.normal(0.0, daily_vol, n_days)

    margin = 40
    pool = np.arange(margin, n_days - margin)
    locs = np.sort(rng.choice(pool, size=min(n_events, pool.size), replace=False))

    per_day = bump / max(revert_days, 1)
    for loc in locs:
        log_ret[loc] += bump                       # the planted event-day bump
        for k in range(1, revert_days + 1):        # fades smoothly over the next days
            if loc + k < n_days:
                log_ret[loc + k] -= per_day

    close = pd.Series(100.0 * np.exp(np.cumsum(log_ret)), index=idx)
    return pd.DataFrame({"Close": close}), idx[locs]
