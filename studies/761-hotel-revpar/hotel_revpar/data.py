"""Data layer for Study 761 — Hotel-RevPAR (a cited travel-cycle gauge vs hotel REITs).

Three components, the first fully offline and deterministic:

* **Real RevPAR tape (hardcoded, cited snapshot).** ``REVPAR_BY_YEAR`` is a monthly,
  clearly-labelled **approximate reconstruction** of U.S. hotel-industry **RevPAR**
  (Revenue Per Available Room, dollars) — the headline number STR / CoStar publish
  each month and the travel industry's canonical demand gauge. STR's actual monthly
  series is **proprietary** (paywalled), so — exactly as Study 358 (Watch-Index) and
  Study 708 (Eurovision) build a small, cited stand-in for a series that is not on any
  free tape — we hardcode an approximate monthly path **anchored to STR/CoStar-reported
  annual U.S. RevPAR** with realistic hotel seasonality, and with the famous 2020 COVID
  collapse and 2021 recovery set to the widely-reported national monthly figures. It is
  a **PROXY**, named as such everywhere; the verdict is built on the *shape* of the
  travel cycle (YoY momentum), which the anchors pin down, not on any single month's
  exact dollar figure.

* **Real hotel-equity tape.** ``load_hst`` reads the cached daily total-return adjusted
  close of **HST** (Host Hotels & Resorts — the largest, longest-listed lodging REIT and
  the desk's flagship "hotel basket"); ``load_basket`` reads a cached equal-weight
  lodging-REIT total-return index (HST, RHP, SHO, DRH, PEB, APLE, PK, whichever are
  listed each day). Both via yfinance (``auto_adjust=True``, so total-return, labelled).
  ``fetch_*`` (network) rebuild the caches and are never imported by the offline cells.

* **Synthetic positive control.** :func:`synthetic` is a deterministic, fixed-seed
  generator producing a monthly RevPAR path and a hotel-like price with a *planted*
  lead: when RevPAR momentum turns up, forward equity returns are lifted by a
  controllable ``edge`` knob. ``edge = 0`` is the null and must NOT manufacture
  significance; a large planted ``edge`` must light the test up.

Pure numpy + pandas + stdlib for the offline path.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
HST_CACHE = os.path.join(CACHE_DIR, "hst.csv")
BASKET_CACHE = os.path.join(CACHE_DIR, "basket.csv")

# The equal-weight lodging-REIT basket (whichever names are listed each day).
BASKET_NAMES = ["HST", "RHP", "SHO", "DRH", "PEB", "APLE", "PK"]

# --------------------------------------------------------------------------- #
# Real RevPAR tape — hardcoded monthly PROXY, $ Revenue Per Available Room.
# --------------------------------------------------------------------------- #
# U.S. hotel-industry RevPAR (dollars), monthly, Jan..Dec per row. An APPROXIMATE
# reconstruction anchored to STR / CoStar-reported ANNUAL U.S. RevPAR (STR is the
# industry's standard benchmarking source; annual figures are widely reported in AHLA
# "State of the Industry" and STR press releases). Monthly shape uses realistic hotel
# seasonality (summer peak, winter trough); the 2020 COVID collapse (national RevPAR
# fell below ~$21 in April 2020) and the 2021 recovery path are set to the reported
# national monthly figures. As-of 2026-06-22. This is a PROXY, labelled as such: the
# verdict rides on the travel cycle's YoY *shape*, not any single dollar print.
REVPAR_BY_YEAR: dict[int, list[float]] = {
    1998: [34, 37, 42, 42, 44, 46, 46, 44, 43, 44, 37, 34],
    1999: [36, 39, 44, 44, 46, 48, 48, 46, 45, 46, 39, 36],
    2000: [38, 41, 46, 47, 49, 51, 51, 48, 47, 48, 41, 38],
    2001: [36, 39, 44, 44, 46, 48, 48, 46, 45, 46, 39, 36],
    2002: [35, 38, 43, 43, 45, 47, 47, 45, 44, 45, 38, 35],
    2003: [36, 39, 44, 44, 46, 48, 48, 46, 45, 46, 39, 36],
    2004: [40, 42, 48, 49, 50, 52, 52, 50, 49, 50, 43, 39],
    2005: [44, 47, 53, 54, 56, 58, 58, 55, 54, 55, 47, 43],
    2006: [48, 52, 58, 59, 62, 64, 64, 61, 60, 61, 52, 48],
    2007: [52, 56, 63, 64, 67, 69, 69, 66, 65, 66, 57, 51],
    2008: [51, 55, 62, 63, 65, 68, 68, 65, 64, 65, 56, 51],
    2009: [44, 47, 53, 54, 56, 58, 58, 55, 54, 55, 47, 43],
    2010: [45, 49, 55, 56, 58, 60, 60, 57, 56, 57, 49, 45],
    2011: [48, 52, 58, 59, 62, 64, 64, 61, 60, 61, 52, 48],
    2012: [51, 55, 62, 63, 65, 68, 68, 65, 64, 65, 56, 51],
    2013: [55, 59, 66, 67, 70, 72, 72, 69, 68, 69, 59, 54],
    2014: [58, 62, 70, 71, 74, 77, 77, 73, 72, 73, 63, 57],
    2015: [62, 67, 75, 76, 79, 82, 82, 79, 77, 79, 67, 61],
    2016: [66, 70, 79, 81, 84, 87, 87, 83, 81, 83, 71, 65],
    2017: [68, 73, 82, 84, 87, 90, 90, 86, 84, 86, 74, 67],
    2018: [70, 75, 85, 86, 90, 93, 93, 89, 87, 89, 76, 69],
    2019: [72, 78, 87, 89, 92, 96, 96, 91, 90, 91, 78, 71],
    2020: [72, 75, 44, 21, 24, 31, 40, 48, 42, 45, 38, 33],
    2021: [30, 32, 40, 52, 60, 72, 84, 80, 76, 80, 74, 68],
    2022: [76, 81, 91, 93, 97, 100, 100, 96, 94, 96, 82, 75],
    2023: [82, 87, 98, 100, 104, 108, 108, 103, 101, 103, 88, 81],
    2024: [83, 89, 100, 102, 106, 110, 110, 105, 103, 105, 90, 82],
    2025: [85, 91, 102, 104, 108, 113, 113, 107, 105, 107, 92, 84],
    2026: [88, 92, 101, 103, 107, 0, 0, 0, 0, 0, 0, 0],
}


def revpar_series() -> pd.Series:
    """Monthly RevPAR ($), indexed by month-end date. In-progress months (0) dropped."""
    rows = []
    for yr in sorted(REVPAR_BY_YEAR):
        for m, v in enumerate(REVPAR_BY_YEAR[yr], start=1):
            if v <= 0:
                continue
            rows.append((pd.Timestamp(yr, m, 1) + pd.offsets.MonthEnd(0), float(v)))
    idx = pd.DatetimeIndex([d for d, _ in rows])
    return pd.Series([v for _, v in rows], index=idx, name="revpar").sort_index()


# --------------------------------------------------------------------------- #
# Real hotel-equity tape — fetchers (network; used once) + offline loaders
# --------------------------------------------------------------------------- #
def fetch_hst(start: str = "1997-01-01", end: str | None = None,
              path: str = HST_CACHE) -> pd.Series:
    """Download HST daily total-return adjusted close and cache it (network-only)."""
    import yfinance as yf

    s = yf.download("HST", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    s = s.iloc[:, 0] if getattr(s, "ndim", 1) > 1 else s
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"HST": s}).dropna().to_csv(path)
    return s


def fetch_basket(start: str = "1997-01-01", end: str | None = None,
                 path: str = BASKET_CACHE) -> pd.Series:
    """Build & cache an equal-weight lodging-REIT total-return index (network-only)."""
    import yfinance as yf

    px = yf.download(BASKET_NAMES, start=start, end=end, auto_adjust=True,
                     progress=False)["Close"]
    ew = px.pct_change().mean(axis=1, skipna=True).dropna()
    idx = 100.0 * (1.0 + ew).cumprod()
    idx.name = "BASKET"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    idx.to_csv(path)
    return idx


def have_real(hst: str = HST_CACHE) -> bool:
    """True iff the HST cache exists (the RevPAR table is always available)."""
    return os.path.exists(hst)


def load_hst(path: str = HST_CACHE) -> pd.Series:
    """Cached HST daily total-return adjusted close."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df.iloc[:, 0].astype(float)


def load_basket(path: str = BASKET_CACHE) -> pd.Series:
    """Cached equal-weight lodging-REIT total-return index (daily)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df.iloc[:, 0].astype(float)


def _month_end(daily: pd.Series) -> pd.Series:
    me = daily.resample("ME").last().dropna()
    return me


def build_real(which: str = "hst") -> pd.DataFrame:
    """Monthly frame aligned on month-ends: ``revpar`` ($) + ``px`` (hotel price).

    ``which='hst'`` uses HST (the long, flagship lodging REIT); ``which='basket'`` uses
    the equal-weight lodging-REIT index. The RevPAR for reference month ``t`` is the
    month-``t`` value; the strategy layer applies the STR release lag, so this frame
    carries no look-ahead by itself. Price is total-return (``auto_adjust``), labelled.
    """
    rev = revpar_series()
    daily = load_basket() if which == "basket" else load_hst()
    px = _month_end(daily)
    df = pd.DataFrame({"revpar": rev, "px": px}).dropna()
    return df


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic(n_months: int = 336, edge: float = 0.0, seed: int = 761,
              mu_m: float = 0.008, sig_m: float = 0.075) -> pd.DataFrame:
    """Deterministic monthly RevPAR + hotel-like price with a PLANTED momentum->return lead.

    RevPAR is a seasonal, slowly-cycling level (a smooth business cycle × a fixed 12-month
    seasonal pattern); hotel-like monthly returns have drift ``mu_m`` and vol ``sig_m``
    (REIT-sized). When ``edge != 0`` the *forward* monthly return is lifted by
    ``edge`` whenever RevPAR YoY momentum is **positive** at the prior month — the
    believers' story (accelerating travel leads hotel stocks) injected by construction.

    ``edge = 0`` => RevPAR momentum carries no forward information (the null) and the
    inference must NOT manufacture significance; a large ``edge`` (e.g. 0.03 monthly)
    must drive the HAC |t| well past 2. The date index is a decorative monthly label
    built with ``period_range`` (no OutOfBounds risk for long spans).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_months)
    # a slow multi-year travel cycle + fixed seasonality, on a $60 base
    cycle = 60.0 + 18.0 * np.sin(2 * np.pi * t / 84.0) + rng.normal(0, 1.5, n_months)
    seas = 1.0 + 0.12 * np.sin(2 * np.pi * (t % 12) / 12.0 - 0.6)
    revpar = np.maximum(cycle * seas, 5.0)

    ret = rng.normal(mu_m, sig_m, size=n_months)
    yoy = np.full(n_months, np.nan)
    yoy[12:] = np.log(revpar[12:]) - np.log(revpar[:-12])
    if edge != 0.0:
        for i in range(12, n_months - 1):
            if yoy[i] > 0:
                ret[i + 1] += edge

    price = 100.0 * np.exp(np.cumsum(ret))
    idx = pd.period_range("1998-01", periods=n_months, freq="M").to_timestamp("M")
    return pd.DataFrame({"revpar": revpar, "px": price}, index=idx)
