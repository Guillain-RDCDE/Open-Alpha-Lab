"""Data layer for Study 769 — "Theme-park attendance/pricing as a DIS consumer tell".

Three sources, all offline-friendly once the caches exist:

* **The parks series (hardcoded, cited, APPROXIMATE).** The TEA/AECOM *Theme Index &
  Museum Index* — the industry-standard annual attendance report — is published as a PDF,
  **not a free API**, so we hardcode a small **annual** series of total Walt Disney
  Attractions worldwide attendance (millions of visits), reconstructed from the public
  Theme Index headline figures and clearly labelled **approximate**. It is anchored on
  facts the desk can cite: steady pre-pandemic growth to ~**156 M** in 2019, the COVID
  crater to ~**47 M** in 2020, and the 2021-23 recovery. Alongside it we hardcode the
  **Walt Disney World Magic Kingdom one-day peak base ticket price** (dollars) — a widely
  reported *pricing-power* series (base $79 in 2010 → $199 by 2024). Sources are listed in
  ``docs/references.md``. **These are PROXIES for the real tape, never the tape.**

  The load-bearing realism is the **release lag**: the Theme Index for calendar year Y is
  published in the *middle of the following year* (historically ~May-July of Y+1). So the
  attendance figure for year Y is only *public* — and only tradable — from ~July Y+1. We
  encode that lag so the study has strictly **no look-ahead**: you learn the print months
  after the year it describes, by which point DIS has already reported ~3 quarters of the
  Parks & Experiences segment.

* **DIS + SPY (yfinance).** Month-end Adj Close for **Disney (`DIS`)** and the benchmark
  **`SPY`**. Cached under ``_cache/`` so the notebooks run offline; on a cache miss with
  network we fetch via yfinance, otherwise we fall back to the frozen headline numbers.

* **Synthetic positive control.** A deterministic fixed-seed generator of an annual
  parks-momentum signal + a DIS-like price with a *knob* for a planted forward edge, used
  to prove the inference engine recovers what it plants. Runs with no network.

Pure numpy/pandas + stdlib; the offline core never touches the network.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Equities + benchmark.
TICKER = "DIS"
BENCH = "SPY"
TICKERS = [TICKER, BENCH]

# The Theme Index for calendar year Y is released ~mid Y+1. We use end-of-July Y+1 as the
# conservative "public as-of" date for year Y's attendance (the strict no-look-ahead lag).
THEME_INDEX_RELEASE_MONTH = 7


# --------------------------------------------------------------------------- #
# The parks series — hardcoded, cited, APPROXIMATE (proxies, not live feeds)
# --------------------------------------------------------------------------- #
# Total Walt Disney Attractions worldwide attendance (MILLIONS of visits), year-end,
# reconstructed from the TEA/AECOM Theme Index headline figures (see docs/references.md).
# The SHAPE is the load-bearing fact: steady pre-COVID growth, the 2020 crater, recovery.
_ATTENDANCE = {
    2010: 118.3,
    2011: 121.4,
    2012: 126.1,
    2013: 132.5,
    2014: 134.3,
    2015: 137.9,
    2016: 140.4,
    2017: 150.0,
    2018: 157.3,
    2019: 156.0,   # pre-pandemic peak
    2020: 47.0,    # COVID crater (parks shuttered ~4 months, capacity caps after)
    2021: 85.0,    # partial reopening
    2022: 122.0,   # recovery
    2023: 142.0,   # near pre-COVID
}

# Walt Disney World Magic Kingdom ONE-DAY PEAK base ticket price ($), a pricing-power tell.
# Public the day of each increase (announced ~Feb, immediately). Cited, approximate.
_TICKET_PRICE = {
    2010: 79, 2011: 85, 2012: 89, 2013: 95, 2014: 99, 2015: 105,
    2016: 124, 2017: 124, 2018: 129, 2019: 159, 2020: 159, 2021: 159,
    2022: 159, 2023: 189, 2024: 199,
}


def _year_end_series(d: dict, name: str) -> pd.Series:
    idx = pd.to_datetime([f"{y}-12-31" for y in d])
    return pd.Series(list(d.values()), index=idx, name=name, dtype=float)


def load_attendance() -> pd.Series:
    """Year-end total Walt Disney Attractions attendance (millions) — a cited PROXY.

    Reconstructed from the public TEA/AECOM Theme Index; a labelled approximation of the
    real tape, never a live feed.
    """
    return _year_end_series(_ATTENDANCE, "attendance")


def load_ticket_price() -> pd.Series:
    """Year-end WDW Magic Kingdom 1-day peak base ticket price ($) — a cited PROXY."""
    return _year_end_series(_TICKET_PRICE, "ticket_price")


def attendance_growth() -> pd.Series:
    """Year-over-year % change of the (approximate, cited) attendance series."""
    s = load_attendance()
    return (s / s.shift(1) - 1.0).dropna() * 100.0


def price_hikes() -> pd.Series:
    """Year-over-year % change of the (approximate, cited) ticket-price series."""
    s = load_ticket_price()
    return (s / s.shift(1) - 1.0).dropna() * 100.0


def release_date(year: int) -> pd.Timestamp:
    """Public 'as-of' date for calendar year ``year``'s Theme Index attendance figure.

    The Theme Index for year Y is published ~mid Y+1; we use end-of-July Y+1. This is the
    date the attendance-growth signal for year Y first becomes tradable (no look-ahead).
    """
    return pd.Timestamp(f"{year + 1}-{THEME_INDEX_RELEASE_MONTH:02d}-28") + pd.offsets.MonthEnd(0)


def _stepped_signal(index: pd.DatetimeIndex, annual: pd.Series,
                    release_lag: bool = True) -> pd.Series:
    """Step an *annual* signal onto a monthly ``index``, honouring the release lag.

    For each month ``m`` the value is the annual figure for the most recent year whose
    figure is already **public** by ``m``. With ``release_lag=True`` (the honest default)
    that means the Theme Index release date (July Y+1); with it False the value would be
    known at the calendar year-end (a LOOK-AHEAD variant we only use to show the lag bites).
    """
    out = pd.Series(np.nan, index=index, name=annual.name)
    for ts in index:
        val = np.nan
        for yr in sorted(annual.index.year):
            asof = release_date(yr) if release_lag else pd.Timestamp(f"{yr}-12-31")
            if asof <= ts:
                val = float(annual[annual.index.year == yr].iloc[0])
        out.loc[ts] = val
    return out


# --------------------------------------------------------------------------- #
# DIS + SPY via yfinance (cached, offline-friendly)
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE, f"{ticker}_monthly.csv")


def have_equities() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in TICKERS)


def fetch_equities(start: str = "2009-12-31", end: str = "2025-01-01") -> dict[str, pd.Series]:
    """Fetch month-end Adj Close for DIS + SPY; cache each. Network only on a cache miss."""
    os.makedirs(CACHE, exist_ok=True)
    need = [t for t in TICKERS if not os.path.exists(_cache_path(t))]
    if need:
        import yfinance as yf  # lazy import so the offline core never needs it
        raw = yf.download(need, start=start, end=end, interval="1mo",
                          auto_adjust=True, progress=False)
        close = raw["Close"] if "Close" in raw else raw
        if isinstance(close, pd.Series):
            close = close.to_frame(need[0])
        for t in need:
            close[t].dropna().to_csv(_cache_path(t))
    return load_equities()


def load_equities() -> dict[str, pd.Series]:
    """Cached month-end Adj Close for DIS + SPY, stamped to month-end (no network)."""
    out = {}
    for t in TICKERS:
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True)
        s = df.iloc[:, 0].dropna().astype(float)
        s.index = s.index + pd.offsets.MonthEnd(0)
        s.name = t
        out[t] = s
    return out


# --------------------------------------------------------------------------- #
# The monthly frame the strategy trades on
# --------------------------------------------------------------------------- #
def build_frame(release_lag: bool = True) -> pd.DataFrame:
    """Monthly frame: ``dis``, ``spy`` (month-end Adj Close) + the release-lagged signals.

    ``pg`` — the currently-*public* parks attendance YoY growth (%), stepped by the Theme
    Index release date. ``ph`` — the currently-public ticket-price hike (%). Both honour
    ``release_lag`` so the frame carries no look-ahead by itself; the strategy layer adds
    the one-month execution lag on top.
    """
    eq = load_equities()
    dis, spy = eq[TICKER], eq[BENCH]
    idx = dis.index
    pg = _stepped_signal(idx, attendance_growth().rename("pg"), release_lag=release_lag)
    ph = _stepped_signal(idx, price_hikes().rename("ph"), release_lag=release_lag)
    out = pd.DataFrame({"dis": dis, "spy": spy, "pg": pg, "ph": ph}).dropna(subset=["dis", "spy"])
    return out


# --------------------------------------------------------------------------- #
# Synthetic positive control (deterministic, fixed seed, no network)
# --------------------------------------------------------------------------- #
def synthetic(n_months: int = 180, edge: float = 0.0, seed: int = 769,
              mu_monthly: float = 0.008, sig_monthly: float = 0.070,
              phi: float = 0.90) -> pd.DataFrame:
    """Deterministic monthly parks-momentum signal + DIS-like price with a PLANTED edge.

    ``pg`` is a persistent AR(1) (the slow, sticky character a real attendance-momentum
    signal has) standardized to ~unit variance. DIS-like monthly returns have drift
    ``mu_monthly`` and vol ``sig_monthly``; if ``edge`` != 0 next month's return gets an
    extra ``edge * pg_t`` — so a high-momentum month genuinely predicts a higher next-month
    return. ``edge = 0`` is the null (the inference must NOT manufacture significance); a
    large ``edge`` must light the test up. Returns a month-end ``DataFrame``.
    """
    rng = np.random.default_rng(seed)
    pg = np.empty(n_months)
    pg[0] = rng.normal()
    for t in range(1, n_months):
        pg[t] = phi * pg[t - 1] + rng.normal(0, np.sqrt(1 - phi ** 2))
    ret = rng.normal(mu_monthly, sig_monthly, size=n_months)
    if edge != 0.0:
        ret[1:] += edge * pg[:-1]
    price = 100.0 * np.exp(np.cumsum(ret))
    idx = pd.period_range("2010-01", periods=n_months, freq="M").to_timestamp(how="end")
    return pd.DataFrame({"pg": pg, "dis": price}, index=idx)
