"""Data layer for Study 755 — JOLTS-Quits (quits-rate tape + SPY + a cyclicals spread).

Three components, the first and third fully offline and deterministic:

* **Real quits-rate tape (hardcoded snapshot).** ``QUITS_RATE_BY_YEAR`` is a hardcoded,
  monthly snapshot of the U.S. **JOLTS Quits Rate — Total Nonfarm, Seasonally Adjusted**
  (percent of employment), Dec 2000..May 2026. Source: U.S. Bureau of Labor Statistics,
  *Job Openings and Labor Turnover Survey* (JOLTS), series ``JTSQUR`` (FRED ``JTSQUR``).
  Unlike the older desk studies where FRED was firewalled, the CSV endpoint *was*
  reachable when this snapshot was pinned (as-of 2026-07-13), so these are the true
  published prints — hardcoded so the study is reproducible and never-revised-in-place.
  The believers' momentum signal is computed *from* this series (see
  :mod:`jolts_quits.strategy`). The quits rate is the canonical worker-confidence gauge
  ("quits = workers voting with their feet") the labour-nowcasting literature quotes.

* **Real equity tapes.** ``load_spy`` reads cached daily **SPY** adjusted close, and
  ``load_cyclicals`` reads cached **XLY** (consumer-discretionary, cyclical) and **XLP**
  (consumer-staples, defensive) adjusted closes, from ``_cache/*.csv`` (yfinance, no key).
  ``fetch_*`` (network) rebuild the caches and are never imported by offline cells. Prices
  are total-return adjusted close (``auto_adjust=True``); labelled as such. From XLY and
  XLP we build a monthly-rebalanced **cyclical-minus-defensive** long-short index (the
  "risk appetite" tape the claim also names).

* **Synthetic positive control.** :func:`synthetic_quits` is a deterministic, fixed-seed
  generator producing a monthly quits path and a daily SPY-like price with a *planted*
  link: when the quits rate is **falling**, forward equity returns are knocked down by a
  controllable ``edge`` knob. ``edge = 0`` is the null (quits momentum and returns are
  independent) and must NOT manufacture significance; a large planted ``edge`` must light
  the test up. The control runs anywhere with no network.

Pure numpy + pandas + stdlib for the offline path.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPY_CACHE = os.path.join(HERE, "..", "_cache", "spy_prices.csv")
DEFAULT_XLY_CACHE = os.path.join(HERE, "..", "_cache", "xly_prices.csv")
DEFAULT_XLP_CACHE = os.path.join(HERE, "..", "_cache", "xlp_prices.csv")


# --------------------------------------------------------------------------- #
# Real quits-rate tape — hardcoded monthly snapshot of FRED JTSQUR (percent, SA)
# --------------------------------------------------------------------------- #
# U.S. JOLTS Quits Rate, Total Nonfarm, SEASONALLY ADJUSTED, in PERCENT of total
# employment. Source: U.S. Bureau of Labor Statistics, JOLTS (FRED series JTSQUR).
# Monthly reference-month value, Jan..Dec per row. As-of: 2026-07-13.
# JOLTS begins Dec 2000; the series has two famous features — the 2009 trough
# (~1.2-1.3%, workers too scared to quit) and the 2021-22 "Great Resignation" peak
# (~3.0%). Values are the published one-decimal prints. Zeros = not-yet-published
# reference months of the final year and are dropped by :func:`quits_series`.
QUITS_RATE_BY_YEAR: dict[int, list[float]] = {
    2000: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2],
    2001: [2.4, 2.3, 2.3, 2.4, 2.3, 2.2, 2.2, 2.2, 2.1, 2.1, 2.0, 2.0],
    2002: [2.2, 2.0, 1.9, 2.0, 1.9, 1.9, 2.0, 2.0, 1.9, 1.9, 1.8, 1.9],
    2003: [1.9, 1.9, 1.8, 1.8, 1.8, 1.8, 1.7, 1.7, 1.8, 1.9, 1.8, 1.9],
    2004: [1.8, 1.9, 2.0, 1.9, 1.8, 2.0, 2.0, 2.0, 1.9, 1.9, 2.1, 2.0],
    2005: [2.1, 2.0, 2.1, 2.1, 2.1, 2.1, 2.0, 2.2, 2.3, 2.1, 2.1, 2.1],
    2006: [2.2, 2.2, 2.2, 2.0, 2.2, 2.2, 2.2, 2.2, 2.1, 2.2, 2.2, 2.2],
    2007: [2.1, 2.1, 2.2, 2.1, 2.2, 2.1, 2.1, 2.2, 1.9, 2.1, 2.0, 2.0],
    2008: [2.1, 2.1, 1.9, 2.1, 1.9, 1.9, 1.8, 1.8, 1.8, 1.7, 1.6, 1.5],
    2009: [1.5, 1.5, 1.4, 1.3, 1.3, 1.3, 1.3, 1.2, 1.2, 1.3, 1.4, 1.4],
    2010: [1.3, 1.4, 1.4, 1.5, 1.4, 1.5, 1.4, 1.4, 1.5, 1.4, 1.4, 1.5],
    2011: [1.4, 1.5, 1.5, 1.4, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
    2012: [1.5, 1.6, 1.6, 1.6, 1.6, 1.6, 1.5, 1.5, 1.4, 1.5, 1.5, 1.5],
    2013: [1.7, 1.7, 1.6, 1.7, 1.6, 1.6, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7],
    2014: [1.7, 1.8, 1.8, 1.8, 1.8, 1.8, 1.9, 1.8, 2.0, 1.9, 1.9, 1.8],
    2015: [2.0, 1.9, 2.0, 1.9, 1.9, 1.9, 1.9, 2.0, 2.0, 2.0, 2.0, 2.1],
    2016: [2.0, 2.1, 2.0, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1],
    2017: [2.2, 2.1, 2.2, 2.1, 2.1, 2.2, 2.1, 2.1, 2.2, 2.2, 2.2, 2.2],
    2018: [2.0, 2.2, 2.2, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3],
    2019: [2.3, 2.4, 2.3, 2.3, 2.3, 2.3, 2.4, 2.3, 2.3, 2.3, 2.3, 2.3],
    2020: [2.3, 2.3, 1.9, 1.5, 1.7, 1.9, 2.2, 2.1, 2.2, 2.3, 2.2, 2.4],
    2021: [2.3, 2.4, 2.5, 2.7, 2.6, 2.8, 2.8, 2.8, 2.9, 2.8, 3.0, 2.9],
    2022: [2.9, 2.8, 2.9, 3.0, 2.8, 2.7, 2.6, 2.8, 2.7, 2.6, 2.6, 2.7],
    2023: [2.5, 2.5, 2.4, 2.3, 2.6, 2.4, 2.3, 2.3, 2.3, 2.3, 2.3, 2.2],
    2024: [2.1, 2.2, 2.1, 2.2, 2.1, 2.1, 2.1, 2.1, 2.0, 2.0, 1.9, 1.9],
    2025: [2.0, 2.0, 2.2, 2.0, 2.1, 2.1, 2.0, 2.0, 1.9, 1.9, 2.0, 2.0],
    2026: [2.0, 1.9, 2.0, 1.9, 1.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}


def have_real(path: str = DEFAULT_SPY_CACHE) -> bool:
    """True iff the SPY cache exists (the quits table is always available)."""
    return os.path.exists(path)


def have_cyclicals() -> bool:
    """True iff both the XLY and XLP caches exist (the cyclical-spread leg)."""
    return os.path.exists(DEFAULT_XLY_CACHE) and os.path.exists(DEFAULT_XLP_CACHE)


def quits_series() -> pd.Series:
    """Monthly JOLTS quits rate (percent), indexed by month-end date.

    The not-yet-published months of the final year (zeros) are dropped so a stamped
    run never includes a partial bar.
    """
    rows = []
    for yr in sorted(QUITS_RATE_BY_YEAR):
        for m, v in enumerate(QUITS_RATE_BY_YEAR[yr], start=1):
            if v <= 0:
                continue
            rows.append((pd.Timestamp(yr, m, 1) + pd.offsets.MonthEnd(0), float(v)))
    idx = pd.DatetimeIndex([d for d, _ in rows])
    return pd.Series([v for _, v in rows], index=idx, name="quits").sort_index()


def _fetch_one(ticker: str, path: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    out = pd.DataFrame({ticker: raw.squeeze()}).dropna()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def fetch_spy(start: str = "2000-12-01", end: str | None = None,
              path: str = DEFAULT_SPY_CACHE) -> pd.DataFrame:
    """Download SPY daily adjusted close via yfinance and cache it (network-only)."""
    return _fetch_one("SPY", path, start, end)


def fetch_cyclicals(start: str = "2000-12-01", end: str | None = None) -> None:
    """Download XLY + XLP daily adjusted close via yfinance and cache (network-only)."""
    _fetch_one("XLY", DEFAULT_XLY_CACHE, start, end)
    _fetch_one("XLP", DEFAULT_XLP_CACHE, start, end)


def load_spy(path: str = DEFAULT_SPY_CACHE) -> pd.Series:
    """Load the cached daily SPY adjusted-close series (total-return adjusted)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df["SPY"].astype(float)


def spy_monthly(path: str = DEFAULT_SPY_CACHE) -> pd.Series:
    """Month-end SPY adjusted close aligned to the monthly quits grid."""
    return load_spy(path).resample("ME").last().dropna()


def _monthly(path: str, col: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df[col].astype(float).resample("ME").last().dropna()


def cyclical_index(xly_path: str = DEFAULT_XLY_CACHE,
                   xlp_path: str = DEFAULT_XLP_CACHE) -> pd.Series:
    """Monthly-rebalanced cyclical-minus-defensive (XLY − XLP) long-short price index.

    Each month the index grows by ``r_XLY − r_XLP`` — the classic "risk appetite" tape:
    it rises when cyclicals lead defensives (investors confident), falls when defensives
    lead. Starts at 100. Total-return adjusted legs; the spread self-finances (long XLY,
    short XLP, gross exposure 1× per leg)."""
    xly = _monthly(xly_path, "XLY")
    xlp = _monthly(xlp_path, "XLP")
    r = (xly.pct_change() - xlp.pct_change()).dropna()
    idx = 100.0 * (1.0 + r).cumprod()
    return idx.rename("cyc")


def load_real(path: str = DEFAULT_SPY_CACHE) -> pd.DataFrame:
    """Monthly frame aligned on month-ends: quits rate + month-end SPY (+ cyclical index).

    Columns: ``quits`` (percent), ``spy`` (price), and — when the XLY/XLP caches exist —
    ``cyc`` (the cyclical-minus-defensive long-short index). Only months present in the
    quits and SPY series are kept; ``cyc`` is joined where available. This is the
    real-tape object the strategy runs on.
    """
    quits = quits_series()
    spy_m = spy_monthly(path)
    df = pd.DataFrame({"quits": quits, "spy": spy_m}).dropna()
    if have_cyclicals():
        df = df.join(cyclical_index())
    return df


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_quits(n_months: int = 300, edge: float = 0.0, seed: int = 755,
                    mu_m: float = 0.007, sig_m: float = 0.040) -> pd.DataFrame:
    """Deterministic monthly quits + SPY-like price with a PLANTED quits->returns link.

    Builds a mean-reverting monthly quits-rate level (AR(1) around ~2.1%) and a monthly
    SPY-like return series. When ``edge != 0`` the *forward* monthly return is perturbed
    by ``-edge`` whenever quits momentum (the 3-month change in the smoothed rate) is
    **falling** — the believers' story (falling quits precede weakness) injected by
    construction, with a knob.

    ``edge = 0`` => quits momentum carries no information about returns (the null); the
    inference must NOT manufacture significance. A large ``edge`` (e.g. 0.04 monthly)
    must drive the Welch t well past 2. The date index is a decorative monthly label
    built with ``period_range`` (no OutOfBounds risk for long spans).
    """
    rng = np.random.default_rng(seed)

    # mean-reverting quits-rate level around 2.1, AR(1), floored so it stays positive
    q = np.empty(n_months)
    q[0] = 2.1
    target = 2.1
    for t in range(1, n_months):
        q[t] = target + 0.90 * (q[t - 1] - target) + rng.normal(0, 0.08)
    q = np.clip(q, 0.8, 3.5)

    # base monthly returns
    ret = rng.normal(mu_m, sig_m, size=n_months)

    # quits momentum: 3-month change of the 3-month-smoothed quits level
    smooth = pd.Series(q).rolling(3, min_periods=1).mean().values
    mom = np.zeros(n_months)
    mom[3:] = smooth[3:] - smooth[:-3]
    falling = mom < 0

    # plant the believers' link: falling quits momentum at t -> lower return at t+1
    if edge != 0.0:
        for t in range(n_months - 1):
            if falling[t]:
                ret[t + 1] -= edge

    price = 100.0 * np.exp(np.cumsum(ret))
    idx = pd.period_range("1990-01", periods=n_months, freq="M").to_timestamp("M")
    return pd.DataFrame({"quits": q, "spy": price}, index=idx)
