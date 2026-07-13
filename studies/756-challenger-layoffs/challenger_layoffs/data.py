"""Data layer for Study 756 — Challenger-Layoffs (job-cut announcements + SPY).

Three components, the first and third fully offline and deterministic:

* **Challenger job-cut tape (LABELLED APPROXIMATE snapshot).**
  ``CHALLENGER_CUTS_BY_YEAR`` is a hardcoded, monthly snapshot of U.S. **announced
  job cuts** (thousands) as published in the **Challenger, Gray & Christmas monthly
  Job Cuts Report** — the widely-cited private tally of layoffs *announced* by U.S.
  employers, released on/around the first Thursday of the following month. Challenger's
  series is **proprietary** (no free FRED / yfinance feed), so — exactly like Study 358
  (Watch-Index) hardcodes a labelled auction-price proxy and Study 708 (Eurovision)
  hardcodes a labelled points table — we hardcode a small, clearly-labelled **approximate
  monthly reconstruction** built from Challenger's published press-release headline
  totals and the well-known monthly spikes (post-9/11 2001, the 2008-09 crisis, the
  2015 energy bust, the COVID-2020 record, the 2023 tech wave, the 2025 federal cuts).
  It is NOT the exact revised vintage — it is an approximate, public-headline series,
  named as a proxy on the Signal axis. The believers' spike signal is computed *from*
  this series (see :mod:`challenger_layoffs.strategy`).

* **Real SPY tape.** ``load_spy`` reads the cached daily SPY adjusted close
  (``_cache/spy_prices.csv``, yfinance, no key). ``fetch_spy`` (network) rebuilds the
  cache and is never imported by the offline notebook cells. Price = total-return
  adjusted close (``auto_adjust=True``); labelled as such.

* **Synthetic positive control.** :func:`synthetic_cuts` is a deterministic, fixed-seed
  generator producing a monthly job-cut path and a daily SPY-like price with a *planted*
  link: when the job-cut spike fires, forward equity returns are knocked down by a
  controllable ``edge`` knob. ``edge = 0`` is the null and must NOT manufacture
  significance; a large planted ``edge`` must light the test up.

Pure numpy + pandas + stdlib for the offline path.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPY_CACHE = os.path.join(HERE, "..", "_cache", "spy_prices.csv")


# --------------------------------------------------------------------------- #
# Challenger job-cut tape — hardcoded LABELLED APPROXIMATE monthly snapshot
# --------------------------------------------------------------------------- #
# U.S. announced job cuts, in THOUSANDS, as reported in the Challenger, Gray &
# Christmas monthly Job Cuts Report. Approximate reconstruction from Challenger's
# published headline press-release totals and the famous monthly spikes. Monthly
# Jan..Dec per row. As-of: 2026-07-13. This is a PROXY (public headlines, not the
# exact revised vintage) — Challenger's data is proprietary and has no free feed.
# The record COVID month (Apr 2020, ~671k announced) is included faithfully.
CHALLENGER_CUTS_BY_YEAR: dict[int, list[float]] = {
    2000: [40, 48, 40, 52, 42, 45, 50, 40, 45, 60, 70, 82],
    2001: [142, 102, 162, 165, 80, 124, 205, 140, 248, 251, 181, 163],
    2002: [219, 128, 120, 142, 116, 101, 84, 119, 102, 177, 98, 100],
    2003: [130, 138, 120, 146, 68, 72, 86, 78, 71, 105, 100, 93],
    2004: [118, 86, 68, 72, 74, 112, 70, 74, 108, 101, 105, 52],
    2005: [92, 109, 86, 57, 82, 111, 103, 74, 72, 90, 101, 96],
    2006: [86, 71, 64, 60, 54, 71, 66, 66, 74, 71, 77, 80],
    2007: [62, 45, 62, 70, 71, 66, 42, 79, 71, 53, 73, 75],
    2008: [75, 72, 53, 90, 104, 82, 103, 89, 95, 112, 182, 167],
    2009: [241, 186, 150, 132, 111, 75, 97, 76, 66, 55, 51, 45],
    2010: [72, 42, 68, 38, 38, 40, 42, 35, 37, 42, 48, 33],
    2011: [38, 50, 41, 37, 38, 42, 67, 52, 116, 42, 43, 42],
    2012: [53, 52, 52, 40, 61, 37, 37, 37, 33, 47, 57, 32],
    2013: [40, 55, 49, 38, 36, 40, 38, 50, 40, 46, 45, 31],
    2014: [46, 42, 35, 40, 53, 32, 47, 40, 31, 52, 35, 32],
    2015: [53, 52, 36, 61, 41, 44, 47, 102, 58, 51, 31, 23],
    2016: [76, 32, 48, 65, 38, 39, 45, 32, 44, 31, 27, 34],
    2017: [46, 36, 44, 37, 52, 32, 28, 33, 33, 29, 31, 32],
    2018: [45, 36, 60, 40, 32, 37, 28, 39, 55, 76, 54, 44],
    2019: [53, 76, 60, 40, 58, 42, 39, 53, 42, 50, 44, 33],
    2020: [46, 56, 222, 671, 397, 170, 63, 116, 118, 80, 64, 77],
    2021: [80, 34, 31, 23, 25, 20, 19, 16, 17, 22, 15, 19],
    2022: [19, 15, 16, 24, 20, 32, 26, 20, 30, 33, 77, 43],
    2023: [103, 180, 90, 67, 80, 41, 24, 75, 47, 37, 45, 34],
    2024: [82, 84, 90, 64, 64, 49, 26, 76, 73, 56, 57, 38],
    2025: [50, 172, 275, 105, 94, 48, 62, 86, 55, 153, 33, 40],
    2026: [92, 88, 76, 71, 64, 58, 0, 0, 0, 0, 0, 0],
}


def have_real(path: str = DEFAULT_SPY_CACHE) -> bool:
    """True iff the SPY cache exists (the job-cut table is always available)."""
    return os.path.exists(path)


def cuts_series() -> pd.Series:
    """Monthly announced job cuts (thousands), indexed by month-end date.

    The in-progress months of the final year (zeros) are dropped so a stamped run
    never includes a partial bar.
    """
    rows = []
    for yr in sorted(CHALLENGER_CUTS_BY_YEAR):
        for m, v in enumerate(CHALLENGER_CUTS_BY_YEAR[yr], start=1):
            if v <= 0:
                continue
            rows.append((pd.Timestamp(yr, m, 1) + pd.offsets.MonthEnd(0), float(v)))
    idx = pd.DatetimeIndex([d for d, _ in rows])
    return pd.Series([v for _, v in rows], index=idx, name="cuts").sort_index()


def fetch_spy(start: str = "1999-01-01", end: str | None = None,
              path: str = DEFAULT_SPY_CACHE) -> pd.DataFrame:
    """Download SPY daily adjusted close via yfinance and cache it (network-only).

    Used once to build ``_cache/spy_prices.csv``. Never imported by offline cells.
    Total-return adjusted close (``auto_adjust=True``).
    """
    import yfinance as yf

    raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    out = pd.DataFrame({"SPY": raw}).dropna()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def load_spy(path: str = DEFAULT_SPY_CACHE) -> pd.Series:
    """Load the cached daily SPY adjusted-close series (total-return adjusted)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df["SPY"].astype(float)


def spy_monthly(path: str = DEFAULT_SPY_CACHE) -> pd.Series:
    """Month-end SPY adjusted close aligned to the monthly job-cut grid."""
    spy = load_spy(path)
    return spy.resample("ME").last().dropna()


def load_real(path: str = DEFAULT_SPY_CACHE) -> pd.DataFrame:
    """Monthly frame aligned on month-ends: Challenger cuts + month-end SPY close.

    Columns: ``cuts`` (thousands announced) and ``spy`` (price). Only months present
    in both series are kept. This is the real-tape object the strategy runs on.
    """
    cuts = cuts_series()
    spy_m = spy_monthly(path)
    df = pd.DataFrame({"cuts": cuts, "spy": spy_m}).dropna()
    return df


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_cuts(n_months: int = 360, edge: float = 0.0, seed: int = 756,
                   mu_m: float = 0.007, sig_m: float = 0.040) -> pd.DataFrame:
    """Deterministic monthly job cuts + SPY-like price with a PLANTED spike->returns link.

    Builds a mean-reverting monthly job-cut level (AR(1) in logs, around ~60k) with
    occasional multiplicative bursts, and a monthly SPY-like return series. When
    ``edge != 0`` the *forward* monthly return is perturbed by ``-edge`` whenever the
    job-cut spike fires at t (cuts above their trailing-12-month average) — the
    believers' story (a cut spike precedes equity weakness) injected by construction.

    ``edge = 0`` => the spike carries no information about returns (the null); the
    inference must NOT manufacture significance. A large ``edge`` (e.g. 0.04 monthly)
    must drive the Welch t well past 2. The date index is a decorative monthly label.
    """
    rng = np.random.default_rng(seed)

    # mean-reverting log-cuts around log(60), AR(1), with occasional bursts
    log_c = np.empty(n_months)
    log_c[0] = np.log(60.0)
    target = np.log(60.0)
    for t in range(1, n_months):
        burst = rng.normal(0, 0.10)
        if rng.random() < 0.05:                      # rare layoff-wave burst
            burst += rng.uniform(0.6, 1.4)
        log_c[t] = target + 0.90 * (log_c[t - 1] - target) + burst
    cuts = np.exp(log_c)

    # base monthly returns
    ret = rng.normal(mu_m, sig_m, size=n_months)

    # spike: cuts_t above trailing-12-month average (excluding t)
    s = pd.Series(cuts)
    trail = s.rolling(12, min_periods=6).mean().shift(1).values
    spike = np.zeros(n_months, dtype=bool)
    ok = ~np.isnan(trail)
    spike[ok] = cuts[ok] > trail[ok]

    # plant the believers' link: spike at t -> lower return at t+1
    if edge != 0.0:
        for t in range(n_months - 1):
            if spike[t]:
                ret[t + 1] -= edge

    price = 100.0 * np.exp(np.cumsum(ret))
    idx = pd.period_range("1990-01", periods=n_months, freq="M").to_timestamp("M")
    return pd.DataFrame({"cuts": cuts, "spy": price}, index=idx)
