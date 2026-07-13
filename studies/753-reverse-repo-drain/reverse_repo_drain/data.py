"""Data layer for Study 753 — Reverse-Repo-Drain (Fed ON RRP balance + SPY).

Two sources, both offline-friendly once the SPY cache exists:

* **Real tape.** SPY daily adjusted closes (yfinance, no key), cached under
  ``_cache/spy.csv`` and resampled to month-end — the equity tape the drain signal trades.

* **A labelled proxy for the liquidity series.** The Fed's **Overnight Reverse Repo
  (ON RRP) facility balance** is not on yfinance. It *is* public — FRED series
  ``RRPONTSYD`` (the daily ON RRP amount, Treasury securities sold by the Fed), sourced
  from the NY Fed's operating desk — but there is no yfinance handle, so we ship a
  **small, clearly-labelled, hardcoded monthly series** (end-of-month levels in USD
  billions), transcribed from the public FRED/NY-Fed prints. It is an *approximate proxy*
  — quarter-end window-dressing spikes are smoothed to round monthly marks — and is named
  a proxy everywhere. It carries the one feature the whole claim rests on: the great
  2021-2022 *fill* to the ~$2.55T Dec-2022 peak and the 2023-2025 *drain* back toward the
  facility's structural floor.

* **Synthetic.** A deterministic, fixed-seed generator that produces a hump-shaped
  RRP-like balance (fill then drain) and an SPY-like price whose drift is *higher during
  the drain* by a KNOWN ``edge`` knob. It is the positive control: ``edge=0`` must NOT
  manufacture significance; a large planted ``edge`` must light the test up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_spy`` (network) is only used
once to build the SPY cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
SPY_CACHE = os.path.join(CACHE_DIR, "spy.csv")

# --------------------------------------------------------------------------- #
# The labelled ON RRP proxy — hardcoded end-of-month levels (USD billions).
# Source: FRED ``RRPONTSYD`` / NY Fed ON RRP operating-desk daily prints, read at each
# month-end and rounded. This is a PROXY (quarter-end spikes smoothed), named as such.
# The shape is the point: a near-zero facility -> the 2021-22 fill to the ~$2,554B
# Dec-2022 all-time peak -> the 2023-25 drain back toward the structural floor.
# --------------------------------------------------------------------------- #
RRP_BILLIONS = {
    "2021-01": 1,    "2021-02": 1,    "2021-03": 3,    "2021-04": 26,
    "2021-05": 479,  "2021-06": 809,  "2021-07": 951,  "2021-08": 1087,
    "2021-09": 1415, "2021-10": 1375, "2021-11": 1591, "2021-12": 1904,
    "2022-01": 1626, "2022-02": 1595, "2022-03": 1723, "2022-04": 1815,
    "2022-05": 2044, "2022-06": 2330, "2022-07": 2266, "2022-08": 2213,
    "2022-09": 2426, "2022-10": 2260, "2022-11": 2145, "2022-12": 2554,
    "2023-01": 2060, "2023-02": 2190, "2023-03": 2280, "2023-04": 2233,
    "2023-05": 2170, "2023-06": 1950, "2023-07": 1727, "2023-08": 1596,
    "2023-09": 1408, "2023-10": 1080, "2023-11": 900,  "2023-12": 1018,
    "2024-01": 619,  "2024-02": 571,  "2024-03": 441,  "2024-04": 431,
    "2024-05": 402,  "2024-06": 664,  "2024-07": 382,  "2024-08": 350,
    "2024-09": 466,  "2024-10": 240,  "2024-11": 180,  "2024-12": 474,
    "2025-01": 176,  "2025-02": 175,  "2025-03": 199,  "2025-04": 122,
    "2025-05": 179,  "2025-06": 231,  "2025-07": 195,
}


# --------------------------------------------------------------------------- #
# Real tape — fetch (network; used once) + offline loaders
# --------------------------------------------------------------------------- #
def fetch_spy(path: str = SPY_CACHE, start: str = "2020-06-01",
              end: str | None = None) -> pd.Series:
    """Download SPY daily adjusted closes and cache them (network; once)."""
    import yfinance as yf

    spy = yf.download("SPY", start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    spy.to_csv(path)
    return spy


def have_real(path: str = SPY_CACHE) -> bool:
    return os.path.exists(path)


def load_spy(path: str = SPY_CACHE) -> pd.Series:
    """Cached SPY daily adjusted close as a Series named 'spy'."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    s = df.iloc[:, 0].astype(float)
    s.name = "spy"
    return s


def spy_month_end(spy: pd.Series | None = None) -> pd.Series:
    """SPY resampled to month-end closes (the monthly tape the drain signal trades)."""
    if spy is None:
        spy = load_spy()
    me = spy.resample("ME").last().dropna()
    me.name = "spy"
    return me


def rrp_series() -> pd.Series:
    """The hardcoded ON RRP proxy as a month-end Series (USD billions)."""
    idx = pd.to_datetime(list(RRP_BILLIONS.keys())) + pd.offsets.MonthEnd(0)
    s = pd.Series(list(RRP_BILLIONS.values()), index=idx, dtype=float)
    s.name = "rrp"
    return s.sort_index()


def build_real() -> pd.DataFrame:
    """Monthly frame with columns ``rrp`` (proxy, $B) and ``spy`` (month-end close), aligned.

    The RRP level for month ``t`` is known at month-end ``t``; ``spy`` is the same month-end
    close. The strategy layer applies the execution lag, so this frame carries no look-ahead.
    """
    rrp = rrp_series()
    spy = spy_month_end()
    out = pd.DataFrame({"rrp": rrp}).join(pd.DataFrame({"spy": spy}), how="inner")
    return out.dropna()


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic(n_months: int = 120, edge: float = 0.0, seed: int = 753,
              mu_monthly: float = 0.007, sig_monthly: float = 0.042,
              peak_frac: float = 0.45) -> pd.DataFrame:
    """Deterministic hump-shaped RRP-like balance + SPY-like price with a PLANTED drain edge.

    The RRP-like series rises to a peak at ``peak_frac`` of the span, then drains back down
    (a smooth tent plus mild noise) — the same fill-then-drain shape the real facility traced.
    SPY-like monthly returns have drift ``mu_monthly`` and vol ``sig_monthly``; if ``edge`` is
    non-zero an *extra* return of ``edge`` is injected in every month where the balance is
    **draining** (level below its value 3 months earlier), so a draining facility genuinely
    predicts a higher return.

    ``edge = 0`` is the null: the drain carries no forward information and the inference must
    NOT manufacture significance. A large ``edge`` must light the test up. The date index is a
    decorative monthly label built with ``period_range`` (no OutOfBounds on long spans).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_months)
    peak = peak_frac * (n_months - 1)
    # symmetric tent to a peak, scaled to a ~2500 top, plus small noise
    tent = 1.0 - np.abs(t - peak) / max(peak, n_months - 1 - peak)
    rrp = np.clip(tent, 0.0, None) * 2500.0 + rng.normal(0, 40.0, n_months)
    rrp = np.clip(rrp, 0.0, None)

    draining = np.zeros(n_months, dtype=bool)
    draining[3:] = rrp[3:] < rrp[:-3]

    ret = rng.normal(mu_monthly, sig_monthly, size=n_months)
    if edge != 0.0:
        # a draining month t -> extra drift in month t+1 (the planted "drain = risk-on")
        ret[1:] += edge * draining[:-1]

    price = 100.0 * np.cumprod(1.0 + ret)
    idx = pd.period_range("2000-01", periods=n_months, freq="M").to_timestamp(how="end")
    return pd.DataFrame({"rrp": rrp, "spy": price}, index=idx)
