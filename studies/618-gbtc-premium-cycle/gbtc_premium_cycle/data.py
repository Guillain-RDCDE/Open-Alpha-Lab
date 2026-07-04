"""Data layer for Study 618 — GBTC Premium Cycle.

Two sources, both offline-friendly once cached:

* **Real tape.** Daily closes for **GBTC** (Grayscale Bitcoin Trust, OTCQX 2015 → NYSE Arca
  spot-ETF 2024) and **BTC-USD** from yfinance, cached under ``_cache/``. The premium/discount
  is reconstructed against a **modeled bitcoin-per-share (BPS)** path built from public trust
  mechanics (hardcoded below with sources) — inception ratio, sponsor-fee accrual in BTC, the
  91-for-1 split, the ETF-conversion fee cut and the 2024 Mini-Trust spin-off. The model is
  **self-validating**: after 2024-01-11 the in-kind create/redeem arb pins the ETF's market
  price to NAV, so the reconstructed post-ETF premium must sit at ≈ 0 — any model error would
  show up as a constant offset there (it comes out at ~2 bp mean).

* **Synthetic.** A deterministic, fixed-seed generator producing a (BTC, GBTC) world with a
  PLANTED three-regime premium path — a premium plateau, a discount trough and a tunable
  **convergence drift** toward zero into a known conversion date — plus a **null** world whose
  premium is pure noise around zero. The positive control for the regime/means/HAC machinery.

Trust mechanics hardcoded (source comments below; URLs in ``docs/references.md``):

* 2013-09-25 — Bitcoin Investment Trust private placement opens at **0.1 BTC per share**
  (Grayscale/SEC Form 10, and every GBTC annual report since).
* Sponsor fee **2.0 %/yr**, accrued daily **in BTC** — bitcoin-per-share decays ~exp(-2 %·t).
* 2018-01-26 — **91-for-1** share split (record 2018-01-22; yfinance carries it on 2018-01-29).
  Yahoo prices are split-adjusted throughout, so the model works in post-split shares from
  inception: 0.1/91 BTC per (post-split) share.
* 2015-05-04 — OTCQX quotation approved (first Yahoo print 2015-05-11).
* 2024-01-11 — uplisted to NYSE Arca as a spot bitcoin **ETF** (SEC approval 2024-01-10);
  sponsor fee cut to **1.5 %/yr**.
* 2024-07-31 — **Grayscale Bitcoin Mini Trust spin-off**: GBTC distributed **10 %** of its
  bitcoin to shareholders as Mini-Trust (ticker BTC) shares. Yahoo folds this into ALL price
  columns as a ×0.90 back-adjustment (no dividend row), so applying the same ×0.90 factor to
  the whole modeled BPS path makes the reconstructed premium exact in both eras.

Model check against disclosed anchors: post-split BPS at the 2018 split date comes out
0.0010076 (Grayscale disclosed ≈0.00101) and at conversion 0.00089442 (Grayscale disclosed
≈0.00089 — ~619k BTC over ~692M shares). The ETF-era premium then averages **≈ -0.02 %** —
the model matches the arb-enforced NAV to basis points.

Pure numpy + pandas for the offline path; ``fetch_tape`` (network) runs only on a cache miss.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
GBTC_CACHE = os.path.join(CACHE_DIR, "gbtc.csv")
BTC_CACHE = os.path.join(CACHE_DIR, "btc.csv")

AS_OF = "2026-06-30"          # last complete month at build time (2026-07-03)

# ------------------------------------------------------------------ trust mechanics
INCEPTION = pd.Timestamp("2013-09-25")   # 0.1 BTC/share at the private placement
SPLIT = 91.0                             # 91-for-1 split, effective 2018-01-26
FEE_TRUST = 0.020                        # sponsor fee while a trust (2 %/yr, accrued in BTC)
FEE_ETF = 0.015                          # sponsor fee after ETF conversion (1.5 %/yr)
CONVERSION = pd.Timestamp("2024-01-11")  # first day trading as a spot ETF on NYSE Arca
SPINOFF_FACTOR = 0.90                    # 2024-07-31 Mini-Trust spin-off: 10 % of BTC out;
                                         # Yahoo back-adjusts all prices ×0.90, so the factor
                                         # applies to the WHOLE modeled BPS path (see docstring)

# Documented regime boundaries (dates from the public record; see docs/references.md)
LISTING = pd.Timestamp("2015-05-11")         # first Yahoo print after the OTCQX approval
DISCOUNT_START = pd.Timestamp("2021-02-23")  # first close of the permanent discount era
EVENTS = {
    # date            label                                        source (references.md)
    "2023-06-15": "BlackRock files the iShares spot-BTC ETF (S-1)",  # SEC EDGAR 2023-06-15
    "2023-06-16": "first close AFTER the BlackRock filing (news hit after hours)",
    "2023-08-29": "Grayscale v. SEC — DC Circuit vacates the denial",  # court opinion
    "2024-01-10": "SEC approves spot bitcoin ETFs",                   # SEC order 34-99306
    "2024-01-11": "GBTC's first day trading as a spot ETF",           # NYSE Arca uplisting
}


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_tape(gbtc_path: str = GBTC_CACHE, btc_path: str = BTC_CACHE) -> None:
    """Download GBTC + BTC-USD daily closes and cache them. Network; cache-miss only."""
    import yfinance as yf

    g = yf.download("GBTC", start="2015-01-01", auto_adjust=True, progress=False)["Close"]
    b = yf.download("BTC-USD", start="2014-09-01", auto_adjust=True, progress=False)["Close"]
    os.makedirs(CACHE_DIR, exist_ok=True)
    g.to_csv(gbtc_path)
    b.to_csv(btc_path)


def have_real(gbtc_path: str = GBTC_CACHE, btc_path: str = BTC_CACHE) -> bool:
    return os.path.exists(gbtc_path) and os.path.exists(btc_path)


def btc_per_share(dates: pd.DatetimeIndex) -> np.ndarray:
    """Modeled bitcoin-per-(post-split)-share on ``dates``, in Yahoo-adjusted terms.

    0.1/91 BTC at inception, decaying at the 2 %/yr sponsor fee until the ETF conversion and
    at 1.5 %/yr after, times the ×0.90 Mini-Trust spin-off factor that Yahoo bakes into every
    price (so premium = price / (BPS × BTC) − 1 is exact in both eras). Deterministic.
    """
    yrs = (dates - INCEPTION).days / 365.25
    yrs_conv = (CONVERSION - INCEPTION).days / 365.25
    pre = np.exp(-FEE_TRUST * np.asarray(yrs, dtype=float))
    post = np.exp(-FEE_TRUST * yrs_conv) * np.exp(-FEE_ETF * (np.asarray(yrs, dtype=float) - yrs_conv))
    path = np.where(dates < CONVERSION, pre, post)
    return SPINOFF_FACTOR * (0.1 / SPLIT) * path


def load_real(as_of: str | None = AS_OF) -> pd.DataFrame:
    """Cache-first aligned frame: gbtc, btc, bps, nav, prem — on GBTC trading days.

    ``prem`` is the reconstructed premium (+0.40 = 40 % above NAV). Sliced to ``as_of`` so the
    published sample never creeps.
    """
    if not have_real():
        fetch_tape()
    g = pd.read_csv(GBTC_CACHE, index_col=0, parse_dates=True).iloc[:, 0]
    b = pd.read_csv(BTC_CACHE, index_col=0, parse_dates=True).iloc[:, 0]
    df = pd.concat({"gbtc": g, "btc": b}, axis=1, sort=True).dropna()
    df["bps"] = btc_per_share(df.index)
    df["nav"] = df["btc"] * df["bps"]
    df["prem"] = df["gbtc"] / df["nav"] - 1.0
    if as_of is not None:
        df = df[df.index <= pd.Timestamp(as_of)]
    return df


def regimes(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the tape into the three documented regimes (dates from the public record)."""
    return {
        "premium era (2015-2021)": df[(df.index >= LISTING) & (df.index < DISCOUNT_START)],
        "discount era (2021-2024)": df[(df.index >= DISCOUNT_START) & (df.index < CONVERSION)],
        "ETF era (2024-)": df[df.index >= CONVERSION],
    }


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_world(drift_on: bool = True, premium_on: bool = True,
                    seed: int = 618, n_days: int = 2400,
                    prem_level: float = 0.40, trough: float = -0.45,
                    noise: float = 0.015) -> pd.DataFrame:
    """Deterministic (BTC, GBTC) world with a PLANTED three-regime premium path.

    * days [0, 55 %)      — premium plateau at ``prem_level`` (AR(1) noise around it);
    * days [55 %, 80 %)   — discount, drifting down to ``trough``;
    * days [80 %, 92.5 %) — the **convergence window**: log(1+prem) drifts linearly back to 0
      iff ``drift_on`` (the planted, tunable effect the HAC test must detect);
    * days [92.5 %, end)  — the ETF era, premium ≈ 0.

    With ``premium_on=False`` the premium is pure AR(1) noise around **zero** for the whole
    sample (the null): the regime means must read ≈ 0 and the convergence-window HAC t must
    NOT manufacture significance. BTC itself is a seeded GBM. Business-day index, well under
    the 250-year pandas span limit. Attaches ``.attrs['conv_window']`` = (start, end) dates.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=n_days)
    btc = 1000.0 * np.exp(np.cumsum(rng.normal(0.0008, 0.035, n_days)))

    i1, i2, i3 = int(n_days * 0.55), int(n_days * 0.80), int(n_days * 0.925)
    target = np.zeros(n_days)
    if premium_on:
        target[:i1] = np.log1p(prem_level)
        target[i1:i2] = np.linspace(np.log1p(prem_level), np.log1p(trough), i2 - i1)
        if drift_on:
            target[i2:i3] = np.linspace(np.log1p(trough), 0.0, i3 - i2)
        else:
            target[i2:i3] = np.log1p(trough)
        target[i3:] = 0.0
    ar = np.zeros(n_days)
    eps = rng.normal(0.0, noise, n_days)
    for t in range(1, n_days):
        ar[t] = 0.90 * ar[t - 1] + eps[t]
    log1p_prem = target + ar
    prem = np.expm1(log1p_prem)

    bps = 0.1 / 91.0 * np.exp(-0.02 * np.arange(n_days) / 252.0)
    nav = btc * bps
    df = pd.DataFrame({"gbtc": nav * (1.0 + prem), "btc": btc, "bps": bps,
                       "nav": nav, "prem": prem}, index=idx)
    df.attrs["conv_window"] = (idx[i2], idx[i3 - 1])
    df.attrs["regime_slices"] = {"premium": (idx[0], idx[i1 - 1]),
                                 "discount": (idx[i1], idx[i2 - 1]),
                                 "etf": (idx[i3], idx[-1])}
    return df
