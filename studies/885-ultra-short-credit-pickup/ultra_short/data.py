"""Data layer for Study 885 — Ultra-Short Credit Pickup.

Two sources, both offline-friendly once cached.

* **Real tape.** Daily auto-adjusted (total-return, net-of-fee) closes for the
  liquid ultra-short vehicles and their cash benchmarks (yfinance, no key):

  - ``JPST`` — JPMorgan Ultra-Short Income ETF (inception 2017-05; active,
    ~AA-/A ultra-short IG corporates & ABS, ~0.25–0.9y duration),
  - ``ICSH`` — BlackRock (iShares) Ultra Short-Term Bond Active ETF (2013-12),
  - ``MINT`` — PIMCO Enhanced Short Maturity Active ETF (2009-11; the granddad
    of the sleeve, the longest live tape),
  - ``BIL``  — SPDR Bloomberg 1-3 Month T-Bill ETF (2007-05; the *cash* leg /
    tradable risk-free — every Sharpe race is excess-of-BIL),
  - ``SHV``  — iShares Short Treasury Bond ETF (2007-01; ~0-1y Treasuries, the
    second cash benchmark, a hair more duration than BIL, still ~zero credit).

  Cached wide as ``usc_prices.csv``; everything downstream runs cache-first.

* **Synthetic world.** A deterministic, fixed-seed generator producing daily
  returns for a "cash" leg and an "ultra-short credit" leg built as
  ``r_credit = r_cash + carry/252 + beta_credit*credit_factor + noise`` with a
  **TUNABLE planted carry** (knob ``pickup_bps_yr``) and a null (``0``). It is the
  machinery proof: the excess-Sharpe / HAC-mean pipeline must recover a planted
  pickup and must NOT manufacture one from zero. Never cited as market evidence.

Pure numpy + pandas + stdlib on the offline path. ``fetch_prices`` (network) is
used once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "usc_prices.csv")

# Ultra-short IG credit vehicles + the two cash benchmarks (BIL/SHV).
TICKERS = ["JPST", "ICSH", "MINT", "BIL", "SHV"]

# The ultra-short credit sleeve (excludes the cash benchmarks).
CREDIT = ["JPST", "ICSH", "MINT"]
CASH = ["BIL", "SHV"]

# Frozen as-of for the headline run: the last complete calendar month before the
# study date. Daily stats never include the partial current month.
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_prices(start: str = "2009-01-01", end: str | None = None,
                 path: str = PRICES_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download the ETF panel (auto-adjusted = total-return closes) and cache it.

    Network-only; used once to build the cache. Guards yfinance flakiness with
    simple retries. Writes a wide CSV (index = date, columns = tickers).
    """
    import yfinance as yf

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                              progress=False)["Close"]
            if raw is not None and len(raw) > 0 and not raw.dropna(how="all").empty:
                break
        except Exception:
            time.sleep(2.0)
    if raw is None:
        raise RuntimeError("yfinance returned no data for the ultra-short panel")
    prices = raw.dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prices.to_csv(path)
    return prices


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE, asof: str | None = AS_OF) -> pd.DataFrame:
    """Cached wide total-return close frame, sliced to the frozen as-of."""
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if asof is not None:
        px = px[px.index <= pd.Timestamp(asof)]
    return px


# --------------------------------------------------------------------------- #
# Synthetic world (machinery proof — planted pickup + null)
# --------------------------------------------------------------------------- #
def synthetic_world(pickup_bps_yr: float = 0.0, n_days: int = 2000, seed: int = 885,
                    beta_credit: float = 1.0, cash_vol_yr: float = 0.004,
                    credit_factor_vol_yr: float = 0.010, idio_vol_yr: float = 0.006,
                    start: str = "2013-01-02") -> pd.DataFrame:
    """Deterministic daily-return world with a PLANTED ultra-short credit pickup.

    Columns: ``CASH`` (a bill-like leg), ``CREDIT_FACTOR`` (a small IG credit
    factor return) and ``CREDIT`` (the ultra-short credit sleeve), built as

        CREDIT_t = CASH_t + pickup/252 + beta_credit * CREDIT_FACTOR_t + eps_t .

    ``pickup_bps_yr`` is the TUNABLE planted annual carry (bps/yr); ``0`` is the
    null — the excess-Sharpe / HAC-mean pipeline must not manufacture a pickup from
    it. A business-day index (``n_days`` <= 10000) stays safely inside the pandas
    ns-Timestamp window; no large monthly ``period_range.to_timestamp`` here.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    # A gently drifting cash rate (bills earn ~2-5%/yr, near-zero vol day to day).
    cash_level = 0.03 / 252.0
    cash = rng.normal(cash_level, cash_vol_yr / np.sqrt(252.0), size=n_days)
    cfac = rng.normal(0.0, credit_factor_vol_yr / np.sqrt(252.0), size=n_days)
    eps = rng.normal(0.0, idio_vol_yr / np.sqrt(252.0), size=n_days)
    credit = cash + pickup_bps_yr / 1e4 / 252.0 + beta_credit * cfac + eps
    return pd.DataFrame(
        {"CASH": cash, "CREDIT_FACTOR": cfac, "CREDIT": credit}, index=idx
    )
