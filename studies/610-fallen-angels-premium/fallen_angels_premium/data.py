"""Data layer for Study 610 — Fallen-Angels-Premium.

Two sources, both offline-friendly once cached:

* **Real tape.** Daily auto-adjusted (total-return) closes for the live fallen-angel
  vehicles and their controls (yfinance, no key):

  - ``ANGL`` — VanEck Fallen Angel High Yield Bond ETF (inception 2012-04; the claim's
    main live vehicle: holds only bonds *downgraded* from investment-grade to junk),
  - ``FALN`` — iShares Fallen Angels USD Bond ETF (inception 2016-06; the confirmation
    vehicle from an independent issuer/index),
  - ``HYG`` / ``JNK`` — the two broad high-yield benchmarks the premium is measured over,
  - ``IEF`` — 7-10y Treasuries, the *duration* control factor (fallen angels carry longer
    duration than broad HY, so a rate factor is the honesty check),
  - ``LQD`` — investment-grade credit, an alternative quality/duration control,
  - ``BIL`` — 1-3m T-bill ETF, the tradable risk-free proxy (excess-vs-excess races).

  Cached wide as ``fap_prices.csv``; everything downstream runs cache-first.

* **Synthetic world.** A deterministic, fixed-seed generator producing monthly returns
  for a "broad HY benchmark", a "rate factor" and a "fallen-angel index" built as
  ``beta_hy * HY + beta_rate * RATE + alpha + noise`` with a **TUNABLE planted alpha**
  (knob ``alpha_bps``) and a null (``alpha_bps=0``). It is the machinery proof: the
  HAC-alpha pipeline must recover a planted alpha and must NOT manufacture one from
  zero. Never cited as market evidence.

Pure numpy + pandas + stdlib on the offline path. ``fetch_prices`` (network) is used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "fap_prices.csv")

# The full ticker set (fallen-angel vehicles + benchmarks + factor controls).
TICKERS = ["ANGL", "FALN", "HYG", "JNK", "IEF", "LQD", "BIL"]

# Frozen as-of for the headline run: the last complete calendar month before the study
# date (2026-07-03). Monthly stats never include a partial month.
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_prices(start: str = "2007-01-01", end: str | None = None,
                 path: str = PRICES_CACHE, retries: int = 3) -> pd.DataFrame:
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
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
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
# Synthetic world (machinery proof — planted alpha + null)
# --------------------------------------------------------------------------- #
def synthetic_world(alpha_bps: float = 0.0, n_months: int = 170, seed: int = 610,
                    beta_hy: float = 1.05, beta_rate: float = 0.25,
                    hy_vol: float = 0.025, rate_vol: float = 0.015,
                    idio_vol: float = 0.006) -> pd.DataFrame:
    """Deterministic monthly-return world with a PLANTED fallen-angel alpha.

    Columns: ``HY`` (broad high-yield benchmark), ``RATE`` (duration factor, an
    IEF-like Treasury return) and ``FA`` (the fallen-angel index), built as

        FA_t = alpha + beta_hy * HY_t + beta_rate * RATE_t + eps_t .

    ``alpha_bps`` is the TUNABLE planted monthly alpha (bps/month); ``alpha_bps=0``
    is the null — the HAC-alpha pipeline must not manufacture significance from it.
    The index is a monthly period_range converted to timestamps (span << 250 years,
    safely inside the pandas ns-Timestamp window).
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2012-05", periods=n_months, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()
    hy = rng.normal(0.004, hy_vol, size=n_months)
    rate = rng.normal(0.002, rate_vol, size=n_months)
    eps = rng.normal(0.0, idio_vol, size=n_months)
    fa = alpha_bps / 1e4 + beta_hy * hy + beta_rate * rate + eps
    return pd.DataFrame({"HY": hy, "RATE": rate, "FA": fa}, index=idx)
