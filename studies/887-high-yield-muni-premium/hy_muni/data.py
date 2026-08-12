"""Data layer for Study 887 — High-Yield Muni Premium.

The claim under test: **high-yield municipal bonds pay a fat, tax-advantaged credit
spread over investment-grade munis** — a paid credit premium in a tax-favored wrapper.
The live test reads the actual packaged vehicles (yfinance, total-return closes):

  * ``HYD`` — VanEck High Yield Muni ETF (inception 2009-02; the HY-muni leg),
  * ``MUB`` — iShares National Muni Bond ETF (inception 2007-09; the IG-muni benchmark
    the credit premium is measured over),
  * ``TFI`` — SPDR Nuveen Bloomberg Municipal Bond ETF (an alternative IG-muni control),
  * ``HYG`` — iShares iBoxx $ HY Corporate Bond ETF (the *taxable* high-yield yardstick
    for the tax-equivalent-yield comparison),
  * ``BIL`` — SPDR 1-3m T-Bill ETF, the tradable risk-free proxy (excess-vs-excess races).

Two flavours of the same closes are cached:

  * ``hym_prices.csv`` — **auto_adjust=True total-return** closes (the headline tape).
  * ``hym_prices_raw.csv`` — **auto_adjust=False price-only** closes. The *income*
    (distribution) return is recovered as ``total_return - price_return`` per month; it
    is the only thing that carries the tax story (muni coupons are federally tax-exempt),
    so it must be measured, not assumed.

Everything downstream runs cache-first and offline. ``fetch`` (network, yfinance) runs
once to build the two caches and is never imported by the notebooks' offline cells.

Synthetic world: a deterministic, fixed-seed monthly generator with a TUNABLE planted
premium (knob ``premium_annual``) plus a shared muni-market factor and idiosyncratic
noise — the positive control (and, at ``premium_annual = 0``, the null that must NOT
fire). The decorative index is a monthly ``period_range`` (span well under the 250-year
ns-Timestamp horizon).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "hym_prices.csv")       # total-return closes
PRICES_RAW_CACHE = os.path.join(CACHE_DIR, "hym_prices_raw.csv")  # price-only closes

# HY-muni + IG-muni benchmarks + taxable-HY yardstick + tradable risk-free.
TICKERS = ["HYD", "MUB", "TFI", "HYG", "BIL"]

# Frozen as-of for the headline run: the last complete calendar month before the study
# date. Monthly stats never include a partial month.
AS_OF = "2026-06-30"

# Top-bracket federal marginal rate on ordinary income used for the tax-equivalent
# comparison: 37% statutory + 3.8% net-investment-income (Medicare) surtax = 40.8%.
# Municipal-bond *income* is exempt from this; taxable-HY (HYG) income is not.
TOP_MARGINAL_RATE = 0.408

__all__ = [
    "TICKERS", "AS_OF", "CACHE_DIR", "PRICES_CACHE", "PRICES_RAW_CACHE",
    "TOP_MARGINAL_RATE",
    "fetch", "have_real", "load_prices", "load_price_only", "synthetic_world",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def _download(tickers, start, end, auto_adjust, retries):
    import yfinance as yf

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=auto_adjust,
                              progress=False)["Close"]
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    return raw.dropna(how="all").sort_index()


def fetch(start: str = "2007-01-01", end: str | None = None, retries: int = 4) -> None:
    """Download both the total-return and price-only ETF panels; cache them.

    Network-only; used once to build the caches. ``auto_adjust=True`` gives the
    total-return (net-of-fee) tape; a second ``auto_adjust=False`` pull gives the
    price-only tape used to back out the (tax-relevant) distribution/income return.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    tr = _download(TICKERS, start, end, True, retries)
    tr.to_csv(PRICES_CACHE)
    pr = _download(TICKERS, start, end, False, retries)
    pr.to_csv(PRICES_RAW_CACHE)


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE, asof: str | None = AS_OF) -> pd.DataFrame:
    """Cached wide **total-return** close frame, sliced to the frozen as-of."""
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if asof is not None:
        px = px[px.index <= pd.Timestamp(asof)]
    return px


def load_price_only(path: str = PRICES_RAW_CACHE, asof: str | None = AS_OF) -> pd.DataFrame:
    """Cached wide **price-only** close frame, sliced to the frozen as-of.

    Used with :func:`load_prices` to back out the monthly income (distribution) return
    as ``total_return - price_return`` — the coupon leg that carries the tax story.
    """
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if asof is not None:
        px = px[px.index <= pd.Timestamp(asof)]
    return px


# --------------------------------------------------------------------------- #
# Synthetic world (machinery proof — planted premium + null)
# --------------------------------------------------------------------------- #
def synthetic_world(premium_annual: float = 0.0, n_months: int = 200, seed: int = 887,
                    muni_vol: float = 0.018, beta_hy: float = 1.15,
                    idio_vol: float = 0.006) -> pd.DataFrame:
    """Deterministic monthly world with a PLANTED HY-muni credit premium.

    A shared IG-muni market return ``MUB`` ~ N(0.3%, muni_vol); the HY-muni leg is

        HYD_t = premium_annual/12 + beta_hy * MUB_t + eps_t ,

    so ``HYD - MUB`` has mean ``premium_annual/12`` plus a credit-beta tilt and noise —
    exactly the excess-spread the estimator targets. ``premium_annual = 0`` is the null:
    the HAC-t / bootstrap pipeline must NOT manufacture a premium from it. Decorative
    monthly index built with ``period_range`` (span << the 250-year ns-Timestamp cap).
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2009-03", periods=n_months, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()
    mub = rng.normal(0.003, muni_vol, size=n_months)
    eps = rng.normal(0.0, idio_vol, size=n_months)
    hyd = premium_annual / 12.0 + beta_hy * mub + eps
    bil = np.full(n_months, 0.0010)  # ~1.2%/yr flat cash leg
    return pd.DataFrame({"HYD": hyd, "MUB": mub, "BIL": bil}, index=idx)
