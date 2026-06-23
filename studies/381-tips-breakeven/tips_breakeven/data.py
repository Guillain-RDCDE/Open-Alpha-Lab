"""Data layer for Study 381 — TIPS-Breakeven (breakeven proxy + asset tape).

Two sources, both offline-friendly:

* **Real tape.** Daily adjusted closes for an inflation-protected Treasury ETF (``TIP``), a
  duration-matched nominal Treasury ETF (``IEF``), equities (``SPY``) and gold (``GLD``)
  (yfinance, no key), cached under ``_cache/proxy_prices.csv`` (wide CSV, one column per
  ticker). From TIP and IEF we build a **breakeven proxy**: ``be = log(TIP / IEF)``. When the
  market's inflation expectations rise, the inflation-protected leg outperforms the nominal
  leg, so ``be`` rises with the true breakeven inflation rate (FRED ``T10YIE``). The true
  series is not reliably fetchable in this environment, so this ratio is an explicit,
  transparent proxy — named as such everywhere.

* **Synthetic.** A deterministic, fixed-seed generator that produces a controllable breakeven
  series and an asset price whose forward return depends on the breakeven signal by a
  **planted edge knob** ``edge``. It is the positive control: with ``edge=0`` the inference
  must NOT manufacture a significant predictive slope; with a large ``edge`` the HAC *t* must
  light up. This proves the engine is faithful and that the real-tape null is a true null,
  not an underpowered one.

Pure numpy + pandas + stdlib for the offline path. ``fetch_assets`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "proxy_prices.csv")

# The four ETFs: the breakeven proxy legs (TIP, IEF) and the assets we ask the signal to time.
TICKERS = ["TIP", "IEF", "SPY", "GLD"]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_assets(start: str = "2003-01-01", end: str | None = None,
                 path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the ETF set via yfinance and cache a wide adjusted-close CSV.

    Network-only; used once to build ``_cache/proxy_prices.csv``. Never imported by the
    offline notebook cells. Restricts to the common overlapping window (GLD lists late 2004).
    """
    import yfinance as yf

    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna()                       # common window for all four ETFs
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide adjusted-close frame (index = date, columns = the ETFs)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def breakeven_proxy(prices: pd.DataFrame) -> pd.Series:
    """The breakeven proxy: ``log(TIP / IEF)`` on the daily tape.

    Rises with the market-implied inflation forecast (inflation-protected over nominal). An
    explicit PROXY for FRED ``T10YIE`` — not the literal nominal-minus-real yield, but the same
    object's return analogue, built from public adjusted closes.
    """
    return np.log(prices["TIP"] / prices["IEF"]).rename("be")


def to_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end frame of the ETFs (the macro signal cadence)."""
    return prices.resample("ME").last()


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Convenience: cached prices -> month-end ETF frame in one call (signal cadence)."""
    return to_monthly(load_prices(path))


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_macro(n_months: int = 240, edge: float = 0.0, seed: int = 381,
                    mu: float = 0.006, sig: float = 0.045,
                    be_sig: float = 0.012) -> pd.DataFrame:
    """Deterministic breakeven + asset series with a PLANTED predictive edge.

    Builds a mean-reverting (AR(1)) breakeven proxy ``be`` and a single risk asset whose
    monthly return is ``mu + edge * z + noise``, where ``z`` is the standardised breakeven
    *level* at the start of the month. ``edge`` is the planted slope of next-month return on
    the breakeven z-score:

      * ``edge = 0``  -> the asset return is independent of the signal; the HAC predictive *t*
        must stay inside the noise (no false positive).
      * large ``edge`` -> the planted predictability must show up as ``|t| >= 2``.

    The columns mirror the real frame enough to flow through the same strategy code: ``be`` is
    the breakeven proxy, ``ASSET`` is the timed asset (and we also expose ``TIP``/``IEF`` so
    ``breakeven_proxy`` round-trips if needed). The date index is a decorative monthly
    ``period_range`` (kept far from pandas' Timestamp bounds).
    """
    rng = np.random.default_rng(seed)
    # AR(1) breakeven proxy around 0, persistence 0.92
    be = np.empty(n_months)
    be[0] = 0.0
    for t in range(1, n_months):
        be[t] = 0.92 * be[t - 1] + rng.normal(0, be_sig)
    z = (be - be.mean()) / be.std()
    # asset monthly returns: planted dependence on the *previous* month's z (no look-ahead)
    ret = np.empty(n_months)
    ret[0] = mu
    ret[1:] = mu + edge * z[:-1] + rng.normal(0, sig, size=n_months - 1)
    price = 100.0 * np.cumprod(1.0 + ret)
    # decorative monthly index well within Timestamp bounds
    idx = pd.period_range("1990-01", periods=n_months, freq="M").to_timestamp(how="end")
    # synthesise TIP/IEF legs whose log-ratio equals `be` (so breakeven_proxy round-trips)
    ief = 100.0 * np.cumprod(1.0 + rng.normal(0.002, 0.01, size=n_months))
    tip = ief * np.exp(be)
    return pd.DataFrame({"be": be, "z": z, "ASSET": price, "TIP": tip, "IEF": ief}, index=idx)
