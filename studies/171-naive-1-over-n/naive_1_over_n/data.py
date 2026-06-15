"""Data layer for Study 171 (Naive-1-Over-N).

Two tapes, one purpose — test whether 1/N equal-weight beats Markowitz optimisers
out of sample:

- ``synthetic_returns`` — a *deterministic, offline* generator. We simulate a
  multivariate normal return process with a planted correlation structure and a
  controllable "signal-to-noise" knob (``signal_strength``). At ``signal_strength=0``
  the true expected returns are all equal — the 1/N portfolio is truly optimal and the
  optimiser's estimation error cannot add value. At ``signal_strength > 0`` the true
  Sharpe ratios differ, so an optimiser with a *perfect* estimate of the mean would win —
  but in practice it must estimate from noisy finite samples. This knob lets us dial
  from "optimiser clearly wins" (oracle) to "estimation error dominates" (reality).
- ``load_real`` — the eleven SPDR sector ETFs (XLB, XLC, XLE, XLF, XLI, XLK, XLP,
  XLRE, XLU, XLV, XLY) from Yahoo Finance via yfinance; cached as parquet.

No look-ahead is baked in here — signal formation and the rolling-window estimation
always happens on the in-sample slice; positions enter at the next period's open.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# ---------------------------------------------------------------------------
# The eleven SPDR sector ETFs — the canonical universe for this study.
# Launch dates: XLC (2018-06-19) is the youngest; we start after its first
# full month so all ETFs have a live history.  Source: https://www.sectorspdr.com
# ---------------------------------------------------------------------------
SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
N_ASSETS = len(SECTOR_ETFS)
UNIVERSE_START = "2018-07-01"  # first full month after XLC launched


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_returns(
    n_periods: int = 504,
    n_assets: int = N_ASSETS,
    signal_strength: float = 0.0,
    vol: float = 0.015,
    correlation: float = 0.40,
    seed: int = 171,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible multivariate-normal return matrix with a known structure.

    The covariance matrix is a constant-correlation model:
        Sigma_ij = vol^2 * correlation  (i != j)
        Sigma_ii = vol^2

    Expected returns are:
        mu_i = 0  when signal_strength = 0  (all assets equal — 1/N is optimal)
        mu_i = signal_strength * (i / n_assets)  otherwise (true alpha gradient)

    ``signal_strength = 0`` is the null: the optimiser has nothing to find, and
    estimation error from a finite sample is pure noise. This is the study's null
    in a bottle — 1/N must tie or beat the optimiser here.

    Returns ``(returns_df, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)

    # Build constant-correlation covariance matrix
    cov = np.full((n_assets, n_assets), vol**2 * correlation)
    np.fill_diagonal(cov, vol**2)

    # True expected returns (per period)
    mu = np.zeros(n_assets)
    if signal_strength != 0.0:
        mu = signal_strength * np.arange(n_assets) / max(n_assets - 1, 1)

    # Draw n_periods of returns
    raw = rng.multivariate_normal(mu, cov, size=n_periods)

    index = pd.date_range("2015-01-02", periods=n_periods, freq="B")
    cols = [f"A{i+1:02d}" for i in range(n_assets)]
    df = pd.DataFrame(raw, index=index, columns=cols)
    df.index.name = "date"

    truth = {
        "n_periods": n_periods,
        "n_assets": n_assets,
        "signal_strength": signal_strength,
        "vol": vol,
        "correlation": correlation,
        "seed": seed,
        "true_mu": mu.tolist(),
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — SPDR sector ETFs via Yahoo Finance; cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "sectors_daily.parquet")


def load_real(
    tickers: list[str] | None = None,
    start: str = UNIVERSE_START,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Daily total-return prices for the SPDR sector ETFs; cache-only unless fetch=True.

    Returns a date-indexed DataFrame of adjusted close prices, one column per ETF.
    The first row is the earliest available date across all tickers (inner join:
    ``start`` date onward, drop any date where any ticker has NaN).  Network is
    touched only on an explicit ``fetch=True`` (result cached as parquet).
    """
    if tickers is None:
        tickers = SECTOR_ETFS

    path = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached sector prices at {path}. "
                f"Call load_real(fetch=True) once to populate the cache."
            )
        prices = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when going to the network

        raw = yf.download(
            tickers,
            start=start,
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            raise RuntimeError("yfinance returned no data for sector ETFs")
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
        prices = prices.dropna(how="all")
        os.makedirs(cache_dir, exist_ok=True)
        prices.to_parquet(path)

    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.index.name = "date"
    # Inner join: only rows where ALL tickers are present
    prices = prices[tickers].dropna()
    prices = prices[prices.index >= pd.Timestamp(start)]
    return prices


def fingerprint(prices: pd.DataFrame) -> str:
    """A short content fingerprint of the price matrix, for the as-of stamp."""
    arr = np.ascontiguousarray(prices.to_numpy().ravel())
    h = hashlib.sha1(arr.tobytes())
    return h.hexdigest()[:12]
