"""Data layer for Study 501 (Idiosyncratic-Volatility).

Two tapes, one schema -- a (date x ticker) daily-price frame plus SPY as the market proxy:

- ``synthetic_panel`` -- a *deterministic, offline* generator. A tunable ``ivol_premium``
  knob controls how strongly LOW-idiosyncratic-vol stocks out-earn HIGH-idiosyncratic-vol
  stocks (the IVOL puzzle is a *negative* IVOL-return relation, so a positive ``ivol_premium``
  plants the puzzle: low-IVOL alpha > 0, high-IVOL alpha < 0). ``ivol_premium = 0`` is the
  null hypothesis. Tests never touch the network.

- ``fetch_panel`` -- real daily price data from yfinance, cached in the study's own
  ``_cache/`` dir (gitignored). Returns ``(prices, spy_prices)`` or empty frames if the cache
  is absent (e.g. on CI).

No look-ahead: idiosyncratic vol is the std of the CAPM residual on a trailing 252-day
window; the long-short is rebalanced monthly using only past-window data, and the holding
period starts one trading day AFTER the signal is known (a single, documented execution lag).

The panel is **survivorship-biased** (current S&P 500 membership projected backwards):
positive results should be read as **upper-bound** estimates.

Universe: ~50 large-cap S&P 500 names (a stable subset), plus SPY as the market proxy.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

# Fixed large-cap universe (stable, representative survivor subset of the S&P 500).
# 50 names spanning sectors so the cross-sectional IVOL dispersion is meaningful.
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "AVGO", "JPM", "LLY", "XOM",
    "UNH", "TSLA", "PG", "MA", "JNJ", "HD", "COST", "ABBV", "MRK", "CVX",
    "BAC", "CRM", "NFLX", "AMD", "PEP", "TMO", "ORCL", "ACN", "WMT", "MCD",
    "CSCO", "ABT", "DHR", "VZ", "TXN", "ADBE", "NEE", "RTX", "INTC", "IBM",
    "CAT", "HON", "GS", "AXP", "ISRG", "GILD", "SBUX", "TJX", "C", "WFC",
]
# Deduplicate while preserving order.
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    ivol_premium: float  # how much low-IVOL out-earns high-IVOL (the planted puzzle)

    @property
    def has_premium(self) -> bool:
        return self.ivol_premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 50,
    n_days: int = 2800,
    ivol_premium: float = 0.06,
    market_vol: float = 0.16,
    seed: int = 501,
) -> tuple[pd.DataFrame, pd.Series, WorldTruth]:
    """A reproducible daily price panel with a tunable IVOL puzzle.

    Each stock follows a market model with a *stable* idiosyncratic-vol level::

        r_i = beta_i * r_mkt + alpha_i + epsilon_i,   eps_i ~ N(0, ivol_i / sqrt(252))

    ``ivol_i`` (the stock's residual vol) is drawn from a wide spread so the cross-section
    has real IVOL dispersion. The planted puzzle: ``alpha_i = -ivol_premium * (ivol_i - mean_ivol)``
    -- high-IVOL stocks carry NEGATIVE alpha and low-IVOL stocks POSITIVE alpha, so a
    low-minus-high long-short earns ``> 0``. ``ivol_premium = 0`` is the null.

    Returns ``(prices, spy_prices, truth)`` -- prices are NAV-style levels (start 100).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-03", periods=n_days)

    betas = rng.lognormal(mean=0.0, sigma=0.30, size=n_stocks)
    # Idiosyncratic vol levels: annualised, spread 12%..55% across the cross-section.
    ivol = rng.uniform(0.12, 0.55, size=n_stocks)
    mean_ivol = float(ivol.mean())
    tickers = [f"S{j:03d}" for j in range(n_stocks)]

    mkt_daily = rng.normal(0.09 / 252, market_vol / np.sqrt(252), size=n_days)
    mkt = pd.Series(mkt_daily, index=dates, name="SPY")

    # Idiosyncratic shocks scaled per-stock by its IVOL level.
    eps = rng.standard_normal((n_days, n_stocks)) * (ivol[None, :] / np.sqrt(252))

    # The planted puzzle: alpha decreasing in IVOL (the negative IVOL-return relation).
    alphas = (-ivol_premium * (ivol - mean_ivol)) / 252.0

    stock_rets = mkt_daily[:, None] * betas[None, :] + alphas[None, :] + eps
    stock_prices = pd.DataFrame((1.0 + stock_rets).cumprod(axis=0) * 100.0,
                                index=dates, columns=tickers)
    spy_prices = pd.Series((1.0 + mkt_daily).cumprod() * 100.0, index=dates, name="SPY")
    return stock_prices, spy_prices, WorldTruth(ivol_premium)


# ---------------------------------------------------------------------------
# Real panel -- yfinance, study-local cache
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    start: str = "2012-01-01",
    end: str = "2026-01-01",
    retries: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Daily adjusted-close prices for the large-cap universe + SPY.

    Cache-first: reads ``<cache_dir>/ivol_prices.parquet`` and ``<cache_dir>/ivol_spy.parquet``
    if present. On cache miss, fetches from yfinance (with retries) and writes the parquets.
    Returns ``(prices, spy_prices)`` or ``(pd.DataFrame(), pd.Series())`` if both cache and
    network fail.

    Prices are adjusted-close (yfinance ``auto_adjust=True``). The caller computes daily
    returns via ``prices.pct_change()``.
    """
    price_path = os.path.join(cache_dir, "ivol_prices.parquet")
    spy_path = os.path.join(cache_dir, "ivol_spy.parquet")

    if os.path.exists(price_path) and os.path.exists(spy_path):
        prices = pd.read_parquet(price_path)
        spy = pd.read_parquet(spy_path).squeeze("columns")
        spy.name = "SPY"
        return prices, spy

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            import yfinance as yf

            all_tickers = list(UNIVERSE) + ["SPY"]
            raw = yf.download(
                all_tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )["Close"]
            if raw.empty:
                raise RuntimeError("yfinance returned an empty frame")

            spy = raw["SPY"].dropna()
            prices = raw.drop(columns=["SPY"], errors="ignore").dropna(how="all")

            # Drop tickers with poor coverage (< 80% of the SPY span).
            coverage = prices.notna().mean()
            prices = prices.loc[:, coverage >= 0.80]
            prices = prices.dropna(how="any")  # common-history rectangle for clean sorts

            os.makedirs(cache_dir, exist_ok=True)
            prices.to_parquet(price_path)
            spy.to_frame("SPY").to_parquet(spy_path)
            return prices, spy
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2.0 * (attempt + 1))

    print(f"fetch_panel: network fetch failed ({last_exc}); returning empty frames.")
    return pd.DataFrame(), pd.Series(dtype=float, name="SPY")


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    """A short content fingerprint of a panel (for the as-of stamp in docs/results.md)."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    arr = np.ascontiguousarray(df.fillna(0).to_numpy())
    h = hashlib.sha1(arr.tobytes())
    return h.hexdigest()[:12]
