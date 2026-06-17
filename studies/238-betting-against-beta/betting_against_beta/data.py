"""Data layer for Study 238 (Betting-Against-Beta).

Two tapes, one schema -- a (date × ticker) daily returns frame plus SPY as the market proxy:

- ``synthetic_panel`` -- a *deterministic, offline* generator. A tunable
  ``bab_premium`` knob controls how strongly low-beta stocks outperform high-beta
  stocks (after beta-adjustment). ``bab_premium = 0`` is the null hypothesis.
  Tests never touch the network.

- ``fetch_panel`` -- real daily price data from yfinance, cached in the study's own
  ``_cache/`` dir (gitignored). Returns ``(prices, spy_prices)`` or empty DataFrames
  if the cache is absent (e.g., on CI).

No look-ahead: rolling betas are computed on a trailing 252-day window (one calendar
year). The BAB portfolio is rebalanced monthly, using only past-window data.

The panel is **survivorship-biased** (current S&P 500 membership projected backwards):
positive results should be read as **upper-bound** estimates.

Universe: ~100 large-cap S&P 500 names (a stable subset drawn from
``_cache/sp500_symbols.json``), plus SPY as the market proxy.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")
SHARED_CACHE = os.path.join(REPO_ROOT, "_cache")

# Fixed large-cap universe (stable, representative subset of S&P 500)
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM",
    "UNH", "XOM", "TSLA", "PG", "MA", "JNJ", "HD", "COST", "ABBV", "MRK",
    "CVX", "BAC", "CRM", "NFLX", "AMD", "PEP", "TMO", "ORCL", "ACN", "WMT",
    "LIN", "MCD", "CSCO", "ABT", "DHR", "VZ", "TXN", "ADBE", "PM", "NEE",
    "RTX", "INTC", "IBM", "GE", "CAT", "HON", "UPS", "SPGI", "LOW", "INTU",
    "GS", "AXP", "BKNG", "ISRG", "MDT", "PLD", "CB", "ADI", "ADP", "GILD",
    "SBUX", "MMC", "TJX", "EOG", "SLB", "BDX", "C", "WFC", "USB", "PNC",
    "SO", "DUK", "SRE", "AEP", "D", "EXC", "XEL", "PCG", "ED", "WEC",
    "T", "CMCSA", "DIS", "PARA", "WBD", "NWSA", "OMC", "IPG", "PH", "EMR",
    "GD", "LMT", "NOC", "BA", "TDG", "HII", "L", "MO", "PM", "BTI",
]
# Deduplicate while preserving order
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    bab_premium: float  # how much low-beta outperforms high-beta after beta-adjustment

    @property
    def has_premium(self) -> bool:
        return self.bab_premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 100,
    n_days: int = 3000,
    bab_premium: float = 0.05,
    market_vol: float = 0.16,
    idio_vol: float = 0.25,
    seed: int = 238,
) -> tuple[pd.DataFrame, pd.Series, WorldTruth]:
    """A reproducible daily returns panel with a tunable BAB premium.

    Stocks have betas drawn from a log-normal centred around 1.0 (reflecting the empirical
    distribution: betas are right-skewed with median near 1). The daily excess return for
    each stock follows a market model:

        r_i = beta_i * r_mkt + alpha_i + epsilon_i

    where ``alpha_i = bab_premium * (1 - beta_i)`` (low-beta stocks have positive alpha,
    high-beta stocks have negative alpha -- the BAB effect). ``bab_premium = 0`` is the null.

    Returns ``(stock_rets, mkt_rets, truth)`` -- all as daily decimal returns.
    The market series is SPY-like (mu=10%/yr, vol=market_vol/yr).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days)

    # True betas: log-normal to ensure positivity and right-skew
    true_betas = rng.lognormal(mean=0.0, sigma=0.35, size=n_stocks)
    tickers = [f"S{j:03d}" for j in range(n_stocks)]

    # Market returns: iid normal (daily)
    mkt_daily = rng.normal(0.10 / 252, market_vol / np.sqrt(252), size=n_days)
    mkt = pd.Series(mkt_daily, index=dates, name="SPY")

    # Idiosyncratic returns
    idio = rng.normal(0.0, idio_vol / np.sqrt(252), size=(n_days, n_stocks))

    # BAB alpha: positive for low-beta, negative for high-beta
    alphas = bab_premium * (1.0 - true_betas) / 252.0

    # r_i = beta_i * r_mkt + alpha_i + epsilon_i
    stock_rets = mkt_daily[:, None] * true_betas[None, :] + alphas[None, :] + idio

    stock_df = pd.DataFrame(stock_rets, index=dates, columns=tickers)
    return stock_df, mkt, WorldTruth(bab_premium)


# ---------------------------------------------------------------------------
# Real panel -- yfinance, study-local cache
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    start: str = "2010-01-01",
    end: str = "2025-12-31",
) -> tuple[pd.DataFrame, pd.Series]:
    """Daily adjusted-close prices for the large-cap universe + SPY.

    Cache-first: reads ``<cache_dir>/bab_prices.parquet`` and
    ``<cache_dir>/bab_spy.parquet`` if present. On cache miss, fetches from
    yfinance and writes the parquets. Returns ``(prices, spy_prices)`` or
    ``(pd.DataFrame(), pd.Series())`` if both cache and network fail.

    Prices are adjusted-close (yfinance auto_adjust=True). The caller computes
    returns via ``prices.pct_change().dropna()``.
    """
    price_path = os.path.join(cache_dir, "bab_prices.parquet")
    spy_path = os.path.join(cache_dir, "bab_spy.parquet")

    if os.path.exists(price_path) and os.path.exists(spy_path):
        prices = pd.read_parquet(price_path)
        spy = pd.read_parquet(spy_path).squeeze()
        spy.name = "SPY"
        return prices, spy

    # Try fetching from yfinance
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
            return pd.DataFrame(), pd.Series(dtype=float, name="SPY")

        spy = raw["SPY"].dropna()
        prices = raw.drop(columns=["SPY"], errors="ignore").dropna(how="all")

        # Drop tickers that are almost entirely missing (< 20% coverage)
        coverage = prices.notna().mean()
        prices = prices.loc[:, coverage >= 0.20]

        os.makedirs(cache_dir, exist_ok=True)
        prices.to_parquet(price_path)
        spy.to_frame("SPY").to_parquet(spy_path)
        return prices, spy

    except Exception:  # noqa: BLE001
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
