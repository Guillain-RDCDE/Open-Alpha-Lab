"""Data layer for Study 509 (Intermediate-Momentum).

One schema -- a (date x ticker) daily adjusted-close price frame plus SPY as the
market proxy -- served by two paths:

- ``synthetic_panel`` -- a *deterministic, offline* generator. A tunable ``mom_premium``
  knob controls how strongly past *intermediate-horizon* (t-12..t-7) winners outperform
  losers in the next month. ``mom_premium = 0`` is the null hypothesis. Tests never touch
  the network.

- ``fetch_panel`` -- real daily price data from yfinance, cached in the study's own
  ``_cache/`` dir (gitignored). Returns ``(prices, spy_prices)`` or empty frames if the
  cache is absent (e.g., on CI).

No look-ahead: the momentum signal at rebalance date *t* uses only returns strictly
before *t*, and the holding period starts one trading day after *t* (one execution lag).

The panel is **survivorship-biased** (current large-cap names projected backwards):
positive results should be read as **upper-bound** estimates.

Universe: ~48 large-cap survivor names, plus SPY as the market proxy.
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

# Fixed large-cap survivor universe (stable, representative; all trading in 2026).
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "JPM", "LLY", "AVGO", "XOM",
    "UNH", "JNJ", "PG", "MA", "HD", "COST", "MRK", "ABBV", "CVX", "PEP",
    "KO", "WMT", "MCD", "CSCO", "ACN", "TMO", "ABT", "DHR", "TXN", "ORCL",
    "NKE", "HON", "UNP", "LIN", "CAT", "GS", "AXP", "LOW", "INTC", "IBM",
    "GE", "BA", "DIS", "VZ", "T", "MMM", "CL", "AMGN",
]
# Deduplicate while preserving order
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    mom_premium: float  # how strongly intermediate-horizon winners beat losers next month

    @property
    def has_premium(self) -> bool:
        return self.mom_premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 48,
    n_days: int = 3000,
    mom_premium: float = 0.06,
    market_vol: float = 0.16,
    idio_vol: float = 0.22,
    seed: int = 509,
) -> tuple[pd.DataFrame, pd.Series, WorldTruth]:
    """A reproducible daily price panel with a tunable *intermediate-horizon* momentum premium.

    The planted effect lives ONLY in the intermediate window (t-12..t-7 months): a stock
    whose intermediate past return is high earns a positive expected return next month,
    proportional to ``mom_premium``. The recent window (t-6..t-1) carries no planted edge,
    so a faithful engine should find the drift in the intermediate sort and not the recent
    one. ``mom_premium = 0`` is the null.

    Returns ``(prices, spy_prices, truth)`` -- adjusted-close-like price levels (start 100).
    """
    rng = np.random.default_rng(seed)
    tickers = [f"S{j:03d}" for j in range(n_stocks)]

    # Market returns: iid normal (daily), SPY-like.
    mkt_daily = rng.normal(0.09 / 252, market_vol / np.sqrt(252), size=n_days)

    # Modest cross-sectional beta dispersion.
    betas = rng.uniform(0.7, 1.3, size=n_stocks)

    # Idiosyncratic daily returns.
    idio = rng.normal(0.0, idio_vol / np.sqrt(252), size=(n_days, n_stocks))

    # Base daily returns = beta*market + idio (no drift yet).
    base = mkt_daily[:, None] * betas[None, :] + idio

    # Inject an intermediate-horizon momentum drift. For each day t (>= 252), rank stocks
    # by their intermediate past return (sum of daily returns over [t-252, t-147), i.e.
    # roughly months t-12..t-7), and add a small expected return proportional to the
    # cross-sectional rank, scaled by mom_premium.
    INT_FAR, INT_NEAR = 252, 147  # ~12mo and ~7mo lookbacks in trading days
    rets = base.copy()
    cumret = np.cumsum(base, axis=0)  # cumulative log-ish; fine for ranking
    for t in range(INT_FAR, n_days):
        far = cumret[t - INT_FAR]
        near = cumret[t - INT_NEAR]
        inter = near - far  # intermediate-window return per stock
        # Cross-sectional rank in [-0.5, +0.5]
        order = inter.argsort().argsort().astype(float)
        rank = order / max(n_stocks - 1, 1) - 0.5
        rets[t] += mom_premium / 252.0 * 2.0 * rank

    prices = pd.DataFrame(
        (1.0 + rets).cumprod(axis=0) * 100.0,
        index=_decorative_index(n_days),
        columns=tickers,
    )
    spy = pd.Series(
        (1.0 + mkt_daily).cumprod() * 100.0,
        index=prices.index,
        name="SPY",
    )
    return prices, spy, WorldTruth(mom_premium)


def _decorative_index(n_days: int) -> pd.DatetimeIndex:
    """A safe daily business-day index for synthetic data (no ns overflow for n<~3000)."""
    return pd.bdate_range("2008-01-02", periods=n_days)


# ---------------------------------------------------------------------------
# Real panel -- yfinance, study-local cache
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    start: str = "2007-01-01",
    end: str = "2026-01-01",
    retries: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Daily adjusted-close prices for the large-cap survivor universe + SPY.

    Cache-first: reads ``<cache_dir>/im_prices.parquet`` and ``<cache_dir>/im_spy.parquet``
    if present. On cache miss, fetches from yfinance (with retries) and writes the parquets.
    Returns ``(prices, spy_prices)`` or ``(pd.DataFrame(), pd.Series())`` if both cache and
    network fail.

    Prices are adjusted-close (yfinance auto_adjust=True).
    """
    price_path = os.path.join(cache_dir, "im_prices.parquet")
    spy_path = os.path.join(cache_dir, "im_spy.parquet")

    if os.path.exists(price_path) and os.path.exists(spy_path):
        prices = pd.read_parquet(price_path)
        spy = pd.read_parquet(spy_path).squeeze()
        spy.name = "SPY"
        return prices, spy

    last_err: Exception | None = None
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
                raise RuntimeError("empty yfinance frame")

            spy = raw["SPY"].dropna()
            prices = raw.drop(columns=["SPY"], errors="ignore").dropna(how="all")

            # Drop tickers with thin coverage (< 80%).
            coverage = prices.notna().mean()
            prices = prices.loc[:, coverage >= 0.80]

            os.makedirs(cache_dir, exist_ok=True)
            prices.to_parquet(price_path)
            spy.to_frame("SPY").to_parquet(spy_path)
            return prices, spy

        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2.0 * (attempt + 1))

    _ = last_err  # silenced; cache-miss path returns empties for CI safety
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
