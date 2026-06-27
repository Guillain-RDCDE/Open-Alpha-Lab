"""Data layer for Study 505 (Left-Tail-Momentum).

One schema -- a (date x ticker) daily-returns frame plus SPY as the market proxy:

- ``synthetic_panel`` -- a *deterministic, offline* generator. A tunable
  ``ltm_premium`` knob controls how strongly the *worst-tail* stocks keep
  underperforming next month (left-tail momentum: crashed stocks keep crashing).
  ``ltm_premium = 0`` is the null hypothesis. Nothing here touches the network.

- ``fetch_panel`` -- real daily adjusted-close prices from yfinance, cached in the
  study's own ``_cache/`` dir (gitignored). Returns ``(prices, spy)`` or empty frames
  if the cache is absent (e.g., on CI).

No look-ahead: the left-tail signal at month *t* is built only from returns up to and
including the last trading day *before* the formation date; the holding return is the
NEXT month, entered one trading day after the signal is public (one execution lag).

The panel is **survivorship-biased** (a fixed list of names still trading in 2026):
the worst left-tail names of the past -- the very firms that crashed into delisting --
are absent, which is exactly the bias that flatters a "short the crashed" book.

Universe: ~48 large-cap S&P 500 names, plus SPY as the market proxy.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

# Fixed large-cap survivor universe (stable subset of the S&P 500, ~48 names).
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "JPM", "LLY", "AVGO", "XOM",
    "UNH", "TSLA", "PG", "MA", "JNJ", "HD", "COST", "ABBV", "MRK", "CVX",
    "BAC", "NFLX", "AMD", "PEP", "TMO", "ORCL", "ACN", "WMT", "MCD", "CSCO",
    "ABT", "DHR", "VZ", "TXN", "ADBE", "NEE", "RTX", "INTC", "IBM", "GE",
    "CAT", "HON", "UPS", "GS", "AXP", "ISRG", "GILD", "BKNG",
]
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    ltm_premium: float  # strength of left-tail momentum (crashed stocks keep underperforming)

    @property
    def has_premium(self) -> bool:
        return self.ltm_premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 48,
    n_days: int = 2600,
    ltm_premium: float = 0.04,
    market_vol: float = 0.16,
    idio_vol: float = 0.28,
    seed: int = 505,
) -> tuple[pd.DataFrame, pd.Series, WorldTruth]:
    """A reproducible daily-returns panel with a tunable left-tail-momentum premium.

    Each stock has a persistent latent "fragility" ``f_i`` in [0, 1]. Fragile stocks
    have (a) fatter left tails -- occasional idiosyncratic crash days -- and (b) a small
    *negative* drift proportional to ``ltm_premium * f_i`` (the left-tail-momentum
    effect: stocks prone to crashing keep underperforming). ``ltm_premium = 0`` removes
    the drift, leaving only the symmetric crash noise -- the null.

    The continuation is what the strategy must detect: sort on realised left-tail risk,
    short the worst tail (high f), long the best tail (low f). Returns
    ``(stock_rets, mkt_rets, truth)`` as daily decimal returns.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-02", periods=n_days)
    tickers = [f"S{j:03d}" for j in range(n_stocks)]

    # Persistent latent fragility per stock (uniform-ish, sorted out by the rng).
    fragility = rng.uniform(0.0, 1.0, size=n_stocks)

    # Market: iid normal daily.
    mkt_daily = rng.normal(0.09 / 252, market_vol / np.sqrt(252), size=n_days)
    mkt = pd.Series(mkt_daily, index=dates, name="SPY")

    betas = rng.uniform(0.8, 1.2, size=n_stocks)

    # Baseline idiosyncratic noise.
    idio = rng.normal(0.0, idio_vol / np.sqrt(252), size=(n_days, n_stocks))

    # Fat left tail: fragile stocks get occasional sharp negative jumps.
    jump_prob = 0.003 + 0.012 * fragility            # per-day crash probability
    jump_mask = rng.random((n_days, n_stocks)) < jump_prob[None, :]
    jump_size = -np.abs(rng.normal(0.0, 0.05, size=(n_days, n_stocks))) * (0.5 + fragility[None, :])
    crashes = jump_mask * jump_size

    # Left-tail-momentum drift: fragile stocks carry a persistent negative drift.
    drift = -ltm_premium * fragility / 252.0

    stock_rets = (
        mkt_daily[:, None] * betas[None, :]
        + drift[None, :]
        + idio
        + crashes
    )
    stock_df = pd.DataFrame(stock_rets, index=dates, columns=tickers)
    return stock_df, mkt, WorldTruth(ltm_premium)


# ---------------------------------------------------------------------------
# Real panel -- yfinance, study-local cache
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    start: str = "2014-01-01",
    end: str = "2026-06-01",
    retries: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Daily adjusted-close prices for the large-cap universe + SPY.

    Cache-first: reads ``<cache_dir>/ltm_prices.parquet`` and
    ``<cache_dir>/ltm_spy.parquet`` if present. On cache miss, fetches from yfinance
    (with retries to guard flakiness) and writes the parquets. Returns
    ``(prices, spy)`` or ``(pd.DataFrame(), pd.Series())`` if both cache and network
    fail. Prices are adjusted-close (auto_adjust=True).
    """
    price_path = os.path.join(cache_dir, "ltm_prices.parquet")
    spy_path = os.path.join(cache_dir, "ltm_spy.parquet")

    if os.path.exists(price_path) and os.path.exists(spy_path):
        prices = pd.read_parquet(price_path)
        spy = pd.read_parquet(spy_path).squeeze("columns")
        spy.name = "SPY"
        return prices, spy

    try:
        import yfinance as yf
    except Exception:  # noqa: BLE001
        return pd.DataFrame(), pd.Series(dtype=float, name="SPY")

    all_tickers = list(UNIVERSE) + ["SPY"]
    raw = None
    for _ in range(max(1, retries)):
        try:
            raw = yf.download(
                all_tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )["Close"]
            if raw is not None and not raw.empty:
                break
        except Exception:  # noqa: BLE001
            raw = None
    if raw is None or raw.empty:
        return pd.DataFrame(), pd.Series(dtype=float, name="SPY")

    spy = raw["SPY"].dropna()
    prices = raw.drop(columns=["SPY"], errors="ignore").dropna(how="all")
    coverage = prices.notna().mean()
    prices = prices.loc[:, coverage >= 0.90]

    os.makedirs(cache_dir, exist_ok=True)
    prices.to_parquet(price_path)
    spy.to_frame("SPY").to_parquet(spy_path)
    return prices, spy


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
