"""Data layer for Study 532 (Firm-Age-Anomaly).

Two tapes, one schema -- a (date x ticker) daily adjusted-close price frame plus SPY
as the market proxy:

- ``synthetic_panel`` -- a *deterministic, offline* generator. A tunable ``age_premium``
  knob controls how strongly old (long-listed) firms outperform young (recently-listed)
  firms. ``age_premium = 0`` is the null hypothesis. Tests never touch the network.

- ``fetch_panel`` -- real daily price data from yfinance, cached in the study's own
  ``_cache/`` dir (gitignored). Returns ``(prices, spy_prices)`` or empty frames if the
  cache is absent (e.g. on CI).

Firm age is proxied from the **first available price date** of each ticker -- a clean,
data-driven proxy that needs no fundamentals pull. At any rebalancing month *t*, a
firm's age is ``(t - first_price_date)`` in years. Because the basket deliberately mixes
old-economy names (price history back to the 1980s) with recent IPOs (2012-2021), there
is genuine cross-sectional dispersion in age to sort on.

**Survivorship bias (named explicitly):** the basket is names still trading in 2026.
Young firms that IPO'd and then failed/delisted -- the very firms the new-list effect
says should underperform worst -- are absent. The age premium is therefore an
**upper-bound is understated / effect is conservatively measured** situation: the
short leg (young firms) is missing its worst names, biasing the old-minus-young spread
*downward*. We name this on the Signal axis.

Universe: ~40 large-cap names spanning four decades of IPO vintages + SPY.
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

# Fixed large-cap universe, chosen to span four decades of IPO/listing vintages so the
# firm-age sort has real cross-sectional dispersion. Old-economy names (pre-1990 listings)
# at the top; mid-cycle (1990s-2000s) in the middle; recent IPOs (2012-2021) at the bottom.
UNIVERSE = [
    # --- Mature / long-listed (price history into the 1980s) ---
    "AAPL", "MSFT", "JNJ", "PG", "KO", "PEP", "WMT", "XOM", "CVX", "JPM",
    "IBM", "GE", "MCD", "HD", "DIS", "INTC", "CSCO", "ORCL", "TXN", "MMM",
    # --- Mid-cycle (1990s-2000s listings) ---
    "AMZN", "NVDA", "GOOGL", "CRM", "NFLX", "MA", "V", "GILD",
    # --- Recent IPOs (2012-2021), survivors to 2026 ---
    "META", "ABNB", "UBER", "SNOW", "PLTR", "COIN", "DASH", "RBLX",
    "ZM", "DOCU", "SHOP", "SQ",
]
# Deduplicate while preserving order
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    age_premium: float  # how much old (long-listed) firms outperform young firms

    @property
    def has_premium(self) -> bool:
        return self.age_premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 40,
    n_days: int = 3000,
    age_premium: float = 0.04,
    market_vol: float = 0.16,
    idio_vol: float = 0.28,
    seed: int = 532,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, WorldTruth]:
    """A reproducible daily returns panel with a tunable firm-age premium.

    Each stock is assigned a *listing date* spread across the panel's first ~40% of days
    (some firms exist from day 0 = "old"; others list later = "young"). A firm's daily
    excess return follows a one-factor model:

        r_i = beta_i * r_mkt + alpha_i + epsilon_i

    where ``alpha_i = age_premium * (age_rank_i - 0.5)`` -- old firms (high age rank,
    near +0.5) earn positive alpha, young firms (low age rank, near -0.5) earn negative
    alpha. ``age_premium = 0`` is the null. Prices before a firm's listing date are NaN,
    so the age proxy (first non-NaN date) is faithful.

    Returns ``(prices, spy_prices, first_dates, truth)``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2008-01-02", periods=n_days)

    betas = rng.normal(1.0, 0.2, size=n_stocks).clip(0.3, 2.0)
    tickers = [f"S{j:03d}" for j in range(n_stocks)]

    # Listing offsets: half the names start at day 0 (old), the rest list across the
    # first ~55% of the panel (young). Older listing => higher age at any time t.
    list_offsets = np.zeros(n_stocks, dtype=int)
    n_young = n_stocks // 2
    young_idx = rng.choice(n_stocks, size=n_young, replace=False)
    list_offsets[young_idx] = rng.integers(int(n_days * 0.10), int(n_days * 0.55), size=n_young)

    # Age rank in [-0.5, +0.5]: earlier listing => older => higher rank.
    order = np.argsort(list_offsets)  # ascending offset = oldest first
    age_rank = np.empty(n_stocks)
    age_rank[order] = np.linspace(0.5, -0.5, n_stocks)  # oldest gets +0.5

    mkt_daily = rng.normal(0.09 / 252, market_vol / np.sqrt(252), size=n_days)
    spy = pd.Series(mkt_daily, index=dates, name="SPY")

    idio = rng.normal(0.0, idio_vol / np.sqrt(252), size=(n_days, n_stocks))
    alphas = age_premium * age_rank / 252.0

    stock_rets = mkt_daily[:, None] * betas[None, :] + alphas[None, :] + idio

    # Build prices, masking pre-listing days with NaN.
    prices = np.full((n_days, n_stocks), np.nan)
    for j in range(n_stocks):
        off = list_offsets[j]
        seg = (1.0 + stock_rets[off:, j]).cumprod() * 100.0
        prices[off:, j] = seg

    price_df = pd.DataFrame(prices, index=dates, columns=tickers)
    spy_px = (1.0 + mkt_daily).cumprod() * 100.0
    spy_series = pd.Series(spy_px, index=dates, name="SPY")

    first_dates = pd.Series(
        {t: price_df[t].first_valid_index() for t in tickers}, name="first_date"
    )
    return price_df, spy_series, first_dates, WorldTruth(age_premium)


# ---------------------------------------------------------------------------
# Real panel -- yfinance, study-local cache
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    start: str = "1985-01-01",
    end: str = "2026-06-01",
    retries: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Daily adjusted-close prices for the mixed-vintage universe + SPY.

    Cache-first: reads ``<cache_dir>/age_prices.parquet`` and
    ``<cache_dir>/age_spy.parquet`` if present. On cache miss, fetches from yfinance
    (with retries to guard flakiness) and writes the parquets. Returns
    ``(prices, spy_prices)`` or empty frames if both cache and network fail.

    Prices are adjusted-close (yfinance auto_adjust=True). Each ticker's first non-NaN
    date is the firm-age proxy. The long pull window (1985-) captures the full listing
    history of the old-economy names so their age is correctly large.
    """
    price_path = os.path.join(cache_dir, "age_prices.parquet")
    spy_path = os.path.join(cache_dir, "age_spy.parquet")

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
    for _ in range(retries):
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

    os.makedirs(cache_dir, exist_ok=True)
    prices.to_parquet(price_path)
    spy.to_frame("SPY").to_parquet(spy_path)
    return prices, spy


def first_price_dates(prices: pd.DataFrame) -> pd.Series:
    """First non-NaN price date per ticker -- the firm-age proxy."""
    return pd.Series(
        {t: prices[t].first_valid_index() for t in prices.columns}, name="first_date"
    )


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
