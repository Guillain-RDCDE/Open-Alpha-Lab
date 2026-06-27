"""Data layer for Study 502 (Betting-Against-Correlation).

Two tapes, one schema -- a (date x ticker) daily returns frame plus SPY as the market proxy.

The angle, and how it differs from its neighbour [238 Betting-Against-Beta]: beta factors as

    beta_i = rho_{i,m} * (sigma_i / sigma_m)

i.e. correlation-to-market times the relative volatility. Asness, Frazzini, Gormsen &
Pedersen (2020) decompose Betting-Against-Beta into a Betting-Against-Correlation (BAC) leg
and a Betting-Against-Volatility (BAV) leg and argue the *correlation* slice carries the
premium. This study isolates BAC: sort each name on its trailing **correlation** to the
market (NOT its beta, NOT its vol), long the low-correlation half, short the high-correlation
half, then beta-neutralise so the residual is not just a hidden beta bet.

- ``synthetic_panel`` -- a *deterministic, offline* generator. A tunable ``bac_premium`` knob
  controls how strongly low-correlation stocks out-earn high-correlation stocks (after
  beta-adjustment). ``bac_premium = 0`` is the null. Tests never touch the network.

- ``fetch_panel`` -- real daily price data from yfinance, cached in the study's own ``_cache/``
  dir. Returns ``(prices, spy_prices)`` or empty frames if the cache is absent (e.g. on CI).

No look-ahead: rolling correlations/betas are computed on a trailing 252-day window using only
past data; the book is rebalanced monthly and enters the close one day after the signal is
public (the lag lives in ``strategy.py``).

The panel is **survivorship-biased** (current large-cap membership projected backwards): every
name is still trading in 2026, so the high-correlation short leg is missing the failed names a
real BAC short would have caught. Positive results are **upper-bound** estimates -- named here,
not hidden, with an opt-in guard note in the strategy.

Universe: ~48 large-cap survivor names (a stable subset), plus SPY as the market proxy.
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
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

# Fixed large-cap survivor universe (~48 names across sectors; all still trading 2026).
# Deliberately spans a wide correlation range: mega-cap tech (high rho to a tech-heavy SPY),
# defensives/utilities/staples (lower rho), energy and gold-miners (lowest rho).
UNIVERSE = [
    # Mega-cap tech / high correlation
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "ORCL", "CRM", "ADBE",
    "AMD", "CSCO", "ACN", "TXN", "INTC", "QCOM",
    # Financials / cyclicals
    "JPM", "BAC", "GS", "AXP", "CAT", "DE", "HON", "GE", "UPS", "FDX",
    # Defensives / staples / health
    "PG", "KO", "PEP", "JNJ", "MRK", "PFE", "ABT", "WMT", "COST", "MCD",
    # Utilities / low correlation
    "NEE", "DUK", "SO", "AEP", "D",
    # Energy / commodity-linked (low correlation to a tech-heavy market)
    "XOM", "CVX", "COP", "SLB", "NEM", "FCX", "GIS",
]
# Deduplicate while preserving order
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    bac_premium: float  # how much low-correlation outperforms high-correlation (beta-adjusted)

    @property
    def has_premium(self) -> bool:
        return self.bac_premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 48,
    n_days: int = 3000,
    bac_premium: float = 0.05,
    market_vol: float = 0.16,
    seed: int = 502,
) -> tuple[pd.DataFrame, pd.Series, WorldTruth]:
    """A reproducible daily returns panel with a tunable correlation-driven premium.

    Each stock is built so that **correlation** and **idiosyncratic vol** vary independently:

        r_i = b_i * r_mkt + alpha_i + epsilon_i,   epsilon_i ~ N(0, idio_i^2)

    where the systematic loading ``b_i`` and the idio vol ``idio_i`` are drawn independently.
    A name's correlation-to-market is then ``rho_i = b_i*sigma_m / sqrt(b_i^2 sigma_m^2 + idio_i^2)``
    -- high when idio vol is *low*, low when idio vol is *high*, for the same systematic loading.
    The planted alpha is keyed to **correlation rank**, not vol:

        alpha_i = bac_premium * (median_rho - rho_i)

    so *low-correlation* names get positive alpha and *high-correlation* names negative --
    the Betting-Against-Correlation effect. ``bac_premium = 0`` is the null.

    Returns ``(stock_rets, mkt_rets, truth)`` as daily decimal returns. A *decorative* index is
    built with ``pd.bdate_range`` for a modest ``n_days`` (well under the ns-Timestamp wall).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days)
    tickers = [f"S{j:03d}" for j in range(n_stocks)]

    # Market returns (daily, iid normal), SPY-like drift.
    mkt_daily = rng.normal(0.08 / 252, market_vol / np.sqrt(252), size=n_days)
    sigma_m = market_vol / np.sqrt(252)

    # Independent systematic loadings and idiosyncratic vols.
    b = rng.uniform(0.6, 1.4, size=n_stocks)                       # systematic loading
    idio = rng.uniform(0.10, 0.45, size=n_stocks) / np.sqrt(252)   # idio vol (daily)

    # Implied population correlation-to-market for each name.
    rho = (b * sigma_m) / np.sqrt((b * sigma_m) ** 2 + idio ** 2)
    med_rho = np.median(rho)

    # Alpha keyed to correlation rank (NOT vol): low-rho long, high-rho short.
    alphas = bac_premium * (med_rho - rho) / 252.0

    eps = rng.normal(0.0, 1.0, size=(n_days, n_stocks)) * idio[None, :]
    stock_rets = mkt_daily[:, None] * b[None, :] + alphas[None, :] + eps

    stock_df = pd.DataFrame(stock_rets, index=dates, columns=tickers)
    mkt = pd.Series(mkt_daily, index=dates, name="SPY")
    return stock_df, mkt, WorldTruth(bac_premium)


# ---------------------------------------------------------------------------
# Real panel -- yfinance, study-local cache
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    fetch: bool = False,
    retries: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Daily adjusted-close prices for the large-cap survivor universe + SPY.

    Cache-first: reads ``<cache_dir>/bac_prices.parquet`` and ``<cache_dir>/bac_spy.parquet``
    if present. On cache miss with ``fetch=True``, downloads from yfinance (with retries) and
    writes the parquets. Returns ``(prices, spy_prices)`` or empty frames if both cache and
    network fail -- so the notebook can fall back to frozen numbers offline.

    Prices are adjusted-close (yfinance auto_adjust=True); the caller takes returns via
    ``prices.pct_change()``.
    """
    price_path = os.path.join(cache_dir, "bac_prices.parquet")
    spy_path = os.path.join(cache_dir, "bac_spy.parquet")

    if os.path.exists(price_path) and os.path.exists(spy_path):
        prices = pd.read_parquet(price_path)
        spy = pd.read_parquet(spy_path).squeeze("columns")
        spy.name = "SPY"
        return prices, spy

    if not fetch:
        return pd.DataFrame(), pd.Series(dtype=float, name="SPY")

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

            # Drop tickers with < 80% coverage over the window (kept names trade the whole span).
            coverage = prices.notna().mean()
            prices = prices.loc[:, coverage >= 0.80]
            prices = prices.dropna()  # common-history frame

            os.makedirs(cache_dir, exist_ok=True)
            prices.to_parquet(price_path)
            spy.to_frame("SPY").to_parquet(spy_path)
            return prices, spy
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2.0 * (attempt + 1))

    print(f"fetch_panel: giving up after {retries} retries ({last_err})")
    return pd.DataFrame(), pd.Series(dtype=float, name="SPY")


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    """A short content fingerprint of a panel (for the as-of stamp in docs/results.md)."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    arr = np.ascontiguousarray(df.fillna(0).to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
