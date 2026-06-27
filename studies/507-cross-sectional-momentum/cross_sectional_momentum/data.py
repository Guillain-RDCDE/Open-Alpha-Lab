"""Data layer for Study 507 (Cross-Sectional-Momentum).

Two tapes, one schema -- a (date x ticker) daily adjusted-close price frame:

- ``synthetic_panel`` -- a *deterministic, offline* generator. A tunable ``mom_strength``
  knob controls how persistently past winners keep winning (a planted relative-strength
  drift). ``mom_strength = 0`` is the null -- prices are pure random walks with no
  cross-sectional persistence, so the winners-minus-losers (WML) book earns nothing.
  Tests and the reproducible core never touch the network.

- ``fetch_panel`` -- real daily adjusted-close prices from yfinance, cached to the study's
  OWN ``_cache/`` dir. Cache-first: returns the cached parquet if present, else an empty
  frame (so notebooks can banner "synthetic tape" on CI) unless ``fetch=True``.

The basket is **survivorship-biased**: it is ~40 large-cap names still trading in 2026,
projected backwards. The natural short candidates of a momentum loser leg -- firms that
trended down into delisting/bankruptcy -- are absent by construction. This is named openly
on the SIGNAL axis (not just tradability), because it directly inflates any apparent
winners-minus-losers premium. See the opt-in guard note in ``strategy.py`` and the README.

No look-ahead lives here: the 12-1 signal (skip the most recent month) and the single
execution lag (enter one trading day after the signal is public) live in ``strategy.py``.
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

# Fixed ~40-name large-cap survivor basket (stable subset of the S&P 500, sector-spread).
# Deliberately smaller than Study 24 (full S&P 500 WML) -- a retail-realistic basket where
# the decile/quintile sort is thin, which is exactly the third-axis question of this study.
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "JPM", "JNJ", "V", "PG",
    "HD", "MA", "XOM", "CVX", "KO", "PEP", "MRK", "ABBV", "WMT", "COST",
    "MCD", "DIS", "CSCO", "INTC", "ORCL", "IBM", "TXN", "QCOM", "HON", "CAT",
    "GE", "BA", "MMM", "UNH", "PFE", "T", "VZ", "GS", "AXP", "LOW",
]
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]

TRADING_DAYS = 252


@dataclass(frozen=True)
class WorldTruth:
    """The planted relative-strength drift for a synthetic panel."""

    mom_strength: float  # annualised persistent winner drift (0 = pure random walk null)

    @property
    def has_momentum(self) -> bool:
        return self.mom_strength != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 40,
    n_days: int = 2600,
    mom_strength: float = 0.0,
    market_vol: float = 0.16,
    idio_vol: float = 0.28,
    persistence: int = 252,
    seed: int = 507,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible daily adjusted-close panel with a tunable momentum drift.

    Each stock loads on a common market factor (beta ~ 1) plus an idiosyncratic component.
    When ``mom_strength > 0`` a *persistent* drift is added: each stock is assigned a slowly
    varying latent "trend" whose sign tends to persist over ``persistence`` days, and the
    drift contribution is ``mom_strength`` (annualised) times that latent trend. Winners thus
    keep winning over the trailing-year window -- exactly the relative-strength persistence
    Jegadeesh-Titman exploit. ``mom_strength = 0`` is the null: pure random walks, no
    cross-sectional persistence, so the WML book earns nothing.

    Returns ``(prices, truth)`` where ``prices`` is a (date x ticker) adjusted-close frame.
    A *decorative* business-day index is built with ``pd.bdate_range`` over a small ``n_days``
    (months, never a huge nanosecond span) -- safely under the ns-Timestamp overflow wall.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-02", periods=n_days)
    tickers = [f"S{j:02d}" for j in range(n_stocks)]

    mkt = rng.normal(0.08 / TRADING_DAYS, market_vol / np.sqrt(TRADING_DAYS), size=n_days)
    idio = rng.normal(0.0, idio_vol / np.sqrt(TRADING_DAYS), size=(n_days, n_stocks))

    # Latent persistent trend: an AR(1)-like sign-persistent process per stock.
    phi = np.exp(-1.0 / max(persistence, 1))  # daily persistence coefficient
    shock_sd = np.sqrt(1.0 - phi**2)
    trend = np.zeros((n_days, n_stocks))
    trend[0] = rng.standard_normal(n_stocks)
    innov = rng.standard_normal((n_days, n_stocks))
    for t in range(1, n_days):
        trend[t] = phi * trend[t - 1] + shock_sd * innov[t]

    drift = mom_strength / TRADING_DAYS * trend  # annualised drift -> daily
    rets = mkt[:, None] + idio + drift

    prices = pd.DataFrame(rets, index=dates, columns=tickers)
    prices = (1.0 + prices).cumprod() * 100.0
    prices.index.name = "date"
    return prices, WorldTruth(mom_strength)


# ---------------------------------------------------------------------------
# Real panel -- yfinance, study-local cache
# ---------------------------------------------------------------------------
def cache_path(cache_dir: str = DEFAULT_CACHE) -> str:
    """Path to the cached real-panel adjusted-close parquet."""
    return os.path.join(cache_dir, "csmom_prices.parquet")


def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    retries: int = 3,
) -> pd.DataFrame:
    """Daily adjusted-close prices for the large-cap survivor basket, cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else an
    **empty** DataFrame (CI/offline path -> notebooks fall back to frozen numbers). Network is
    touched only on ``fetch=True`` (guarded with retries against yfinance flakiness), then the
    result is cached as adjusted close (auto_adjust=True). The caller computes returns via
    ``prices.pct_change()``.
    """
    p = cache_path(cache_dir)
    if os.path.exists(p):
        df = pd.read_parquet(p)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy: only on an explicit network pull

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                list(UNIVERSE),
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )["Close"]
            if raw.empty:
                raise RuntimeError("empty yfinance frame")
            raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
            # Keep names with >= 80% coverage over the window (drop late IPOs / gaps).
            coverage = raw.notna().mean()
            raw = raw.loc[:, coverage >= 0.80].dropna(how="all")
            raw = raw.ffill().dropna(how="any")
            raw.index.name = "date"
            os.makedirs(cache_dir, exist_ok=True)
            raw.to_parquet(p)
            return raw
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"yfinance fetch failed after {retries} tries: {last_err}")


def drop_partial_last_month(prices: pd.DataFrame, asof_day: int = 25) -> pd.DataFrame:
    """Drop trailing in-progress-month rows (house rule: no partial month in a stamp).

    If the last bar falls in the current calendar month before ``asof_day``, the whole
    current-month tail is dropped so the stamped run contains only completed months.
    """
    if prices.empty:
        return prices
    last = prices.index[-1]
    today = pd.Timestamp.today().normalize()
    if last.year == today.year and last.month == today.month and today.day < asof_day:
        keep = ~((prices.index.year == today.year) & (prices.index.month == today.month))
        return prices.loc[keep]
    return prices


def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    """A short content fingerprint of a panel (for the as-of stamp in docs/results.md)."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    arr = np.ascontiguousarray(df.fillna(0.0).to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
