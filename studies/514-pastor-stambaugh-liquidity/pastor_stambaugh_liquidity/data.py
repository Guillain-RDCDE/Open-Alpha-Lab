"""Data layer for Study 514 (Pastor-Stambaugh Liquidity Risk).

Two tapes, one schema -- a (date x ticker) daily OHLCV panel plus SPY as the market
proxy. We need price AND dollar-volume because the aggregate liquidity series is an
Amihud-style market average (|return| / dollar-volume), so the panel carries both a
``close`` frame and a ``dollar_volume`` frame.

- ``synthetic_panel`` -- a *deterministic, offline* generator. A tunable
  ``liq_premium`` knob controls how strongly high-liquidity-beta stocks out-earn
  low-liquidity-beta stocks. ``liq_premium = 0`` is the null hypothesis. Tests never
  touch the network.

- ``fetch_panel`` -- real daily OHLCV from yfinance, cached in the study's own
  ``_cache/`` dir. Returns ``(close, dollar_volume, spy)`` or empty frames if the
  cache is absent (e.g., on CI).

No look-ahead: liquidity betas are estimated on a trailing window of monthly data; the
sort is rebalanced monthly using only past-window data; positions enter the close one
trading day after the signal is public (one execution lag, applied in ``strategy``).

The panel is **survivorship-biased** (current large-cap membership projected
backwards): positive results should be read as **upper-bound** estimates.
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

# Fixed large-cap survivor universe (same family as Studies 238 / 330 / 363).
# ~45 names spanning sectors so the cross-section of liquidity betas has dispersion.
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "JPM", "LLY", "AVGO", "UNH",
    "XOM", "TSLA", "PG", "MA", "JNJ", "HD", "COST", "ABBV", "MRK", "CVX",
    "BAC", "CRM", "NFLX", "AMD", "PEP", "TMO", "ORCL", "ACN", "WMT", "MCD",
    "CSCO", "ABT", "DHR", "VZ", "TXN", "ADBE", "NEE", "RTX", "INTC", "IBM",
    "GE", "CAT", "HON", "UPS", "GS",
]
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    liq_premium: float  # how much high-liquidity-beta stocks out-earn low-liquidity-beta

    @property
    def has_premium(self) -> bool:
        return self.liq_premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 45,
    n_days: int = 3000,
    liq_premium: float = 0.06,
    market_vol: float = 0.16,
    idio_vol: float = 0.22,
    liq_shock_vol: float = 1.0,
    seed: int = 514,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, WorldTruth]:
    """A reproducible daily OHLCV-like panel with a tunable liquidity-risk premium.

    We plant a common *aggregate liquidity shock* factor ``L_t`` (a standardised daily
    series). Each stock has a true liquidity beta ``gamma_i`` drawn N(0, 1). Its daily
    return follows::

        r_i = beta_mkt_i * r_mkt + gamma_i * L_t + alpha_i + epsilon_i

    where ``alpha_i = liq_premium * gamma_i / 252`` plants a premium proportional to the
    liquidity beta (high-gamma stocks out-earn low-gamma stocks). ``liq_premium = 0`` is
    the null. The aggregate liquidity series the *engine* later reconstructs from
    dollar-volume should correlate with the planted ``L_t``.

    Returns ``(close, dollar_volume, spy, truth)``. ``close`` and ``spy`` are price
    levels; ``dollar_volume`` is a positive synthetic dollar-volume frame whose
    cross-sectional |return|/volume tracks the planted liquidity shock so the engine's
    reconstructed Amihud-average is informative.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days)
    tickers = [f"S{j:03d}" for j in range(n_stocks)]

    mkt_beta = rng.lognormal(mean=0.0, sigma=0.30, size=n_stocks)
    gamma = rng.normal(0.0, 1.0, size=n_stocks)  # true liquidity betas

    mkt_daily = rng.normal(0.08 / 252, market_vol / np.sqrt(252), size=n_days)
    spy = pd.Series((1 + mkt_daily).cumprod() * 100.0, index=dates, name="SPY")

    # Aggregate liquidity shock: standardised, persistent-ish series
    raw_L = rng.normal(0.0, liq_shock_vol, size=n_days)
    L = (raw_L - raw_L.mean()) / (raw_L.std() + 1e-12)

    idio = rng.normal(0.0, idio_vol / np.sqrt(252), size=(n_days, n_stocks))
    liq_scale = 0.004  # daily sensitivity scaling so gamma * L is a modest return component
    alphas = liq_premium * gamma / 252.0

    rets = (
        mkt_daily[:, None] * mkt_beta[None, :]
        + liq_scale * L[:, None] * gamma[None, :]
        + alphas[None, :]
        + idio
    )
    close = pd.DataFrame((1 + rets).cumprod(axis=0) * 100.0, index=dates, columns=tickers)

    # Synthetic dollar-volume: HIGH on liquid days (L high), so |ret|/dollar_vol is LOW
    # and the engine's sign-flipped Amihud average reconstructs a *liquidity* level whose
    # innovations align with the planted shock L (positive L = more liquid).
    base_dv = rng.uniform(5e8, 5e9, size=n_stocks)
    dv = base_dv[None, :] * np.exp(0.35 * L[:, None] + 0.10 * idio)
    dollar_volume = pd.DataFrame(np.abs(dv), index=dates, columns=tickers)

    return close, dollar_volume, spy, WorldTruth(liq_premium)


# ---------------------------------------------------------------------------
# Real panel -- yfinance, study-local cache
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    retries: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Daily adjusted close + dollar-volume for the universe, plus SPY.

    Cache-first: reads three parquets under ``cache_dir`` if present. On cache miss,
    fetches from yfinance (with retries to guard flakiness) and writes the parquets.
    Returns ``(close, dollar_volume, spy)`` or empty frames if both cache and network
    fail.

    ``dollar_volume`` = Close x Volume (raw, un-adjusted volume x adjusted close), the
    ingredient for the Amihud-style aggregate liquidity series. ``close`` is
    auto-adjusted close (the return tape).
    """
    close_path = os.path.join(cache_dir, "ps_close.parquet")
    dv_path = os.path.join(cache_dir, "ps_dollar_volume.parquet")
    spy_path = os.path.join(cache_dir, "ps_spy.parquet")

    if os.path.exists(close_path) and os.path.exists(dv_path) and os.path.exists(spy_path):
        close = pd.read_parquet(close_path)
        dv = pd.read_parquet(dv_path)
        spy = pd.read_parquet(spy_path).squeeze()
        spy.name = "SPY"
        return close, dv, spy

    try:
        import yfinance as yf
    except Exception:  # noqa: BLE001
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float, name="SPY")

    all_tickers = list(UNIVERSE) + ["SPY"]
    raw = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                all_tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw is not None and not raw.empty:
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2.0 * (attempt + 1))

    if raw is None or raw.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float, name="SPY")

    close_all = raw["Close"]
    vol_all = raw["Volume"]

    spy = close_all["SPY"].dropna()
    close = close_all.drop(columns=["SPY"], errors="ignore").dropna(how="all")
    vol = vol_all.drop(columns=["SPY"], errors="ignore").reindex_like(close)

    # Drop tickers with poor coverage
    coverage = close.notna().mean()
    keep = coverage[coverage >= 0.20].index
    close = close[keep]
    vol = vol[keep]

    dollar_volume = (close * vol).astype("float64")

    os.makedirs(cache_dir, exist_ok=True)
    close.to_parquet(close_path)
    dollar_volume.to_parquet(dv_path)
    spy.to_frame("SPY").to_parquet(spy_path)
    return close, dollar_volume, spy


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    """A short content fingerprint of a frame (for the as-of stamp in docs/results.md)."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    arr = np.ascontiguousarray(df.fillna(0).to_numpy(dtype="float64"))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
