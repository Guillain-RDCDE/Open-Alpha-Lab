"""Data layer for Study 208 (Gold-Miners).

Two tapes:

- ``synthetic_daily`` — a *deterministic, offline* generator. Simulates GLD and GDX price
  series where GDX is defined as leveraged GLD with a configurable beta and an operational
  drag (alpha). The ``leverage`` knob controls the amplification factor: at ``leverage=1``
  GDX moves 1:1 with GLD; at ``leverage=2`` it doubles gold moves; at ``asymmetry > 0`` the
  downside beta exceeds the upside beta (the "negative-alpha leveraged beta" claim). At
  ``leverage=0`` GDX is uncorrelated with GLD — the null.

- ``fetch_prices`` — daily adjusted closes from Yahoo! Finance (``yfinance``), cache-only
  by default. ``fetch=False`` raises ``FileNotFoundError`` if the parquet is absent, so the
  test-suite and offline core never touch the network. ``fetch=True`` downloads and caches.
  GDX starts 2006-05-22; GLD starts 2004-11-18.

No look-ahead is baked in here — that discipline lives in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# GDX IPO date — data before this is not available
GDX_START = "2006-05-22"
GLD_START = "2004-11-18"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 15,
    gold_drift_ann: float = 0.07,   # annualised GLD drift
    gold_vol_ann: float = 0.18,     # annualised GLD volatility
    leverage: float = 1.5,          # GDX beta to GLD
    asymmetry: float = 0.3,         # extra downside beta vs upside (0 = symmetric)
    alpha_ann: float = -0.04,       # GDX annual operational drag (negative alpha)
    idio_vol_ann: float = 0.10,     # GDX idiosyncratic volatility
    start: str = "2006-05-22",
    seed: int = 208,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible pair of daily (GLD-like, GDX-like) price series.

    GDX log-returns follow:
        r_GDX = alpha_daily + beta * r_GLD + idio_noise
    where ``beta = leverage + asymmetry`` when ``r_GLD < 0`` (downside) and
    ``beta = leverage`` when ``r_GLD >= 0`` (upside). This plants the asymmetry
    the study's control tests for.

    At ``leverage=0``, GDX is uncorrelated with GLD — the null. At ``asymmetry=0``
    leverage is symmetric. ``alpha_ann < 0`` represents the operational drag of
    running gold mines vs simply holding physical gold.

    Returns ``(prices, truth)`` where ``prices`` has columns ``gld`` and ``gdx``,
    indexed by business dates starting from ``start``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    dates = pd.bdate_range(start=start, periods=n_days)

    # Daily moments
    mu_gld = gold_drift_ann / TRADING_DAYS_PER_YEAR
    sig_gld = gold_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    alpha_daily = alpha_ann / TRADING_DAYS_PER_YEAR
    sig_idio = idio_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)

    r_gld = rng.normal(mu_gld, sig_gld, n_days)

    # Asymmetric beta: downside leverage exceeds upside
    beta = np.where(r_gld < 0, leverage + asymmetry, leverage)
    idio = rng.normal(0.0, sig_idio, n_days)
    r_gdx = alpha_daily + beta * r_gld + idio

    gld = 100.0 * np.exp(np.cumsum(r_gld))
    gdx = 100.0 * np.exp(np.cumsum(r_gdx))

    prices = pd.DataFrame(
        {"gld": gld, "gdx": gdx},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "leverage": leverage,
        "asymmetry": asymmetry,
        "alpha_ann": alpha_ann,
        "gold_drift_ann": gold_drift_ann,
        "gold_vol_ann": gold_vol_ann,
        "idio_vol_ann": idio_vol_ann,
        "n_years": n_years,
        "n_days": n_days,
        "seed": seed,
        "planted_upside_beta": float(leverage),
        "planted_downside_beta": float(leverage + asymmetry),
    }
    return prices, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo! Finance daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch_prices(
    ticker: str,
    start: str = "2006-05-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Daily adjusted close prices for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``). Returns a frame with a single ``close`` column indexed
    by date. Uses ``auto_adjust=True`` (split- and dividend-adjusted total-return close).
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily prices for {ticker} at {path}. "
                f"Call fetch_prices({ticker!r}, fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, start=start, end=end, interval="1d",
            auto_adjust=True, progress=False,
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        df = raw[["close"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(path)

    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.dropna(subset=["close"])
    return df


def load_pair(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    start: str = GDX_START,
) -> pd.DataFrame:
    """Load GLD and GDX adjusted closes, aligned on common dates from ``start``.

    Returns a DataFrame with columns ``gld`` and ``gdx``, index = date. The common
    start date is the GDX IPO (2006-05-22) by default so both series share the full
    history; GLD has ~18 months of pre-GDX data that is excluded here.
    """
    gld = fetch_prices("GLD", start=start, fetch=fetch, cache_dir=cache_dir)
    gdx = fetch_prices("GDX", start=start, fetch=fetch, cache_dir=cache_dir)
    prices = gld.rename(columns={"close": "gld"}).join(
        gdx.rename(columns={"close": "gdx"}), how="inner"
    )
    return prices.dropna()


def fingerprint(prices: pd.DataFrame, col: str = "gld") -> str:
    """A short content fingerprint of a price series, for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(prices[col].to_numpy()).tobytes())
    return h.hexdigest()[:12]
