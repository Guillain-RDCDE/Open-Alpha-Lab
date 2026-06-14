"""Data layer for Study 145 (Home-Bias).

Two tapes, one shape (a daily price-return panel):

- ``synthetic_panel`` — a deterministic, offline generator. Each asset has a
  configurable expected-return (``mu_*``), volatility (``vol_*``), and correlation
  to the US market (``corr_*``). Setting ``corr_intl=1.0`` produces a perfectly
  correlated world where global diversification has zero benefit; lower values
  plant the free-lunch the claim relies on. ``mu_spread=0`` means the US and
  international have the same expected return (the null); a positive ``mu_spread``
  plants the US outperformance observed since 2009.

- ``fetch_panel`` — daily total-return price data for SPY (US large-cap), EFA
  (developed ex-US) and EEM (emerging markets), drawn from the shared
  ``cross_asset_etfs.parquet`` cache (cache-only by default; ``fetch=True`` hits
  Yahoo and refreshes).

No look-ahead: the portfolio weights are computed from trailing data up to the
start of each rebalance period; the return is earned in the next period.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")

# The three ETFs this study cares about.
ASSETS = ["SPY", "EFA", "EEM"]

# Default global-blend weights (60/25/15), rebalanced annually.
BLEND_WEIGHTS = {"SPY": 0.60, "EFA": 0.25, "EEM": 0.15}

# One share = one trading day.
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 5000,
    mu_us: float = 0.07 / TRADING_DAYS,       # US daily drift
    mu_intl: float = 0.07 / TRADING_DAYS,     # international daily drift (null: same)
    vol_us: float = 0.16 / TRADING_DAYS**0.5, # US daily vol (~16%/yr)
    vol_intl: float = 0.18 / TRADING_DAYS**0.5, # intl daily vol (~18%/yr)
    corr_intl: float = 0.70,                  # correlation intl vs US
    start: str = "2003-04-14",
    seed: int = 145,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily return panel with tunable correlation and drift.

    Three synthetic assets represent SPY (US), EFA (developed ex-US), EEM
    (emerging). The US asset drives a common factor; international assets load
    onto it with weight ``corr_intl``. ``mu_intl - mu_us`` is the planted
    drift spread: negative means US outperforms (the post-2009 reality we
    want to test against); zero is the null where diversification should help
    via variance reduction alone.

    Returns ``(returns_df, truth)`` where ``truth`` records the planted parameters
    and the in-sample diversification gain vs US-only (measured by Sharpe ratio).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days, name="date")

    common = rng.standard_normal(n_days)  # shared factor

    def _asset(mu, vol, corr):
        idio = rng.standard_normal(n_days)
        shock = corr * common + np.sqrt(max(1 - corr**2, 0.0)) * idio
        return mu + vol * shock

    r_spy = _asset(mu_us, vol_us, 1.0)
    r_efa = _asset(mu_intl, vol_intl, corr_intl)
    r_eem = _asset(mu_intl, vol_intl * 1.3, corr_intl * 0.85)

    ret = pd.DataFrame({"SPY": r_spy, "EFA": r_efa, "EEM": r_eem}, index=idx)

    truth = {
        "n_days": n_days,
        "mu_us": float(mu_us),
        "mu_intl": float(mu_intl),
        "vol_us": float(vol_us),
        "vol_intl": float(vol_intl),
        "corr_intl": float(corr_intl),
        "seed": seed,
        "mu_spread_ann": float((mu_us - mu_intl) * TRADING_DAYS),
    }
    return ret, truth


# ---------------------------------------------------------------------------
# Real tape — cross-asset parquet (EFA + EEM available from 2003-04-14)
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    assets: list[str] | None = None,
) -> pd.DataFrame:
    """Daily total-return prices for SPY, EFA, EEM; cache-first.

    Reads the shared ``cross_asset_etfs.parquet`` under ``_cache/``; only
    rows where all three tickers are non-null are returned (from 2003-04-14
    when EEM IPO'd). Network is touched only on ``fetch=True``.

    Returns a DataFrame of log-returns (columns = tickers, index = date).
    """
    cache = os.path.join(cache_dir, "cross_asset_etfs.parquet")
    if assets is None:
        assets = ASSETS

    if os.path.exists(cache):
        px = pd.read_parquet(cache)
        px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    elif fetch:
        import yfinance as yf

        raw = yf.download(
            assets, period="max", auto_adjust=True, progress=False
        )["Close"]
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
        os.makedirs(cache_dir, exist_ok=True)
        raw.to_parquet(cache)
        px = raw
    else:
        raise FileNotFoundError(
            f"No cached cross-asset panel at {cache}. "
            "Call fetch_panel(fetch=True) once to populate the cache."
        )

    # Keep only our three tickers and rows where all are present.
    cols_avail = [a for a in assets if a in px.columns]
    px = px[cols_avail].dropna()
    px.index.name = "date"

    # Convert prices to log-returns.
    ret = np.log(px / px.shift(1)).dropna()
    return ret


def fingerprint(ret: pd.DataFrame) -> str:
    """A short content fingerprint of a return panel, for the as-of stamp."""
    arr = ret.to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
