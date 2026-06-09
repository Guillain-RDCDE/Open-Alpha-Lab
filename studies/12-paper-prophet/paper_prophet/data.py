"""Data layer for Paper-Prophet — SPY daily, offline from the desk cache.

The article fetches ``yfinance`` SPY at ``period="5y"``; we use the desk's cached
``_cache/SPY_split_only.parquet`` instead so the headline run is fully reproducible and pinned by
an as-of date and a content fingerprint (house rule: *pin the as-of, publish the fingerprint*).
Adjustment mode is ``split_only`` — the desk default — which keeps the genuine ex-dividend gaps a
price-taker actually lives through; it is immaterial for a *direction* study on daily returns but
we state it as a decision, not a detail.

The one thing the article gets unambiguously right is the **stationarity discipline**: a price
series has a unit root (regress price-on-price and you get a spurious R² ≈ 0.99), its log-returns
do not. We reproduce the Augmented Dickey–Fuller check as a sanity step, not a finding, and model
**returns**, never price levels.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(os.environ.get("OVERNIGHT_CACHE", "_cache"))

# The article scales log-returns by 100 (percent) before fitting — GARCH optimisers behave better
# on percent-scale data — so we keep its convention exactly.
RETURN_SCALE = 100.0


def load_spy(asof: str | None = "2026-06-01",
             cache_dir: Path | str = CACHE_DIR) -> pd.DataFrame:
    """Load cached daily SPY (split-only), sliced to ``asof``.

    Returns a frame with a ``Close`` column and a ``DatetimeIndex``. Offline by construction:
    reads the desk parquet, never hits the network, so the published fingerprint is stable.
    """
    path = Path(cache_dir) / "SPY_split_only.parquet"
    if not path.exists():  # pragma: no cover - depends on the desk cache being present
        raise FileNotFoundError(
            f"{path} not found. The headline run reads the desk's cached SPY tape; "
            "populate it via quantlab.data.fetch('SPY') or set OVERNIGHT_CACHE."
        )
    df = pd.read_parquet(path)
    df = df[["Close"]].dropna()
    if asof is not None:
        df = df[df.index <= pd.Timestamp(asof)]
    return df.copy()


def log_returns(prices: pd.Series, scale: float = RETURN_SCALE) -> pd.Series:
    """Log-returns in percent — the article's ``np.log(P/P.shift(1)).dropna() * 100``."""
    return (np.log(prices / prices.shift(1)).dropna() * scale).rename("ret")


def adf_pvalue(series: pd.Series) -> float:
    """Augmented Dickey–Fuller p-value. p < 0.05 ⇒ reject the unit root (stationary)."""
    from statsmodels.tsa.stattools import adfuller

    return float(adfuller(np.asarray(series.dropna(), dtype=float))[1])


def stationarity_check(prices: pd.Series) -> dict:
    """The article's beat-3 sanity step: price has a unit root, returns don't.

    Returns the two ADF p-values. The expected pattern (which we confirm, not discover):
    ``price_adf_p`` > 0.05 (non-stationary) and ``returns_adf_p`` < 0.05 (stationary) — so the
    only honest object to model is returns.
    """
    rets = log_returns(prices)
    return {
        "price_adf_p": adf_pvalue(prices),
        "returns_adf_p": adf_pvalue(rets),
        "n_prices": int(prices.dropna().shape[0]),
        "n_returns": int(rets.shape[0]),
    }
