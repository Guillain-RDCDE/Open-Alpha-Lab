"""Data layer for Study 157 (Kelly-Sizing).

Two tapes, one shape (a daily total-return series for a single risky asset):

- ``synthetic_returns`` — a *deterministic, offline* generator. A Sharpe-ratio
  knob (``sharpe``) lets us dial the only thing the Kelly formula can possibly
  optimise: the signal-to-noise ratio of a stationary return stream.
  ``sharpe=0`` ⇒ no edge ⇒ any Kelly fraction is a pure gamble.
  ``sharpe>0`` ⇒ a real positive-expectancy signal the Kelly formula is meant
  for. This is the study's null in a bottle.

- ``fetch_spy`` — the real SPY daily total-return tape (``yfinance``), cache-only
  by default so the test-suite and reproducible core never touch the network.
  SPY from its 1993-01-29 inception gives us the longest freely available
  US-equity daily return series.

No look-ahead is baked in here — that discipline lives in ``strategy.py``
(the Kelly fraction is estimated from returns up to *t*, positions entered from
*t+1* onward in a strict walk-forward window).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
STUDY_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_returns(
    n_years: float = 20.0,
    sharpe: float = 0.5,
    annual_vol: float = 0.16,
    start: str = "2004-01-02",
    seed: int = 157,
) -> tuple[pd.Series, dict]:
    """A reproducible daily return series with a known information ratio.

    Returns follow i.i.d. Normal(mu, sigma) with:
        sigma = annual_vol / sqrt(252)
        mu    = sharpe * sigma                   (daily Sharpe = sharpe / sqrt(252))

    The *annual* Sharpe of the series equals ``sharpe`` exactly in expectation.
    ``sharpe=0`` ⇒ a fair game (Kelly fraction = 0); ``sharpe=0.5`` ⇒ roughly
    matching S&P 500's long-run equity premium.

    Returns ``(series, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    n_days = int(round(n_years * TRADING_DAYS))
    dates = pd.bdate_range(start=start, periods=n_days, name="date")

    sigma_d = annual_vol / np.sqrt(TRADING_DAYS)
    # Annual Sharpe = (mu_d / sigma_d) * sqrt(252), so to plant annual Sharpe:
    #   mu_d = (sharpe / sqrt(252)) * sigma_d
    mu_d = (sharpe / np.sqrt(TRADING_DAYS)) * sigma_d
    r = rng.normal(mu_d, sigma_d, n_days)
    series = pd.Series(r, index=dates, name="ret")

    truth = {
        "sharpe": sharpe,
        "annual_vol": annual_vol,
        "mu_daily": float(mu_d),
        "sigma_daily": float(sigma_d),
        "n_days": n_days,
        "seed": seed,
    }
    return series, truth


# ---------------------------------------------------------------------------
# Real tape — SPY daily total return, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "spy_daily_kelly.parquet")


def fetch_spy(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.Series:
    """Real SPY daily total-return series; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as a parquet under ``_cache/``). Returns a Series of simple daily
    returns, index = date (tz-naive), name = 'ret', from SPY's inception.
    """
    path = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached SPY daily tape at {path}. "
                f"Call fetch_spy(fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(path)
        return df["ret"]
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download("SPY", period="max", interval="1d",
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError("yfinance returned no daily bars for SPY")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        px = raw["Close"].rename("close")
        px.index = pd.DatetimeIndex(px.index).tz_localize(None)
        px.index.name = "date"
        ret = px.pct_change().dropna()
        ret.name = "ret"
        os.makedirs(cache_dir, exist_ok=True)
        ret.to_frame().to_parquet(path)
        return ret


def fingerprint(returns: pd.Series) -> str:
    """A short content fingerprint of a daily return series, for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(returns.to_numpy()).tobytes())
    return h.hexdigest()[:12]
