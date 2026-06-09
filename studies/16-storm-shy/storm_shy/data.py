"""Data access for the vol-targeting study — the price tape and where it comes from.

The whole study runs on one thing: a series of **daily returns** (read off a close series). The
overlay and every diagnostic are pure functions of those returns, so the data layer is deliberately
thin — the desk's standing split between an offline synthetic tape and a cache-only real reader:

    * :func:`synthetic_prices` — fully **offline**. Daily returns with a *constant* drift and a
      **two-state Markov volatility regime** (a calm vol ``sigma_lo`` and a stormy ``sigma_hi``,
      persisting with probability ``stay_prob``). Two facts are baked in on purpose. First, vol
      *clusters*: a persistent regime makes realized variance strongly autocorrelated — the thing
      the overlay needs to forecast. Second, the drift is **regime-independent**, so a stormy
      stretch carries the *same* expected return as a calm one but far more risk — which is exactly
      the condition (Moreira–Muir 2017) under which scaling exposure by inverse variance raises the
      Sharpe. Pass ``sigma_lo == sigma_hi`` for the **flat-vol null**: no clustering, nothing for
      the overlay to read, no gain. Deterministic given ``seed``; no network.
    * :func:`fetch_prices` — pull real daily closes from Yahoo! Finance, cache each ticker to
      parquet, and return the single ``close`` column. **Cache-only** unless ``fetch=True``: a
      missing cache is skipped, never a silent stall, so the offline core (and CI) never imports
      ``yfinance``.

Daily fidelity and **split/dividend-adjusted** closes (``auto_adjust=True``) are the decisions
here, named up front per the house rule: vol-targeting is a position-sizing overlay studied on
daily data in the source literature, total-return closes are the right input for a long-horizon
risk-premium harvest, and daily history gives the multi-decade window the decay test wants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Synthetic tape — offline, with baked-in volatility clustering
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GroundTruth:
    """What the synthetic generator baked in, so a test can check the diagnostics."""
    drift_daily: float        # constant daily drift, the SAME in both vol regimes
    sigma_lo: float           # calm-regime per-bar return vol
    sigma_hi: float           # stormy-regime per-bar return vol
    stay_calm: float          # P(calm -> calm): how sticky the calm regime is
    stay_storm: float         # P(storm -> storm): how sticky the storm regime is
    n_bars: int

    @property
    def has_regime(self) -> bool:
        """True when there is a genuine vol regime to read (storm != calm)."""
        return self.sigma_hi != self.sigma_lo

    @property
    def calm_fraction(self) -> float:
        """Stationary share of time in the calm regime (markets spend most time calm)."""
        out_calm = 1.0 - self.stay_calm
        out_storm = 1.0 - self.stay_storm
        denom = out_calm + out_storm
        return float(out_storm / denom) if denom > 0 else 1.0

    @property
    def theoretical_sharpe_gain(self) -> float:
        """The Sharpe multiple a *perfect-foresight* vol target earns over buy-&-hold.

        With a regime-independent drift ``mu`` and stationary regime weights ``π``, buy-&-hold
        Sharpe ≈ ``mu / sqrt(E[sigma^2])`` while a constant-risk overlay earns ≈ ``mu * E[1/sigma]``.
        Their ratio is ``E[1/sigma] · sqrt(E[sigma^2])`` — ≥ 1 by Cauchy–Schwarz, strictly > 1
        whenever the regimes differ. This is the *ceiling* the realized, past-only overlay
        approaches but cannot exceed (it pays for a noisy, lagged, capped forecast).
        """
        pc = self.calm_fraction
        sl, sh = self.sigma_lo, self.sigma_hi
        e_inv = pc * (1.0 / sl) + (1.0 - pc) * (1.0 / sh)
        e_var = pc * sl**2 + (1.0 - pc) * sh**2
        return float(e_inv * np.sqrt(e_var))


def synthetic_prices(
    n_bars: int = 5040,
    drift_daily: float = 0.0005,
    sigma_lo: float = 0.005,
    sigma_hi: float = 0.025,
    stay_calm: float = 0.985,
    stay_storm: float = 0.95,
    start_price: float = 100.0,
    seed: int = 0,
) -> tuple[pd.Series, GroundTruth]:
    """A toy daily close whose volatility **clusters** in persistent regimes around a steady drift.

    The return process is::

        regime[t] in {calm, storm}, a Markov chain (sticky: P(stay) per regime below)
        r[t] = drift_daily + sigma(regime[t]) * eps[t],   eps ~ N(0, 1)

    The drift is identical in both regimes, so storms add *risk without return* — the precise setup
    (Moreira–Muir 2017) where down-sizing in high vol and up-sizing in low vol raises the Sharpe.
    The chain is **asymmetric on purpose**: ``stay_calm`` > ``stay_storm`` makes calm the dominant
    regime (~77% of the time at the defaults), so unconditional vol lands near a realistic ~20%/yr
    rather than a 50/50 blend. Persistence makes realized variance autocorrelated, i.e.
    **forecastable** — the only input the past-only overlay has. Set ``sigma_lo == sigma_hi`` for
    the flat-vol null where there is nothing to read and the overlay must add nothing.

    Returns ``(close, truth)`` — ``close`` is a strictly-positive price ``pd.Series`` on a
    business-day index named ``date``; ``truth`` carries the baked-in parameters, the stationary
    calm fraction and the perfect-foresight Sharpe ceiling. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2016-01-04", periods=n_bars, name="date")

    # Asymmetric two-state Markov regime path: 0 = calm, 1 = storm.
    sig = np.empty(n_bars)
    state = 0
    for t in range(n_bars):
        stay = stay_calm if state == 0 else stay_storm
        if rng.random() > stay:
            state = 1 - state
        sig[t] = sigma_lo if state == 0 else sigma_hi

    rets = drift_daily + sig * rng.standard_normal(n_bars)
    close = pd.Series(start_price * np.cumprod(1.0 + rets), index=idx, name="close")
    truth = GroundTruth(
        drift_daily=drift_daily, sigma_lo=sigma_lo, sigma_hi=sigma_hi,
        stay_calm=stay_calm, stay_storm=stay_storm, n_bars=n_bars,
    )
    return close, truth


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! daily closes, cached per ticker
# --------------------------------------------------------------------------- #

def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "_").replace("^", "_")
    return os.path.join(cache_dir, f"close_{safe}.parquet")


def fetch_prices(
    ticker: str,
    period: str = "max",
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
) -> pd.Series:
    """Return (and cache) the daily **close** series for ``ticker``, on a ``date`` index.

    **Cache-only by default**: if a parquet exists it is returned; otherwise an empty series is
    returned *unless* ``fetch=True``, in which case Yahoo is hit once (``auto_adjust=True``, so the
    closes are split/dividend-adjusted — a data-choice the house rules ask us to state) and the
    result cached. The network import is lazy, so the offline core never imports ``yfinance``.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path) and not fetch:
        return pd.read_parquet(path)["close"]

    if not fetch:
        return pd.Series(dtype="float64", name="close")

    import yfinance as yf  # lazy: offline core never imports it

    raw = yf.download(ticker, period=period, interval="1d",
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.Series(dtype="float64", name="close")

    col = raw["Close"]
    if isinstance(col, pd.DataFrame):                # yfinance MultiIndex on a single ticker
        col = col.iloc[:, 0]
    close = col.astype("float64")
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    close.index.name = "date"
    close.name = "close"
    close = close[~close.index.duplicated(keep="last")].sort_index()

    os.makedirs(cache_dir, exist_ok=True)
    close.to_frame().to_parquet(path)
    return close
