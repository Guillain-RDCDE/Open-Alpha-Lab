"""Data access for the trend-following study — the multi-asset tape and where it comes from.

Time-series momentum lives across a *basket*: each asset is timed on its own past return, and the
edge is the diversified average of those timed bets (Moskowitz, Ooi & Pedersen 2012). So the tape is a
panel of daily returns, and the data layer keeps the desk's standing split between an offline synthetic
universe and a cache-only real reader:

    * :func:`synthetic_panel` — fully **offline**. Each asset is ``r_{i,t} = mu_{i,t} + sigma_i *
      eps``, where the *drift* ``mu_{i,t}`` is a slow, persistent AR(1) process (zero-mean, so trends go
      both ways). Because the drift wanders slowly, an asset's **past-T return predicts its next
      return** — the time-series momentum the strategy harvests. ``trend_strength`` sets the drift's
      amplitude; pass ``trend_strength = 0`` for the **null** — driftless noise where past returns
      carry no information and a trend rule must add nothing. Deterministic given ``seed``.
    * :func:`fetch_basket` — read cached daily closes for a basket of liquid, diversifying ETFs
      (equities, bonds, gold, oil, FX, country funds) from the shared
      ``_cache/<TICKER>_split_only.parquet`` files, differenced to returns. **Cache-only** unless
      ``fetch=True``: a missing cache is skipped, never a silent download, so the offline core (and CI)
      never imports ``yfinance``.

Two data choices, named up front. **A diversified multi-asset basket**: TSMOM is a *portfolio* effect —
its Sharpe comes from averaging many low-correlation timed bets, so a single index would understate it;
the basket spans equities, bonds, commodities and FX, the canonical managed-futures menu. **Split-only
closes**: the signal is a slow (months-long) return sign, so dividends shift the long-run drift a
little but not the *timing*; split-only is stated, not hidden, and a total-return variant is a fork.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Synthetic universe — offline, with baked-in time-series momentum
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PanelTruth:
    """What the synthetic generator baked in, so a test can check the diagnostics recover it."""
    n_assets: int
    n_bars: int
    trend_strength: float     # amplitude of the persistent drift; 0 == driftless-noise null
    phi: float                # AR(1) persistence of the drift process
    noise_daily: float

    @property
    def has_trend(self) -> bool:
        """True when there is a persistent drift to ride (trend_strength != 0)."""
        return self.trend_strength != 0.0


def synthetic_panel(
    n_assets: int = 12,
    n_bars: int = 5040,
    trend_strength: float = 0.0006,
    phi: float = 0.99,
    noise_daily: float = 0.010,
    seed: int = 0,
) -> tuple[pd.DataFrame, PanelTruth]:
    """A toy daily multi-asset tape whose **past returns predict future returns**, by construction.

    For each asset ``i`` the drift follows a stationary, *persistent* AR(1)::

        mu_{i,t}  = phi * mu_{i,t-1} + sqrt(1 - phi^2) * trend_strength * nu_{i,t}
        r_{i,t}   = mu_{i,t} + sigma_i * eps_{i,t}

    The drift is zero-mean (trends run up *and* down) but slow (``phi`` near 1), so a trailing-window
    average return is a good read on the *current* drift, and the drift persists into the next period —
    exactly the time-series-momentum condition. Per-asset vols ``sigma_i`` are spread out, so the
    inverse-vol sizing in the strategy has something to do. With ``trend_strength = 0`` the drift is
    identically zero: returns are i.i.d. noise, past returns predict nothing, and a trend rule must earn
    nothing (the null).

    Returns ``(panel, truth)`` — ``panel`` is a ``dates x asset`` returns DataFrame (columns
    ``A00...``). Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2005-01-03", periods=n_bars, name="date")

    sigmas = np.linspace(0.006, 0.016, n_assets)
    rng.shuffle(sigmas)

    mu = np.zeros((n_bars, n_assets))
    innov = trend_strength * np.sqrt(1.0 - phi**2)
    nu = rng.standard_normal((n_bars, n_assets))
    for t in range(1, n_bars):
        mu[t] = phi * mu[t - 1] + innov * nu[t]
    eps = rng.standard_normal((n_bars, n_assets))
    rets = mu + sigmas[None, :] * eps

    cols = [f"A{i:02d}" for i in range(n_assets)]
    panel = pd.DataFrame(rets, index=idx, columns=cols)
    truth = PanelTruth(
        n_assets=n_assets, n_bars=n_bars, trend_strength=trend_strength,
        phi=phi, noise_daily=noise_daily,
    )
    return panel, truth


# --------------------------------------------------------------------------- #
# Real tape — cached ETF closes, one parquet per ticker
# --------------------------------------------------------------------------- #

DEFAULT_BASKET = [
    "SPY", "QQQ", "TLT", "GLD", "USO", "UUP",
    "EWG", "EWH", "EWJ", "EWQ", "EWU", "EWZ", "FXI", "INDA",
]


def fetch_basket(
    tickers: list[str] | None = None,
    cache_dir: str = DEFAULT_CACHE,
    min_days: int = 1500,
    fetch: bool = False,
) -> pd.DataFrame:
    """Return a ``dates x ticker`` **daily-returns** panel for a diversifying ETF basket, cache-first.

    Reads each ``<TICKER>_split_only.parquet`` close, keeps names with at least ``min_days`` of
    history, and differences to returns. **Cache-only by default**: a ticker with no cached parquet is
    skipped unless ``fetch=True`` (which pulls it from Yahoo once). The network import stays lazy, so
    the offline core never imports ``yfinance``.
    """
    tickers = tickers or DEFAULT_BASKET
    closes: dict[str, pd.Series] = {}
    for tk in tickers:
        path = os.path.join(cache_dir, f"{tk}_split_only.parquet")
        if os.path.exists(path):
            c = pd.read_parquet(path)["Close"].dropna()
            if len(c) >= min_days:
                c.index = pd.DatetimeIndex(c.index).tz_localize(None)
                closes[tk] = c
            continue
        if not fetch:
            continue
        import yfinance as yf  # lazy: offline core never imports it

        raw = yf.download(tk, period="max", interval="1d", auto_adjust=False, progress=False)
        if raw is None or raw.empty:
            continue
        c = raw["Close"].dropna()
        c.index = pd.DatetimeIndex(c.index).tz_localize(None)
        os.makedirs(cache_dir, exist_ok=True)
        raw[["Open", "High", "Low", "Close", "Volume"]].to_parquet(path)
        if len(c) >= min_days:
            closes[tk] = c
    if not closes:
        return pd.DataFrame()
    prices = pd.DataFrame(closes).sort_index()
    rets = prices.pct_change().dropna(how="all")
    rets.index.name = "date"
    return rets
