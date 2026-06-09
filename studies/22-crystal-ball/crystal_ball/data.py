"""Data access for the HP-filter study — the price tape and where it comes from.

This study is about a *backtest trap*, so the tape is designed to expose it. We need a series with a
**known** structure: either a genuine mean-reverting cycle (which an honest filter should find) or a
pure random walk (which has *nothing* to find — so anything a strategy "earns" on it is an artefact).
The data layer keeps the desk's offline/cache split:

    * :func:`synthetic_prices` — fully **offline**. A log-price built as ``trend_t + amp * cycle_t``,
      where the trend is a slow random walk and the cycle is an AR(1) with persistence ``revert_rho``.
      With ``revert_rho < 1`` the cycle **mean-reverts** (deviations from trend pull back — a real,
      tradable signal an honest filter recovers). With ``revert_rho = 1`` the cycle is itself a random
      walk, so the whole price is a **random walk with no extractable signal** — the null, on which a
      look-ahead filter still manufactures a fake edge. Deterministic given ``seed``.
    * :func:`fetch_closes` — cached daily closes for liquid ETFs from the shared
      ``_cache/<TICKER>_split_only.parquet`` files. **Cache-only** unless ``fetch=True``; the network
      import stays lazy, so the offline core never imports ``yfinance``.

Data choice, named up front: **split-only closes**, used in log space — the HP filter and the trading
rule act on the *shape* of the (log) price path, and the comparison (two-sided vs one-sided filter)
charges both the identical series, so the only thing that differs is *what data the filter is allowed
to see*.
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


@dataclass(frozen=True)
class CycleTruth:
    """What the synthetic generator baked in, so a test can check the diagnostics recover it."""
    n_bars: int
    revert_rho: float         # AR(1) persistence of the cycle; <1 mean-reverts, ==1 is a random walk
    amp: float
    trend_vol: float

    @property
    def has_reversion(self) -> bool:
        """True when the cycle genuinely mean-reverts (an honest filter can find a real edge)."""
        return self.revert_rho < 1.0


def synthetic_prices(
    n_bars: int = 5040,
    revert_rho: float = 1.0,
    amp: float = 0.06,
    trend_vol: float = 0.004,
    cycle_innov: float = 0.012,
    start_price: float = 100.0,
    seed: int = 0,
) -> tuple[pd.Series, CycleTruth]:
    """A log-price = slow random-walk trend + an AR(1) cycle. ``revert_rho<1`` reverts; ``==1`` is a RW.

    The cycle ``c_t = revert_rho * c_{t-1} + cycle_innov * eps`` is the only thing a detrending strategy
    could trade: when it is stationary (``revert_rho < 1``) deviations from the trend pull back, a real
    signal; when ``revert_rho == 1`` it is a random walk and the price has **no** extractable structure.
    The price is ``exp(trend + amp_scaled * cycle)`` so it looks like a real instrument. Returns
    ``(close, truth)``; deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2003-01-02", periods=n_bars, name="date")

    trend = np.cumsum(trend_vol * rng.standard_normal(n_bars))     # slow random-walk trend (log)
    c = np.empty(n_bars)
    c[0] = 0.0
    eps = rng.standard_normal(n_bars)
    for t in range(1, n_bars):
        c[t] = revert_rho * c[t - 1] + cycle_innov * eps[t]
    # normalise the cycle to a comparable amplitude regardless of rho
    csd = c.std() if c.std() > 0 else 1.0
    log_price = np.log(start_price) + trend + amp * (c / csd)
    close = pd.Series(np.exp(log_price), index=idx, name="close")
    truth = CycleTruth(n_bars=n_bars, revert_rho=revert_rho, amp=amp, trend_vol=trend_vol)
    return close, truth


DEFAULT_TICKERS = ["SPY", "QQQ", "TLT", "GLD", "EWJ", "FXI"]


def fetch_closes(tickers: list[str] | None = None, cache_dir: str = DEFAULT_CACHE,
                 min_days: int = 1500, fetch: bool = False) -> dict[str, pd.Series]:
    """Return ``{ticker: close-series}`` for liquid ETFs, cache-first (see Study 21's reader)."""
    tickers = tickers or DEFAULT_TICKERS
    out: dict[str, pd.Series] = {}
    for tk in tickers:
        path = os.path.join(cache_dir, f"{tk}_split_only.parquet")
        if os.path.exists(path):
            c = pd.read_parquet(path)["Close"].dropna()
            if len(c) >= min_days:
                c.index = pd.DatetimeIndex(c.index).tz_localize(None)
                c.index.name = "date"
                out[tk] = c.rename("close")
            continue
        if not fetch:
            continue
        import yfinance as yf  # lazy

        raw = yf.download(tk, period="max", interval="1d", auto_adjust=False, progress=False)
        if raw is None or raw.empty:
            continue
        raw[["Open", "High", "Low", "Close", "Volume"]].to_parquet(path)
        c = raw["Close"].dropna()
        c.index = pd.DatetimeIndex(c.index).tz_localize(None)
        if len(c) >= min_days:
            out[tk] = c.rename("close")
    return out
