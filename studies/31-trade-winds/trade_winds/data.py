"""Data for the trend study — an offline synthetic trending panel, and a cached real futures basket.

The desk's offline/cache split:

  * :func:`synthetic_trends` — fully **offline, deterministic**. A panel of synthetic markets, each
    with a slowly **regime-switching drift** (persistent up / flat / down trends) plus noise — so the
    *sign of the recent return predicts the next one*, by construction. ``trend_strength`` sets the
    premium; ``trend_strength = 0`` is the null (a pure random walk, no trend to follow).
  * :func:`fetch_futures` — a diversified basket of liquid **continuous futures** (equities, bonds,
    commodities, FX) from Yahoo, **cache-first**; network only on an explicit cache miss with
    ``fetch=True``. The real run lives in ``docs/results.md``.

Data choice: daily returns; the universe spans four asset classes so the portfolio's edge comes from
*diversified* trends, not one market; futures so the cost story is honest (cheap, deep, no full-notional
financing — the carry is in the price). Stated, not hidden.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
TRADING_DAYS = 252

# A diversified continuous-futures basket (Yahoo tickers), four asset classes.
FUTURES = {
    "ES=F": "equity", "NQ=F": "equity", "YM=F": "equity",          # US equity indices
    "ZN=F": "rates", "ZB=F": "rates", "ZF=F": "rates",             # Treasury notes/bonds
    "CL=F": "commodity", "GC=F": "commodity", "SI=F": "commodity",  # oil, gold, silver
    "HG=F": "commodity", "NG=F": "commodity", "ZC=F": "commodity",  # copper, natgas, corn
    "ZS=F": "commodity", "ZW=F": "commodity",                      # soybeans, wheat
    "6E=F": "fx", "6J=F": "fx", "6B=F": "fx", "6A=F": "fx",         # EUR, JPY, GBP, AUD
}


@dataclass(frozen=True)
class PanelTruth:
    n_markets: int
    n_days: int
    trend_strength: float

    @property
    def has_trend(self) -> bool:
        return self.trend_strength != 0.0


def synthetic_trends(n_markets: int = 18, n_days: int = 252 * 20, trend_strength: float = 0.12,
                     regime_days: float = 60.0, vol_ann: float = 0.16, seed: int = 31
                     ) -> tuple[pd.DataFrame, PanelTruth]:
    """A panel of synthetic markets whose **recent return sign predicts the next**, by construction.

    Each market follows a slow 3-state drift regime (up / flat / down) that switches on average every
    ``regime_days`` days; in a trending regime the daily drift is ``±trend_strength·σ_daily``, so a
    market that has been rising tends to keep rising (the time-series-momentum premium). Returns a
    ``days × market`` daily-return frame and the truth. ``trend_strength = 0`` is the null.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2005-01-03", periods=n_days, name="date")
    sig_d = vol_ann / np.sqrt(TRADING_DAYS)
    p_switch = 1.0 / regime_days
    rets = np.empty((n_days, n_markets))
    states = np.zeros(n_markets, dtype=int)             # -1 down, 0 flat, +1 up
    states[:] = rng.choice([-1, 0, 1], size=n_markets)
    for t in range(n_days):
        flip = rng.random(n_markets) < p_switch
        states = np.where(flip, rng.choice([-1, 0, 1], size=n_markets), states)
        drift = trend_strength * sig_d * states
        rets[t] = drift + sig_d * rng.standard_normal(n_markets)
    cols = [f"MKT{i:02d}" for i in range(n_markets)]
    return pd.DataFrame(rets, index=idx, columns=cols), PanelTruth(n_markets, n_days, trend_strength)


def fetch_futures(cache_dir: str = DEFAULT_CACHE, fetch: bool = False, min_years: float = 12.0
                  ) -> pd.DataFrame:
    """Daily returns of the diversified continuous-futures basket, cache-first.

    **Cache-only** unless ``fetch=True`` (downloads each ticker in :data:`FUTURES` from Yahoo). Keeps
    only markets with at least ``min_years`` of history. Returns a ``date × ticker`` daily-return frame
    (empty on a cache miss with ``fetch=False`` so offline runs never hit the network).
    """
    cache = os.path.join(cache_dir, "trade_winds_futures.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf  # lazy
    closes = {}
    for tk in FUTURES:
        try:
            raw = yf.download(tk, period="max", interval="1d", auto_adjust=False, progress=False)
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            s = raw["Close"].dropna()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            if (s.index[-1] - s.index[0]).days / 365.25 >= min_years:
                closes[tk] = s.astype(float)
        except Exception:
            continue
    if not closes:
        return pd.DataFrame()
    prices = pd.DataFrame(closes).sort_index()
    rets = prices.pct_change()
    # Data hygiene: clip daily returns to ±25%. Continuous-futures feeds carry roll glitches and the
    # notorious negative-WTI print of Apr-2020 (CL went to -$37, which makes pct_change meaningless);
    # a real future is halted long before a clean ±25% day. Stated, not hidden (see docs/references.md).
    rets = rets.clip(-0.25, 0.25).dropna(how="all")
    rets.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    rets.to_parquet(cache)
    return rets


def asset_class(ticker: str) -> str:
    return FUTURES.get(ticker, "other")
