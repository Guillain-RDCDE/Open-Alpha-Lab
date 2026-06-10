"""Data for the reversion study — an offline synthetic mean-reverting panel, and the cached real
futures basket (the SAME 18 liquid continuous futures as Study 31, so trend and reversion are tested
on identical markets and the only difference is the sign of the signal).

The desk's offline/cache split:

  * :func:`synthetic_reversion` — fully **offline, deterministic**. A panel of synthetic markets whose
    daily returns carry **negative lag-1 autocorrelation**, so the *sign of yesterday's return predicts
    the opposite today*, by construction. ``revert_strength`` sets the effect; ``revert_strength = 0``
    is the null (a pure random walk — nothing to fade).
  * :func:`fetch_futures` — the diversified continuous-futures basket from Yahoo, **cache-first**;
    reuses Study 31's cached ``trade_winds_futures.parquet`` so the two studies share one tape. Network
    only on an explicit cache miss with ``fetch=True``. The real run lives in ``docs/results.md``.

Data choice: daily returns; the same four-asset-class basket as the trend study, so any difference in
the verdict is the *strategy*, not the universe. Stated, not hidden.
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

# The diversified continuous-futures basket (Yahoo tickers), four asset classes — identical to Study 31.
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
    revert_strength: float

    @property
    def has_reversion(self) -> bool:
        return self.revert_strength != 0.0


def synthetic_reversion(n_markets: int = 18, n_days: int = 252 * 20, revert_strength: float = 0.06,
                        vol_ann: float = 0.16, seed: int = 32) -> tuple[pd.DataFrame, PanelTruth]:
    """A panel of synthetic markets whose **price overshoots and then drifts back over several days**.

    Each market's log-price is a mean-reverting (Ornstein-Uhlenbeck) process around a flat mean:
    ``r_t = -revert_strength · dev_{t-1} + ε_t`` where ``dev`` is the running log-price deviation from
    the mean. A market that has risen above its mean tends to fall back — and crucially the pull
    *persists for several days* (half-life ≈ ``ln 2 / revert_strength``), so a contrarian signal can
    still catch it even with a realistic execution lag (unlike a pure one-day bounce). Returns a
    ``days × market`` daily-return frame and the truth. ``revert_strength = 0`` is the null (a random
    walk — no overshoot to fade).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2005-01-03", periods=n_days, name="date")
    sig_d = vol_ann / np.sqrt(TRADING_DAYS)
    rets = np.empty((n_days, n_markets))
    dev = np.zeros(n_markets)                       # running log-price deviation from the mean
    for t in range(n_days):
        eps = sig_d * rng.standard_normal(n_markets)
        cur = -revert_strength * dev + eps          # pull back toward the mean
        rets[t] = cur
        dev = dev + cur                             # accumulate the level
    cols = [f"MKT{i:02d}" for i in range(n_markets)]
    return pd.DataFrame(rets, index=idx, columns=cols), PanelTruth(n_markets, n_days, revert_strength)


def fetch_futures(cache_dir: str = DEFAULT_CACHE, fetch: bool = False, min_years: float = 12.0
                  ) -> pd.DataFrame:
    """Daily returns of the diversified continuous-futures basket, cache-first.

    Reuses Study 31's ``trade_winds_futures.parquet`` if present (the identical basket, so trend and
    reversion run on one tape). Falls back to a Rip-Tide-specific cache, and only downloads from Yahoo
    on an explicit cache miss with ``fetch=True`` (returns empty otherwise, so offline runs never touch
    the network).
    """
    shared = os.path.join(cache_dir, "trade_winds_futures.parquet")
    own = os.path.join(cache_dir, "rip_tide_futures.parquet")
    for cache in (own, shared):
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
    # Same hygiene as Study 31: clip daily returns to ±25% (continuous-feed roll glitches + the
    # Apr-2020 negative-WTI print make raw pct_change meaningless). Stated, not hidden.
    rets = prices.pct_change().clip(-0.25, 0.25).dropna(how="all")
    rets.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    rets.to_parquet(own)
    return rets


def asset_class(ticker: str) -> str:
    return FUTURES.get(ticker, "other")
