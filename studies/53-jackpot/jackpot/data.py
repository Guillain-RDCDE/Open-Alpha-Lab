"""Data for the lottery (MAX) study — an offline synthetic daily panel, and a cached real panel.

  * :func:`synthetic_panel` — **offline, deterministic**. Each stock has a fixed "lottery score": high
    scorers get occasional large positive daily spikes (raising MAX) *and* a drift penalty of strength
    ``lottery_premium`` — so high-MAX stocks underperform, by construction. 0 = the null. Pins the
    machinery offline.
  * :func:`fetch_panel` — daily returns for current S&P 500 members with long history (Yahoo),
    **cache-first**. Survivorship-biased, large-cap (stated). Fingerprinted run in ``docs/results.md``.
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


@dataclass(frozen=True)
class WorldTruth:
    lottery_premium: float

    @property
    def has_premium(self) -> bool:
        return self.lottery_premium != 0.0


def synthetic_panel(
    n_stocks: int = 120, n_years: int = 22, lottery_premium: float = 0.0008, vol_ann: float = 0.25,
    seed: int = 53
) -> tuple[pd.DataFrame, WorldTruth]:
    """A daily stock panel with a lottery (MAX) structure — deterministic given ``seed``.

    Each stock gets a fixed lottery score in [0,1]; high scorers have **higher volatility** (so a higher
    MAX, since MAX scales with vol) *and* a per-day drift penalty ``-lottery_premium·score`` — so the
    high-MAX, high-vol names earn less, the lottery story. ``lottery_premium = 0`` is the null. (That
    vol drives MAX is itself the real-world point: MAX ≈ idiosyncratic vol, see Study 54.) Returns a
    (date × stock) daily-return frame.
    """
    rng = np.random.default_rng(seed)
    n = n_years * TRADING_DAYS
    idx = pd.bdate_range("2003-01-01", periods=n, name="date")
    score = rng.uniform(0, 1, n_stocks)
    per_stock_vol = (vol_ann / np.sqrt(TRADING_DAYS)) * (0.5 + score)   # high score → high vol → high MAX
    drift = 0.0006 - lottery_premium * score                            # high score → low drift
    R = drift[None, :] + per_stock_vol[None, :] * rng.standard_normal((n, n_stocks))
    cols = [f"S{j:03d}" for j in range(n_stocks)]
    return pd.DataFrame(R, index=idx, columns=cols), WorldTruth(lottery_premium)


def fetch_panel(cache_dir: str = DEFAULT_CACHE, fetch: bool = False, min_days: int = TRADING_DAYS * 15
                ) -> pd.DataFrame:
    """Daily returns of current S&P 500 members with long history, cache-first.

    **Cache-only** unless ``fetch=True`` (Wikipedia membership + Yahoo daily). Survivorship-biased and
    large-cap — stated, since the lottery effect is documented strongest in small/illiquid names.
    The bias is opted into explicitly (``allow_survivorship_bias=True``): magnitudes are upper bounds.
    """
    cache = os.path.join(cache_dir, "jackpot_panel.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    if not fetch:
        return pd.DataFrame()
    import sys
    sys.path.insert(0, REPO_ROOT)
    from quantlab.universe import sp500_symbols
    import yfinance as yf

    syms = sp500_symbols(allow_survivorship_bias=True)
    px = yf.download(syms, period="max", interval="1d", auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    ret = px.pct_change().loc["1999-01-01":]
    keep = [c for c in ret.columns if ret[c].dropna().shape[0] >= min_days]
    ret = ret[keep]
    ret.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_parquet(cache)
    return ret
