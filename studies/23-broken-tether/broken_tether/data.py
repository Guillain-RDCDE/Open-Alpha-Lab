"""Data access for the pairs-trading study — the two price series and where they come from.

A pairs trade lives on a *relationship*: two assets that move together, so the spread between them
mean-reverts. The tape is therefore a pair of daily closes, and the data layer keeps the desk's
offline/cache split:

    * :func:`synthetic_pair` — fully **offline**. Two log-prices built from a shared random-walk trend
      ``f_t`` plus a spread ``s_t``: ``A = f + ½·amp·s``, ``B = f − ½·amp·s``, so ``A − B ∝ s``. With
      ``revert_rho < 1`` the spread is a *stationary* AR(1) — the pair is genuinely **cointegrated** and
      a spread trade has something to mean-revert against. With ``revert_rho = 1`` the spread is itself a
      random walk, so the two series are independent I(1) walks that merely *drift together for a while*:
      a **spurious pair** (the null), where any apparent cointegration is luck. Deterministic given
      ``seed``.
    * :func:`fetch_closes` — cached daily closes for liquid ETFs from the shared
      ``_cache/<TICKER>_split_only.parquet`` files; the study forms pairs from them. **Cache-only**
      unless ``fetch=True``; the network import stays lazy, so the offline core never imports
      ``yfinance``.

Data choice, named up front: **split-only closes, worked in log space** — the hedge ratio and spread
are estimated on log-prices (so the relationship is multiplicative/return-based), and both legs of the
pair are charged the same series.
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
class PairTruth:
    """What the synthetic generator baked in, so a test can check the diagnostics recover it."""
    n_bars: int
    revert_rho: float         # AR(1) persistence of the spread; <1 cointegrated, ==1 spurious
    amp: float

    @property
    def is_cointegrated(self) -> bool:
        """True when the spread genuinely mean-reverts (a real, tradable pair)."""
        return self.revert_rho < 1.0


def synthetic_pair(
    n_bars: int = 4032,
    revert_rho: float = 0.93,
    amp: float = 0.15,
    trend_vol: float = 0.008,
    spread_innov: float = 0.03,
    idio: float = 0.004,
    start_price: float = 50.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, PairTruth]:
    """Two log-prices sharing a trend, with a spread that mean-reverts (``revert_rho<1``) or wanders (``==1``).

    ``A = f + ½·amp·s_norm + idio·η``, ``B = f − ½·amp·s_norm + idio·η`` where ``f`` is a shared
    random-walk trend and ``s`` an AR(1) spread. With ``revert_rho < 1`` the spread is stationary so the
    pair is **cointegrated**; with ``revert_rho = 1`` the spread is a random walk and the pair is
    **spurious** (two independent walks). Returns ``(prices, truth)`` with columns ``A, B`` (price
    levels). Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2008-01-02", periods=n_bars, name="date")
    f = np.cumsum(trend_vol * rng.standard_normal(n_bars))           # shared trend (log)
    s = np.empty(n_bars); s[0] = 0.0
    eps = rng.standard_normal(n_bars)
    for t in range(1, n_bars):
        s[t] = revert_rho * s[t - 1] + spread_innov * eps[t]
    s_norm = s / (s.std() if s.std() > 0 else 1.0)
    la = np.log(start_price) + f + 0.5 * amp * s_norm + idio * rng.standard_normal(n_bars)
    lb = np.log(start_price) + f - 0.5 * amp * s_norm + idio * rng.standard_normal(n_bars)
    prices = pd.DataFrame({"A": np.exp(la), "B": np.exp(lb)}, index=idx)
    return prices, PairTruth(n_bars=n_bars, revert_rho=revert_rho, amp=amp)


DEFAULT_TICKERS = ["SPY", "QQQ", "TLT", "GLD", "EWJ", "EWG", "EWQ", "EWU", "FXI", "USO"]


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
                out[tk] = c.rename(tk)
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
            out[tk] = c.rename(tk)
    return out
