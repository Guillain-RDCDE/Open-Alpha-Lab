"""Data access for the sector-rotation study — the sector panel and where it comes from.

Sector rotation ranks the equity sectors by recent performance and piles into the leaders. So the tape
is a panel of daily sector returns, and the data layer keeps the desk's offline/cache split:

    * :func:`synthetic_sectors` — fully **offline**. A small panel of synthetic "sectors": a shared
      market factor plus a slow, *persistent* per-sector relative drift (so a sector that has led tends
      to keep leading — the rotation momentum). ``mom_strength`` sets the drift amplitude;
      ``mom_strength = 0`` is the **null** (no persistence, rotation chases noise). Deterministic.
    * :func:`fetch_sectors` — cached daily closes for the 11 SPDR sector ETFs from the shared
      ``_cache/<TICKER>_split_only.parquet`` files, differenced to returns. **Cache-only** unless
      ``fetch=True``; the network import stays lazy.

Data choice: split-only closes for the SPDR sectors (XLK, XLF, XLE, XLV, XLI, XLP, XLY, XLU, XLB, XLRE,
XLC). The benchmark a rotation book must beat is *holding all of them equal-weight* — sector rotation
is only worth the turnover and concentration if it beats the diversified sector basket.
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

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC"]


@dataclass(frozen=True)
class PanelTruth:
    n_sectors: int
    n_bars: int
    mom_strength: float
    phi: float

    @property
    def has_momentum(self) -> bool:
        return self.mom_strength != 0.0


def synthetic_sectors(n_sectors: int = 11, n_bars: int = 4032, mom_strength: float = 0.0011,
                      phi: float = 0.985, mkt_drift: float = 0.0003, mkt_vol: float = 0.009,
                      idio: float = 0.006, seed: int = 0) -> tuple[pd.DataFrame, PanelTruth]:
    """A panel of synthetic sectors where the leaders **persist** — the rotation momentum, by construction."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2009-01-02", periods=n_bars, name="date")
    market = mkt_drift + mkt_vol * rng.standard_normal(n_bars)
    betas = np.clip(rng.normal(1.0, 0.15, n_sectors), 0.6, 1.4)
    theta = np.zeros((n_bars, n_sectors))
    innov = mom_strength * np.sqrt(1.0 - phi**2)
    nu = rng.standard_normal((n_bars, n_sectors))
    for t in range(1, n_bars):
        theta[t] = phi * theta[t - 1] + innov * nu[t]
    rets = betas[None, :] * market[:, None] + theta + idio * rng.standard_normal((n_bars, n_sectors))
    panel = pd.DataFrame(rets, index=idx, columns=[f"SEC{i:02d}" for i in range(n_sectors)])
    return panel, PanelTruth(n_sectors=n_sectors, n_bars=n_bars, mom_strength=mom_strength, phi=phi)


def fetch_sectors(tickers: list[str] | None = None, cache_dir: str = DEFAULT_CACHE,
                  min_days: int = 500, fetch: bool = False) -> pd.DataFrame:
    """Return a ``dates x sector`` daily-returns panel for the SPDR sectors, cache-first."""
    tickers = tickers or SECTORS
    closes = {}
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
        import yfinance as yf  # lazy
        raw = yf.download(tk, period="max", interval="1d", auto_adjust=False, progress=False)
        if raw is None or raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw[["Open", "High", "Low", "Close", "Volume"]].to_parquet(path)
        c = raw["Close"].dropna(); c.index = pd.DatetimeIndex(c.index).tz_localize(None)
        if len(c) >= min_days:
            closes[tk] = c
    if not closes:
        return pd.DataFrame()
    rets = pd.DataFrame(closes).sort_index().pct_change().dropna(how="all")
    rets.index.name = "date"
    return rets
