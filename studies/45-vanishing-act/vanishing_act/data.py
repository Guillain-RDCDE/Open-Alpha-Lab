"""Data for the size-premium study — an offline synthetic world, and a cached real pairs panel.

  * :func:`synthetic_smb` — **offline, deterministic**. A market factor drives a small-cap and a
    large-cap series; the small leg carries an annual ``premium`` that can ``decay`` to zero (or
    reverse) over the sample. ``premium = 0`` is the null. Pins the SMB machinery offline.
  * :func:`fetch_pairs` — monthly total returns for the real size pairs (**^RUT/^GSPC** for the long
    history, **IWM/SPY** and **IJR/IVV** total-return ETF pairs), **cache-first**. Fingerprinted run in
    ``docs/results.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
MONTHS = 12

# Real size pairs (Yahoo): index pair for length, ETF pairs for tradable total return.
TICKERS = ["^RUT", "^GSPC", "IWM", "SPY", "IJR", "IVV"]
PAIRS = {"rut_gspc": ("^RUT", "^GSPC"), "iwm_spy": ("IWM", "SPY"), "ijr_ivv": ("IJR", "IVV")}


@dataclass(frozen=True)
class WorldTruth:
    premium: float
    decay: bool

    @property
    def has_premium(self) -> bool:
        return self.premium != 0.0


def synthetic_smb(
    n_years: int = 38, premium: float = 0.04, decay: bool = True,
    mkt_vol: float = 0.16, idio_vol: float = 0.10, seed: int = 45
) -> tuple[pd.DataFrame, WorldTruth]:
    """A small/large monthly world — deterministic given ``seed``.

    A market factor drives both legs (beta 1). The small leg adds an annual ``premium`` alpha; if
    ``decay`` is set, that premium ramps linearly from ``premium`` at the start to ``−premium`` at the
    end (worked, then reversed — the Banz story). ``premium = 0`` is the null. Returns a monthly-return
    frame with columns ``small`` and ``large``.
    """
    rng = np.random.default_rng(seed)
    n = n_years * MONTHS
    idx = pd.date_range("1988-01-31", periods=n, freq="ME", name="date")
    mkt = (mkt_vol / np.sqrt(MONTHS)) * rng.standard_normal(n)
    ramp = np.linspace(1.0, -1.0, n) if decay else np.ones(n)
    small_alpha = (premium / MONTHS) * ramp
    small = mkt + small_alpha + (idio_vol / np.sqrt(MONTHS)) * rng.standard_normal(n)
    large = mkt + (idio_vol / np.sqrt(MONTHS)) * rng.standard_normal(n)
    return pd.DataFrame({"small": small, "large": large}, index=idx), WorldTruth(premium, decay)


def fetch_pairs(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Monthly total returns for the real size pairs, cache-first.

    **Cache-only** unless ``fetch=True``. Returns a monthly-return frame with columns
    ``^RUT, ^GSPC, IWM, SPY, IJR, IVV`` (empty on a cache miss with ``fetch=False``).
    """
    cache = os.path.join(cache_dir, "vanishing_act_pairs.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf  # lazy

    px = yf.download(TICKERS, period="max", interval="1mo", auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    ret = px.resample("ME").last().pct_change().dropna(how="all")
    ret.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_parquet(cache)
    return ret
