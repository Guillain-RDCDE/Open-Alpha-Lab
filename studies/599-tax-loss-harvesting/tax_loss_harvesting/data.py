"""Data layer for Study 599 — Tax-Loss Harvesting.

Two sources, both offline-friendly once cached:

* **Real tape.** SPY daily **total-return** closes (yfinance ``auto_adjust=True``) from the
  1993 inception — the asset the tax-loss-harvesting (TLH) engine trades — plus IVV daily
  total-return closes (2000-05 on) used ONLY to justify the "twin fund" simplification: SPY and
  IVV track the same index, their daily returns correlate at ~0.999+, so a wash-sale-safe
  SPY↔IVV swap keeps the investor's market exposure economically unchanged while realising the
  tax loss. Both cached as CSV under ``_cache/``.

* **Synthetic.** A deterministic, seeded GBM world with a **tunable planted effect**: the
  volatility knob ``sigma`` drives how many harvestable drawdowns the path serves up, so TLH
  alpha must rise with ``sigma`` (the planted effect) and must collapse to ~zero minus costs
  when the tax rates are 0/0 or when ``sigma`` ≈ 0 (the nulls). Pure numpy; spans ≤ 250 years.

``fetch()`` (network) is used only once to build the cache and is never imported by the
notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
SPY_CACHE = os.path.join(CACHE_DIR, "tlh_spy.csv")
IVV_CACHE = os.path.join(CACHE_DIR, "tlh_ivv.csv")

# The frozen as-of for every headline run (June 2026 is the last complete month).
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "1993-01-01", end: str | None = None,
          spy_path: str = SPY_CACHE, ivv_path: str = IVV_CACHE,
          retries: int = 3) -> tuple[pd.Series, pd.Series]:
    """Download SPY + IVV total-return daily closes and cache them (network, run once)."""
    import yfinance as yf

    def _get(ticker: str) -> pd.Series:
        for _ in range(retries):
            try:
                raw = yf.download(ticker, start=start, end=end, auto_adjust=True,
                                  progress=False)["Close"]
                if raw is not None and len(raw) > 0:
                    s = raw.iloc[:, 0] if isinstance(raw, pd.DataFrame) else raw
                    s.index = pd.DatetimeIndex(s.index).tz_localize(None)
                    return s.dropna().sort_index()
            except Exception:
                time.sleep(2.0)
        raise RuntimeError(f"could not fetch {ticker}")

    spy = _get("SPY")
    ivv = _get("IVV")
    os.makedirs(CACHE_DIR, exist_ok=True)
    spy.rename("SPY").to_csv(spy_path)
    ivv.rename("IVV").to_csv(ivv_path)
    return spy, ivv


def have_real(spy_path: str = SPY_CACHE, ivv_path: str = IVV_CACHE) -> bool:
    return os.path.exists(spy_path) and os.path.exists(ivv_path)


def load_real(as_of: str | None = AS_OF) -> tuple[pd.Series, pd.Series]:
    """Cached SPY + IVV total-return closes, sliced to the frozen as-of."""
    spy = pd.read_csv(SPY_CACHE, index_col=0, parse_dates=True).iloc[:, 0].sort_index()
    ivv = pd.read_csv(IVV_CACHE, index_col=0, parse_dates=True).iloc[:, 0].sort_index()
    if as_of is not None:
        cut = pd.Timestamp(as_of)
        spy, ivv = spy[spy.index <= cut], ivv[ivv.index <= cut]
    return spy, ivv


def twin_check(spy: pd.Series, ivv: pd.Series) -> dict:
    """Justify the twin-fund simplification: SPY vs IVV daily-return agreement."""
    both = pd.concat([spy.pct_change(), ivv.pct_change()], axis=1, join="inner").dropna()
    both.columns = ["SPY", "IVV"]
    corr = float(both["SPY"].corr(both["IVV"]))
    diff = both["SPY"] - both["IVV"]
    te_ann = float(diff.std() * np.sqrt(252) * 1e4)          # tracking diff vol, bps/yr
    gap_ann = float(diff.mean() * 252 * 1e4)                 # mean return gap, bps/yr
    return {"corr": corr, "te_ann_bps": te_ann, "gap_ann_bps": gap_ann,
            "n_days": int(len(both))}


# --------------------------------------------------------------------------- #
# Synthetic world — tunable planted effect (volatility) + nulls
# --------------------------------------------------------------------------- #
def synthetic_path(n_years: int = 10, mu: float = 0.06, sigma: float = 0.18,
                   seed: int = 599) -> pd.Series:
    """Deterministic GBM daily price path.

    ``sigma`` is the TUNABLE planted effect: TLH harvests drawdowns, so its after-tax alpha
    must rise with ``sigma`` and vanish as ``sigma`` → 0 (one null). The other null is running
    the engine with tax rates 0/0 on any path — the alpha must then be exactly the trading
    costs, i.e. ~zero. Business-day index kept well under the 250-year pandas ns-Timestamp cap.
    """
    rng = np.random.default_rng(seed)
    n = int(252 * n_years)
    dt = 1.0 / 252
    z = rng.standard_normal(n)
    logret = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    px = 100.0 * np.exp(np.cumsum(logret))
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(px, index=idx, name="SYN")
