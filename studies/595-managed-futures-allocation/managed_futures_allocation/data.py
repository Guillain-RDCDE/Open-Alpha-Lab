"""Data layer for Study 595 — Managed-Futures Sleeve.

Three sources, all offline-friendly once cached:

* **Shared futures cache.** The desk's 18-contract continuous-futures daily-return panel
  (``REPO/_cache/trade_winds_futures.parquet``, built by study 31-trade-winds): equity index
  (ES/NQ/YM), rates (ZN/ZB/ZF), commodities (CL/GC/SI/HG/NG/ZC/ZS/ZW) and FX (6E/6J/6B/6A),
  2000-07 → 2026-06. This is the raw material for the simple 12-month TSMOM replication book.
  Read-only — this study never rewrites the shared cache.

* **ETF tape (yfinance, no key).** Daily **total-return** (auto-adjusted) closes for
  SPY (equities), VBMFX (Vanguard Total Bond, long history to cover 2000+), DBMF and KMLM
  (the live managed-futures ETFs, 2019-05 / 2020-12 inceptions) plus ^IRX (13-week T-bill
  discount yield, the risk-free leg). Cached wide under the study's own ``_cache/``.

* **Synthetic world.** A deterministic, seeded joint (stock, bond, MF) monthly generator with
  a TUNABLE planted MF Sharpe (the diversification benefit knob) and a tunable MF-bond
  correlation. ``mf_sharpe=0`` is the null: a zero-edge, zero-correlation sleeve must NOT
  let the ΔSharpe bootstrap manufacture an improvement. It is the machinery proof only —
  never cited in support of a stamp.

Pure numpy + pandas + stdlib on the offline path; ``fetch_etfs`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
ETF_CACHE = os.path.join(CACHE_DIR, "mf_etfs.csv")

# The shared futures cache lives at the repo root; historical layouts have used both
# _cache/ and _cache/_cache/ — probe both, never write either.
_REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FUTURES_PATHS = [
    os.path.join(_REPO, "_cache", "trade_winds_futures.parquet"),
    os.path.join(_REPO, "_cache", "_cache", "trade_winds_futures.parquet"),
]

ETF_TICKERS = ["SPY", "VBMFX", "DBMF", "KMLM", "^IRX"]

# Contract sectors (for the attribution / bonds-in-disguise axis)
SECTORS = {
    "ES=F": "equity", "NQ=F": "equity", "YM=F": "equity",
    "ZN=F": "rates", "ZB=F": "rates", "ZF=F": "rates",
    "CL=F": "commodity", "GC=F": "commodity", "SI=F": "commodity", "HG=F": "commodity",
    "NG=F": "commodity", "ZC=F": "commodity", "ZS=F": "commodity", "ZW=F": "commodity",
    "6E=F": "fx", "6J=F": "fx", "6B=F": "fx", "6A=F": "fx",
}


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def futures_path() -> str | None:
    """Path of the shared futures parquet, or None if absent."""
    for p in FUTURES_PATHS:
        if os.path.exists(p):
            return p
    return None


def load_futures() -> pd.DataFrame:
    """Daily simple returns, 18 continuous futures (wide; index = date)."""
    p = futures_path()
    if p is None:
        raise FileNotFoundError("shared trade_winds_futures.parquet not found at repo root _cache")
    df = pd.read_parquet(p)
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def fetch_etfs(start: str = "1999-01-01", end: str | None = None,
               path: str = ETF_CACHE, retries: int = 3) -> pd.DataFrame:
    """Download the ETF/rate tape once and cache it (network-only path).

    SPY/VBMFX/DBMF/KMLM are auto-adjusted closes (TOTAL-RETURN); ^IRX is a 13-week T-bill
    discount-yield LEVEL in percent (not a price).
    """
    import yfinance as yf

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(ETF_TICKERS, start=start, end=end, auto_adjust=True,
                              progress=False)["Close"]
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    raw = raw.dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_etfs(path: str = ETF_CACHE) -> bool:
    return os.path.exists(path)


def load_etfs(path: str = ETF_CACHE) -> pd.DataFrame:
    """Cached wide frame: SPY/VBMFX/DBMF/KMLM total-return closes + ^IRX yield level (%)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def have_real() -> bool:
    return futures_path() is not None and have_etfs()


# --------------------------------------------------------------------------- #
# Monthly building blocks
# --------------------------------------------------------------------------- #
def monthly_from_daily_returns(daily: pd.DataFrame, min_days: int = 15) -> pd.DataFrame:
    """Compound daily simple returns into calendar-month returns (NaN if < min_days obs)."""
    out = (1.0 + daily).resample("ME").prod() - 1.0
    cnt = daily.resample("ME").count()
    return out.where(cnt >= min_days)


def monthly_total_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end-to-month-end simple returns from a (total-return) price frame."""
    m = prices.resample("ME").last()
    return m.pct_change()


def monthly_rf(irx: pd.Series) -> pd.Series:
    """Monthly risk-free return from the ^IRX 13-week discount-yield level.

    rf for month t = the yield level observed at the END of month t-1 (annual %, /100/12) —
    known in advance, no look-ahead.
    """
    y = irx.dropna().resample("ME").last() / 100.0
    return (y / 12.0).shift(1)


def drop_partial_last_month(monthly: pd.DataFrame | pd.Series,
                            last_daily_date: pd.Timestamp):
    """Drop the final monthly bucket if the underlying daily tape stops mid-month."""
    if len(monthly) == 0:
        return monthly
    last_bucket = monthly.index.max()
    if pd.Timestamp(last_daily_date) < last_bucket:  # bucket label is the true month-end
        return monthly.iloc[:-1]
    return monthly


# --------------------------------------------------------------------------- #
# Synthetic world (machinery proof only)
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 312, mf_sharpe: float = 0.0, rho_bond: float = 0.0,
                    seed: int = 595) -> pd.DataFrame:
    """Deterministic joint monthly world: stock, bond, MF excess returns + a flat rf.

    * stock: 5%/yr excess mean, 15% vol; bond: 1.5%/yr excess mean, 5% vol, corr +0.1
      to stocks (calm-regime style).
    * mf: 12% vol; annualised **planted Sharpe = ``mf_sharpe``** (the tunable effect) and
      correlation ``rho_bond`` to the bond leg, zero to stocks. ``mf_sharpe=0`` is the null.

    Returns a DataFrame (index = month-end timestamps via ``period_range`` — spans stay far
    below the 250-year ns-Timestamp trap) with columns ``stock``, ``bond``, ``mf`` (all
    monthly EXCESS returns) and ``rf`` (flat 2%/yr monthly).
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2000-01", periods=n_months, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()

    z = rng.standard_normal((n_months, 3))
    zs, zb_raw, zm_raw = z[:, 0], z[:, 1], z[:, 2]
    zb = 0.1 * zs + np.sqrt(1 - 0.1 ** 2) * zb_raw          # bond, corr .1 to stock
    zm = rho_bond * zb + np.sqrt(max(1 - rho_bond ** 2, 0.0)) * zm_raw  # mf, corr to bond

    stock = 0.05 / 12.0 + (0.15 / np.sqrt(12.0)) * zs
    bond = 0.015 / 12.0 + (0.05 / np.sqrt(12.0)) * zb
    mf_vol_m = 0.12 / np.sqrt(12.0)
    mf = (mf_sharpe * 0.12) / 12.0 + mf_vol_m * zm

    return pd.DataFrame({"stock": stock, "bond": bond, "mf": mf,
                         "rf": 0.02 / 12.0}, index=idx)
