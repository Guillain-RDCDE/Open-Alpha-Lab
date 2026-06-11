"""Data for the accruals study — an offline synthetic panel, and the cached EDGAR fundamentals.

  * :func:`synthetic_panel` — **offline, deterministic**. Firms with a random accruals score; next-year
    return loads **negatively** on accruals by ``accruals_premium``; 0 = the null.
  * :func:`fetch_panel` — accruals (EDGAR NetIncome − OperatingCashFlow ÷ Assets) and aligned next-year
    returns for current S&P 500 members, **cache-first** (reads the desk's shared EDGAR pull). Honest
    caveats: short XBRL sample (~2007+) and survivorship — stated in the verdict.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")


@dataclass(frozen=True)
class WorldTruth:
    accruals_premium: float

    @property
    def has_premium(self) -> bool:
        return self.accruals_premium != 0.0


def synthetic_panel(
    n_firms: int = 200, n_years: int = 20, accruals_premium: float = 0.05, base_ret: float = 0.09,
    ret_vol: float = 0.30, seed: int = 52
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A firm panel for the accruals sort — deterministic given ``seed``.

    Each firm-year draws an accruals score ~N(0,1); next year's return is
    ``base_ret − accruals_premium·z(accruals) + noise``, so high-accruals firms underperform when the
    premium is on. ``accruals_premium = 0`` is the null. Returns ``(signal, fwd_ret, truth)``.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2000 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]
    a = rng.standard_normal((n_years, n_firms))
    z = (a - a.mean(axis=1, keepdims=True)) / (a.std(axis=1, keepdims=True) + 1e-9)
    fwd = base_ret - accruals_premium * z + ret_vol * rng.standard_normal((n_years, n_firms))
    return (pd.DataFrame(a, index=pd.Index(years, name="year"), columns=firms),
            pd.DataFrame(fwd, index=pd.Index(years, name="year"), columns=firms),
            WorldTruth(accruals_premium))


def fetch_panel(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accruals signal + aligned next-year returns, cache-first (reads the desk's shared EDGAR pull).

    **Cache-only** unless ``fetch=True``. Reads ``_edgar_NetIncomeLoss``,
    ``_edgar_NetCashProvidedByUsedInOperatingActivities``, ``_edgar_Assets``, ``_edgar_yrret``. Returns
    ``(signal, fwd_ret)`` aligned so ``fwd_ret.loc[y]`` is the year after the fundamentals in
    ``signal.loc[y]``.
    """
    ni_p = os.path.join(cache_dir, "_edgar_NetIncomeLoss.parquet")
    cfo_p = os.path.join(cache_dir, "_edgar_NetCashProvidedByUsedInOperatingActivities.parquet")
    a_p = os.path.join(cache_dir, "_edgar_Assets.parquet")
    r_p = os.path.join(cache_dir, "_edgar_yrret.parquet")
    if not all(os.path.exists(p) for p in (ni_p, cfo_p, a_p, r_p)):
        if not fetch:
            return pd.DataFrame(), pd.DataFrame()
        _crawl_edgar(cache_dir)
    ni, cfo, assets = pd.read_parquet(ni_p), pd.read_parquet(cfo_p), pd.read_parquet(a_p)
    yr_ret = pd.read_parquet(r_p)
    from .strategy import accruals
    sig = accruals(ni, cfo, assets)
    fwd = yr_ret.reindex(sig.index + 1)
    fwd.index = sig.index
    fwd.index.name = "year"
    return sig, fwd


def _crawl_edgar(cache_dir: str) -> None:  # pragma: no cover - network
    """Rebuild the shared EDGAR caches (NetIncome, CFO, Assets) + annual returns. Slow; network.

    Uses *current* S&P 500 membership projected backwards (explicit
    ``allow_survivorship_bias=True`` opt-in): magnitudes read as upper bounds.
    """
    import json
    import sys
    import time
    import urllib.request
    sys.path.insert(0, REPO_ROOT)
    from quantlab.universe import sp500_symbols
    import yfinance as yf

    ua = {"User-Agent": "OpenAlphaLab research guillain@poulpe.us"}

    def get(url):
        for _ in range(3):
            try:
                return urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=30).read()
            except Exception:
                time.sleep(1.0)
        return None

    syms = sp500_symbols(allow_survivorship_bias=True)[:420]
    t2c = {v["ticker"]: str(v["cik_str"]).zfill(10)
           for v in json.loads(get("https://www.sec.gov/files/company_tickers.json")).values()}
    concepts = ["NetIncomeLoss", "NetCashProvidedByUsedInOperatingActivities", "Assets"]

    def annual(cik, concept):
        j = get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json")
        if not j:
            return None
        try:
            d = json.loads(j)
            rows = [(u["end"], u["val"]) for u in d["units"]["USD"]
                    if u.get("form") == "10-K" and u.get("fp") == "FY"]
            s = pd.Series(dict(rows))
            s.index = pd.to_datetime(s.index)
            return s[~s.index.duplicated(keep="last")].sort_index().groupby(lambda x: x.year).last()
        except Exception:
            return None

    store = {c: {} for c in concepts}
    for tk in syms:
        cik = t2c.get(tk) or t2c.get(tk.replace("-", "."))
        if not cik:
            continue
        for c in concepts:
            s = annual(cik, c)
            if s is not None:
                store[c][tk] = s
            time.sleep(0.05)
    px = yf.download(syms, period="max", interval="1mo", auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    yr_ret = px.resample("YE").last().pct_change()
    yr_ret.index = yr_ret.index.year
    os.makedirs(cache_dir, exist_ok=True)
    pd.DataFrame(store["NetIncomeLoss"]).to_parquet(os.path.join(cache_dir, "_edgar_NetIncomeLoss.parquet"))
    pd.DataFrame(store["NetCashProvidedByUsedInOperatingActivities"]).to_parquet(
        os.path.join(cache_dir, "_edgar_NetCashProvidedByUsedInOperatingActivities.parquet"))
    pd.DataFrame(store["Assets"]).to_parquet(os.path.join(cache_dir, "_edgar_Assets.parquet"))
    yr_ret.to_parquet(os.path.join(cache_dir, "_edgar_yrret.parquet"))
