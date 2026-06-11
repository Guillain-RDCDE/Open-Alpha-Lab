"""Data for the quality study — an offline synthetic panel, and the cached EDGAR fundamentals.

  * :func:`synthetic_panel` — **offline, deterministic**. Firms with a random quality score; the
    next-year return loads **positively** on quality by ``quality_premium``; 0 = the null.
  * :func:`fetch_panel` — gross profitability (EDGAR GrossProfit ÷ Assets) and aligned next-year
    returns for current S&P 500 members, **cache-first** (reads the desk's shared EDGAR pull). Honest
    caveats: short XBRL sample (~2007+) and survivorship (current members) — stated in the verdict.
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
    quality_premium: float

    @property
    def has_premium(self) -> bool:
        return self.quality_premium != 0.0


def synthetic_panel(
    n_firms: int = 200, n_years: int = 20, quality_premium: float = 0.05, base_ret: float = 0.09,
    ret_vol: float = 0.30, seed: int = 51
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A firm panel for the quality sort — deterministic given ``seed``.

    Each firm-year draws a quality score ~U(0,1) (a stand-in for gross profitability); next year's
    return is ``base_ret + quality_premium·z(quality) + noise``, so high-quality firms out-earn when the
    premium is on. ``quality_premium = 0`` is the null. Returns ``(signal, fwd_ret, truth)`` aligned.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2000 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]
    q = rng.uniform(0, 1, (n_years, n_firms))
    z = (q - q.mean(axis=1, keepdims=True)) / (q.std(axis=1, keepdims=True) + 1e-9)
    fwd = base_ret + quality_premium * z + ret_vol * rng.standard_normal((n_years, n_firms))
    return (pd.DataFrame(q, index=pd.Index(years, name="year"), columns=firms),
            pd.DataFrame(fwd, index=pd.Index(years, name="year"), columns=firms),
            WorldTruth(quality_premium))


def fetch_panel(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gross-profitability signal (GP/Assets) + aligned next-year returns, cache-first.

    **Cache-only** unless ``fetch=True``. Reads the desk's shared EDGAR pull (``_edgar_GrossProfit``,
    ``_edgar_Assets``, ``_edgar_yrret``). Returns ``(signal, fwd_ret)`` aligned so ``fwd_ret.loc[y]`` is
    the year after the fundamentals in ``signal.loc[y]``. Empty frames on a cache miss with
    ``fetch=False``.
    """
    gp_p = os.path.join(cache_dir, "_edgar_GrossProfit.parquet")
    a_p = os.path.join(cache_dir, "_edgar_Assets.parquet")
    r_p = os.path.join(cache_dir, "_edgar_yrret.parquet")
    if not (os.path.exists(gp_p) and os.path.exists(a_p) and os.path.exists(r_p)):
        if not fetch:
            return pd.DataFrame(), pd.DataFrame()
        _crawl_edgar(cache_dir)
    gp, assets = pd.read_parquet(gp_p), pd.read_parquet(a_p)
    yr_ret = pd.read_parquet(r_p)
    from .strategy import gross_profitability
    sig = gross_profitability(gp, assets)
    fwd = yr_ret.reindex(sig.index + 1)
    fwd.index = sig.index
    fwd.index.name = "year"
    return sig, fwd


def _crawl_edgar(cache_dir: str) -> None:  # pragma: no cover - network
    """Rebuild the shared EDGAR caches (GrossProfit, Assets) + annual returns. Slow; network.

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

    gp, assets = {}, {}
    for tk in syms:
        cik = t2c.get(tk) or t2c.get(tk.replace("-", "."))
        if not cik:
            continue
        for concept, store in (("GrossProfit", gp), ("Assets", assets)):
            s = annual(cik, concept)
            if s is not None:
                store[tk] = s
            time.sleep(0.05)
    px = yf.download(syms, period="max", interval="1mo", auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    yr_ret = px.resample("YE").last().pct_change()
    yr_ret.index = yr_ret.index.year
    os.makedirs(cache_dir, exist_ok=True)
    pd.DataFrame(gp).to_parquet(os.path.join(cache_dir, "_edgar_GrossProfit.parquet"))
    pd.DataFrame(assets).to_parquet(os.path.join(cache_dir, "_edgar_Assets.parquet"))
    yr_ret.to_parquet(os.path.join(cache_dir, "_edgar_yrret.parquet"))
