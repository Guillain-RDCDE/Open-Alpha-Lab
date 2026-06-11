"""Data for the asset-growth study — an offline synthetic panel, and a cached EDGAR + price panel.

  * :func:`synthetic_panel` — **offline, deterministic**. Firms with random yearly asset growth whose
    next-year return carries a ``growth_penalty`` (fast growers underperform); 0 = the null. Pins the
    long-short machinery offline.
  * :func:`fetch_panel` — the real panel: **total assets from SEC EDGAR** (us-gaap ``Assets``, 10-K)
    and **July→June annual total returns from Yahoo** for a large-cap survivor universe, **cache-first**
    (the heavy EDGAR crawl runs only on ``fetch=True``). Fingerprinted run in ``docs/results.md``.

**No look-ahead by construction.** Portfolios are formed on **June 30 of year y+1** from fiscal-year-y
total assets — the Cooper-Gulen-Schill (2008) convention — and we enforce it with EDGAR's ``filed``
dates: a firm-year enters the sort only if its 10-K was actually on file by the formation date (10-Ks
land in February–March; late filers are dropped, not peeked at). The forward return is the July(y+1) →
June(y+2) window, so the sort never uses a number the market hadn't seen.

Honest caveat, stated not hidden: the real universe is *current* S&P 500 members, so it carries
**survivorship bias** and is all large-cap. Censoring the delisted high-asset-growth disasters props up
the measured return of the *short* leg, which biases the hedge **against** the effect — a panel on
which a null (or reversed-sign) hedge is expected even if the premium is real elsewhere. That direction
is part of the verdict, not swept under it.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
SEC_UA = {"User-Agent": "OpenAlphaLab research guillain@poulpe.us"}


@dataclass(frozen=True)
class WorldTruth:
    growth_penalty: float

    @property
    def has_penalty(self) -> bool:
        return self.growth_penalty != 0.0


def synthetic_panel(
    n_firms: int = 200, n_years: int = 20, growth_penalty: float = 0.06,
    base_ret: float = 0.09, ret_vol: float = 0.30, seed: int = 44
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A firm panel for the asset-growth sort — deterministic given ``seed``.

    Each firm-year draws an asset growth ~N(0.08, 0.25); the next year's return is
    ``base_ret − growth_penalty·z(growth) + noise``, where ``z`` is the cross-sectionally standardised
    growth — so high growers underperform when the penalty is on. ``growth_penalty = 0`` is the null.
    Returns ``(ag, fwd_ret, truth)`` as (year × firm) frames, already aligned (``fwd_ret.loc[y]`` is
    what ``ag.loc[y]`` predicts).
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2000 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]
    ag = 0.08 + 0.25 * rng.standard_normal((n_years, n_firms))
    z = (ag - ag.mean(axis=1, keepdims=True)) / (ag.std(axis=1, keepdims=True) + 1e-9)
    fwd = base_ret - growth_penalty * z + ret_vol * rng.standard_normal((n_years, n_firms))
    return (pd.DataFrame(ag, index=pd.Index(years, name="year"), columns=firms),
            pd.DataFrame(fwd, index=pd.Index(years, name="year"), columns=firms),
            WorldTruth(growth_penalty))


def fetch_panel(cache_dir: str = DEFAULT_CACHE, fetch: bool = False, max_names: int = 400
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Real asset-growth panel (EDGAR total assets + Yahoo July→June returns), cache-first.

    **Cache-only** unless ``fetch=True`` (then crawls SEC EDGAR for ``Assets`` on up to ``max_names``
    current S&P 500 names — survivorship-biased and large-cap, stated openly — and downloads monthly
    total-return prices). Returns ``(ag, fwd_ret)`` indexed by **fiscal year** ``y``:

      * ``ag.loc[y]`` — fiscal-year-``y`` asset growth, masked to NaN unless the 10-K reporting it was
        **filed by June 30 of y+1** (the formation date — no look-ahead);
      * ``fwd_ret.loc[y]`` — the **July(y+1) → June(y+2)** total return that a June-30 formation earns.

    Prices are pinned to the desk's as-of (``quantlab.repro.DEFAULT_AS_OF``) and return windows that
    haven't completed by the as-of are dropped, so a partial year can never leak into a headline mean.
    Empty frames on a cache miss with ``fetch=False``.
    """
    ag_path = os.path.join(cache_dir, "growth_spurt_ag.parquet")
    ret_path = os.path.join(cache_dir, "growth_spurt_ret.parquet")
    if os.path.exists(ag_path) and os.path.exists(ret_path):
        return pd.read_parquet(ag_path), pd.read_parquet(ret_path)
    if not fetch:
        return pd.DataFrame(), pd.DataFrame()

    import sys
    sys.path.insert(0, REPO_ROOT)
    from quantlab.repro import DEFAULT_AS_OF
    from quantlab.universe import sp500_symbols
    import yfinance as yf

    asof = pd.Timestamp(DEFAULT_AS_OF)

    def _get(url, tries=3):
        for _ in range(tries):
            try:
                return urllib.request.urlopen(urllib.request.Request(url, headers=SEC_UA), timeout=30).read()
            except Exception:
                time.sleep(1.0)
        return None

    # Explicit opt-in: the bias is documented in the README/results and is itself part of the verdict
    # (censoring delisted high-growth disasters biases the hedge AGAINST the effect).
    syms = sp500_symbols(allow_survivorship_bias=True)[:max_names]
    cmap = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
    t2c = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in cmap.values()}

    assets, filed = {}, {}
    for tk in syms:
        cik = t2c.get(tk) or t2c.get(tk.replace("-", "."))
        if not cik:
            continue
        j = _get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/Assets.json")
        if not j:
            continue
        try:
            d = json.loads(j)
            rows = [(u["end"], u["val"], u.get("filed"))
                    for u in d["units"]["USD"] if u.get("form") == "10-K" and u.get("fp") == "FY"]
            if rows:
                s = pd.Series({end: val for end, val, _ in rows})
                s.index = pd.to_datetime(s.index)
                assets[tk] = s[~s.index.duplicated(keep="last")].sort_index()
                # first date each fiscal-year-end value was on file (comparatives re-file it later)
                f = pd.DataFrame(rows, columns=["end", "val", "filed"]).dropna(subset=["filed"])
                f["end"] = pd.to_datetime(f["end"])
                f["filed"] = pd.to_datetime(f["filed"])
                filed[tk] = f.groupby("end")["filed"].min().sort_index()
        except Exception:
            pass
        time.sleep(0.08)

    ag = pd.DataFrame({tk: s.groupby(s.index.year).last().pct_change() for tk, s in assets.items()})
    ag.index.name = "year"

    # No look-ahead: fiscal-year-y growth enters the June-30(y+1) formation only if the 10-K
    # reporting it was filed by then. Missing filed dates are dropped, not assumed timely.
    filed_yr = pd.DataFrame({tk: f.groupby(f.index.year).min() for tk, f in filed.items()})
    filed_yr = filed_yr.reindex(index=ag.index, columns=ag.columns)
    cutoffs = pd.Series({y: pd.Timestamp(int(y) + 1, 6, 30) for y in ag.index})
    on_file = filed_yr.notna() & filed_yr.le(cutoffs, axis=0)
    ag = ag.where(on_file)

    px = yf.download(list(assets), period="max", interval="1mo", auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    px = px[px.index <= asof]
    jun = px.resample("YE-JUN").last()
    yr_ret = jun.pct_change()  # July→June windows, labelled by the ending June
    yr_ret.index = yr_ret.index.year
    # drop the in-progress window (its nominal June-30 end is after the as-of)
    yr_ret = yr_ret[[pd.Timestamp(int(y), 6, 30) <= asof for y in yr_ret.index]]
    # align: fiscal-year-y assets → June-30(y+1) formation → return over July(y+1)→June(y+2)
    fwd = yr_ret.reindex(ag.index + 2)
    fwd.index = ag.index
    fwd.index.name = "year"

    os.makedirs(cache_dir, exist_ok=True)
    ag.to_parquet(ag_path)
    fwd.to_parquet(ret_path)
    return ag, fwd
