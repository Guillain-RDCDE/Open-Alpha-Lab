"""Data for the Fed-Model study — an offline synthetic world, and the cached Shiller series.

  * :func:`synthetic_world` — **offline, deterministic**. An earnings yield, a 10-year yield, and an
    equity return whose next-month value loads on the **E/P deviation** (valuation) by ``ep_predicts``
    — and *not* on the bond yield. So the Fed signal (E/P − y10) can never beat E/P alone by
    construction; ``ep_predicts = 0`` is the null. Pins the "the bond term is inert" logic offline.
  * :func:`fetch_shiller` — Robert Shiller's monthly S&P series (price, earnings, dividend, 10-year
    yield, CPI) via the key-free datahub mirror, back to 1871; **cache-first**. Fingerprinted run in
    ``docs/results.md``.
"""

from __future__ import annotations

import io
import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
SHILLER_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"


@dataclass(frozen=True)
class WorldTruth:
    ep_predicts: float

    @property
    def ep_informative(self) -> bool:
        return self.ep_predicts != 0.0


def synthetic_world(
    n_years: int = 100, ep_predicts: float = 0.5, ep_mean: float = 0.06, y10_mean: float = 0.05,
    seed: int = 47
) -> tuple[pd.DataFrame, WorldTruth]:
    """A monthly world where only E/P (not the bond yield) forecasts equities — deterministic by seed.

    E/P mean-reverts around ``ep_mean``; the 10-year yield wanders around ``y10_mean`` *independently*.
    Next month's equity return loads on the E/P deviation by ``ep_predicts`` plus noise — so the Fed
    signal (E/P − y10), which adds the irrelevant yield, can only dilute E/P's information.
    ``ep_predicts = 0`` is the null. Returns a frame with ``tr, ep, y10, bond``.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("1925-01-31", periods=n, freq="ME", name="date")

    def ar1(mean, vol, phi):
        x = np.empty(n)
        x[0] = mean
        for t in range(1, n):
            x[t] = mean + phi * (x[t - 1] - mean) + vol * rng.standard_normal()
        return x

    # STATIONARY (mean-reverting) E/P and 10y — a random walk would manufacture spurious regressions.
    ep = ar1(ep_mean, 0.004, 0.97)
    y10 = ar1(y10_mean, 0.004, 0.98)         # independent of E/P by construction
    ep_dev = ep - ep_mean
    tr = 0.005 + ep_predicts * np.concatenate([[0.0], ep_dev[:-1]]) + 0.04 * rng.standard_normal(n)
    bond = y10 / 12.0
    return (pd.DataFrame({"tr": tr, "ep": ep, "y10": y10, "bond": bond}, index=idx),
            WorldTruth(ep_predicts))


def fetch_shiller(cache_dir: str = DEFAULT_CACHE, fetch: bool = False, start_year: int = 1900
                  ) -> pd.DataFrame:
    """Shiller monthly S&P series → a frame with ``tr, ep, y10, bond, infl``, cache-first.

    **Cache-only** unless ``fetch=True`` (downloads the key-free datahub mirror). ``tr`` is the monthly
    total return (price change + dividend/12), ``ep`` the trailing earnings yield, ``y10`` the 10-year
    yield, ``bond`` a monthly cash/bond proxy (y10/12), ``infl`` trailing-12m CPI inflation.
    """
    cache = os.path.join(cache_dir, "paper_moon_shiller.parquet")
    if os.path.exists(cache):
        d = pd.read_parquet(cache)
        return d[d.index.year >= start_year]
    if not fetch:
        return pd.DataFrame()
    raw = urllib.request.urlopen(urllib.request.Request(SHILLER_URL, headers={"User-Agent": "Mozilla/5.0"}),
                                 timeout=30).read()
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    sp, earn = df["SP500"].astype(float), df["Earnings"].astype(float)
    y10 = df["Long Interest Rate"].astype(float) / 100.0
    div, cpi = df["Dividend"].astype(float), df["Consumer Price Index"].astype(float)
    out = pd.DataFrame({
        "tr": sp.pct_change() + (div / sp) / 12.0,
        "ep": earn / sp,
        "y10": y10,
        "bond": y10 / 12.0,
        "infl": cpi.pct_change(12),
    })
    out.index.name = "date"
    out = out.dropna(subset=["ep", "y10"])
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(cache)
    return out[out.index.year >= start_year]
