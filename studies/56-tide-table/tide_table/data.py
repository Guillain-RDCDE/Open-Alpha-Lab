"""Data for the CAPE study — an offline synthetic world, and the cached Shiller CAPE series.

  * :func:`synthetic_world` — **offline, deterministic**. A stationary CAPE series and a forward
    real-return series that loads **negatively** on CAPE by ``predicts``; 0 = the null. Pins the
    machinery offline.
  * :func:`fetch_shiller` — Shiller's CAPE (PE10) and the forward 1-year and 10-year real total
    returns, from the key-free datahub mirror; **cache-first**. Fingerprinted run in ``docs/results.md``.
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
    predicts: float

    @property
    def cape_informative(self) -> bool:
        return self.predicts != 0.0


def synthetic_world(
    n_years: int = 116, predicts: float = 0.9, cape_mean: float = 18.0, seed: int = 56
) -> tuple[pd.DataFrame, WorldTruth]:
    """A monthly CAPE/forward-return world — deterministic given ``seed``.

    CAPE mean-reverts around ``cape_mean`` (AR(1)); the forward 10-year real return loads on the
    **earnings yield** (1/CAPE) by ``predicts`` plus noise — so a high CAPE precedes low returns.
    ``predicts = 0`` is the null. Returns a frame with ``cape`` and ``fwd10``.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("1900-01-31", periods=n, freq="ME", name="date")
    cape = np.empty(n)
    cape[0] = cape_mean
    for t in range(1, n):
        cape[t] = cape_mean + 0.99 * (cape[t - 1] - cape_mean) + 1.2 * rng.standard_normal()
    cape = np.clip(cape, 4.0, 50.0)
    fwd10 = 0.0 + predicts * (1.0 / cape) + 0.03 * rng.standard_normal(n)
    return pd.DataFrame({"cape": cape, "fwd10": fwd10}, index=idx), WorldTruth(predicts)


def fetch_shiller(cache_dir: str = DEFAULT_CACHE, fetch: bool = False, start_year: int = 1900
                  ) -> pd.DataFrame:
    """Shiller CAPE (PE10) + forward 1y/10y real total return, cache-first.

    **Cache-only** unless ``fetch=True`` (downloads the datahub mirror). Real total return reinvests
    real dividends monthly; ``fwd1``/``fwd10`` are the annualised forward real returns over 1 and 10
    years. Returns a frame with ``cape, fwd1, fwd10`` (forward columns NaN near the end of the sample).
    """
    cache = os.path.join(cache_dir, "tide_table_cape.parquet")
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
    cape = df["PE10"].astype(float).replace(0.0, np.nan)
    rp, rd = df["Real Price"].astype(float), df["Real Dividend"].astype(float)
    rtr = (1.0 + (rp.pct_change() + (rd / rp) / 12.0).fillna(0.0)).cumprod()
    out = pd.DataFrame({
        "cape": cape,
        "fwd1": (rtr.shift(-12) / rtr) ** (1 / 1) - 1.0,
        "fwd10": (rtr.shift(-120) / rtr) ** (1 / 10) - 1.0,
    })
    out.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(cache)
    return out[out.index.year >= start_year]
