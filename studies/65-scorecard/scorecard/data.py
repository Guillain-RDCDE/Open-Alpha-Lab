"""Data for the F-score study — an offline synthetic panel, and the cached EDGAR concept pull."""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np, pandas as pd

from .strategy import CONCEPTS, piotroski_fscore

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")


@dataclass(frozen=True)
class WorldTruth:
    fscore_premium: float

    @property
    def has_premium(self) -> bool:
        return self.fscore_premium != 0.0


def synthetic_panel(n_firms=200, n_years=18, fscore_premium=0.04, base_ret=0.09, ret_vol=0.30, seed=65):
    """A firm panel with a known F-score effect — deterministic given ``seed``. Each firm-year draws an
    integer F-score in 0–9 (uniform-ish); next year's return is
    ``base_ret + fscore_premium*(F-4.5)/4.5 + noise`` so high-F firms outperform when the premium is on
    (the textbook direction). 0 = the null. Returns (fscore_frame, fwd_ret, truth)."""
    rng = np.random.default_rng(seed)
    years = np.arange(2008, 2008 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]
    f = rng.integers(0, 10, size=(n_years, n_firms)).astype(float)
    fwd = base_ret + fscore_premium * (f - 4.5) / 4.5 + ret_vol * rng.standard_normal((n_years, n_firms))
    return (pd.DataFrame(f, index=pd.Index(years, name="year"), columns=firms),
            pd.DataFrame(fwd, index=pd.Index(years, name="year"), columns=firms),
            WorldTruth(fscore_premium))


def fetch_panel(cache_dir=DEFAULT_CACHE, fetch=False):
    """F-score signal + aligned next-year returns, cache-first (reads the shared EDGAR pull: the nine
    ``_edgar_<concept>`` frames + ``_edgar_yrret``)."""
    paths = {c: os.path.join(cache_dir, f"_edgar_{c}.parquet") for c in CONCEPTS}
    r_p = os.path.join(cache_dir, "_edgar_yrret.parquet")
    if not (all(os.path.exists(p) for p in paths.values()) and os.path.exists(r_p)):
        return pd.DataFrame(), pd.DataFrame()
    panels = {c: pd.read_parquet(p) for c, p in paths.items()}
    yr_ret = pd.read_parquet(r_p)
    f = piotroski_fscore(panels)
    fwd = yr_ret.reindex(f.index + 1); fwd.index = f.index; fwd.index.name = "year"
    return f, fwd
