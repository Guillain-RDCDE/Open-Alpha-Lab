"""Data for the net-issuance study — an offline synthetic panel, and the cached EDGAR shares pull."""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")


@dataclass(frozen=True)
class WorldTruth:
    issuance_premium: float

    @property
    def has_premium(self) -> bool:
        return self.issuance_premium != 0.0


def synthetic_panel(n_firms=200, n_years=18, issuance_premium=0.05, base_ret=0.09, ret_vol=0.30, seed=64):
    """A firm panel for the issuance sort — deterministic given ``seed``. Each firm-year draws a net
    issuance ~N(0, 0.1); next year's return is ``base_ret - issuance_premium*z(issuance) + noise`` so
    high issuers underperform when the premium is on (the textbook direction). 0 = the null."""
    rng = np.random.default_rng(seed)
    years = np.arange(2008, 2008 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]
    iss = 0.1 * rng.standard_normal((n_years, n_firms))
    z = (iss - iss.mean(axis=1, keepdims=True)) / (iss.std(axis=1, keepdims=True) + 1e-9)
    fwd = base_ret - issuance_premium * z + ret_vol * rng.standard_normal((n_years, n_firms))
    return (pd.DataFrame(iss, index=pd.Index(years, name="year"), columns=firms),
            pd.DataFrame(fwd, index=pd.Index(years, name="year"), columns=firms),
            WorldTruth(issuance_premium))


def fetch_panel(cache_dir=DEFAULT_CACHE, fetch=False):
    """Net-issuance signal + aligned next-year returns, cache-first (reads the shared EDGAR pull:
    ``_edgar_WeightedAverageNumberOfDilutedSharesOutstanding`` + ``_edgar_yrret``)."""
    sh_p = os.path.join(cache_dir, "_edgar_WeightedAverageNumberOfDilutedSharesOutstanding.parquet")
    r_p = os.path.join(cache_dir, "_edgar_yrret.parquet")
    if not (os.path.exists(sh_p) and os.path.exists(r_p)):
        return pd.DataFrame(), pd.DataFrame()
    from .strategy import net_issuance
    sh = pd.read_parquet(sh_p); yr_ret = pd.read_parquet(r_p)
    sig = net_issuance(sh)
    fwd = yr_ret.reindex(sig.index + 1); fwd.index = sig.index; fwd.index.name = "year"
    return sig, fwd
