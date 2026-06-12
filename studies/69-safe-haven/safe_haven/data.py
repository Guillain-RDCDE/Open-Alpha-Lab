"""Data for the gold safe-haven study — an offline synthetic world, and the cached gold / equity / CPI
series."""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
MONTHS = 12


@dataclass(frozen=True)
class WorldTruth:
    inflation_beta: float

    @property
    def hedges_inflation(self) -> bool:
        return self.inflation_beta != 0.0


def synthetic_world(n_years=20, inflation_beta=0.8, seed=69):
    """A monthly world: CPI follows a slow inflation path; gold's YoY return loads on YoY inflation with
    coefficient ``inflation_beta`` (plus noise). ``inflation_beta=0`` ⇒ gold ignores inflation (the
    null). Returns (gold_level, eq_ret, cpi_level, truth)."""
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("2005-01-31", periods=n, freq="ME", name="date")
    # CPI: a smooth multi-year inflation cycle (deterministic) + a small drift — so YoY inflation is a
    # clean wave a hedge can track, and pure noise robustly does not.
    t = np.arange(n)
    infl_m = 0.0025 + 0.004 * np.sin(2 * np.pi * t / 60.0)      # ~5-year cycle, 0.1–0.7%/mo
    cpi = 100 * np.cumprod(1 + infl_m)
    # gold monthly return: base + beta on the monthly inflation swing + noise. When beta>0 gold's YoY
    # return tracks YoY inflation; beta=0 ⇒ gold is pure noise (the null).
    g_m = 0.004 + inflation_beta * (infl_m - infl_m.mean()) + 0.012 * rng.standard_normal(n)
    gold = 100 * np.cumprod(1 + g_m)
    eq = 0.007 + 0.04 * rng.standard_normal(n)        # equities, ~independent of gold
    return (pd.Series(gold, index=idx, name="gold"), pd.Series(eq, index=idx, name="eq"),
            pd.Series(cpi, index=idx, name="cpi"), WorldTruth(inflation_beta))


def fetch_panel(cache_dir=DEFAULT_CACHE, fetch=False):
    """Monthly gold (GLD) level, equity (SPY) monthly return, and CPI level, cache-first (from the shared
    ``cross_asset_etfs`` and ``macro_us`` pulls). Returns (gold_m, eq_ret_m, cpi_m)."""
    ca_p = os.path.join(cache_dir, "cross_asset_etfs.parquet")
    mac_p = os.path.join(cache_dir, "macro_us.parquet")
    if os.path.exists(ca_p) and os.path.exists(mac_p):
        ca = pd.read_parquet(ca_p); ca.index = pd.DatetimeIndex(ca.index).tz_localize(None)
        gold_m = ca["GLD"].dropna().resample("ME").last()
        eq_ret = ca["SPY"].dropna().resample("ME").last().pct_change()
        cpi = pd.read_parquet(mac_p)["cpi"].dropna()
        cpi.index = pd.DatetimeIndex(cpi.index); cpi_m = cpi.resample("ME").last()
        return gold_m, eq_ret, cpi_m
    if not fetch:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    import yfinance as yf
    px = yf.download(["GLD", "SPY"], period="max", auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    gold_m = px["GLD"].dropna().resample("ME").last()
    eq_ret = px["SPY"].dropna().resample("ME").last().pct_change()
    return gold_m, eq_ret, pd.Series(dtype=float)
