"""Data for the yield-curve study — an offline synthetic world, and a cached rates/equity panel."""
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
    predicts: float

    @property
    def curve_informative(self) -> bool:
        return self.predicts != 0.0


def synthetic_world(n_years=40, predicts=0.04, seed=66):
    """A monthly world where the curve slope (AR(1), occasionally inverting) forecasts the next 18m of
    equity returns: a more inverted curve precedes lower forward returns by ``predicts``. 0 = the null."""
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("1985-01-31", periods=n, freq="ME", name="date")
    slope = np.empty(n); slope[0] = 1.5
    for t in range(1, n):
        slope[t] = 0.97 * slope[t - 1] + 0.25 * rng.standard_normal()
    # equity monthly return: base, plus a slope-driven drift smeared over the prior 18 months
    base = 0.008
    drive = predicts * (slope - slope.mean()) / 12.0
    eq = base + np.roll(drive, 6) + (0.15 / np.sqrt(12)) * rng.standard_normal(n)
    return pd.DataFrame({"curve": slope, "eq": eq}, index=idx), WorldTruth(predicts)


def fetch_panel(cache_dir=DEFAULT_CACHE, fetch=False):
    """Monthly 10y (^TNX), 13-week (^IRX) yields and S&P 500 (^GSPC) returns, cache-first.
    Returns a frame with ``long_yield, short_yield, eq`` (eq = monthly S&P return)."""
    cache = os.path.join(cache_dir, "inverted_panel.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf
    px = yf.download(["^TNX", "^IRX", "^GSPC"], period="max", interval="1mo", auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    m = px.resample("ME").last()
    out = pd.DataFrame({"long_yield": m["^TNX"], "short_yield": m["^IRX"], "eq": m["^GSPC"].pct_change()}).dropna()
    out.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(cache)
    return out
