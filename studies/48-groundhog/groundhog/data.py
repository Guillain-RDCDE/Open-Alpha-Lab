"""Data for the return-seasonality study — an offline synthetic cross-section, and a cached stock panel.

  * :func:`synthetic_panel` — **offline, deterministic**. A cross-section of stocks each given a fixed
    per-(stock, calendar-month) seasonal bias of strength ``seasonality`` plus noise — so the same
    month repeats by construction (and the *other*-month control shouldn't). ``seasonality = 0`` is the
    null. Pins the machinery offline.
  * :func:`fetch_panel` — monthly returns for current S&P 500 members with ≥20y history (Yahoo),
    **cache-first** and **guarded**: the panel is the current survivor universe projected backwards
    *and* filtered on ≥20y of history (double-conditioned on the future), so it raises
    :class:`quantlab.hf_data.SurvivorshipBiasError` unless the caller opts in with
    ``allow_survivorship_bias=True`` — the same pattern as ``quantlab.hf_data.load_panel``.
    Fingerprinted run in ``docs/results.md``.
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
    seasonality: float

    @property
    def has_seasonality(self) -> bool:
        return self.seasonality != 0.0


def synthetic_panel(
    n_stocks: int = 120, n_years: int = 28, seasonality: float = 0.02, vol: float = 0.06, seed: int = 48
) -> tuple[pd.DataFrame, WorldTruth]:
    """A monthly stock panel with a fixed (stock × calendar-month) seasonal bias — deterministic by seed.

    Each (stock, month-of-year) cell draws a constant bias ~N(0, ``seasonality``); a stock's return in
    that month = its fixed bias + noise, so the same calendar month repeats while other months don't.
    ``seasonality = 0`` is the null (pure noise). Returns a (date × stock) monthly-return frame.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("1996-01-31", periods=n, freq="ME", name="date")
    bias = seasonality * rng.standard_normal((12, n_stocks))   # fixed per (calendar-month, stock)
    month0 = idx.month.to_numpy() - 1
    R = bias[month0, :] + vol * rng.standard_normal((n, n_stocks))
    cols = [f"S{j:03d}" for j in range(n_stocks)]
    return pd.DataFrame(R, index=idx, columns=cols), WorldTruth(seasonality)


def fetch_panel(
    cache_dir: str = DEFAULT_CACHE, fetch: bool = False, min_months: int = 240,
    allow_survivorship_bias: bool = False,
) -> pd.DataFrame:
    """Monthly returns of current S&P 500 members with long history, cache-first — **GUARDED**.

    This panel is **survivorship-biased twice over**: it takes *current* S&P 500 membership
    (delisted losers are absent) and then keeps only names with at least ``min_months`` of history
    (another condition on having survived). Any cross-sectional magnitude measured on it is biased
    upward — which is why, like ``quantlab.hf_data.load_panel``, it refuses to hand the panel over
    unless you pass ``allow_survivorship_bias=True`` and document the bias where you use it. The
    bias is a property of the panel, not of the download, so the guard applies cache-first too.

    **Cache-only** unless ``fetch=True`` (Wikipedia membership + Yahoo monthly).
    """
    if not allow_survivorship_bias:
        import sys
        sys.path.insert(0, REPO_ROOT)
        from quantlab.hf_data import SurvivorshipBiasError

        raise SurvivorshipBiasError(
            "Refusing to build/load the groundhog panel: it is current S&P 500 membership "
            "projected backwards, then filtered on >=20y of history — double-conditioned on the "
            "future, so cross-sectional magnitudes are biased upward. Pass "
            "allow_survivorship_bias=True to override (and state the bias where the numbers land)."
        )
    cache = os.path.join(cache_dir, "groundhog_panel.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    if not fetch:
        return pd.DataFrame()
    import sys
    sys.path.insert(0, REPO_ROOT)
    from quantlab.universe import sp500_symbols
    import yfinance as yf

    try:  # the universe helper may carry the same opt-in guard — pass it through if so
        syms = sp500_symbols(allow_survivorship_bias=True)
    except TypeError:
        syms = sp500_symbols()
    px = yf.download(syms, period="max", interval="1mo", auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    ret = px.resample("ME").last().pct_change()
    keep = [c for c in ret.columns if ret[c].dropna().shape[0] >= min_months]
    ret = ret[keep].loc["1995-01-01":]
    ret.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_parquet(cache)
    return ret
