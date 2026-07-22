"""Data layer for Study 788 — Overnight / Intraday Tug of War.

Two ingredients, both offline-friendly once cached.

* **Real tape — a liquid US cross-section.** Daily OHLC for a fixed list of ~50
  liquid US large-caps (``UNIVERSE`` below), pulled with yfinance through the
  ``quantlab.universe`` **survivorship guard** (``download_panel(...,
  allow_survivorship_bias=True)``). ``auto_adjust=True`` (total-return prices);
  the same daily adjustment factor scales Open and Close, so the exact night/day
  identity survives it. The panel parquet is cached under this study's OWN
  ``_cache/`` (we point ``quantlab.universe``'s cache there via
  ``OVERNIGHT_CACHE`` *before* importing it).

  **Survivorship — named on the Signal axis.** ``UNIVERSE`` is a *current*
  membership list of names that are liquid mega-caps *today*; feeding it to a
  backward-looking panel omits the delisted / de-rated names and biases any
  cross-sectional result **upward**. The guard forces the opt-in; the caveat
  travels with every published number.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) whose overnight and intraday legs are driven by a
  persistent latent per-stock "overnight-demand" factor with a TUNABLE tug knob:
  the overnight leg loads ``+tug`` on the factor (persistence), the intraday leg
  loads ``-tug`` (reversal). ``tug = 0`` is the null world — trailing overnight
  return carries no information about either forward leg, and the sort must NOT
  manufacture a tug. ``tug > 0`` plants exactly the Lou-Polk-Skouras pattern.

The offline path is pure numpy + pandas + stdlib. ``fetch()`` (network) runs
once to build the cache and is never imported by the notebooks' offline cells;
``load_panel()`` reads the cached parquet directly (no yfinance import).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
# Point quantlab.universe's cache at THIS study's _cache/ before importing it.
os.environ.setdefault("OVERNIGHT_CACHE", CACHE_DIR)

from quantlab.universe import (  # noqa: E402  (after the env var is set)
    SurvivorshipBiasError,
    download_panel,
    panel_cache_path,
)

START = "2010-01-01"        # panel start (matches quantlab.universe default)
AS_OF = "2026-06-30"        # last complete calendar month at publication (2026-07)

# A fixed list of ~50 liquid US large-caps. This is *current* membership -> a
# survivor set (see the module docstring): delisted / de-rated names are absent,
# so any cross-sectional result is an upper bound. Named on the Signal axis.
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "HD", "BAC", "XOM", "CVX", "KO", "PEP", "ABBV",
    "COST", "MRK", "PFE", "CSCO", "ORCL", "ADBE", "CRM", "NKE", "DIS", "MCD",
    "TXN", "INTC", "QCOM", "AMD", "IBM", "GE", "CAT", "BA", "MMM", "HON",
    "UNH", "T", "VZ", "WFC", "GS", "MS", "C", "AXP", "LMT", "UPS",
]

__all__ = [
    "SurvivorshipBiasError",
    "UNIVERSE", "START", "AS_OF", "CACHE_DIR",
    "fetch", "have_real", "load_panel", "synthetic_panel",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START) -> None:
    """Download the cross-section panel through the survivorship guard; cache it.

    Network; runs once. ``allow_survivorship_bias=True`` is the *documented*
    opt-in (``UNIVERSE`` is a current-membership survivor set — see module
    docstring). Writes the batched parquet under this study's ``_cache/``.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    download_panel(
        UNIVERSE, start=start, use_cache=True, allow_survivorship_bias=True,
    )


def have_real() -> bool:
    return os.path.exists(panel_cache_path(UNIVERSE, START))


def load_panel(start: str = START, asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """Cached panel as ``{ticker: DataFrame[Open, High, Low, Close]}``, sliced to
    ``[start, asof]``. Reads the parquet directly — OFFLINE, no yfinance import.

    Mirrors ``quantlab.universe.download_panel``'s post-processing so the two
    paths return the same shape.
    """
    cache = panel_cache_path(UNIVERSE, start)
    raw = pd.read_parquet(cache)
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    panel: dict[str, pd.DataFrame] = {}
    for s in UNIVERSE:
        if s not in raw.columns.get_level_values(0):
            continue
        df = raw[s][["Open", "High", "Low", "Close"]].dropna()
        df = df[(df.index >= lo) & (df.index <= hi)]
        if not df.empty:
            panel[s] = df
    return panel


# --------------------------------------------------------------------------- #
# Synthetic world — planted Lou-Polk-Skouras tug (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    tug: float = 0.0,
    seed: int = 788,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    daily_vol: float = 0.012,
    drift: float = 0.06 / 252,
    factor_rho: float = 0.92,
    factor_sd: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC panel with a TUNABLE planted overnight/intraday tug.

    Each asset ``i`` carries a persistent latent "overnight-demand" factor
    ``d_i[t]`` — an AR(1) with autocorrelation ``factor_rho`` — that a trailing
    overnight-return sort can proxy. The two legs load on it with opposite signs:

        r_overnight[i,t] = drift/2 + tug * d_i[t] + noise_on
        r_intraday [i,t] = drift/2 - tug * d_i[t] + noise_id

    So a stock with a high recent overnight run (high ``d``) keeps earning
    overnight (**persistence**) and gives it back intraday (**reversal**) — the
    Lou-Polk-Skouras tug of war. ``tug = 0`` is the null: both legs are pure
    noise, trailing overnight return predicts neither forward leg, and the sort
    must find nothing. Returns synthetic OHLC (Open/Close consistent with the
    two legs; High/Low bracket them) so the same ``strategy`` code runs on it.

    Business-day index; span well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    panel: dict[str, pd.DataFrame] = {}

    # Per-asset persistent latent overnight-demand factor (AR(1)).
    innov_sd = factor_sd * np.sqrt(1.0 - factor_rho ** 2)
    for i in range(n_assets):
        d = np.empty(n_days)
        d[0] = rng.normal(0.0, factor_sd)
        eps = rng.normal(0.0, innov_sd, n_days)
        for t in range(1, n_days):
            d[t] = factor_rho * d[t - 1] + eps[t]

        r_on = drift / 2 + tug * d + rng.normal(0.0, daily_vol, n_days)
        r_id = drift / 2 - tug * d + rng.normal(0.0, daily_vol, n_days)

        prev_close = 100.0 * np.ones(n_days)
        open_ = np.empty(n_days)
        close = np.empty(n_days)
        pc = 100.0
        for t in range(n_days):
            prev_close[t] = pc
            o = pc * (1.0 + r_on[t])
            c = o * (1.0 + r_id[t])
            open_[t], close[t] = o, c
            pc = c
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close}, index=idx
        )
    return panel
