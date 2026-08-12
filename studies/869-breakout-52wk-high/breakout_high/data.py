"""Data layer for Study 869 — 52-Week-High Breakout Drift.

The claim under test: when a stock **closes at a fresh 52-week high** — a breakout
*event*, not merely being *near* its high (that is George-Hwang, study 236) — does it
go on to **drift up** (breakout momentum) or **fade** (resistance / anchoring)? We flag
each name's new-52-week-high days point-in-time and measure the forward 5/20-day return
of the just-broke-out book relative to the rest of the cross-section.

Two ingredients, both offline-friendly once cached.

* **Real tape — a liquid US cross-section.** Daily OHLC for a fixed list of ~50
  liquid US large-caps (``UNIVERSE`` below), pulled with yfinance through the
  ``quantlab.universe`` **survivorship guard** (``download_panel(...,
  allow_survivorship_bias=True)``). ``auto_adjust=True`` (total-return prices). The
  panel parquet is cached under this study's OWN ``_cache/`` (we point
  ``quantlab.universe``'s cache there via ``OVERNIGHT_CACHE`` *before* importing it).

  **Survivorship — named on the Signal axis.** ``UNIVERSE`` is a *current*
  membership list of names that are liquid mega-caps *today*; feeding it to a
  backward-looking panel omits the delisted / de-rated names and biases any
  cross-sectional result. It also makes fresh 52-week highs *more* frequent (survivors
  trend up), so the breakout-event base rate here is an upper bound. The guard forces
  the opt-in; the caveat travels with every published number.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) with a TUNABLE knob ``edge``: whenever a name printed a fresh
  52-week-high *yesterday*, ``edge > 0`` adds a small extra drift to its return today
  (a planted breakout->forward-drift relation). ``edge = 0`` is the null world:
  breakouts still happen (survivor-style up-drifting names print new highs) but carry
  **no** information about the forward return, and the event sort must find nothing.

The offline path is pure numpy + pandas + stdlib. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells; ``load_panel()``
reads the cached parquet directly (no yfinance import).
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
AS_OF = "2026-06-30"        # last complete calendar month at publication

# A fixed list of ~50 liquid US large-caps — *current* membership, a survivor set.
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
    """Download the cross-section panel through the survivorship guard; cache it."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    download_panel(
        UNIVERSE, start=start, use_cache=True, allow_survivorship_bias=True,
    )


def have_real() -> bool:
    return os.path.exists(panel_cache_path(UNIVERSE, START))


def load_panel(start: str = START, asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """Cached panel as ``{ticker: DataFrame[Open, High, Low, Close, Volume]}``, sliced
    to ``[start, asof]``. Reads the parquet directly — OFFLINE, no yfinance import."""
    cache = panel_cache_path(UNIVERSE, start)
    raw = pd.read_parquet(cache)
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    panel: dict[str, pd.DataFrame] = {}
    for s in UNIVERSE:
        if s not in raw.columns.get_level_values(0):
            continue
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"]
                if c in raw[s].columns]
        df = raw[s][cols].dropna(subset=["Close"])
        df = df[(df.index >= lo) & (df.index <= hi)]
        if not df.empty:
            panel[s] = df
    return panel


# --------------------------------------------------------------------------- #
# Synthetic world — planted breakout->forward-drift relation (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 869,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    daily_vol: float = 0.012,
    drift: float = 0.05 / 252,
    lookback: int = 252,
    persist: int = 20,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC panel with a TUNABLE planted breakout-drift relation.

    Each asset follows a lightly-drifting random walk. We build the path forward one day
    at a time; whenever a name printed a **fresh ``lookback``-day high** on day ``t-1``,
    the knob ``edge > 0`` adds a small extra drift to the next ``persist`` days' returns
    — so the names that *just broke out* go on to earn more (a breakout-momentum world).
    ``edge = 0`` is the null: fresh highs still occur (drifting survivors print them) but
    predict nothing about the forward return, and the event sort must find nothing.

    Business-day index; span well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)

    z = rng.normal(0.0, 1.0, size=(n_days, n_assets))
    close = np.empty((n_days, n_assets), dtype=float)
    close[0] = 100.0
    boost = np.zeros(n_assets, dtype=float)          # remaining boosted days per asset
    for t in range(1, n_days):
        extra = edge * (boost > 0)                   # planted forward drift after a breakout
        r = drift + extra + daily_vol * z[t]
        close[t] = close[t - 1] * (1.0 + r)
        boost = np.maximum(boost - 1.0, 0.0)
        # Did today print a fresh lookback-day high? If so, arm the boost for `persist` days.
        if t >= lookback:
            prior_max = close[t - lookback:t].max(axis=0)   # excludes today
            broke = close[t] > prior_max
            boost = np.where(broke, float(persist), boost)

    panel: dict[str, pd.DataFrame] = {}
    for i in range(n_assets):
        c = close[:, i]
        prev_close = np.concatenate([[100.0], c[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0.0, daily_vol / 3, n_days))
        hi = np.maximum(open_, c) * (1.0 + np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        lo = np.minimum(open_, c) * (1.0 - np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        vol = np.abs(rng.normal(1e6, 2e5, n_days))
        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": c, "Volume": vol}, index=idx
        )
    return panel
