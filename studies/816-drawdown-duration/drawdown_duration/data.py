"""Data layer for Study 816 — Drawdown Duration.

The claim under test: the **fraction of the trailing year** each name spent
**underwater** — its cumulative total return sitting **below its running high-water
mark** — is a *persistent-drawdown* risk signal. Sort the cross-section into fractiles
on time-underwater and test the forward long-short spread: does the market **pay** for
bearing persistent-drawdown names (a positive high-minus-low spread, a risk premium), or
do they simply **keep sinking** (a negative spread)? Honest sign — we report the tape.

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
  cross-sectional result. The guard forces the opt-in; the caveat travels with every
  published number. A drawdown-duration sort is *especially* exposed: the names that
  spent the most time underwater and then **died** are exactly the ones a survivor
  panel drops, so the survivor magnitudes are an upper bound on the good news.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) with a TUNABLE ``knob``: each name carries a persistent latent
  "drift tilt" ``c_i``. When ``knob > 0`` a low-tilt name drifts **down**, so it slips
  and stays **below its high-water mark** — spending **more of the trailing year
  underwater** — *and*, because the tilt persists, it **keeps sinking** (a lower forward
  mean). So a high time-underwater goes with a **lower** forward return: the
  "persistent-drawdown names keep sinking" plant, recovered as a **negative**
  long-high-underwater / short-low-underwater spread. ``knob = 0`` is the null world: all
  names share one drift, time-underwater becomes pure path noise (time-below-HWM is
  scale-invariant, so the independent per-name volatility does **not** move it), and the
  sort must find nothing.

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
    """Cached panel as ``{ticker: DataFrame[Open, High, Low, Close]}``, sliced to
    ``[start, asof]``. Reads the parquet directly — OFFLINE, no yfinance import."""
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
# Synthetic world — planted time-underwater -> return relation (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    knob: float = 0.0,
    seed: int = 816,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    daily_vol: float = 0.010,
    drift: float = 0.12 / 252,
    vol_gain: float = 0.25,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC panel with a TUNABLE planted time-underwater relation.

    Each asset ``i`` carries a fixed latent "drift tilt" ``c_i ~ N(0,1)`` and an
    **independent** idiosyncratic volatility ``vol_i = daily_vol * exp(vol_gain * v_i)``.
    All names share the base ``drift``; only when ``knob > 0`` does the tilt shift the
    mean:

        z ~ N(0,1)
        r[i,t] = drift + knob * c_i + vol_i * z

    A low-tilt (negative ``c_i``) name then drifts **down**, slips and *stays* below its
    high-water mark — spending **more of the trailing year underwater** — and, because
    the tilt persists, it **keeps sinking** (a lower forward mean). So a high
    time-underwater goes with a **lower** forward return, which a long-high-underwater /
    short-low-underwater sort recovers as a **negative** spread. ``knob = 0`` is the null:
    every name shares one drift, so time-underwater is pure path noise (time-below-HWM is
    scale-invariant, so ``vol_i`` does not move it) and predicts nothing. Business-day
    index; span well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    panel: dict[str, pd.DataFrame] = {}

    for i in range(n_assets):
        c = float(rng.normal(0.0, 1.0))            # fixed per-name drift tilt
        v = float(rng.normal(0.0, 1.0))            # independent vol draw
        vol_i = daily_vol * float(np.exp(vol_gain * v))
        z = rng.normal(0.0, 1.0, n_days)
        r = drift + knob * c + vol_i * z

        close = 100.0 * np.cumprod(1.0 + r)
        prev_close = np.concatenate([[100.0], close[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0.0, vol_i / 3, n_days))
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, vol_i / 2, n_days)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, vol_i / 2, n_days)))
        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close}, index=idx
        )
    return panel
