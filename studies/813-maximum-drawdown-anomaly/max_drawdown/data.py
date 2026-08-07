"""Data layer for Study 813 — the Maximum-Drawdown Anomaly.

The claim under test: sort a cross-section of stocks on each name's **trailing
12-month maximum drawdown** (the largest peak-to-trough decline of its cumulative
total return) and ask whether the recently **distressed** names — the ones that just
suffered the deepest drawdown — go on to **under-earn** (a distress / flight-to-quality
premium) or **rebound** (a reversal). The desk takes no prior on the sign; the point is
to sort into fractiles, measure the forward long-short spread, and be honest about which
way it points.

Two ingredients, both offline-friendly once cached.

* **Real tape — a liquid US cross-section.** Daily OHLC for a fixed list of ~50 liquid
  US large-caps (``UNIVERSE`` below), pulled with yfinance through the
  ``quantlab.universe`` **survivorship guard** (``download_panel(...,
  allow_survivorship_bias=True)``). ``auto_adjust=True`` (total-return prices, so a
  drawdown of the cumulative total return is exactly what the cache stores). The panel
  parquet is cached under this study's OWN ``_cache/`` (we point ``quantlab.universe``'s
  cache there via ``OVERNIGHT_CACHE`` *before* importing it).

  **Survivorship — named on the Signal axis.** ``UNIVERSE`` is a *current* membership
  list of names that are liquid mega-caps *today*; feeding it to a backward-looking panel
  omits the delisted / de-rated names and biases any cross-sectional result. This bites
  a drawdown study especially hard: the deepest drawdowns of all — the names that fell
  and never recovered — are the ones survivorship deletes. The guard forces the opt-in;
  the caveat travels with every published number.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) with a TUNABLE knob ``edge``: each name carries a persistent
  latent "fragility" ``f_i`` that both (a) deepens its drawdowns (higher volatility and
  larger downside jumps, so a trailing-MaxDD sort proxies ``f_i``) and (b) — only when
  ``edge > 0`` — depresses its forward mean return. ``edge = 0`` is the null world:
  drawdowns still vary across names but carry **no** information about forward returns,
  and the sort must find nothing. ``edge > 0`` plants the distress-underperforms pattern
  (deep-drawdown names under-earn). The jump component is de-meaned per name so it only
  shapes the drawdown, never the arithmetic daily mean — keeping the null exactly flat.

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
        df = raw[s][["Open", "High", "Low", "Close", "Volume"]].dropna()
        df = df[(df.index >= lo) & (df.index <= hi)]
        if not df.empty:
            panel[s] = df
    return panel


# --------------------------------------------------------------------------- #
# Synthetic world — planted distress->underperformance relation (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 813,
    n_assets: int = 40,
    n_days: int = 1500,
    start: str = "2010-01-04",
    daily_vol: float = 0.010,
    drift: float = 0.06 / 252,
    vol_gain: float = 1.6,
    jump_p: float = 0.02,
    jump_scale: float = 0.045,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC panel with a TUNABLE planted MaxDD->return relation.

    Each asset ``i`` carries a persistent latent "fragility" ``f_i ~ U(0,1)`` drawn once.
    Fragile names run **hotter** — daily vol ``daily_vol*(1 + vol_gain*f_i)`` — and take
    **larger, more frequent downside jumps** (probability ``jump_p*(0.3 + f_i)``, size
    ``-jump_scale*(0.5 + f_i)*|N(1,0.3)|``). Both deepen the trailing maximum drawdown,
    so a MaxDD sort proxies ``f_i``. The jump stream is **de-meaned per name** so it adds
    only downside *shape*, never drift; the daily mean is therefore exactly

        r[i,t] = drift - edge * f_i + vol_i * z[t] + (jump[t] - mean(jump))

    So a high-fragility name has deep drawdowns *and* (with ``edge > 0``) a lower forward
    mean — the distress-underperforms pattern: high past MaxDD, low forward return.
    ``edge = 0`` is the null: drawdowns still vary across names but predict nothing (the
    arithmetic daily mean is ``drift`` for every name, deep-drawdown or calm).
    Business-day index; span well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    panel: dict[str, pd.DataFrame] = {}

    for i in range(n_assets):
        f = float(rng.uniform(0.0, 1.0))          # persistent fragility
        vol_i = daily_vol * (1.0 + vol_gain * f)

        z = rng.normal(0.0, 1.0, n_days)
        hit = rng.random(n_days) < jump_p * (0.3 + f)
        size = -jump_scale * (0.5 + f) * np.abs(rng.normal(1.0, 0.3, n_days))
        jump = np.where(hit, size, 0.0)
        jump = jump - jump.mean()                 # de-mean: shape only, no drift bias

        r = drift - edge * f + vol_i * z + jump

        close = 100.0 * np.cumprod(1.0 + r)
        prev_close = np.concatenate([[100.0], close[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0.0, daily_vol / 3, n_days))
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        vol = np.abs(rng.normal(1e6, 2e5, n_days))
        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close, "Volume": vol},
            index=idx,
        )
    return panel
