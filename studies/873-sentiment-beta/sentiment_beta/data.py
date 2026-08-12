"""Data layer for Study 873 — Sentiment Beta.

The claim under test (Baker & Wurgler 2006, 2007 — *"Investor Sentiment and the
Cross-Section of Stock Returns"*): the stocks whose returns **co-move most with
market sentiment** — high *sentiment beta* — are the speculative, hard-to-value names
that get **over-priced in euphoria** and **under-perform afterwards**, especially once
sentiment has peaked. Sort the cross-section on each name's beta to a sentiment gauge;
the theory says a long **low-sentiment-beta** / short **high-sentiment-beta** book
earns a *positive* spread (the euphoria-chasers under-earn).

Two ingredients, both offline-friendly once cached.

* **Real tape — a liquid US cross-section.** Daily OHLC(V) for a fixed list of ~50
  liquid US large-caps (``UNIVERSE`` below), pulled with yfinance through the
  ``quantlab.universe`` **survivorship guard** (``download_panel(...,
  allow_survivorship_bias=True)``). ``auto_adjust=True`` (total-return prices). The
  panel parquet is cached under this study's OWN ``_cache/`` (we point
  ``quantlab.universe``'s cache there via ``OVERNIGHT_CACHE`` *before* importing it).

  **The sentiment gauge is built from the same real tape.** Rather than reach for an
  external index that cannot be fetched offline, we proxy market sentiment with a
  **tradable high-minus-low-volatility spread** computed *inside* ``strategy`` from the
  panel itself: the daily return of the most-volatile (speculative) tercile minus the
  least-volatile (safe) tercile. It rises in risk-on euphoria (the lottery names are
  bid up) and falls in risk-off — a self-contained, tradable sentiment gauge, and one
  of the two proxies the task names (the other being the inverse of VIX). So the gauge
  is REAL data, not a fabrication.

  **Survivorship — named on the Signal axis.** ``UNIVERSE`` is a *current* membership
  list of names that are liquid mega-caps *today*; feeding it to a backward-looking
  panel omits the delisted / de-rated names and biases any cross-sectional result. The
  guard forces the opt-in; the caveat travels with every published number.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) with a TUNABLE knob ``edge``: a common latent *sentiment
  factor* ``f[t]`` drives every name, and each name carries a persistent loading
  ``b_i`` on it (its true sentiment beta). The loading shapes how strongly the name
  co-moves with the factor and — only when ``edge > 0`` — **depresses its forward mean
  return** (high-beta names under-earn, the Baker-Wurgler pattern). ``edge = 0`` is the
  null world: sentiment betas still vary across names but carry **no** information about
  forward returns, and the sort must find nothing.

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
# Synthetic world — planted sentiment-beta->return relation (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 873,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    daily_vol: float = 0.010,
    drift: float = 0.06 / 252,
    factor_vol: float = 0.011,
    factor_rho: float = 0.0,
    beta_spread: float = 1.2,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC panel with a TUNABLE planted sentiment-beta relation.

    A single common **sentiment factor** ``f[t]`` (an AR(1) with autocorrelation
    ``factor_rho`` — by default ``0.0``, i.e. iid daily shocks, matching the low
    autocorrelation of a real daily high-minus-low-vol spread) drives the whole
    cross-section. Each asset ``i`` carries a
    persistent loading ``b_i`` (its *true* sentiment beta, drawn once, dispersed by
    ``beta_spread``). The factor enters each name's daily return through ``b_i``, and —
    only when ``edge > 0`` — a name's forward mean return is **depressed in proportion
    to its loading**:

        f[t]  = factor_rho * f[t-1] + factor_vol * sqrt(1-rho^2) * u[t]
        r[i,t] = drift - edge * b_i + b_i * f[t] + daily_vol * z[i,t]

    So a high loading makes the name co-move strongly with the speculative factor
    (high **estimated** sentiment beta, since the high-|b| names populate the
    high-volatility tercile that defines the gauge) *and* — with ``edge > 0`` —
    lower-mean: the Baker-Wurgler pattern (high sentiment beta, low forward return).
    ``edge = 0`` is the null: betas still vary but predict nothing.

    Business-day index; span well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)

    # Persistent common sentiment factor (AR(1), unconditional sd ~ factor_vol).
    innov_sd = factor_vol * np.sqrt(1.0 - factor_rho ** 2)
    f = np.empty(n_days)
    f[0] = rng.normal(0.0, factor_vol)
    fu = rng.normal(0.0, innov_sd, n_days)
    for t in range(1, n_days):
        f[t] = factor_rho * f[t - 1] + fu[t]

    # Persistent per-name loadings (the true sentiment betas), dispersed & centred.
    loadings = rng.normal(0.0, beta_spread, n_assets)

    panel: dict[str, pd.DataFrame] = {}
    for i in range(n_assets):
        b = float(loadings[i])
        z = rng.normal(0.0, 1.0, n_days)
        r = drift - edge * b + b * f + daily_vol * z

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
