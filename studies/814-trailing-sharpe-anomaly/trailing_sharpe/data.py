"""Data layer for Study 814 — Trailing-Sharpe Anomaly.

The claim under test (risk-adjusted momentum, e.g. **Rachev / Biglova et al 2004**,
and the *residual/idiosyncratic Sharpe* strand): rank a cross-section on each name's
**trailing 12-month Sharpe ratio** — the mean of its daily returns divided by their
standard deviation, measured over the ~12-month formation window but **skipping the most
recent month** (à la Jegadeesh-Titman 12-1 momentum). Buy the **high-Sharpe** names,
short the **low-Sharpe** names. The interesting question is honest: does risk-adjusting
momentum *add* anything, or is a Sharpe sort just **plain 12-1 momentum + low-vol
repackaged**?

Two ingredients, both offline-friendly once cached.

* **Real tape — a liquid US cross-section.** Daily OHLC for a fixed list of ~50 liquid
  US large-caps (``UNIVERSE`` below), pulled with yfinance through the
  ``quantlab.universe`` **survivorship guard** (``download_panel(...,
  allow_survivorship_bias=True)``). ``auto_adjust=True`` (total-return prices). The panel
  parquet is cached under this study's OWN ``_cache/`` (we point ``quantlab.universe``'s
  cache there via ``OVERNIGHT_CACHE`` *before* importing it).

  **Survivorship — named on the Signal axis.** ``UNIVERSE`` is a *current* membership
  list of names that are liquid mega-caps *today*; feeding it to a backward-looking panel
  omits the delisted / de-rated names and biases any cross-sectional result. The guard
  forces the opt-in; the caveat travels with every published number.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) with a TUNABLE knob ``edge``: each name carries a persistent
  latent "quality tilt" ``c_i`` that (a) sets the **volatility** of its daily returns —
  so a trailing-Sharpe sort proxies ``c_i`` — and (b) — only when ``edge > 0`` — lifts
  its **forward mean** return. ``edge = 0`` is the null world: Sharpe still varies across
  names (through the persistent vol structure) but carries **no** information about
  forward returns, and the sort must find nothing. ``edge > 0`` plants a clean
  high-Sharpe → high-forward-return relation for the machinery to recover.

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
# Synthetic world — planted high-Sharpe -> high-return relation (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 814,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    daily_vol: float = 0.012,
    drift: float = 0.06 / 252,
    factor_rho: float = 0.995,
    vol_gain: float = 0.55,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC panel with a TUNABLE planted Sharpe->return relation.

    Each asset ``i`` carries a persistent latent "quality tilt" ``c_i[t]`` — a
    slowly-mean-reverting AR(1) with autocorrelation ``factor_rho`` (near-unit so the
    tilt is stable across a 12-month formation window). The tilt sets the **volatility**
    of the daily return (high tilt → low vol), so its trailing **Sharpe** ratio is
    monotone in ``c_i``; and — only when ``edge > 0`` — the tilt lifts the **forward
    mean**:

        vol_i[t] = daily_vol * exp(-vol_gain * c_i[t])       # high tilt -> low vol
        z ~ N(0,1)
        r[i,t] = drift + edge * c_i[t] + vol_i[t] * z

    A high positive tilt makes returns **low-volatility** (hence high trailing Sharpe,
    since the mean drift is divided by a small std) *and* (with ``edge > 0``) higher-mean
    — so a long-high-Sharpe / short-low-Sharpe sort earns a positive spread. ``edge = 0``
    is the null: the forward mean is the constant ``drift`` for every name, so trailing
    Sharpe (which still varies through the vol structure) predicts **nothing**. This
    deliberately entangles Sharpe with low-vol — exactly the confound the real-tape test
    must disentangle. Business-day index; span well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    panel: dict[str, pd.DataFrame] = {}

    innov_sd = np.sqrt(1.0 - factor_rho ** 2)
    for i in range(n_assets):
        c = np.empty(n_days)
        c[0] = rng.normal(0.0, 1.0)
        eps = rng.normal(0.0, innov_sd, n_days)
        for t in range(1, n_days):
            c[t] = factor_rho * c[t - 1] + eps[t]

        vol = daily_vol * np.exp(-vol_gain * c)
        z = rng.normal(0.0, 1.0, n_days)
        r = drift + edge * c + vol * z

        close = 100.0 * np.cumprod(1.0 + r)
        prev_close = np.concatenate([[100.0], close[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0.0, daily_vol / 3, n_days))
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close}, index=idx
        )
    return panel
