"""Data layer for Study 810 — Price Delay.

The claim under test (**Hou & Moskowitz 2005**, *"Market Frictions, Price Delay, and
the Cross-Section of Expected Returns"*): stocks into which market-wide information
diffuses **slowly** — whose price responds to the market with a lag — earn a **return
premium** over stocks that price the same information promptly. The "delay" measure is
built per name from a weekly regression of the stock's return on the contemporaneous
market return plus four weekly lags of the market: a name whose lagged-market
coefficients add a lot of explanatory power is a **high-delay** name.

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
  forces the opt-in; the caveat travels with every published number. Delay is documented
  as a **small, illiquid, neglected-stock** effect, so a 50-mega-cap survivor panel is
  exactly where it is *least* expected to appear.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) with a TUNABLE knob ``knob``: each name carries a persistent
  latent **delay parameter** ``d_i`` that (a) makes its daily return load on the market's
  **lagged** value (so the weekly delay regression recovers ``d_i``) and (b) — only when
  ``knob > 0`` — lifts its forward mean return. ``knob = 0`` is the null world: delay
  still varies across names but carries **no** information about forward returns, and the
  sort must find nothing. ``knob > 0`` plants the Hou-Moskowitz slow-diffusion premium.

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
# Synthetic world — planted slow-diffusion (delay) premium (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    knob: float = 0.0,
    seed: int = 810,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    mkt_vol: float = 0.012,
    idio_vol: float = 0.004,
    drift: float = 0.06 / 252,
    beta: float = 1.0,
    lag: int = 5,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC panel with a TUNABLE planted price-delay premium.

    A single daily market factor ``mkt[t]`` drives every name. Each asset ``i`` carries a
    persistent **delay parameter** ``d_i in [0, dmax]`` that splits its market exposure
    between the **contemporaneous** and the ``lag``-day-**lagged** factor, plus an idio
    shock and — only when ``knob > 0`` — a forward-return premium proportional to ``d_i``:

        r[i,t] = drift + knob * d_i
                 + beta * ((1 - d_i) * mkt[t] + d_i * mkt[t - lag])
                 + idio[i,t]

    A high ``d_i`` name therefore prices the market with a one-week lag (``lag=5`` trading
    days) — so a weekly delay regression (contemporaneous market + 4 weekly lags) recovers
    a **high** delay — and, with ``knob > 0``, earns a **higher** forward mean: the
    Hou-Moskowitz slow-diffusion premium. ``knob = 0`` is the null: delay still varies
    across names but predicts nothing. Business-day index; span well below the pandas
    ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    mkt = rng.normal(0.0, mkt_vol, n_days)
    lagged = np.concatenate([np.zeros(lag), mkt[:-lag]])
    d = rng.uniform(0.0, 0.8, n_assets)   # per-name delay parameter

    panel: dict[str, pd.DataFrame] = {}
    for i in range(n_assets):
        di = d[i]
        idio = rng.normal(0.0, idio_vol, n_days)
        r = (drift + knob * di
             + beta * ((1.0 - di) * mkt + di * lagged)
             + idio)

        close = 100.0 * np.cumprod(1.0 + r)
        prev_close = np.concatenate([[100.0], close[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0.0, idio_vol / 3, n_days))
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, idio_vol / 2, n_days)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, idio_vol / 2, n_days)))
        vol = rng.integers(1_000_000, 5_000_000, n_days).astype(float)
        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close, "Volume": vol},
            index=idx,
        )
    return panel
