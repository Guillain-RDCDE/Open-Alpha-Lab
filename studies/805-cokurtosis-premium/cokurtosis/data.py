"""Data layer for Study 805 — Cokurtosis Premium.

The claim under test (Fang & Lai 1997, *"Co-Kurtosis and Capital Asset Pricing"*, and
the four-moment CAPM tradition): beyond beta (co-variance) and co-skewness, a security's
**systematic kurtosis** — its **cokurtosis** with the market — is a priced risk. A name
whose returns amplify the market's *cubed* deviations (its return spikes precisely when
the market has a fat-tailed move) loads on market tail-co-movement, an undesirable
exposure that investors must be paid to hold. So high cokurtosis should command a
**positive** risk premium, and a book that is **long high-cokurtosis / short
low-cokurtosis** should earn a positive spread.

The cokurtosis of name ``i`` over a trailing window is

    cokurt_i = E[(r_i - μ_i)(r_m - μ_m)^3] / (σ_i · σ_m^3)

where the market ``r_m`` is the **equal-weight mean of the panel**. It is a *fourth*-order
co-moment (one power on the name, three on the market) — distinct from co-**skewness**
(one power on the name, **two** on the market; study 504) and from the low-beta trade
(study 238 BAB, a co-**variance** tilt).

Two ingredients, both offline-friendly once cached.

* **Real tape — a liquid US cross-section.** Daily OHLC for a fixed list of ~50 liquid US
  large-caps (``UNIVERSE`` below), pulled with yfinance through the ``quantlab.universe``
  **survivorship guard** (``download_panel(..., allow_survivorship_bias=True)``).
  ``auto_adjust=True`` (total-return prices). The panel parquet is cached under this
  study's OWN ``_cache/`` (we point ``quantlab.universe``'s cache there via
  ``OVERNIGHT_CACHE`` *before* importing it).

  **Survivorship — named on the Signal axis.** ``UNIVERSE`` is a *current* membership list
  of names that are liquid mega-caps *today*; feeding it to a backward-looking panel omits
  the delisted / de-rated names and biases any cross-sectional result. The guard forces
  the opt-in; the caveat travels with every published number.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) with a TUNABLE knob ``knob``: each name carries a persistent latent
  "tail-co-movement tilt" ``c_i`` that both (a) makes the name's return load on the *cubed*
  market shock (so a trailing-cokurtosis sort proxies ``c_i``) and (b) — only when
  ``knob > 0`` — *raises* its forward mean return (the priced premium). ``knob = 0`` is the
  null world: cokurtosis still varies across names but carries **no** information about
  forward returns, and the sort must find nothing. ``knob > 0`` plants the Fang-Lai
  positive cokurtosis→return relation.

The offline path is pure numpy + pandas + stdlib. ``fetch()`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells; ``load_panel()`` reads the
cached parquet directly (no yfinance import).
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
    """Cached panel as ``{ticker: DataFrame[Open, High, Low, Close, Volume]}``, sliced to
    ``[start, asof]``. Reads the parquet directly — OFFLINE, no yfinance import."""
    cache = panel_cache_path(UNIVERSE, start)
    raw = pd.read_parquet(cache)
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    panel: dict[str, pd.DataFrame] = {}
    for s in UNIVERSE:
        if s not in raw.columns.get_level_values(0):
            continue
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"]
                if c in raw[s].columns]
        df = raw[s][cols].dropna()
        df = df[(df.index >= lo) & (df.index <= hi)]
        if not df.empty:
            panel[s] = df
    return panel


# --------------------------------------------------------------------------- #
# Synthetic world — planted positive cokurtosis->return relation (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    knob: float = 0.0,
    seed: int = 805,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    market_vol: float = 0.010,
    idio_vol: float = 0.010,
    drift: float = 0.06 / 252,
    tail_gain: float = 0.9,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC panel with a TUNABLE planted cokurtosis->return relation.

    A single common **market shock** ``g[t] ~ N(0,1)`` drives every name (so the
    equal-weight panel mean tracks it). Each asset ``i`` carries a persistent latent
    "tail-co-movement tilt" ``c_i`` (drawn once, fixed across time). The name's daily
    return loads on the market shock *and* on the **cubed** market shock scaled by ``c_i``,
    so its return spikes exactly when the market has a fat-tailed move — i.e. its
    **cokurtosis** with the market rises with ``c_i``:

        g ~ N(0,1)                                    # common market shock
        r[i,t] = drift + knob*c_i
                 + market_vol*g[t]                     # beta (co-variance) leg
                 + tail_gain*c_i*market_vol*g[t]**3    # tail-co-movement (cokurtosis) leg
                 + idio_vol*e[i,t]

    So a high tilt makes the name load on ``g**3`` (high cokurtosis) *and* — only when
    ``knob > 0`` — raises its **forward mean** (the priced premium: high cokurtosis pays).
    ``knob = 0`` is the null: cokurtosis still varies across names but predicts nothing.
    Business-day index; span well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    panel: dict[str, pd.DataFrame] = {}

    g = rng.normal(0.0, 1.0, n_days)          # common market shock, shared by all names
    c = rng.normal(0.0, 1.0, n_assets)        # per-name tail-co-movement tilt (fixed)
    c = c - c.mean()                          # zero-mean tilt: the equal-weight market
    #                                           carries no net g**3 skew (a clean market)
    g3 = g ** 3
    for i in range(n_assets):
        e = rng.normal(0.0, idio_vol, n_days)
        r = (drift + knob * c[i]
             + market_vol * g
             + tail_gain * c[i] * market_vol * g3
             + e)

        close = 100.0 * np.cumprod(1.0 + r)
        prev_close = np.concatenate([[100.0], close[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0.0, idio_vol / 3, n_days))
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, idio_vol / 2, n_days)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, idio_vol / 2, n_days)))
        vol = np.abs(rng.normal(1e6, 2e5, n_days))
        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close, "Volume": vol},
            index=idx,
        )
    return panel
