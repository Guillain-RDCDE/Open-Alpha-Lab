"""Data layer for Study 811 — Zero-Return Illiquidity.

The claim under test (Lesmond, Ogden & Trzcinka 1999, *"A New Estimate of
Transaction Costs"*): the **proportion of trading days with an exactly-zero daily
return** is a cheap, price-only **transaction-cost / illiquidity** proxy — a name
whose price sits still on many days is one where the round-trip cost of trading
exceeds the information arriving, so the marginal trader stays out and the observed
return is zero. Illiquid names should command an **illiquidity premium** (Amihud &
Mendelson 1986), so a long **high**-zero-proportion / short **low**-zero-proportion
book should earn a positive spread.

The honest wrinkle, stated up front: liquid mega-caps almost never print an
exactly-zero adjusted daily return, so on a 50-mega-cap survivor panel the signal is
**near-degenerate** — the whole cross-section clusters just above zero. That is
exactly the regime where an illiquidity premium should *not* appear, so we expect
**None**, and we say so.

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
  published number.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) with a TUNABLE knob ``edge``: each name carries a persistent
  latent "illiquidity" ``q_i`` (the probability its daily return is stale-to-zero)
  that both (a) sets the frequency of exactly-zero return days and (b) — only when
  ``edge > 0`` — *raises* its forward mean return (the illiquidity premium). ``edge
  = 0`` is the null world: zero-return frequency still varies across names but
  carries **no** information about forward returns, and the sort must find nothing.

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
    to ``[start, asof]``. Reads the parquet directly — OFFLINE, no yfinance import.
    ``Volume`` is carried through so a liquidity cross-check is available, but the
    zero-return signal itself is built from ``Close`` alone."""
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
# Synthetic world — planted zero-return -> illiquidity-premium (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 811,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    daily_vol: float = 0.012,
    drift: float = 0.06 / 252,
    q_max: float = 0.20,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC panel with a TUNABLE planted zero-return -> return
    relation.

    Each asset ``i`` is dealt a persistent latent **illiquidity** ``q_i`` drawn
    uniformly on ``[0, q_max]`` — the probability that, on any given day, its price is
    stale and the observed daily return is **exactly zero**. That mechanically makes
    the trailing zero-return *proportion* a clean proxy for ``q_i``. On non-stale days
    the return carries a drift lifted by the illiquidity premium (only when
    ``edge > 0``)::

        stale ~ Bernoulli(q_i)
        r[i,t] = 0                                        if stale
        r[i,t] = drift + edge * (q_i - qbar) + vol * z    otherwise

    So a high ``q_i`` makes a name both **often-zero** *and* (with ``edge > 0``)
    higher-earning on the days it does move — the Amihud illiquidity premium expressed
    through the LOT zero-return proxy. ``edge = 0`` is the null: zero-return frequency
    still varies across names but predicts nothing. Business-day index; span well below
    the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    q = rng.uniform(0.0, q_max, n_assets)
    qbar = float(q.mean())
    panel: dict[str, pd.DataFrame] = {}

    for i in range(n_assets):
        z = rng.normal(0.0, 1.0, n_days)
        r = drift + edge * (q[i] - qbar) + daily_vol * z
        stale = rng.random(n_days) < q[i]
        r = np.where(stale, 0.0, r)

        close = 100.0 * np.cumprod(1.0 + r)
        prev_close = np.concatenate([[100.0], close[:-1]])
        open_ = np.where(stale, close, prev_close * (1.0 + rng.normal(0.0, daily_vol / 3, n_days)))
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        vol = rng.integers(1_000_000, 5_000_000, n_days).astype(float)
        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close, "Volume": vol},
            index=idx,
        )
    return panel
