"""Data layer for Study 812 — Corwin-Schultz Spread.

The claim under test (Corwin & Schultz 2012, *"A Simple Way to Estimate Bid-Ask Spreads
from Daily High and Low Prices"*, Journal of Finance): a stock's effective bid-ask
**spread** can be recovered from nothing but its daily **high** and **low** prices, and
that estimated spread proxies illiquidity. Sorting a cross-section on the estimated
spread and buying the **illiquid** (high-spread) names against the **liquid** (low-spread)
names is the textbook illiquidity-premium bet — high-spread names should earn a premium.

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
  published number. The illiquidity premium in particular is documented as a
  *small-and-illiquid-stock* effect, so a 50-mega-cap survivor panel is exactly where it
  is least likely to appear.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) with a TUNABLE knob ``edge``: each name carries a persistent
  latent **spread level** ``s_i`` that both (a) inflates its observed daily high-low
  range (the high prints at the ask, the low at the bid — the Corwin-Schultz observation
  model) and (b) — only when ``edge > 0`` — lifts its forward mean return (the
  illiquidity premium). ``edge = 0`` is the null world: spreads still vary across names
  and the estimator still recovers them, but they carry **no** information about forward
  returns, and the sort must find nothing. ``edge > 0`` plants the illiquidity premium.

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
# Synthetic world — planted illiquidity premium (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 812,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    daily_vol: float = 0.012,
    drift: float = 0.06 / 252,
    intraday_vol: float = 0.008,
    spread_lo: float = 0.0005,
    spread_hi: float = 0.030,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC panel with a TUNABLE planted illiquidity->return relation.

    Each asset ``i`` is assigned a persistent latent **spread level** ``s_i`` drawn once
    from ``[spread_lo, spread_hi]``. Two independent channels use it:

    * **Observation model (always on).** The true intraday high/low are formed from an
      ``intraday_vol`` bar around the open/close; the *observed* high is then marked up by
      ``s_i / 2`` and the *observed* low marked down by ``s_i / 2`` — the Corwin-Schultz
      premise that the high transacts at the ask and the low at the bid. So a high-``s_i``
      name has a persistently wider high-low range, and the CS estimator recovers ``s_i``.
    * **Return channel (only when ``edge > 0``).** The daily mean return is lifted by
      ``edge * s_i`` — the illiquidity premium: illiquid (wide-spread) names out-earn.

        r[i,t] = drift + edge * s_i + daily_vol * z

    ``edge = 0`` is the null: spreads still vary and the estimator still recovers them,
    but they predict nothing, so a spread sort must find nothing. ``edge > 0`` plants the
    premium (long high-spread / short low-spread should light up).
    Business-day index; span well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    panel: dict[str, pd.DataFrame] = {}

    for i in range(n_assets):
        s_i = rng.uniform(spread_lo, spread_hi)          # persistent per-name spread

        z = rng.normal(0.0, 1.0, n_days)
        r = drift + edge * s_i + daily_vol * z
        close = 100.0 * np.cumprod(1.0 + r)
        prev_close = np.concatenate([[100.0], close[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0.0, daily_vol / 3, n_days))

        # True intraday extremes from an intraday-vol bar around open/close.
        up = np.abs(rng.normal(0.0, intraday_vol, n_days))
        dn = np.abs(rng.normal(0.0, intraday_vol, n_days))
        true_hi = np.maximum(open_, close) * (1.0 + up)
        true_lo = np.minimum(open_, close) * (1.0 - dn)
        # Corwin-Schultz observation model: high at the ask, low at the bid.
        hi = true_hi * (1.0 + s_i / 2.0)
        lo = true_lo * (1.0 - s_i / 2.0)

        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close}, index=idx
        )
    return panel
