"""Data layer for Study 818 — Trend Factor.

The claim under test (Han, Zhou & Zhu 2016, *"A Trend Factor: Any Economic Gains from
Using Information over Investment Horizons?"*, Review of Financial Studies): a single
**trend factor** that blends moving-average price signals across *many* horizons — short
(3, 5, 10 days), intermediate (20, 50), and long (100, 200) — beats any single-horizon
moving-average timing rule *and* the standard momentum sort. The recipe:

1. for each name form normalized moving-average signals ``A_L = MA_L(price) / price`` for
   ``L`` in {3, 5, 10, 20, 50, 100, 200};
2. each period run a **cross-sectional regression** of the next return on the ``A_L``
   vector, giving a time series of predictive slopes;
3. average the *past* slopes (a rolling Fama-MacBeth-lite expectation) and dot them into
   today's ``A_L`` vector → the fitted expected return, the **trend factor**;
4. sort **long high-trend / short low-trend**.

Two ingredients, both offline-friendly once cached.

* **Real tape — a liquid US cross-section.** Daily OHLC for a fixed list of ~50 liquid US
  large-caps (``UNIVERSE`` below), pulled with yfinance through the ``quantlab.universe``
  **survivorship guard** (``download_panel(..., allow_survivorship_bias=True)``).
  ``auto_adjust=True`` (total-return prices). The panel parquet is cached under this
  study's OWN ``_cache/`` (we point ``quantlab.universe``'s cache there via
  ``OVERNIGHT_CACHE`` *before* importing it).

  **Survivorship — named on the Signal axis.** ``UNIVERSE`` is a *current* membership list
  of names that are liquid mega-caps *today*; feeding it to a backward-looking panel omits
  the delisted / de-rated names and biases any cross-sectional result. The guard forces the
  opt-in; the caveat travels with every published number.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) with a TUNABLE knob ``edge``: each name carries a persistent latent
  "trend state" ``m_i[t]`` (an AR(1)) that — only when ``edge > 0`` — is added to the daily
  return, so the price genuinely *trends* (its moving averages line up) AND the trend
  predicts the next return. ``edge = 0`` is the null world: prices are random walks, the
  moving-average signals carry **no** forward information, and the fitted trend factor must
  find nothing. ``edge > 0`` plants the Han-Zhou-Zhu trend->return relation.

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

# The trend factor's moving-average horizons (Han-Zhou-Zhu 2016, Table 1).
MA_LAGS = (3, 5, 10, 20, 50, 100, 200)

__all__ = [
    "SurvivorshipBiasError",
    "UNIVERSE", "START", "AS_OF", "CACHE_DIR", "MA_LAGS",
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
# Synthetic world — planted trend->return relation (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 818,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    daily_vol: float = 0.012,
    drift: float = 0.06 / 252,
    trend_rho: float = 0.985,
    trend_sd: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC panel with a TUNABLE planted trend->return relation.

    Each asset ``i`` carries a persistent latent "trend state" ``m_i[t]`` — an AR(1) with
    autocorrelation ``trend_rho`` and stationary sd ``trend_sd``. When ``edge > 0`` the
    trend is *added to the daily return*:

        m_i[t] = trend_rho * m_i[t-1] + innovation
        r[i,t] = drift + edge * m_i[t] + daily_vol * z

    Because ``m`` is highly persistent, a positive trend both (a) drives the price **up in a
    sustained way**, so its moving averages line up (``MA_L / P < 1``), and (b) predicts the
    **next** day's return (``m[t] ≈ m[t+1]``). So the fitted trend factor — the cross-
    sectional regression's dot-product of the ``A_L`` vector with the averaged past slopes —
    should rank the strong up-trenders' forward returns high, and a long-high / short-low
    trend book earns a positive spread. ``edge = 0`` is the null: prices are random walks,
    the moving-average signals carry no forward information, and the sort must find nothing.
    Business-day index; span well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    panel: dict[str, pd.DataFrame] = {}

    innov_sd = trend_sd * np.sqrt(1.0 - trend_rho ** 2)
    for i in range(n_assets):
        m = np.empty(n_days)
        m[0] = rng.normal(0.0, trend_sd)
        eps = rng.normal(0.0, innov_sd, n_days)
        for t in range(1, n_days):
            m[t] = trend_rho * m[t - 1] + eps[t]

        z = rng.normal(0.0, 1.0, n_days)
        r = drift + edge * m + daily_vol * z

        close = 100.0 * np.cumprod(1.0 + r)
        prev_close = np.concatenate([[100.0], close[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0.0, daily_vol / 3, n_days))
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close}, index=idx
        )
    return panel
