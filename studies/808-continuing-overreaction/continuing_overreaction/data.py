"""Data layer for Study 808 — Continuing Overreaction.

The claim under test (Byun, Lim & Yun 2016, *"Continuing Overreaction and Stock
Return Predictability"*): a **weighted signed-momentum** measure that puts more weight
on the *signs* of the more recent monthly returns predicts the cross-section of
returns **positively**. A name on a persistent recent up-streak (high "continuing
overreaction", CO) keeps rising for a while — a momentum-style continuation that is
stronger than plain past-return momentum because it counts the *consistency* of the
run, not its magnitude.

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
  latent monthly "trend state" ``s_i[m]`` (an AR(1) across months). When ``edge > 0``
  that state drives BOTH the sign of the recent monthly returns (so a weighted signed
  momentum proxies it) AND — because the state is persistent — the *forward* monthly
  return, planting the Byun-Lim-Yun continuation. ``edge = 0`` is the null world:
  monthly returns are pure noise, the signed-momentum score still varies across names
  but carries **no** information about forward returns, and the sort must find nothing.

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
# Synthetic world — planted continuing-overreaction relation (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 808,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    daily_vol: float = 0.012,
    drift: float = 0.06 / 252,
    state_rho: float = 0.85,
    days_per_month: int = 21,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC panel with a TUNABLE planted CO->return relation.

    Each asset ``i`` carries a persistent latent monthly **trend state** ``s_i[m]`` —
    an AR(1) across months with autocorrelation ``state_rho``. Every calendar month is
    ``days_per_month`` business days long; within month ``m`` the daily return is

        r[i,t] = drift + edge * s_i[m] / days_per_month + daily_vol * z ,   z ~ N(0,1)

    so when ``edge > 0`` a positive state tilts that month's return **positive** (the
    *sign* the CO measure reads) AND — because ``s_i`` is persistent — the *next*
    month's return too. A weighted signed-momentum score built from the recent monthly
    signs therefore proxies ``s_i`` and predicts the forward month: the Byun-Lim-Yun
    continuation. ``edge = 0`` is the null: ``drift`` is common, monthly signs are pure
    coin-flips, and the score predicts nothing. Business-day index; span well below the
    pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    month_of_day = np.arange(n_days) // days_per_month
    n_months = int(month_of_day[-1]) + 1
    panel: dict[str, pd.DataFrame] = {}

    innov_sd = np.sqrt(1.0 - state_rho ** 2)
    for i in range(n_assets):
        s = np.empty(n_months)
        s[0] = rng.normal(0.0, 1.0)
        eps = rng.normal(0.0, innov_sd, n_months)
        for m in range(1, n_months):
            s[m] = state_rho * s[m - 1] + eps[m]

        state_daily = s[month_of_day]            # broadcast month state onto its days
        z = rng.normal(0.0, 1.0, n_days)
        r = drift + edge * state_daily / days_per_month + daily_vol * z

        close = 100.0 * np.cumprod(1.0 + r)
        prev_close = np.concatenate([[100.0], close[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0.0, daily_vol / 3, n_days))
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close}, index=idx
        )
    return panel
