"""Data layer for Study 807 — Salience-Theory Returns.

The claim under test (Cosemans & Frehen 2021, *"Salience Theory and Stock Prices:
Empirical Evidence"*, applying the Bordalo-Gennaioli-Shleifer salience model): over the
trailing month, a name whose returns were **salient on the UPSIDE** — its big, attention-
grabbing days were up-days relative to the market — is bid up by salience-thinking
investors and goes on to earn **lower** returns. The signal is the name's
**salience-theory value** ST (a salience-weighted mean of its market-excess returns); a
high ST (salient upside) is a **negative** predictor, so a long **low-ST** / short
**high-ST** book should earn a positive spread.

Two ingredients, both offline-friendly once cached.

* **Real tape — a liquid US cross-section.** Daily OHLC for a fixed list of ~50 liquid
  US large-caps (``UNIVERSE`` below), pulled with yfinance through the
  ``quantlab.universe`` **survivorship guard** (``download_panel(...,
  allow_survivorship_bias=True)``). ``auto_adjust=True`` (total-return prices). The panel
  parquet is cached under this study's OWN ``_cache/`` (we point ``quantlab.universe``'s
  cache there via ``OVERNIGHT_CACHE`` *before* importing it).

  **Survivorship — named on the Signal axis.** ``UNIVERSE`` is a *current* membership list
  of names that are liquid mega-caps *today*; feeding it to a backward-looking panel omits
  the delisted / de-rated names and biases any cross-sectional result. The guard forces the
  opt-in; the caveat travels with every published number.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) with a TUNABLE knob ``edge``: each name carries a persistent latent
  "salient-upside tilt" ``c_i`` that both (a) shapes the **upside salience** of its daily
  returns (a right-skewed tape whose salient days are up-days, so the salience-theory value
  ST proxies ``c_i``) and (b) — only when ``edge > 0`` — depresses its forward mean return.
  ``edge = 0`` is the null world: ST still varies across names but carries **no** information
  about forward returns, and the sort must find nothing. ``edge > 0`` plants the
  Cosemans-Frehen negative salient-upside → return relation.

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
# Synthetic world — planted negative salient-upside->return relation (control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 807,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    daily_vol: float = 0.012,
    drift: float = 0.06 / 252,
    factor_rho: float = 0.95,
    skew_gain: float = 1.4,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC panel with a TUNABLE planted salient-upside->return relation.

    Each asset ``i`` carries a persistent latent "salient-upside tilt" ``c_i[t]`` — an AR(1)
    with autocorrelation ``factor_rho``. The tilt shapes the daily return so a high positive
    tilt yields a **right-skewed** tape whose big, salient days are **up**-days (so the
    salience-theory value ST — a salience-weighted mean of market-excess returns — proxies
    ``c_i``), and — only when ``edge > 0`` — depresses the **forward mean**:

        z ~ N(0,1)
        skewed_shock = daily_vol * (z + skew_gain * c_i[t] * (z**2 - 1))
        r[i,t] = drift - edge * c_i[t] + skewed_shock

    So a high positive tilt makes the salient (large-magnitude) days upside days *and* (with
    ``edge > 0``) lowers the mean — the Cosemans-Frehen pattern: salient upside, low forward
    return. ``edge = 0`` is the null: ST still varies across names but predicts nothing.
    Business-day index; span well below the pandas ns-timestamp horizon.
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

        z = rng.normal(0.0, 1.0, n_days)
        skewed = daily_vol * (z + skew_gain * c * (z ** 2 - 1.0))
        r = drift - edge * c + skewed

        close = 100.0 * np.cumprod(1.0 + r)
        prev_close = np.concatenate([[100.0], close[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0.0, daily_vol / 3, n_days))
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close}, index=idx
        )
    return panel
