"""Data layer for Study 875 — Idiosyncratic-Vol Change.

The claim under test: distinct from the idiosyncratic-vol **level** puzzle (Ang-Hodrick-
Xing-Zhang, study 501), does the **change** in idiosyncratic volatility predict returns?
Estimate each name's **residual** (market-model) vol over a recent window vs a prior
window; a **rising** idio-vol (a deteriorating information environment / rising
disagreement) may precede **lower** returns. So a long **falling-idio-vol** / short
**rising-idio-vol** book should earn a positive spread. The honest question is whether
this delta-IVOL is anything beyond the idio-vol *level* effect (501) or the *total*-vol
trend (817).

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
  (``synthetic_panel``) with a TUNABLE knob ``edge``: a common market factor plus, for
  each name, a persistent latent "idio-vol tilt" ``c_i[t]`` (an AR(1)) that modulates
  the **amplitude** of the name's *residual* (idiosyncratic) return volatility over
  time (so its recent-vs-prior residual vol — the delta-IVOL — proxies the direction in
  which ``c_i`` is moving), and — only when ``edge > 0`` — depresses the name's forward
  mean when its idio-vol is **rising**. ``edge = 0`` is the null world: the idio-vol
  change still varies across names but carries **no** information about forward returns,
  and the sort must find nothing. ``edge > 0`` plants the claimed rising-idio-vol →
  lower-return relation.

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
# Synthetic world — planted rising-idio-vol->de-rate relation (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 875,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    base_ivol: float = 0.011,
    mkt_vol: float = 0.009,
    drift: float = 0.06 / 252,
    factor_rho: float = 0.97,
    vol_gain: float = 0.6,
    change_window: int = 21,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC panel with a TUNABLE planted delta-IVOL->return relation.

    A single common **market factor** ``m_t`` drives every name (each with a beta near
    one), so the market-model residuals recover a genuine *idiosyncratic* component.
    Each asset ``i`` also carries a persistent latent "idio-vol tilt" ``c_i[t]`` — an
    AR(1) with autocorrelation ``factor_rho`` — that modulates the **amplitude** of its
    residual return volatility over time:

        m_t     ~ N(drift, mkt_vol)                       # common market factor
        z ~ N(0,1)
        ivol_t  = base_ivol * exp(vol_gain * c_i[t])      # rising c -> rising idio-vol
        dtrend  = c_i[t] - c_i[t-change_window]           # recent CHANGE in the tilt
        r[i,t]  = beta_i * m_t + drift - edge * dtrend + ivol_t * z

    So when a name's idio-vol is **rising** (``dtrend > 0``, which the recent-vs-prior
    residual-vol delta measures) its forward mean is depressed (with ``edge > 0``) — the
    claimed pattern: rising idio-vol de-rates, falling idio-vol re-rates. ``edge = 0`` is
    the null: the idio-vol change still varies across names but predicts nothing.
    Business-day index; span well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)

    # One common market factor shared by every name.
    m = rng.normal(drift, mkt_vol, n_days)
    betas = rng.uniform(0.8, 1.2, n_assets)

    panel: dict[str, pd.DataFrame] = {}
    innov_sd = np.sqrt(1.0 - factor_rho ** 2)
    for i in range(n_assets):
        c = np.empty(n_days)
        c[0] = rng.normal(0.0, 1.0)
        eps = rng.normal(0.0, innov_sd, n_days)
        for t in range(1, n_days):
            c[t] = factor_rho * c[t - 1] + eps[t]

        # recent CHANGE in the latent vol tilt (rising vs falling idio-vol)
        dtrend = np.zeros(n_days)
        dtrend[change_window:] = c[change_window:] - c[:-change_window]

        z = rng.normal(0.0, 1.0, n_days)
        ivol_t = base_ivol * np.exp(vol_gain * c)
        r = betas[i] * m + drift - edge * dtrend + ivol_t * z

        close = 100.0 * np.cumprod(1.0 + r)
        prev_close = np.concatenate([[100.0], close[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0.0, base_ivol / 3, n_days))
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, base_ivol / 2, n_days)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, base_ivol / 2, n_days)))
        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close}, index=idx
        )
    return panel
