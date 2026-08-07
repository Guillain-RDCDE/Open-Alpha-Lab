"""Data layer for Study 821 — Turnover Volatility.

The claim under test (Tarun **Chordia**, Avanidhar **Subrahmanyam** & V. Ravi
**Anshuman** 2001, *"Trading activity and expected stock returns"*, JFE): once you
control for the *level* of trading activity, its **variability** carries a robust
**negative** cross-sectional premium — stocks whose daily **share turnover** is most
erratic (the **coefficient of variation** of turnover, std/mean over a trailing
window, is high) go on to earn **lower** returns. The reading is a liquidity-risk
discount: unpredictable liquidity is a cost, so erratic-turnover names are priced up
and under-earn. A long **low** turnover-vol / short **high** turnover-vol book should
therefore earn a *positive* spread.

Two ingredients, both offline-friendly once cached.

* **Real tape — a liquid US cross-section (with Volume).** Daily OHLC **and Volume**
  for a fixed list of ~50 liquid US large-caps (``UNIVERSE`` below), pulled with
  yfinance through the ``quantlab.universe`` **survivorship guard**
  (``download_panel(..., allow_survivorship_bias=True)``). ``auto_adjust=True``
  (total-return prices). The panel parquet is cached under this study's OWN
  ``_cache/`` (we point ``quantlab.universe``'s cache there via ``OVERNIGHT_CACHE``
  *before* importing it).

  **Survivorship — named on the Signal axis.** ``UNIVERSE`` is a *current*
  membership list of names that are liquid mega-caps *today*; feeding it to a
  backward-looking panel omits the delisted / de-rated names and biases any
  cross-sectional result. The guard forces the opt-in; the caveat travels with every
  published number.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) with a TUNABLE knob ``edge``: each name carries a persistent
  latent "turnover-vol tilt" ``c_i`` that both (a) inflates the **variability** of its
  daily volume (so a trailing CV-of-turnover sort proxies ``c_i``) and (b) — only when
  ``edge > 0`` — depresses its forward mean return. ``edge = 0`` is the null world:
  turnover-vol still varies across names but carries **no** information about forward
  returns, and the sort must find nothing. ``edge > 0`` plants the Chordia-et-al
  negative turnover-vol->return relation.

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
    ``Volume`` is carried because this study's signal is a function of it."""
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
# Synthetic world — planted negative turnover-vol->return relation (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 821,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    daily_vol: float = 0.012,
    drift: float = 0.06 / 252,
    base_volume: float = 5.0e6,
    vol_disp: float = 0.9,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLCV panel with a TUNABLE planted turnover-vol->return
    relation.

    Each asset ``i`` carries a fixed positive latent "turnover-vol tilt" ``c_i`` drawn
    once. It governs the **dispersion** of the name's daily volume via a log-normal
    with per-name log-scale ``c_i * vol_disp`` — a larger ``c_i`` means a more erratic
    volume series and hence a larger **coefficient of variation** of turnover
    (``sqrt(exp(s**2) - 1)``, monotone in ``c_i``). The tilt — only when ``edge > 0`` —
    also depresses the **forward mean** return:

        c_i  ~ Uniform(0.3, 1.5)            (per-name, fixed over time)
        volume[i,t] = base_volume * exp(c_i * vol_disp * u[i,t]),  u ~ N(0,1)
        r[i,t]      = drift - edge * (c_i - c_mean) + daily_vol * z[i,t]

    So a high-tilt name has **erratic turnover** *and* (with ``edge > 0``) a lower mean
    — the Chordia-Subrahmanyam-Anshuman pattern: high turnover-vol, low forward return.
    ``edge = 0`` is the null: turnover-vol still varies across names but predicts
    nothing. Business-day index; span well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    c = rng.uniform(0.3, 1.5, n_assets)
    c_mean = float(c.mean())
    panel: dict[str, pd.DataFrame] = {}

    for i in range(n_assets):
        z = rng.normal(0.0, 1.0, n_days)
        r = drift - edge * (c[i] - c_mean) + daily_vol * z

        close = 100.0 * np.cumprod(1.0 + r)
        prev_close = np.concatenate([[100.0], close[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0.0, daily_vol / 3, n_days))
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, daily_vol / 2, n_days)))

        u = rng.normal(0.0, 1.0, n_days)
        volume = base_volume * np.exp(c[i] * vol_disp * u)

        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close, "Volume": volume},
            index=idx,
        )
    return panel
