"""Data layer for Study 815 — Variance-Ratio Reversal.

The claim under test (Andrew W. **Lo & A. Craig MacKinlay** 1988, *"Stock Market Prices
Do Not Follow Random Walks: Evidence from a Simple Specification Test"*): the **variance
ratio** ``VR(q) = Var(q-day return) / (q * Var(1-day return))`` measures how far a price
series departs from a random walk. Under the random-walk null ``VR(q) = 1``; ``VR < 1``
flags **mean reversion** (negative return autocorrelation), ``VR > 1`` flags
**trending / positive autocorrelation**. The cross-sectional question we pose: sort a
universe by trailing VR and ask whether the **low-VR (mean-reverting)** names offer a
tradable **reversal** premium (or the **high-VR** names a continuation). We report the
forward long-short spread honestly.

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
  latent "mean-reversion tilt" ``c_i`` that both (a) sets the sign/size of its daily
  return autocorrelation (via an MA(1) term, so a trailing VR sort proxies ``c_i``)
  and (b) — only when ``edge > 0`` — lifts the **forward mean** of the mean-reverting
  (low-VR) names. ``edge = 0`` is the null world: VR still varies across names but
  carries **no** information about forward returns, and the sort must find nothing.
  ``edge > 0`` plants the "low-VR names out-earn" reversal premium.

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
# Synthetic world — planted VR->return reversal relation (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 815,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    daily_vol: float = 0.012,
    drift: float = 0.06 / 252,
    ma_gain: float = 0.32,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC panel with a TUNABLE planted VR->return relation.

    Each asset ``i`` carries a persistent latent "mean-reversion tilt" ``c_i`` (a fixed
    per-name standard-normal draw). It sets an MA(1) coefficient ``theta_i = ma_gain *
    c_i`` on the daily return, so the return has **lag-1 autocorrelation**
    ``rho_1 = theta/(1+theta**2)`` of the *same sign* as ``c_i``:

        eps ~ N(0, daily_vol)
        r[i,t] = drift - edge * c_i + eps[t] + theta_i * eps[t-1]

    A **negative** tilt (``c_i < 0``) makes returns negatively autocorrelated ⇒ a
    **variance ratio below 1** (mean-reverting); a positive tilt ⇒ ``VR > 1`` (trending).
    Because ``theta`` is fixed per name, a trailing VR estimate proxies ``c_i`` cleanly.
    With ``edge > 0`` the forward mean gets ``- edge * c_i``, i.e. the mean-reverting
    (low-VR, ``c_i < 0``) names earn **more** — the "low-VR reversal premium" a
    long-low-VR / short-high-VR book is meant to harvest. ``edge = 0`` is the null: VR
    still varies across names but predicts nothing.

    ``theta`` is clipped to ``[-0.8, 0.8]`` to keep the MA(1) invertible. Business-day
    index; span well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    panel: dict[str, pd.DataFrame] = {}

    for i in range(n_assets):
        c = float(rng.normal(0.0, 1.0))
        theta = float(np.clip(ma_gain * c, -0.8, 0.8))

        eps = rng.normal(0.0, daily_vol, n_days + 1)
        ma = eps[1:] + theta * eps[:-1]           # MA(1) daily innovations
        r = drift - edge * c + ma

        close = 100.0 * np.cumprod(1.0 + r)
        prev_close = np.concatenate([[100.0], close[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0.0, daily_vol / 3, n_days))
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close}, index=idx
        )
    return panel
