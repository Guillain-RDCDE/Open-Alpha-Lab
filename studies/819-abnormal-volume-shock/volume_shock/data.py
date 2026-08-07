"""Data layer for Study 819 — Abnormal-Volume Shock.

The claim under test (Garfinkel & Sokobin 2006, *"Volume, Opinion Divergence, and
Returns: A Study of Post-Earnings Announcement Drift"*): **unusual trading volume** is a
proxy for **attention / opinion divergence**, and names that print abnormally high volume
go on to earn a **positive subsequent drift**. Volume that cannot be explained by its
recent norm signals a burst of disagreement/attention, and the disagreement is resolved
in favour of the informed side — a forward return.

Two ingredients, both offline-friendly once cached.

* **Real tape — a liquid US cross-section.** Daily OHLC **and Volume** for a fixed list
  of ~50 liquid US large-caps (``UNIVERSE`` below), pulled with yfinance through the
  ``quantlab.universe`` **survivorship guard** (``download_panel(...,
  allow_survivorship_bias=True)``). ``auto_adjust=True`` (total-return prices). The panel
  parquet is cached under this study's OWN ``_cache/`` (we point ``quantlab.universe``'s
  cache there via ``OVERNIGHT_CACHE`` *before* importing it).

  **Survivorship — named on the Signal axis.** ``UNIVERSE`` is a *current* membership
  list of names that are liquid mega-caps *today*; feeding it to a backward-looking panel
  omits the delisted / de-rated names and biases any cross-sectional result. The guard
  forces the opt-in; the caveat travels with every published number.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) with a TUNABLE knob ``edge``: each name carries a persistent
  latent "attention shock" ``a_i[t]`` (an AR(1)) that both (a) inflates that day's
  **volume** and (b) — only when ``edge > 0`` — lifts its forward mean return.
  ``edge = 0`` is the null world: abnormal volume still varies across names but carries
  **no** information about forward returns, and the sort must find nothing. ``edge > 0``
  plants the Garfinkel-Sokobin attention → drift relation.

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
    """Cached panel as ``{ticker: DataFrame[Open, High, Low, Close, Volume]}``, sliced to
    ``[start, asof]``. Reads the parquet directly — OFFLINE, no yfinance import.

    ``Volume`` is carried alongside the OHLC because the abnormal-volume signal is built
    from it (this is the one column added versus the sibling price studies)."""
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
# Synthetic world — planted attention(volume)->return relation (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 819,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    daily_vol: float = 0.012,
    drift: float = 0.06 / 252,
    factor_rho: float = 0.92,
    vol_gain: float = 0.7,
    base_vol: float = 5.0e6,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC+Volume panel with a TUNABLE planted attention→drift
    relation.

    Each asset ``i`` carries a persistent latent "attention shock" ``a_i[t]`` — an AR(1)
    with autocorrelation ``factor_rho`` and unit stationary variance. The shock inflates
    that day's **traded volume** (log-linear, so volume stays positive), and — only when
    ``edge > 0`` — lifts the **forward mean** return:

        a[i,t] = factor_rho*a[i,t-1] + sqrt(1-rho^2)*eps
        volume[i,t] = base_i * exp(vol_gain * a[i,t] + noise)          (always)
        r[i,t] = drift + edge * a[i,t] + N(0, daily_vol)              (edge>0 only)

    Because ``a`` is persistent, a trailing few-day mean of the **standardised abnormal
    volume** (which tracks ``a``) known at ``t-1`` proxies ``a[t]`` and therefore the
    day-``t`` return — so a high-abnormal-volume / low-abnormal-volume sort recovers the
    Garfinkel-Sokobin drift. ``edge = 0`` is the null: abnormal volume still varies across
    names but predicts nothing. Business-day index; span well below the pandas
    ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    panel: dict[str, pd.DataFrame] = {}

    innov_sd = np.sqrt(1.0 - factor_rho ** 2)
    for i in range(n_assets):
        a = np.empty(n_days)
        a[0] = rng.normal(0.0, 1.0)
        eps = rng.normal(0.0, innov_sd, n_days)
        for t in range(1, n_days):
            a[t] = factor_rho * a[t - 1] + eps[t]

        shock = rng.normal(0.0, daily_vol, n_days)
        r = drift + edge * a + shock

        close = 100.0 * np.cumprod(1.0 + r)
        prev_close = np.concatenate([[100.0], close[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0.0, daily_vol / 3, n_days))
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, daily_vol / 2, n_days)))

        base_i = base_vol * (0.5 + rng.random())
        vnoise = rng.normal(0.0, 0.25, n_days)
        volume = base_i * np.exp(vol_gain * a + vnoise)

        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close,
             "Volume": volume},
            index=idx,
        )
    return panel
