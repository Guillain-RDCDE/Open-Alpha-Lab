"""Data layer for Study 871 — The Rank Effect.

The claim under test (Samuel M. **Hartzmark**, 2015, *"The Worst, the Best, Ignoring
All the Rest: The Rank Effect and Trading Behavior"*, Review of Financial Studies):
investors disproportionately **sell the best- and worst-ranked positions** in their
portfolio — the salience of the extremes drives the trade, not the raw return level.
That behaviour puts predictable **selling pressure** on the names an investor ranks at
the top and bottom of her book.

Turned into a self-contained cross-sectional test: each period **rank the names by
trailing return**; ask whether the **extreme-ranked** names (rank 1, the best; rank N,
the worst) go on to **underperform the middle** of the ranking next period — a
*rank-extremity short* (long the middle, short both tails) — while **controlling for
the raw trailing-return level** so the test isolates rank *position* from momentum /
reversal.

Two ingredients, both offline-friendly once cached.

* **Real tape — a liquid US cross-section.** Daily OHLC for a fixed list of ~50 liquid
  US large-caps (``UNIVERSE`` below), pulled with yfinance through the
  ``quantlab.universe`` **survivorship guard** (``download_panel(...,
  allow_survivorship_bias=True)``). ``auto_adjust=True`` (total-return prices). The
  panel parquet is cached under this study's OWN ``_cache/`` (we point
  ``quantlab.universe``'s cache there via ``OVERNIGHT_CACHE`` *before* importing it).

  **Survivorship — named on the Signal axis.** ``UNIVERSE`` is a *current* membership
  list of names that are liquid mega-caps *today*; feeding it to a backward-looking
  panel omits the delisted / de-rated names and biases any cross-sectional result. The
  guard forces the opt-in; the caveat travels with every published number.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) with a TUNABLE knob ``edge``: names sitting at an **extreme
  trailing-return rank** carry a forward-return **penalty** proportional to their rank
  *extremity* — only when ``edge > 0``. ``edge = 0`` is the null world: trailing
  returns still disperse across names and get ranked, but rank extremity carries **no**
  forward information and the sort must find nothing. ``edge > 0`` plants the Hartzmark
  rank-extremity relation (extremes under-earn the middle).

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
# Synthetic world — planted rank-extremity penalty (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 871,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    daily_vol: float = 0.012,
    drift: float = 0.06 / 252,
    factor_rho: float = 0.97,
    window: int = 42,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC panel with a TUNABLE planted rank-extremity relation.

    Each asset ``i`` earns a baseline daily return ``r0[i,t]`` = market drift + a
    persistent AR(1) name factor + idiosyncratic noise, so trailing returns **disperse**
    across names and can be ranked. From ``r0`` we compute each name's ``window``-day
    trailing return, rank the cross-section each day, and form a rank-**extremity** score
    ``ext ∈ [0, 1]`` (0 in the middle of the ranking, 1 at either tail). The planted
    return is

        r[i,t] = r0[i,t] − edge * ext0[i, t−1]

    so — only when ``edge > 0`` — the names sitting at an **extreme trailing-return rank**
    at ``t−1`` are penalised at ``t``: the Hartzmark pattern (best- and worst-ranked names
    under-earn the middle). Because the penalty is tiny relative to ``r0``, the ranking the
    strategy re-derives from ``r`` is essentially the clean ``ext0``, so the effect is
    detectable. ``edge = 0`` is the null: names are still ranked, but rank extremity
    predicts nothing. Business-day index; span well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)

    # Persistent AR(1) name factor -> dispersed trailing returns across names.
    innov_sd = np.sqrt(1.0 - factor_rho ** 2)
    f = np.empty((n_days, n_assets))
    f[0] = rng.normal(0.0, 1.0, n_assets)
    eps = rng.normal(0.0, innov_sd, (n_days, n_assets))
    for t in range(1, n_days):
        f[t] = factor_rho * f[t - 1] + eps[t]
    noise = rng.normal(0.0, daily_vol, (n_days, n_assets))
    r0 = drift + 0.5 * daily_vol * f + noise                       # [T, N]

    # Trailing window return per name (cumulative simple return over `window` days).
    logr = np.log1p(r0)
    csum = np.cumsum(logr, axis=0)
    trail = np.full((n_days, n_assets), np.nan)
    trail[window:] = np.expm1(csum[window:] - csum[:-window])

    # Cross-sectional fractional rank u in [0,1]; extremity = |2u-1| (V-shaped).
    ext0 = np.full((n_days, n_assets), np.nan)
    for t in range(window, n_days):
        row = trail[t]
        order = np.argsort(row, kind="stable")
        rank = np.empty(n_assets)
        rank[order] = np.arange(n_assets)
        u = rank / (n_assets - 1.0)
        ext0[t] = np.abs(2.0 * u - 1.0)

    # Plant: extremes at t-1 under-earn at t (only when edge>0).
    penalty = np.zeros_like(r0)
    penalty[1:] = edge * np.nan_to_num(ext0[:-1])
    r = r0 - penalty

    close = 100.0 * np.cumprod(1.0 + r, axis=0)
    prev_close = np.vstack([np.full((1, n_assets), 100.0), close[:-1]])
    open_ = prev_close * (1.0 + rng.normal(0.0, daily_vol / 3, (n_days, n_assets)))
    hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, daily_vol / 2, (n_days, n_assets))))
    lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, daily_vol / 2, (n_days, n_assets))))

    panel: dict[str, pd.DataFrame] = {}
    for i in range(n_assets):
        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_[:, i], "High": hi[:, i], "Low": lo[:, i], "Close": close[:, i]},
            index=idx,
        )
    return panel
