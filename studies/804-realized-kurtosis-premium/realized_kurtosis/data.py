"""Data layer for Study 804 — Realized-Kurtosis Premium.

The claim under test (Amaya, Christoffersen, Jacobs & Vasquez 2015, *"Does Realized
Skewness Predict the Cross-Section of Equity Returns?"*): alongside the strong negative
realized-**skewness** relation, the same paper also tests realized **kurtosis** — a
name's recent fat-tailedness. There the evidence is **weak / ambiguous**: realized
kurtosis is only marginally priced and is largely *subsumed* by skewness and volatility
once they are controlled for. We take the self-contained daily version: sort a liquid US
cross-section on trailing realized kurtosis and test whether a long-high-kurt /
short-low-kurt book earns anything at all. We expect Weak/None honestly.

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
  latent "kurtosis tilt" ``c_i`` that both (a) inflates the fat-tailedness of its daily
  returns and (b) — only when ``edge > 0`` — lifts its forward mean return. ``edge = 0``
  is the null world: realized kurtosis still varies across names but carries **no**
  information about forward returns, and the sort must find nothing. ``edge > 0`` plants
  a positive kurt→return relation (long high-kurt), the sign the ``strategy`` sort is
  built to detect; the machinery is thus proven able to fire when an effect is present.

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
# Synthetic world — planted positive kurt->return relation (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 804,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    daily_vol: float = 0.012,
    drift: float = 0.06 / 252,
    factor_rho: float = 0.95,
    kurt_gain: float = 2.2,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC panel with a TUNABLE planted kurtosis->return relation.

    Each asset ``i`` carries a persistent latent "kurtosis tilt" ``c_i[t]`` — an AR(1)
    with autocorrelation ``factor_rho``, squashed to ``[0, 1]`` via a logistic so it acts
    as a fat-tail *intensity*. The tilt inflates the **kurtosis** of the daily return by
    scaling a heavy-tailed (Student-t-like) shock (so a trailing realized-kurtosis sort
    proxies ``c_i``), and — only when ``edge > 0`` — lifts the **forward mean**:

        g   = logistic(c_i[t])                       # fat-tail intensity in [0,1]
        s   = mix of a narrow and a wide Gaussian, mixing weight rising with g
        r[i,t] = drift + edge * (g - 0.5) + daily_vol * s

    So a high tilt makes returns **fat-tailed / high-kurtosis** *and* (with ``edge > 0``)
    higher-mean — a positive realized-kurt→return relation the sort can detect. ``edge =
    0`` is the null: kurtosis still varies across names but predicts nothing.
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

        # Fat-tail intensity in (0,1): high g -> more mass in the wide component.
        g = 1.0 / (1.0 + np.exp(-kurt_gain * c))
        # Two-component Gaussian mixture: a heavy tail switched on with prob ~g.
        z_narrow = rng.normal(0.0, 0.6, n_days)
        z_wide = rng.normal(0.0, 3.0, n_days)
        pick_wide = rng.random(n_days) < (0.15 * g)
        # Rescale so the marginal variance stays ~1 regardless of g mixing.
        s = np.where(pick_wide, z_wide, z_narrow)
        s = s / np.sqrt(np.mean(s ** 2)) if np.mean(s ** 2) > 0 else s
        r = drift + edge * (g - 0.5) + daily_vol * s

        close = 100.0 * np.cumprod(1.0 + r)
        prev_close = np.concatenate([[100.0], close[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0.0, daily_vol / 3, n_days))
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, daily_vol / 2, n_days)))
        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close}, index=idx
        )
    return panel
