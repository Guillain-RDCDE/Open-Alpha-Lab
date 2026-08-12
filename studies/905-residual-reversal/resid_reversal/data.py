"""Data layer for Study 905 — Residual Reversal.

The claim under test (Blitz, Huij, Lansdorp & Verbeek 2013, *"Short-Term Residual
Reversal"*): short-horizon reversal — buy last week's losers, sell last week's winners —
is *contaminated*. Part of a name's weekly move is (a) **bid-ask bounce** (a mechanical
one-print artefact that fakes reversal) and (b) a **common factor** move (a name that
sank because the whole market sank is not mispriced). Strip the factor with a
market-model regression, keep the **residual** weekly return, and reverse on that. The
residual reversal is a cleaner signal that, they argue, survives costs where the raw
version — which mostly harvests bounce and factor noise — does not.

Two ingredients, both offline-friendly once cached.

* **Real tape — a liquid US cross-section.** Daily OHLC **+ Volume** for a fixed list of
  ~50 liquid US large-caps (``UNIVERSE`` below), pulled with yfinance through the
  ``quantlab.universe`` **survivorship guard** (``download_panel(...,
  allow_survivorship_bias=True)``). ``auto_adjust=True`` (total-return prices). The panel
  parquet is cached under this study's OWN ``_cache/`` (we point ``quantlab.universe``'s
  cache there via ``OVERNIGHT_CACHE`` *before* importing it).

  **Survivorship — named on the Signal axis.** ``UNIVERSE`` is a *current* membership
  list of names that are liquid mega-caps *today*; feeding it to a backward-looking panel
  omits the delisted / de-rated names and biases any cross-sectional result. The guard
  forces the opt-in; the caveat travels with every published number.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) with a TUNABLE knob ``edge``: each name loads on a common market
  factor (``beta_i``) and carries a weekly **residual** whose next value mean-reverts on
  its last — an AR(1) with coefficient ``-edge``. ``edge = 0`` is the null world:
  residuals are i.i.d., the reversal sort must find nothing. ``edge > 0`` plants the
  Blitz-et-al residual reversal (last week's residual loser out-earns next week), while
  the market-factor component muddies the *raw* return so the machinery must clean it.

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
    ``[start, asof]``. Reads the parquet directly — OFFLINE, no yfinance import. Volume is
    carried for the dollar-volume liquidity screen."""
    cache = panel_cache_path(UNIVERSE, start)
    raw = pd.read_parquet(cache)
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    want = ["Open", "High", "Low", "Close", "Volume"]
    panel: dict[str, pd.DataFrame] = {}
    for s in UNIVERSE:
        if s not in raw.columns.get_level_values(0):
            continue
        cols = [c for c in want if c in raw[s].columns]
        df = raw[s][cols].dropna(subset=["Close"])
        df = df[(df.index >= lo) & (df.index <= hi)]
        if not df.empty:
            panel[s] = df
    return panel


# --------------------------------------------------------------------------- #
# Synthetic world — planted weekly residual reversal (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 905,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    mkt_vol: float = 0.040,
    resid_week_sd: float = 0.020,
    daily_noise: float = 0.004,
    drift: float = 0.06 / 252,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC+Volume panel with a TUNABLE planted residual reversal.

    Each asset ``i`` loads on a single common market factor with beta ``beta_i`` and
    carries its own **weekly residual** ``u[i, w]`` built as an AR(1) with coefficient
    ``-edge``::

        u[i, w] = -edge * u[i, w-1] + eps[i, w],   eps ~ N(0, resid_week_sd)

    so a positive ``edge`` makes this week's residual **mean-revert** on last week's — the
    Blitz-et-al residual reversal (last week's residual loser tends to be next week's
    winner). The weekly residual is spread uniformly across that week's five trading days
    and combined with the daily market factor plus small daily idiosyncratic noise::

        r[i, t] = drift + beta_i * f[t] + u[i, week(t)] / 5 + daily_noise * z

    Because the raw return is dominated by ``beta_i * f[t]``, a **raw** last-week-return
    reversal sort sees mostly factor + noise, whereas the **market-model residual** sort
    recovers the planted mean-reversion cleanly. ``edge = 0`` is the null: residuals are
    i.i.d., the sort must find nothing. Business-day index; span well below the pandas
    ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    n_weeks = n_days // 5 + 2
    week_of_day = np.arange(n_days) // 5          # each day -> its week block index

    f = rng.normal(drift, mkt_vol, n_days)        # common market factor (daily)
    betas = rng.uniform(0.7, 1.3, n_assets)
    base_vol = rng.uniform(4e6, 4e7, n_assets)    # per-name typical share volume

    panel: dict[str, pd.DataFrame] = {}
    for i in range(n_assets):
        eps = rng.normal(0.0, resid_week_sd, n_weeks)
        u = np.empty(n_weeks)
        u[0] = eps[0]
        for w in range(1, n_weeks):
            u[w] = -edge * u[w - 1] + eps[w]
        daily_resid = u[week_of_day] / 5.0 + rng.normal(0.0, daily_noise, n_days)
        r = drift + betas[i] * f + daily_resid

        close = 100.0 * np.cumprod(1.0 + r)
        prev_close = np.concatenate([[100.0], close[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0.0, daily_noise / 2, n_days))
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, daily_noise, n_days)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, daily_noise, n_days)))
        vol = base_vol[i] * (1.0 + np.abs(rng.normal(0.0, 0.3, n_days)))
        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close, "Volume": vol},
            index=idx,
        )
    return panel
