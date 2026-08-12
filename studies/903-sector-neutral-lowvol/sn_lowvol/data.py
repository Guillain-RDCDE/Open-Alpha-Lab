"""Data layer for Study 903 — Sector-Neutral Low-Vol.

The claim under test (Baker-Bradley-Wurgler 2011; Frazzini-Pedersen 2014; and the
critique of them): the **low-volatility anomaly** — calm stocks out-earn wild ones on a
risk-adjusted basis — is *partly* an artefact of a **sector bet**. A naive low-vol sort
loads up on the structurally calm sectors (utilities, staples) and shorts the naturally
wild ones (tech, energy), so much of the "edge" is really a defensive-sector tilt. Strip
that out: rank each name on its trailing volatility **within its own sector** (demean by
the sector's cross-sectional median), long the low-vol / short the high-vol names
**sector-neutrally**, and ask whether a genuine *stock-level* low-vol effect survives.

Two ingredients, both offline-friendly once cached.

* **Real tape — a liquid US cross-section.** Daily OHLC for a fixed list of ~50 liquid US
  large-caps (``UNIVERSE`` below), pulled with yfinance through the ``quantlab.universe``
  **survivorship guard** (``download_panel(..., allow_survivorship_bias=True)``).
  ``auto_adjust=True`` (total-return prices). The panel parquet is cached under this
  study's OWN ``_cache/`` (we point ``quantlab.universe``'s cache there via
  ``OVERNIGHT_CACHE`` *before* importing it). Each name carries a fixed **GICS sector**
  label (``SECTORS`` below) so the cross-section can be demeaned within sector.

  **Survivorship — named on the Signal axis.** ``UNIVERSE`` is a *current* membership list
  of names that are liquid mega-caps *today*; feeding it to a backward-looking panel omits
  the delisted / de-rated names and biases any cross-sectional result. The guard forces the
  opt-in; the caveat travels with every published number.

* **Synthetic world — the positive control.** A deterministic, seeded panel
  (``synthetic_panel``) with a TUNABLE knob ``edge`` and an explicit sector structure: each
  name carries a persistent latent volatility level ``v_i`` and belongs to a sector with its
  own base vol and mean premium. When ``edge > 0`` a name's forward mean falls with its
  volatility **relative to its own sector** — a genuine *stock-level* low-vol effect that a
  sector-neutral sort must recover. ``edge = 0`` is the null: within-sector vol carries **no**
  forward information and the sector-neutral sort must find nothing. A companion
  ``synthetic_sectors`` reproduces the (deterministic) sector assignment.

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

# Static GICS-style sector labels for the current-membership universe (a fixed map; the
# desk's sector *bet* is exactly what we neutralise). "Defensive" sectors — Staples,
# Utilities (none here), Health Care — are the structurally calm ones the naive low-vol
# sort over-weights; the sector-neutral sort strips that tilt.
SECTORS: dict[str, str] = {
    # Information Technology
    "AAPL": "InfoTech", "MSFT": "InfoTech", "NVDA": "InfoTech", "V": "InfoTech",
    "MA": "InfoTech", "CSCO": "InfoTech", "ORCL": "InfoTech", "ADBE": "InfoTech",
    "CRM": "InfoTech", "TXN": "InfoTech", "INTC": "InfoTech", "QCOM": "InfoTech",
    "AMD": "InfoTech", "IBM": "InfoTech",
    # Communication Services
    "GOOGL": "CommServices", "META": "CommServices", "DIS": "CommServices",
    "T": "CommServices", "VZ": "CommServices",
    # Consumer Discretionary
    "AMZN": "ConsDisc", "TSLA": "ConsDisc", "HD": "ConsDisc", "NKE": "ConsDisc",
    "MCD": "ConsDisc",
    # Consumer Staples (defensive)
    "WMT": "Staples", "PG": "Staples", "KO": "Staples", "PEP": "Staples",
    "COST": "Staples",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
    "MS": "Financials", "C": "Financials", "AXP": "Financials",
    # Health Care (defensive)
    "JNJ": "HealthCare", "ABBV": "HealthCare", "MRK": "HealthCare", "PFE": "HealthCare",
    "UNH": "HealthCare",
    # Energy
    "XOM": "Energy", "CVX": "Energy",
    # Industrials
    "GE": "Industrials", "CAT": "Industrials", "BA": "Industrials", "MMM": "Industrials",
    "HON": "Industrials", "UPS": "Industrials", "LMT": "Industrials",
}

# The desk-conventional "defensive" buckets — the calm sectors a naive low-vol sort loads.
DEFENSIVE_SECTORS = ("Staples", "HealthCare")

__all__ = [
    "SurvivorshipBiasError",
    "UNIVERSE", "SECTORS", "DEFENSIVE_SECTORS", "START", "AS_OF", "CACHE_DIR",
    "fetch", "have_real", "load_panel", "sector_series",
    "synthetic_panel", "synthetic_sectors",
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
    ``[start, asof]``. Reads the parquet directly — OFFLINE, no yfinance import."""
    cache = panel_cache_path(UNIVERSE, start)
    raw = pd.read_parquet(cache)
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    want = ["Open", "High", "Low", "Close"]
    has_vol = "Volume" in set(raw.columns.get_level_values(1))
    if has_vol:
        want = want + ["Volume"]
    panel: dict[str, pd.DataFrame] = {}
    for s in UNIVERSE:
        if s not in raw.columns.get_level_values(0):
            continue
        df = raw[s][want].dropna(subset=["Close"])
        df = df[(df.index >= lo) & (df.index <= hi)]
        if not df.empty:
            panel[s] = df
    return panel


def sector_series(panel: dict[str, pd.DataFrame]) -> pd.Series:
    """The sector label for every name present in ``panel`` (index=ticker)."""
    return pd.Series({s: SECTORS.get(s, "Other") for s in panel}, name="sector")


# --------------------------------------------------------------------------- #
# Synthetic world — planted stock-level low-vol effect within sectors (control)
# --------------------------------------------------------------------------- #
def synthetic_sectors(n_assets: int = 40, n_sectors: int = 8) -> dict[str, str]:
    """Deterministic sector assignment mirroring :func:`synthetic_panel` (round-robin)."""
    return {f"SYN{i:02d}": f"SEC{i % n_sectors}" for i in range(n_assets)}


def synthetic_panel(
    edge: float = 0.0,
    seed: int = 903,
    n_assets: int = 40,
    n_days: int = 3000,
    start: str = "2010-01-04",
    n_sectors: int = 8,
    base_vol: float = 0.011,
    within_vol_spread: float = 0.5,
    sector_vol_spread: float = 0.5,
    sector_prem_ann: float = 0.08,
    drift: float = 0.05 / 252,
) -> dict[str, pd.DataFrame]:
    """Deterministic seeded OHLC panel with a planted *stock-level* low-vol effect.

    Structure that lets the sector-neutral sort be tested honestly:

    * Each asset ``i`` is assigned round-robin to one of ``n_sectors`` sectors (see
      :func:`synthetic_sectors`). A sector ``s`` has a persistent base-vol multiplier
      ``sm_s`` (so some sectors are structurally calm, some wild) and a mean premium
      ``sp_s`` that is *correlated with its calmness* — the calm sectors carry a higher
      mean, exactly the defensive-sector confound the study targets.
    * Within a sector each name carries a persistent idiosyncratic vol tilt ``w_i`` around
      the sector level, so its realized daily vol is ``v_i = base_vol * sm_s * (1 + w_i)``.
    * Daily return:  ``r[i,t] = drift + sp_s - edge * (v_i - mean_s(v)) + v_i * z``
      with ``z ~ N(0,1)``. When ``edge > 0`` a name's forward mean falls with its vol
      **relative to its own sector's mean vol** — a genuine within-sector low-vol effect.
      ``edge = 0`` is the null: within-sector vol predicts nothing.

    So a *sector-neutral* sort (demean the vol by sector median) recovers the ``edge``
    effect; a *raw* sort additionally reaps ``sp_s`` (the sector premium) — the machinery
    can therefore show how much of a raw low-vol spread is the sector bet.
    Business-day index; span well below the pandas ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)

    # Sector-level parameters (calm sectors carry a higher mean premium).
    sm = rng.uniform(1.0 - sector_vol_spread, 1.0 + sector_vol_spread, n_sectors)
    # premium inversely related to sector vol multiplier (calm -> higher premium): a pure
    # defensive-SECTOR bet that a raw low-vol sort reaps but a sector-neutral sort must not.
    sp = -(sm - sm.mean()) / max(sm.std(), 1e-9) * (sector_prem_ann / 252)

    # Per-name idiosyncratic within-sector vol tilt and its per-name volatility level.
    w = rng.uniform(-within_vol_spread, within_vol_spread, n_assets)
    sec_of = np.array([i % n_sectors for i in range(n_assets)])
    v = base_vol * sm[sec_of] * (1.0 + w)

    # Within-sector mean vol (for the sector-relative low-vol effect).
    sec_mean_v = np.zeros(n_assets)
    for s in range(n_sectors):
        m = sec_of == s
        if m.any():
            sec_mean_v[m] = v[m].mean()

    panel: dict[str, pd.DataFrame] = {}
    for i in range(n_assets):
        z = rng.normal(0.0, 1.0, n_days)
        r = drift + sp[sec_of[i]] - edge * (v[i] - sec_mean_v[i]) + v[i] * z
        close = 100.0 * np.cumprod(1.0 + r)
        prev_close = np.concatenate([[100.0], close[:-1]])
        open_ = prev_close * (1.0 + rng.normal(0.0, v[i] / 3, n_days))
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, v[i] / 2, n_days)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, v[i] / 2, n_days)))
        vol = np.abs(rng.normal(1e6, 2e5, n_days))
        panel[f"SYN{i:02d}"] = pd.DataFrame(
            {"Open": open_, "High": hi, "Low": lo, "Close": close, "Volume": vol},
            index=idx,
        )
    return panel
