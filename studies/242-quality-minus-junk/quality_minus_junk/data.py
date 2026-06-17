"""Data layer for Study 242 (Quality-Minus-Junk).

Two tapes, one schema — a (year × ticker) QMJ composite score plus aligned next-year
returns:

- ``synthetic_panel`` — a *deterministic, offline* generator. A tunable ``qmj_premium``
  knob controls how strongly the composite quality score predicts next-year returns;
  ``qmj_premium = 0`` is the null hypothesis. This lets tests assert the long-short only
  outperforms when the premium is planted.
- ``fetch_panel`` — the real EDGAR fundamentals, aligned to next-year annual returns, read
  from the desk's shared caches. ``fetch=False`` reads the staged parquets; ``fetch=True``
  is a no-op (the EDGAR cache is maintained centrally). Returns ``(score, fwd_ret)`` or
  empty DataFrames if caches are absent.

The QMJ composite implemented here approximates Asness, Frazzini & Pedersen (2019) using
three of their four sub-pillars (payout is excluded — no dividend/share-count cache):

  - **Profitability** (z-score of GP/A + ROA + net-margin + CFO/Assets, equal-weight)
  - **Growth** (z-score of year-over-year changes in the profitability metrics)
  - **Safety** (z-score of equity / assets as a low-leverage proxy)

The composite is the equal-weight average of the three pillar z-scores, re-ranked
cross-sectionally each year.

No look-ahead: signal at year *y* uses fiscal year-*y* EDGAR data; the strategy enters
at Jan 1 of year *y+1* (one full calendar-year reporting lag, following the Novy-Marx /
AQR convention).

The panel is **survivorship-biased** (current S&P 500 projected backwards): positive
results are **upper-bound estimates**.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    qmj_premium: float  # slope: how much QMJ z-score lifts next-year return

    @property
    def has_premium(self) -> bool:
        return self.qmj_premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 200,
    n_years: int = 18,
    qmj_premium: float = 0.06,
    base_ret: float = 0.10,
    ret_vol: float = 0.30,
    seed: int = 242,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A reproducible (year × ticker) QMJ composite panel with a known effect.

    Each firm-year draws a composite quality score from a standard normal (the
    QMJ composite is already z-scored). Next year's return follows
    ``base_ret + qmj_premium * z(quality_t) + noise``.  ``qmj_premium = 0`` is the null.

    Returns ``(score, fwd_ret, truth)`` — parallel (year × ticker) DataFrames where
    ``fwd_ret.loc[y]`` is the return that followed ``score.loc[y]``.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2008, 2008 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Quality score: composite z-score ~ standard normal.
    raw = rng.standard_normal((n_years, n_firms))
    # Cross-sectional z-score (already roughly normal but re-standardise for safety).
    z = (raw - raw.mean(axis=1, keepdims=True)) / (raw.std(axis=1, keepdims=True) + 1e-9)
    # Next-year returns load on the z-scored quality signal.
    fwd = base_ret + qmj_premium * z + ret_vol * rng.standard_normal((n_years, n_firms))

    score = pd.DataFrame(raw, index=pd.Index(years, name="year"), columns=firms)
    fwd_ret = pd.DataFrame(fwd, index=pd.Index(years, name="year"), columns=firms)
    return score, fwd_ret, WorldTruth(qmj_premium)


# ---------------------------------------------------------------------------
# Real panel — EDGAR caches, shared desk infrastructure
# ---------------------------------------------------------------------------
def _zscore_cross(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score each year, broadcast to (year × ticker) frame."""
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,  # noqa: ARG001
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """QMJ composite score + aligned next-year returns from the shared EDGAR caches.

    Constructs a three-pillar composite:
      - Profitability: GP/A + ROA + net-margin + CFO/Assets (equal-weight z-score)
      - Growth: year-over-year change in the profitability composite
      - Safety: equity / assets (low-leverage proxy)

    The composite QMJ = equal-weight(pillar_z_profit, pillar_z_growth, pillar_z_safety).
    All z-scores are cross-sectional (per-year rank).

    Returns ``(score, fwd_ret)`` where ``score.loc[y]`` = QMJ_y and ``fwd_ret.loc[y]``
    is the calendar return in year *y+1* (one-year lag). Returns empty DataFrames if any
    required cache is absent.

    **Survivorship caveat:** universe = current S&P 500 projected backwards.
    Positive results should be read as upper-bound estimates.
    """
    needed = {
        "gp": "_edgar_GrossProfit.parquet",
        "assets": "_edgar_Assets.parquet",
        "ni": "_edgar_NetIncomeLoss.parquet",
        "rev": "_edgar_Revenues.parquet",
        "eq": "_edgar_StockholdersEquity.parquet",
        "cfo": "_edgar_NetCashProvidedByUsedInOperatingActivities.parquet",
        "yrret": "_edgar_yrret.parquet",
    }
    paths = {k: os.path.join(cache_dir, v) for k, v in needed.items()}
    if not all(os.path.exists(p) for p in paths.values()):
        return pd.DataFrame(), pd.DataFrame()

    gp = pd.read_parquet(paths["gp"])
    assets = pd.read_parquet(paths["assets"])
    ni = pd.read_parquet(paths["ni"])
    rev = pd.read_parquet(paths["rev"])
    eq = pd.read_parquet(paths["eq"])
    cfo = pd.read_parquet(paths["cfo"])
    yr_ret = pd.read_parquet(paths["yrret"])

    # Restrict to years with reasonable coverage; exclude partial 2026.
    year_min, year_max = 2008, 2025

    def _trim(df: pd.DataFrame) -> pd.DataFrame:
        return df[(df.index >= year_min) & (df.index <= year_max)]

    gp = _trim(gp)
    assets = _trim(assets)
    ni = _trim(ni)
    rev = _trim(rev)
    eq = _trim(eq)
    cfo = _trim(cfo)

    # ---- Profitability pillar -------------------------------------------
    gpa = gp / assets.replace(0, np.nan)           # GrossProfit / Assets
    roa = ni / assets.replace(0, np.nan)            # NetIncome / Assets
    npm = ni / rev.replace(0, np.nan)               # Net margin
    cfoa = cfo / assets.replace(0, np.nan)          # CFO / Assets

    # Align all tickers to common universe across all four metrics.
    all_cols = (
        gpa.columns
        .intersection(roa.columns)
        .intersection(npm.columns)
        .intersection(cfoa.columns)
        .intersection(eq.columns)
    )

    gpa = gpa[all_cols]
    roa = roa[all_cols]
    npm = npm[all_cols]
    cfoa = cfoa[all_cols]
    eq_c = eq[all_cols]
    assets_c = assets[all_cols]

    # Cross-sectional z-scores per year.
    z_gpa = _zscore_cross(gpa)
    z_roa = _zscore_cross(roa)
    z_npm = _zscore_cross(npm)
    z_cfoa = _zscore_cross(cfoa)

    # Equal-weight profitability composite.
    z_profit = (z_gpa + z_roa + z_npm + z_cfoa) / 4.0

    # ---- Growth pillar ---------------------------------------------------
    z_profit_lag = z_profit.shift(1)
    growth = z_profit - z_profit_lag           # year-over-year change in composite
    z_growth = _zscore_cross(growth)

    # ---- Safety pillar ---------------------------------------------------
    leverage = eq_c / assets_c.replace(0, np.nan)    # equity ratio (higher = safer)
    z_safety = _zscore_cross(leverage)

    # ---- Composite QMJ --------------------------------------------------
    # Use the aligned index of z_growth (starts one year later due to lag).
    common_idx = z_profit.index.intersection(z_growth.index).intersection(z_safety.index)
    qmj = (
        z_profit.reindex(common_idx)
        + z_growth.reindex(common_idx)
        + z_safety.reindex(common_idx)
    ) / 3.0

    # Drop rows with too few non-NaN (< 30 firms).
    row_valid = qmj.notna().sum(axis=1) >= 30
    qmj = qmj.loc[row_valid]

    # Align returns: signal in year y → return in year y+1.
    fwd = yr_ret.reindex(qmj.index + 1)
    fwd.index = qmj.index
    fwd.index.name = "year"
    qmj.index.name = "year"

    return qmj, fwd


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a panel (for the as-of stamp in docs/results.md)."""
    arr = np.ascontiguousarray(df.fillna(0).to_numpy())
    h = hashlib.sha1(arr.tobytes())
    return h.hexdigest()[:12]
