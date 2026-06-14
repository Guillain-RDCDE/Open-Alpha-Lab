"""Data layer for Study 154 (Leverage-Anomaly).

Two tapes, one shape (a year × ticker panel of leverage + forward returns):

- ``synthetic_panel`` — a *deterministic, offline* generator.  A ``premium`` knob
  plants the exact effect the leverage-anomaly literature claims: low-leverage firms
  outperform high-leverage firms by a known amount.  ``premium = 0`` is the null:
  the leverage rank carries no predictive information and the bottom decile (low
  leverage) should not beat a random draw of the same size.  This is the study's
  null in a bottle.

- ``fetch_panel`` — the real EDGAR fundamentals (LongTermDebtNoncurrent, Assets,
  Liabilities, StockholdersEquity) from the desk's shared cache files at
  ``_cache/_edgar_<Concept>.parquet``, plus the annual-return panel
  ``_cache/_edgar_yrret.parquet``.  Returns ``(lev1, lev2, fwd_ret)`` where
  ``lev1`` is the LTD/Assets ratio and ``lev2`` is the Liabilities/StockholdersEquity
  ratio (both year × ticker), and ``fwd_ret.loc[y]`` is calendar-year y+1 returns.

**Survivorship-bias caveat**: the EDGAR caches cover the *current* S&P 500
membership projected backwards — every firm in the panel survived to 2026.  Positive
results are *upper bounds*; the true live effect is weaker.  This is named on the
Signal axis and in the results.

No look-ahead: ``lev1.loc[y]`` / ``lev2.loc[y]`` are built from fiscal year y's
annual 10-K; ``fwd_ret.loc[y]`` is the return in calendar year y+1 (a conservative
one-year-plus reporting lag that assumes 10-K data is not actionable until the
following calendar year).
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

# Concepts drawn from the shared EDGAR cache
CONCEPTS = (
    "LongTermDebtNoncurrent",   # numerator of LTD/Assets
    "Assets",                   # denominator of LTD/Assets
    "Liabilities",              # numerator of D/E ratio
    "StockholdersEquity",       # denominator of D/E ratio
)


@dataclass(frozen=True)
class WorldTruth:
    """Records what was planted in the synthetic panel."""

    premium: float  # excess annual return of low-leverage over high-leverage (0 = null)

    @property
    def has_premium(self) -> bool:
        return self.premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 200,
    n_years: int = 15,
    premium: float = 0.05,
    base_ret: float = 0.10,
    ret_vol: float = 0.30,
    seed: int = 154,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A firm × year panel with a known leverage-anomaly effect — deterministic given ``seed``.

    Each firm-year draws a leverage score from a standard normal.  Next year's return is::

        base_ret - premium * z(leverage) + noise

    where ``z(·)`` normalises leverage scores to zero mean / unit variance cross-sectionally
    so that the *highest* leverage maps to the *lowest* expected return (the anomaly
    direction).  ``premium = 0`` is the null (the top and bottom leverage decile earn the
    same expected return).

    Returns ``(lev_signal, fwd_ret_df, truth)`` where ``lev_signal.loc[y]`` is the raw
    leverage proxy (higher = more debt), and the anomaly says the *low* tail should
    beat the *high* tail when ``premium > 0``.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2009, 2009 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Leverage score — plain normal, cross-sectionally
    lev = rng.standard_normal((n_years, n_firms))

    # Cross-sectional z-score of leverage
    mu_l = lev.mean(axis=1, keepdims=True)
    sd_l = lev.std(axis=1, keepdims=True) + 1e-9
    z = (lev - mu_l) / sd_l

    # Forward returns: *high* leverage → *lower* return (anomaly direction)
    fwd = base_ret - premium * z + ret_vol * rng.standard_normal((n_years, n_firms))

    lev_signal = pd.DataFrame(
        lev,
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    fwd_ret = pd.DataFrame(
        fwd,
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    return lev_signal, fwd_ret, WorldTruth(premium)


# ---------------------------------------------------------------------------
# Real panel — shared EDGAR caches, cache-first
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leverage signals + aligned next-year returns from the shared EDGAR pull.

    Reads four ``_edgar_<Concept>.parquet`` frames plus ``_edgar_yrret.parquet``.
    Returns ``(lev1, lev2, fwd_ret)`` where:

    - ``lev1`` — LongTermDebtNoncurrent / Assets (financial-leverage ratio; year × ticker).
    - ``lev2`` — Liabilities / StockholdersEquity (total-leverage / D-E ratio; year × ticker;
      rows where equity ≤ 0 are masked out as the ratio is undefined or misleading).
    - ``fwd_ret`` — calendar-year y+1 return aligned to the signal year y index.

    The anomaly direction: **low-lev** (bottom decile) should beat **high-lev** (top decile)
    if the Penman-Richardson-Tuna (2007) effect is present.

    Returns three *empty* DataFrames if any required cache file is absent — so the
    test-suite and the synthetic demo work offline.

    **Survivorship-bias warning**: the EDGAR panel covers only current S&P 500 members
    projected backwards.  All results are upper bounds.
    """
    paths = {c: os.path.join(cache_dir, f"_edgar_{c}.parquet") for c in CONCEPTS}
    r_path = os.path.join(cache_dir, "_edgar_yrret.parquet")

    if not all(os.path.exists(p) for p in paths.values()) or not os.path.exists(r_path):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ltd = pd.read_parquet(paths["LongTermDebtNoncurrent"])
    assets = pd.read_parquet(paths["Assets"])
    liab = pd.read_parquet(paths["Liabilities"])
    equity = pd.read_parquet(paths["StockholdersEquity"])
    yr_ret = pd.read_parquet(r_path)

    # --- LEV1: LTD / Assets --- #
    common1 = sorted(set(ltd.columns) & set(assets.columns) & set(yr_ret.columns))
    lev1 = ltd[common1] / assets[common1]
    # Trim to fiscal years with a usable forward return and non-trivial coverage
    lev1 = lev1.reindex([y for y in lev1.index if y + 1 in yr_ret.index and y <= 2024])

    # --- LEV2: Liabilities / Equity (D/E) --- #
    common2 = sorted(set(liab.columns) & set(equity.columns) & set(yr_ret.columns))
    eq2 = equity[common2].where(equity[common2] > 0)   # mask non-positive equity
    lev2 = liab[common2] / eq2
    # Drop extreme outliers (D/E > 20 is likely data error or financials)
    lev2 = lev2.where(lev2 <= 20)
    lev2 = lev2.reindex([y for y in lev2.index if y + 1 in yr_ret.index and y <= 2024])

    # --- Forward returns aligned to signal year --- #
    # Signal year y → returns in calendar year y+1
    all_years = sorted(set(lev1.index) | set(lev2.index))
    fwd_ret = yr_ret.reindex([y + 1 for y in all_years])
    fwd_ret.index = pd.Index(all_years, name="year")

    return lev1, lev2, fwd_ret


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint for any DataFrame (for the as-of stamp)."""
    arr = df.to_numpy(dtype=float, na_value=0.0)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
