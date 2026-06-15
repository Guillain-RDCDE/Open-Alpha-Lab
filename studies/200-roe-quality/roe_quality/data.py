"""Data layer for Study 200 (ROE-Quality).

Two tapes, one shape (a year × ticker panel of ROE signal + forward returns):

- ``synthetic_panel`` — a *deterministic, offline* generator. A ``premium`` knob plants
  the exact effect quality-factor theory predicts: high-ROE firms earn higher forward
  returns. ``premium = 0`` is the null: the ROE signal carries no information and the
  top quintile should not differ from a random draw of the same size. This is the
  study's null in a bottle.

- ``fetch_panel`` — the real EDGAR fundamentals (NetIncomeLoss, StockholdersEquity,
  GrossProfit, Assets) from the desk's shared cache files at
  ``_cache/_edgar_<Concept>.parquet``. Returns ``(roe, gp_ratio, fwd_ret)`` aligned so
  ``fwd_ret.loc[y]`` is calendar-year y+1 returns.

**ROE definition**::

    ROE_y = NetIncomeLoss_y / StockholdersEquity_{y-1}

Denominator lagged one year to avoid using end-of-period equity that is partly shaped by
the income being measured. Firms with non-positive lagged book equity are dropped
(negative book equity distorts ROE sign interpretation).

**Gross profitability (for head-to-head comparison)**::

    GP_ratio_y = GrossProfit_y / Assets_y

Following Novy-Marx (2013) — a cleaner profitability signal than ROE because it is
harder to manage and closer to economic profit.

**Survivorship-bias caveat**: the EDGAR caches cover the *current* S&P 500 membership
projected backwards, so every firm in the panel survived to 2026. Positive or negative
results from the real tape are upper/lower bounds — the true live effect is weaker in
magnitude. This is named on the Signal axis and in the results.

No look-ahead: the ``roe.loc[y]`` fundamentals are from fiscal year y's annual 10-K;
``fwd_ret.loc[y]`` is the return in calendar year y+1 (a conservative one-year-plus
reporting lag that assumes fundamentals are not actionable until the following year).

ROE is winsorized cross-sectionally at the 1st and 99th percentile to reduce the
influence of outlier firms with near-zero equity.
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

# Concepts required from the shared EDGAR cache
CONCEPTS = (
    "NetIncomeLoss",         # numerator of ROE
    "StockholdersEquity",    # denominator of ROE (lagged by one year)
    "GrossProfit",           # gross profitability comparison (Novy-Marx 2013)
    "Assets",                # denominator for gross profitability
)


@dataclass(frozen=True)
class WorldTruth:
    """Records what was planted in the synthetic panel."""

    premium: float  # return premium per unit of cross-sectional ROE z-rank
    # premium > 0 means high ROE -> higher returns (the quality story)
    # premium = 0 is the null: ROE carries no signal

    @property
    def has_premium(self) -> bool:
        return self.premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 200,
    n_years: int = 17,
    premium: float = 0.05,
    base_ret: float = 0.10,
    ret_vol: float = 0.30,
    seed: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A firm × year panel with a known ROE effect — deterministic given ``seed``.

    Each firm-year draws an independent standard-normal ROE z-score. Next year's return is::

        base_ret + premium * z(ROE_rank) + noise

    where ``z(·)`` normalises ranks to zero mean / unit variance cross-sectionally.
    With ``premium = 0`` the ROE rank carries zero signal (the null). With ``premium > 0``
    the high-ROE firms outperform (the quality-factor prediction). A synthetic gross
    profitability panel (GP_ratio) is also generated, independently, with the same
    planted premium so the head-to-head comparison is symmetric.

    Returns ``(roe_df, gp_df, fwd_ret_df, truth)`` where ``roe_df`` and ``gp_df`` carry
    the cross-sectional signals (year × firm), and ``fwd_ret_df`` the next-year returns.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2008, 2008 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Synthetic ROE-like scores (higher = higher return on equity)
    roe_scores = rng.standard_normal((n_years, n_firms))

    # Synthetic GP/Assets scores (independent signal)
    gp_scores = rng.standard_normal((n_years, n_firms))

    # Cross-sectional z-score of ROE signal
    mu_r = roe_scores.mean(axis=1, keepdims=True)
    sd_r = roe_scores.std(axis=1, keepdims=True) + 1e-9
    z_roe = (roe_scores - mu_r) / sd_r

    # Forward returns driven by ROE z-rank
    fwd = base_ret + premium * z_roe + ret_vol * rng.standard_normal((n_years, n_firms))

    roe_df = pd.DataFrame(
        roe_scores,
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    gp_df = pd.DataFrame(
        gp_scores,
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    fwd_ret_df = pd.DataFrame(
        fwd,
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    return roe_df, gp_df, fwd_ret_df, WorldTruth(premium)


# ---------------------------------------------------------------------------
# Real panel — shared EDGAR caches, cache-first
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ROE signal, GP/Assets signal, and aligned next-year returns from the EDGAR pull.

    Reads four ``_edgar_<Concept>.parquet`` frames (columns = tickers, index = fiscal year)
    and the ``_edgar_yrret.parquet`` annual-return panel. Returns ``(roe, gp_ratio, fwd_ret)``
    where ``roe.loc[y]`` is the winsorised ROE for fiscal year y (higher = higher quality),
    ``gp_ratio.loc[y]`` is GP/Assets for fiscal year y, and ``fwd_ret.loc[y]`` is
    calendar-year y+1 returns (the next full year, conservative lag).

    If any required cache file is absent the function returns three empty DataFrames so
    the test-suite and the synthetic demo work offline.

    **Survivorship-bias warning**: the EDGAR panel covers only current S&P 500 members
    projected backwards. All results should be treated as upper bounds in magnitude.
    """
    paths = {c: os.path.join(cache_dir, f"_edgar_{c}.parquet") for c in CONCEPTS}
    r_path = os.path.join(cache_dir, "_edgar_yrret.parquet")

    if not all(os.path.exists(p) for p in paths.values()) or not os.path.exists(r_path):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ni = pd.read_parquet(paths["NetIncomeLoss"])
    se = pd.read_parquet(paths["StockholdersEquity"])
    gp = pd.read_parquet(paths["GrossProfit"])
    assets = pd.read_parquet(paths["Assets"])
    yr_ret = pd.read_parquet(r_path)

    from .strategy import roe_signal, gp_signal

    common = sorted(
        set(ni.columns) & set(se.columns) & set(yr_ret.columns)
    )
    common_gp = sorted(
        set(common) & set(gp.columns) & set(assets.columns)
    )

    roe = roe_signal(
        ni.reindex(columns=common),
        se.reindex(columns=common),
    )
    gpr = gp_signal(
        gp.reindex(columns=common_gp),
        assets.reindex(columns=common_gp),
    )

    # Align: signal from year y -> returns in year y+1
    fwd = yr_ret.reindex(roe.index + 1)
    fwd.index = roe.index
    fwd.index.name = "year"

    return roe, gpr, fwd


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint for any DataFrame, for the as-of stamp."""
    arr = df.to_numpy(dtype=float, na_value=0.0)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
