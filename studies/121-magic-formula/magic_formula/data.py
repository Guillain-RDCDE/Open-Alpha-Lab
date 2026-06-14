"""Data layer for Study 121 (Magic-Formula).

Two tapes, one shape (a year × ticker panel of fundamentals + forward returns):

- ``synthetic_panel`` — a *deterministic, offline* generator. A ``premium`` knob plants
  the exact effect the Magic Formula claims: high-quality/cheap firms outperform by a known
  amount. ``premium = 0`` is the null: the combined rank carries no information and the top
  decile should not beat a random draw of the same size. This is the study's null in a bottle.

- ``fetch_panel`` — the real EDGAR fundamentals (OperatingIncomeLoss, Assets,
  LiabilitiesCurrent, LongTermDebtNoncurrent, CashAndCashEquivalentsAtCarryingValue,
  StockholdersEquity) from the desk's shared cache files at ``_cache/_edgar_<Concept>.parquet``.
  Returns (signal, fwd_ret) aligned so ``fwd_ret.loc[y]`` is calendar-year y+1 returns.

**Survivorship-bias caveat**: the EDGAR caches cover the *current* S&P 500 membership
projected backwards, so every firm in the panel survived to 2026. Positive results from
the real tape are upper bounds — the true live effect is weaker. This is named on the
Signal axis and in the results.

No look-ahead: the ``signal.loc[y]`` fundamentals are from fiscal year y's annual 10-K;
``fwd_ret.loc[y]`` is the return in calendar year y+1 (a conservative one-year-plus
reporting lag that assumes fundamentals are not actionable until the following year).
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
    "OperatingIncomeLoss",          # EBIT proxy
    "Assets",                       # total assets
    "LiabilitiesCurrent",           # current liabilities
    "LongTermDebtNoncurrent",       # interest-bearing debt
    "CashAndCashEquivalentsAtCarryingValue",  # cash (reduces EV and IC)
    "StockholdersEquity",           # book equity (used in EV proxy)
)


@dataclass(frozen=True)
class WorldTruth:
    """Records what was planted in the synthetic panel."""
    premium: float  # annualised alpha per unit of normalised rank (0 = null)

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
    seed: int = 121,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A firm × year panel with a known Magic-Formula effect — deterministic given ``seed``.

    Each firm-year draws two independent standard-normal scores (quality and cheapness);
    the combined rank is their average (mimicking Greenblatt's equal-weight combo). Next
    year's return is::

        base_ret + premium * z(combined_rank) + noise

    where ``z(·)`` normalises ranks to zero mean / unit variance cross-sectionally.
    ``premium = 0`` is the null (the top decile should beat a random equal-size draw
    by zero). Returns ``(signal_df, fwd_ret_df, truth)`` where signal_df carries the
    combined rank (year × firm, higher is better).
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2008, 2008 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Two independent quality/cheapness axes — combined rank is their average percentile
    quality = rng.standard_normal((n_years, n_firms))
    cheapness = rng.standard_normal((n_years, n_firms))
    combined = (quality + cheapness) / 2.0

    # Cross-sectional z-score of the combined signal
    mu_c = combined.mean(axis=1, keepdims=True)
    sd_c = combined.std(axis=1, keepdims=True) + 1e-9
    z = (combined - mu_c) / sd_c

    # Forward returns: higher rank → higher return when premium > 0
    fwd = base_ret + premium * z + ret_vol * rng.standard_normal((n_years, n_firms))

    signal = pd.DataFrame(
        combined,
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    fwd_ret = pd.DataFrame(
        fwd,
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    return signal, fwd_ret, WorldTruth(premium)


# ---------------------------------------------------------------------------
# Real panel — shared EDGAR caches, cache-first
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Magic-Formula signal + aligned next-year returns from the shared EDGAR pull.

    Reads six ``_edgar_<Concept>.parquet`` frames (columns = tickers, index = fiscal year)
    and the ``_edgar_yrret.parquet`` annual-return panel. Returns ``(signal, fwd_ret)``
    where ``signal.loc[y]`` is the combined MF rank for fiscal year y (higher rank =
    better), and ``fwd_ret.loc[y]`` is calendar-year y+1 returns (the next full year,
    conservative lag).

    If any required cache file is absent the function returns two empty DataFrames so
    the test-suite and the synthetic demo work offline.

    **Survivorship-bias warning**: the EDGAR panel covers only current S&P 500 members
    projected backwards. All results should be treated as upper bounds.
    """
    paths = {c: os.path.join(cache_dir, f"_edgar_{c}.parquet") for c in CONCEPTS}
    r_path = os.path.join(cache_dir, "_edgar_yrret.parquet")

    if not all(os.path.exists(p) for p in paths.values()) or not os.path.exists(r_path):
        return pd.DataFrame(), pd.DataFrame()

    panels = {c: pd.read_parquet(p) for c, p in paths.items()}
    yr_ret = pd.read_parquet(r_path)

    from .strategy import magic_formula_rank
    sig = magic_formula_rank(panels)

    # Align: signal from year y → returns in year y+1
    fwd = yr_ret.reindex(sig.index + 1)
    fwd.index = sig.index
    fwd.index.name = "year"
    return sig, fwd


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint for any DataFrame, for the as-of stamp."""
    arr = df.to_numpy(dtype=float, na_value=0.0)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
