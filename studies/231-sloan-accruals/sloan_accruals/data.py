"""Data layer for Study 231 (Sloan Accruals).

Two tapes, one shape (a year x ticker panel of accruals signal + forward returns):

- ``synthetic_panel`` — a *deterministic, offline* generator. A ``premium`` knob plants
  the exact effect Sloan (1996) claims: high-accrual firms underperform because earnings
  quality is low. ``premium = 0`` is the null: the accruals signal carries no information.

- ``fetch_panel`` — the real EDGAR fundamentals from the desk's shared cache files at
  ``_cache/_edgar_<Concept>.parquet``. Returns ``(acc, fwd_ret)`` aligned so
  ``fwd_ret.loc[y]`` is calendar-year y+1 returns.

**Sloan (1996) accruals definition (cash-flow-statement approach)**::

    Accruals = Net Income - Cash From Operations
    Accruals_scaled = Accruals / average(Total Assets_t, Total Assets_{t-1})

    HIGH accruals => earnings are supported by non-cash items => LOW future returns.
    LOW accruals  => earnings are backed by real cash flows => HIGHER future returns.

This is the cash-flow-statement version of the Sloan accrual, as used in the post-SFAS 95
literature and recommended by Richardson et al. (2005) as the cleaner measure.

**Survivorship-bias caveat**: the EDGAR caches cover the *current* S&P 500 membership
projected backwards, so every firm in the panel survived to 2026. Positive results from
the real tape are upper bounds — the true live effect is weaker. This is named on the
Signal axis and in the results.

No look-ahead: ``acc.loc[y]`` fundamentals are from fiscal year y's annual 10-K;
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
    "NetIncomeLoss",                              # net income (earnings)
    "NetCashProvidedByUsedInOperatingActivities", # operating cash flow
    "Assets",                                     # total assets (scale denominator)
)


@dataclass(frozen=True)
class WorldTruth:
    """Records what was planted in the synthetic panel."""
    premium: float  # annualised alpha per unit of normalised accruals rank (0 = null)
    # Note: premium < 0 means high accruals => lower returns (Sloan's claim).

    @property
    def has_premium(self) -> bool:
        return self.premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 200,
    n_years: int = 17,
    premium: float = -0.05,
    base_ret: float = 0.10,
    ret_vol: float = 0.30,
    seed: int = 231,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A firm x year panel with a known accruals effect — deterministic given ``seed``.

    Each firm-year draws an independent standard-normal accruals z-score. Next year's
    return is::

        base_ret + premium * z(acc_rank) + noise

    where ``z(.)`` normalises ranks to zero mean / unit variance cross-sectionally.
    With ``premium = 0`` the accruals rank carries zero signal (the null). With
    ``premium < 0`` high-accrual firms underperform (the Sloan prediction).

    Returns ``(acc_df, fwd_ret_df, truth)`` where ``acc_df`` carries the cross-sectional
    accruals signal (year x firm, higher = more accrual / less cash backing), and
    ``fwd_ret_df`` the next-year return.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2008, 2008 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Synthetic accruals-like scores (higher = more accruals, less cash backing)
    acc_scores = rng.standard_normal((n_years, n_firms))

    # Cross-sectional z-score of the accruals signal
    mu_a = acc_scores.mean(axis=1, keepdims=True)
    sd_a = acc_scores.std(axis=1, keepdims=True) + 1e-9
    z = (acc_scores - mu_a) / sd_a

    # Forward returns: higher accruals rank => lower return when premium < 0
    fwd = base_ret + premium * z + ret_vol * rng.standard_normal((n_years, n_firms))

    acc_df = pd.DataFrame(
        acc_scores,
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    fwd_ret_df = pd.DataFrame(
        fwd,
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    return acc_df, fwd_ret_df, WorldTruth(premium)


# ---------------------------------------------------------------------------
# Real panel — shared EDGAR caches, cache-first
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accruals signal + aligned next-year returns from the shared EDGAR pull.

    Reads three ``_edgar_<Concept>.parquet`` frames (columns = tickers, index = fiscal
    year) and the ``_edgar_yrret.parquet`` annual-return panel. Returns ``(acc, fwd_ret)``
    where ``acc.loc[y]`` is the scaled accruals ratio for fiscal year y
    (higher = more accrual / less cash backing), and ``fwd_ret.loc[y]`` is calendar-year
    y+1 returns (the next full year, conservative lag).

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

    from .strategy import accruals_signal
    acc = accruals_signal(panels)

    # Align: signal from year y => returns in year y+1
    fwd = yr_ret.reindex(acc.index + 1)
    fwd.index = acc.index
    fwd.index.name = "year"
    return acc, fwd


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint for any DataFrame, for the as-of stamp."""
    arr = df.to_numpy(dtype=float, na_value=0.0)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
