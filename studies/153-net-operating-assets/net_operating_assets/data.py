"""Data layer for Study 153 (Net-Operating-Assets).

Two tapes, one shape (a year × ticker panel of NOA signal + forward returns):

- ``synthetic_panel`` — a *deterministic, offline* generator. A ``premium`` knob plants
  the exact effect Hirshleifer et al. (2004) claim: high-NOA firms underperform because
  investors over-extrapolate accounting earnings. ``premium = 0`` is the null: the NOA
  signal carries no information and the top quintile should not differ from a random
  draw of the same size. This is the study's null in a bottle.

- ``fetch_panel`` — the real EDGAR fundamentals (Assets, Liabilities,
  LongTermDebtNoncurrent, CashAndCashEquivalentsAtCarryingValue) from the desk's shared
  cache files at ``_cache/_edgar_<Concept>.parquet``. Returns ``(noa, fwd_ret)`` aligned
  so ``fwd_ret.loc[y]`` is calendar-year y+1 returns.

**Net Operating Assets definition (Hirshleifer et al. 2004)**::

    Operating Assets  = Total Assets - Cash
    Operating Liabs   = Total Liabilities - Long-Term Financial Debt
    NOA_raw           = Operating Assets - Operating Liabs
    NOA               = NOA_raw / lagged Total Assets   (scale for comparability)

High NOA = balance-sheet "bloat" = firm has accumulated large operating-asset surplus
relative to its asset base. The paper predicts HIGH NOA → LOW future returns (investors
over-extrapolate accounting-based performance signals embedded in past earnings).

**Survivorship-bias caveat**: the EDGAR caches cover the *current* S&P 500 membership
projected backwards, so every firm in the panel survived to 2026. Positive results from
the real tape are upper bounds — the true live effect is weaker. This is named on the
Signal axis and in the results.

No look-ahead: the ``noa.loc[y]`` fundamentals are from fiscal year y's annual 10-K;
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
    "Assets",                                         # total assets (numerator and lag denominator)
    "CashAndCashEquivalentsAtCarryingValue",           # cash (reduces operating assets)
    "Liabilities",                                    # total liabilities (operating liabilities proxy)
    "LongTermDebtNoncurrent",                         # financial debt (excluded from operating liabs)
)


@dataclass(frozen=True)
class WorldTruth:
    """Records what was planted in the synthetic panel."""
    premium: float  # annualised alpha per unit of normalised NOA rank (0 = null)
    # Note: premium < 0 means high NOA → lower returns (the paper's claim).
    # We store the *direction* as planted: premium = -0.05 means the top NOA quintile
    # underperforms by ~5%/yr per unit z-rank.

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
    seed: int = 153,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A firm × year panel with a known NOA effect — deterministic given ``seed``.

    Each firm-year draws an independent standard-normal NOA z-score. Next year's return is::

        base_ret + premium * z(NOA_rank) + noise

    where ``z(·)`` normalises ranks to zero mean / unit variance cross-sectionally.
    With ``premium = 0`` the NOA rank carries zero signal (the null). With ``premium < 0``
    the high-NOA firms underperform (the Hirshleifer et al. prediction).

    Returns ``(noa_df, fwd_ret_df, truth)`` where ``noa_df`` carries the cross-sectional
    NOA signal (year × firm, higher = more bloat), and ``fwd_ret_df`` the next-year return.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2008, 2008 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Synthetic NOA-like scores (higher = more balance-sheet bloat)
    noa_scores = rng.standard_normal((n_years, n_firms))

    # Cross-sectional z-score of the NOA signal
    mu_n = noa_scores.mean(axis=1, keepdims=True)
    sd_n = noa_scores.std(axis=1, keepdims=True) + 1e-9
    z = (noa_scores - mu_n) / sd_n

    # Forward returns: higher NOA rank → lower return when premium < 0
    fwd = base_ret + premium * z + ret_vol * rng.standard_normal((n_years, n_firms))

    noa_df = pd.DataFrame(
        noa_scores,
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    fwd_ret_df = pd.DataFrame(
        fwd,
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    return noa_df, fwd_ret_df, WorldTruth(premium)


# ---------------------------------------------------------------------------
# Real panel — shared EDGAR caches, cache-first
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """NOA signal + aligned next-year returns from the shared EDGAR pull.

    Reads four ``_edgar_<Concept>.parquet`` frames (columns = tickers, index = fiscal year)
    and the ``_edgar_yrret.parquet`` annual-return panel. Returns ``(noa, fwd_ret)``
    where ``noa.loc[y]`` is the NOA ratio for fiscal year y (higher = more bloat),
    and ``fwd_ret.loc[y]`` is calendar-year y+1 returns (the next full year,
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

    from .strategy import noa_signal
    noa = noa_signal(panels)

    # Align: signal from year y → returns in year y+1
    fwd = yr_ret.reindex(noa.index + 1)
    fwd.index = noa.index
    fwd.index.name = "year"
    return noa, fwd


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint for any DataFrame, for the as-of stamp."""
    arr = df.to_numpy(dtype=float, na_value=0.0)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
