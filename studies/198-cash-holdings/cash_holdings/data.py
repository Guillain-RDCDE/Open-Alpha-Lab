"""Data layer for Study 198 (Cash-Holdings).

Two tapes, one shape (a year × ticker panel of Cash-to-Assets signal + forward returns):

- ``synthetic_panel`` — a *deterministic, offline* generator. A ``premium`` knob plants
  the exact effect Palazzo (2012) claims: high-cash firms earn higher returns (a financial-
  constraint premium). ``premium = 0`` is the null: the cash ratio carries no information
  and the top quintile should not differ from a random draw of the same size. This is the
  study's null in a bottle.

- ``fetch_panel`` — the real EDGAR fundamentals (Assets,
  CashAndCashEquivalentsAtCarryingValue) from the desk's shared cache files at
  ``_cache/_edgar_<Concept>.parquet``. Returns ``(cash_ratio, fwd_ret)`` aligned so
  ``fwd_ret.loc[y]`` is calendar-year y+1 returns.

**Cash-to-Assets definition (Palazzo 2012)**::

    Cash-to-Assets = CashAndCashEquivalentsAtCarryingValue / Total Assets

High Cash-to-Assets = cash-rich = Palazzo's financial-constraint proxy. The paper predicts
HIGH Cash-to-Assets → HIGH future returns (a financial-constraint risk premium).

**Survivorship-bias caveat**: the EDGAR caches cover the *current* S&P 500 membership
projected backwards, so every firm in the panel survived to 2026. Positive results from
the real tape are upper bounds — the true live effect is weaker. This is named on the
Signal axis and in the results.

No look-ahead: the ``cash_ratio.loc[y]`` fundamentals are from fiscal year y's annual
10-K; ``fwd_ret.loc[y]`` is the return in calendar year y+1 (a conservative one-year-plus
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
    "Assets",                                     # total assets (denominator)
    "CashAndCashEquivalentsAtCarryingValue",       # cash (numerator)
)


@dataclass(frozen=True)
class WorldTruth:
    """Records what was planted in the synthetic panel."""

    premium: float  # annualised alpha per unit of normalised Cash-to-Assets rank (0 = null)
    # Note: premium > 0 means high cash → higher returns (Palazzo's claim).
    # premium = 0 is the null (no anomaly).

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
    seed: int = 198,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A firm × year panel with a known cash-holdings effect — deterministic given ``seed``.

    Each firm-year draws an independent standard-normal cash-ratio z-score. Next year's
    return is::

        base_ret + premium * z(cash_rank) + noise

    where ``z(·)`` normalises ranks to zero mean / unit variance cross-sectionally.
    With ``premium = 0`` the cash rank carries zero signal (the null). With ``premium > 0``
    the high-cash firms outperform (the Palazzo 2012 prediction).

    Returns ``(cash_df, fwd_ret_df, truth)`` where ``cash_df`` carries the cross-sectional
    Cash-to-Assets signal (year × firm, higher = more cash), and ``fwd_ret_df`` the
    next-year return.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2008, 2008 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Synthetic cash-to-assets scores in [0, 1] range (realistic: cash ratio bounded)
    cash_raw = rng.beta(2, 5, (n_years, n_firms))  # right-skewed, mostly < 0.5

    # Cross-sectional z-score of the signal
    mu_n = cash_raw.mean(axis=1, keepdims=True)
    sd_n = cash_raw.std(axis=1, keepdims=True) + 1e-9
    z = (cash_raw - mu_n) / sd_n

    # Forward returns: higher cash rank → higher return when premium > 0
    fwd = base_ret + premium * z + ret_vol * rng.standard_normal((n_years, n_firms))

    cash_df = pd.DataFrame(
        cash_raw,
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    fwd_ret_df = pd.DataFrame(
        fwd,
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    return cash_df, fwd_ret_df, WorldTruth(premium)


# ---------------------------------------------------------------------------
# Real panel — shared EDGAR caches, cache-first
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cash-to-Assets signal + aligned next-year returns from the shared EDGAR pull.

    Reads two ``_edgar_<Concept>.parquet`` frames (columns = tickers, index = fiscal year)
    and the ``_edgar_yrret.parquet`` annual-return panel. Returns ``(cash_ratio, fwd_ret)``
    where ``cash_ratio.loc[y]`` is the Cash-to-Assets ratio for fiscal year y (higher =
    more cash), and ``fwd_ret.loc[y]`` is calendar-year y+1 returns (the next full year,
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

    assets = pd.read_parquet(paths["Assets"])
    cash = pd.read_parquet(paths["CashAndCashEquivalentsAtCarryingValue"])
    yr_ret = pd.read_parquet(r_path)

    # Compute cash-to-assets ratio
    from .strategy import cash_ratio_signal
    ratio = cash_ratio_signal(assets, cash)

    # Align: signal from year y → returns in year y+1
    fwd = yr_ret.reindex(ratio.index + 1)
    fwd.index = ratio.index
    fwd.index.name = "year"

    return ratio, fwd


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint for any DataFrame, for the as-of stamp."""
    arr = df.to_numpy(dtype=float, na_value=0.0)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
