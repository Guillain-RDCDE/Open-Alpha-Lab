"""Data layer for Study 244 (Asset-Growth).

Two tapes, one shape (a year × ticker panel of asset-growth signal + forward returns):

- ``synthetic_panel`` — a *deterministic, offline* generator. A ``premium`` knob plants
  the exact effect Cooper, Gulen & Schill (2008) claim: high-total-asset-growth firms
  underperform because investors over-extrapolate past investment growth. ``premium = 0``
  is the null: the asset-growth signal carries no information and the top quintile should
  not differ from a random draw of the same size. This is the study's null in a bottle.

- ``fetch_panel`` — the real EDGAR total-assets panel from the desk's shared cache file
  at ``_cache/_edgar_Assets.parquet``. Computes YoY total-asset growth and returns
  ``(ag, fwd_ret)`` aligned so ``fwd_ret.loc[y]`` is calendar-year y+1 returns.

**Asset Growth definition (Cooper, Gulen & Schill 2008)**::

    AG_t = (Total_Assets_t - Total_Assets_{t-1}) / Total_Assets_{t-1}

High AG = aggressive expansion. The paper predicts HIGH AG → LOW future returns
(managers over-invest; market extrapolates growth; disappointed when returns diminish).

**Survivorship-bias caveat**: the EDGAR caches cover the *current* S&P 500 membership
projected backwards, so every firm in the panel survived to 2026. Positive results from
the real tape are upper bounds — the true live effect is weaker. This is named on the
Signal axis and in the results.

No look-ahead: the ``ag.loc[y]`` growth rate uses fiscal years y and y-1 from the
annual 10-K; ``fwd_ret.loc[y]`` is the return in calendar year y+1 (a conservative
one-year-plus reporting lag that assumes fundamentals are not actionable until the
following year).
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

# Concept required from the shared EDGAR cache
CONCEPTS = ("Assets",)


@dataclass(frozen=True)
class WorldTruth:
    """Records what was planted in the synthetic panel."""
    premium: float  # annualised alpha per unit of normalised AG rank (0 = null)
    # premium < 0 means high asset-growth → lower returns (the paper's claim).

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
    seed: int = 244,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A firm × year panel with a known asset-growth effect — deterministic given ``seed``.

    Each firm-year draws an independent standard-normal AG z-score. Next year's return is::

        base_ret + premium * z(AG_rank) + noise

    where ``z(·)`` normalises ranks to zero mean / unit variance cross-sectionally.
    With ``premium = 0`` the AG rank carries zero signal (the null). With ``premium < 0``
    the high-AG firms underperform (the Cooper et al. prediction).

    Returns ``(ag_df, fwd_ret_df, truth)`` where ``ag_df`` carries the cross-sectional
    asset-growth signal (year × firm, higher = faster growth), and ``fwd_ret_df`` the
    next-year return.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2008, 2008 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Synthetic asset-growth-like scores (higher = more aggressive expansion)
    ag_scores = rng.standard_normal((n_years, n_firms))

    # Cross-sectional z-score of the AG signal
    mu_n = ag_scores.mean(axis=1, keepdims=True)
    sd_n = ag_scores.std(axis=1, keepdims=True) + 1e-9
    z = (ag_scores - mu_n) / sd_n

    # Forward returns: higher AG rank → lower return when premium < 0
    fwd = base_ret + premium * z + ret_vol * rng.standard_normal((n_years, n_firms))

    ag_df = pd.DataFrame(
        ag_scores,
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    fwd_ret_df = pd.DataFrame(
        fwd,
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    return ag_df, fwd_ret_df, WorldTruth(premium)


# ---------------------------------------------------------------------------
# Real panel — shared EDGAR caches, cache-first
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Asset-growth signal + aligned next-year returns from the shared EDGAR pull.

    Reads ``_edgar_Assets.parquet`` (columns = tickers, index = fiscal year) and
    ``_edgar_yrret.parquet`` annual-return panel. Computes YoY total-asset growth rate
    (AG = (Assets_t - Assets_{t-1}) / Assets_{t-1}) and returns ``(ag, fwd_ret)``
    where ``ag.loc[y]`` is the asset-growth rate for fiscal year y (higher = more
    aggressive expansion), and ``fwd_ret.loc[y]`` is calendar-year y+1 returns.

    If any required cache file is absent the function returns two empty DataFrames so
    the test-suite and the synthetic demo work offline.

    **Survivorship-bias warning**: the EDGAR panel covers only current S&P 500 members
    projected backwards. All results should be treated as upper bounds.
    """
    a_path = os.path.join(cache_dir, "_edgar_Assets.parquet")
    r_path = os.path.join(cache_dir, "_edgar_yrret.parquet")

    if not os.path.exists(a_path) or not os.path.exists(r_path):
        return pd.DataFrame(), pd.DataFrame()

    assets = pd.read_parquet(a_path)
    yr_ret = pd.read_parquet(r_path)

    from .strategy import ag_signal
    ag = ag_signal(assets)

    # Align: signal from year y → returns in year y+1
    fwd = yr_ret.reindex(ag.index + 1)
    fwd.index = ag.index
    fwd.index.name = "year"
    return ag, fwd


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint for any DataFrame, for the as-of stamp."""
    arr = df.to_numpy(dtype=float, na_value=0.0)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
