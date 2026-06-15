"""Data layer for Study 199 (Sales-Growth).

Two tapes, one shape (a year x ticker panel of sales-growth signal + forward returns):

- ``synthetic_panel`` — a *deterministic, offline* generator. A ``premium`` knob plants
  the exact effect Lakonishok, Shleifer & Vishny (1994) claim: high-sales-growth
  "glamour" stocks UNDERperform low-growth "value" stocks. ``premium = 0`` is the null:
  the growth signal carries no information and the top quintile should not differ from a
  random draw of the same size. ``premium < 0`` plants the LSV hypothesis (high growth
  → lower returns). This is the study's null in a bottle.

- ``fetch_panel`` — the real EDGAR fundamentals (Revenues) from the desk's shared cache
  at ``_cache/_edgar_Revenues.parquet`` plus ``_cache/_edgar_yrret.parquet`` for annual
  returns. Returns ``(sales_growth, fwd_ret)`` aligned so ``fwd_ret.loc[y]`` is
  calendar-year y+1 returns for signal year y.

**Sales growth definition**::

    SalesGrowth_y = Revenues_y / Revenues_{y-1} - 1

High sales growth = "glamour" stock. LSV (1994) predicts HIGH growth → LOW future
returns (investors over-extrapolate the growth trajectory).

**Reporting lag**: fundamentals from fiscal year y are used to predict calendar-year
y+1 returns — a conservative one-year-plus reporting lag that assumes financials are not
actionable until the following year.

**Survivorship-bias caveat**: the EDGAR caches cover the *current* S&P 500 membership
projected backwards, so every firm in the panel survived to 2026. Positive results from
the real tape are upper bounds — the true live effect is weaker. This is named on the
Signal axis and in the results.
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
CONCEPTS = ("Revenues",)


@dataclass(frozen=True)
class WorldTruth:
    """Records what was planted in the synthetic panel."""

    premium: float  # annualised alpha per unit of normalised rank of sales growth.
    # premium < 0 means high growth -> lower returns (LSV claim).
    # premium = 0 means the null: growth carries no information.

    @property
    def has_premium(self) -> bool:
        return self.premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 200,
    n_years: int = 18,
    premium: float = -0.05,
    base_ret: float = 0.10,
    ret_vol: float = 0.30,
    seed: int = 199,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A firm x year panel with a known sales-growth effect — deterministic given ``seed``.

    Each firm-year draws an independent standard-normal sales-growth z-score. Next year's
    return is::

        base_ret + premium * z(SalesGrowth_rank) + noise

    where ``z(.)`` normalises ranks to zero mean / unit variance cross-sectionally.
    With ``premium = 0`` the sales-growth rank carries zero signal (the null).
    With ``premium < 0`` the high-growth firms underperform (the LSV prediction).

    Returns ``(sales_growth_df, fwd_ret_df, truth)`` where ``sales_growth_df`` carries the
    cross-sectional sales-growth signal (year x firm, higher = faster growth), and
    ``fwd_ret_df`` the next-year return.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2007, 2007 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Synthetic sales-growth scores (higher = faster revenue growth)
    sg_scores = rng.standard_normal((n_years, n_firms))

    # Cross-sectional z-score of the sales-growth signal
    mu_n = sg_scores.mean(axis=1, keepdims=True)
    sd_n = sg_scores.std(axis=1, keepdims=True) + 1e-9
    z = (sg_scores - mu_n) / sd_n

    # Forward returns: higher growth rank -> lower return when premium < 0
    fwd = base_ret + premium * z + ret_vol * rng.standard_normal((n_years, n_firms))

    sg_df = pd.DataFrame(
        sg_scores,
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    fwd_ret_df = pd.DataFrame(
        fwd,
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    return sg_df, fwd_ret_df, WorldTruth(premium)


# ---------------------------------------------------------------------------
# Real panel — shared EDGAR cache, cache-first
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sales-growth signal + aligned next-year returns from the shared EDGAR pull.

    Reads ``_edgar_Revenues.parquet`` (columns = tickers, index = fiscal year) and the
    ``_edgar_yrret.parquet`` annual-return panel. Returns ``(sales_growth, fwd_ret)``
    where ``sales_growth.loc[y]`` is the trailing one-year revenue growth for fiscal
    year y (higher = faster growth = "glamour"), and ``fwd_ret.loc[y]`` is calendar-year
    y+1 returns (the next full year, conservative lag).

    If any required cache file is absent the function returns two empty DataFrames so
    the test-suite and the synthetic demo work offline.

    **Survivorship-bias warning**: the EDGAR panel covers only current S&P 500 members
    projected backwards. All results should be treated as upper bounds.
    """
    rev_path = os.path.join(cache_dir, "_edgar_Revenues.parquet")
    r_path = os.path.join(cache_dir, "_edgar_yrret.parquet")

    if not os.path.exists(rev_path) or not os.path.exists(r_path):
        return pd.DataFrame(), pd.DataFrame()

    rev = pd.read_parquet(rev_path)
    yr_ret = pd.read_parquet(r_path)

    from .strategy import sales_growth_signal

    sg = sales_growth_signal(rev)

    # Align: signal from year y -> returns in year y+1
    # We build fwd_ret indexed by the *signal* year y, containing year y+1 returns.
    common_years = sorted(set(sg.index) & {y - 1 for y in yr_ret.index})
    fwd_rows = {}
    for y in sg.index:
        if y + 1 in yr_ret.index:
            fwd_rows[y] = yr_ret.loc[y + 1]
    if not fwd_rows:
        return pd.DataFrame(), pd.DataFrame()

    fwd = pd.DataFrame(fwd_rows).T
    fwd.index.name = "year"
    fwd.index = fwd.index.astype(int)

    # Keep only signal years that have forward returns
    valid_years = sorted(set(sg.index) & set(fwd.index))
    sg = sg.loc[valid_years]
    fwd = fwd.loc[valid_years]
    return sg, fwd


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint for any DataFrame, for the as-of stamp."""
    arr = df.to_numpy(dtype=float, na_value=0.0)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
