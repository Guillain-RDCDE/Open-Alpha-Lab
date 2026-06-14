"""Data layer for Study 155 (Asset-Turnover).

Two tapes, one shape (a year × ticker panel of signals + forward returns):

- ``synthetic_panel`` — a *deterministic, offline* generator.  A ``premium`` knob plants
  the exact effect the asset-turnover factor claims: high-efficiency firms (high Revenues /
  Assets) outperform by a known amount.  ``premium = 0`` is the null: the AT rank carries
  no information and the top quintile should not beat a random draw of the same size.
  This is the study's null in a bottle.

- ``fetch_panel`` — the real EDGAR fundamentals (Revenues, Assets, NetIncomeLoss) from the
  desk's shared cache files at ``_cache/_edgar_<Concept>.parquet``.  Returns ``(at_signal,
  roa_signal, fwd_ret)`` aligned so ``fwd_ret.loc[y]`` is calendar-year y+1 returns
  (a one-year lag that ensures the 10-K is filed and publicly available before we act on
  it).

**Survivorship-bias caveat**: the EDGAR caches cover the *current* S&P 500 membership
projected backwards, so every firm in the panel survived to 2026.  Positive results from
the real tape are upper bounds — the true live edge is weaker.  This is named explicitly
on the Signal axis and in docs/results.md.

No look-ahead: ``at_signal.loc[y]`` fundamentals are from fiscal year y's annual 10-K;
``fwd_ret.loc[y]`` is the return in calendar year y+1 (the next full year, conservative
reporting lag — roughly 12–20 months before the signal is acted upon).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")

# EDGAR concept names required from the shared cache
CONCEPTS = (
    "Revenues",       # net revenues / total sales
    "Assets",         # total assets
    "NetIncomeLoss",  # net income (for the ROA comparison)
)

# Signal year range (where EDGAR data is dense; 2026 is partial so we cap at 2023 for
# the forward-return window — we need y+1 returns to be complete)
SIGNAL_YEAR_START = 2008
SIGNAL_YEAR_END = 2023


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 200,
    n_years: int = 16,
    premium: float = 0.04,
    base_ret: float = 0.12,
    ret_vol: float = 0.28,
    seed: int = 155,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A firm × year panel with a known asset-turnover effect — deterministic given ``seed``.

    Each firm-year draws an asset-turnover signal (log-normal, mimicking the real ratio
    distribution which is right-skewed with sector clusters).  Next year's return is::

        base_ret + premium * z(AT_rank) + noise

    where ``z(·)`` normalises the cross-sectional AT rank to zero mean / unit variance.
    ``premium = 0`` is the null: the top quintile should beat a random equal-size draw
    by zero.

    Returns ``(at_signal, fwd_ret, truth)`` where ``at_signal`` is year × firm with the
    asset-turnover ratio, ``fwd_ret`` is year × firm with next-year log returns, and
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2008, 2008 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Asset turnover is log-normally distributed (values ~ 0.1 to 3.0 in real data)
    log_at = rng.standard_normal((n_years, n_firms))  # cross-sectional variation

    # Cross-sectional z-score of the rank (rank to avoid outlier sensitivity)
    rank_z = np.zeros_like(log_at)
    for t in range(n_years):
        r = pd.Series(log_at[t]).rank(pct=True)
        rank_z[t] = (r - r.mean()) / (r.std() + 1e-9)

    # Forward returns: higher AT rank → higher return when premium > 0
    fwd = base_ret + premium * rank_z + ret_vol * rng.standard_normal((n_years, n_firms))

    at_signal = pd.DataFrame(
        np.exp(log_at),  # back to ratio-scale for realism
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    fwd_ret = pd.DataFrame(
        fwd,
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    truth = {
        "premium": premium,
        "n_firms": n_firms,
        "n_years": n_years,
        "seed": seed,
    }
    return at_signal, fwd_ret, truth


# ---------------------------------------------------------------------------
# Real panel — shared EDGAR caches, cache-first
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    signal_year_start: int = SIGNAL_YEAR_START,
    signal_year_end: int = SIGNAL_YEAR_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Asset-Turnover and ROA signals + aligned next-year returns from the shared EDGAR pull.

    Reads ``_edgar_Revenues.parquet``, ``_edgar_Assets.parquet``, and
    ``_edgar_NetIncomeLoss.parquet`` (columns = tickers, index = fiscal year) and the
    ``_edgar_yrret.parquet`` annual-return panel.

    Returns ``(at_signal, roa_signal, fwd_ret)`` where:

    - ``at_signal.loc[y]``  = Revenues_y / Assets_y (the efficiency ratio, winsorised at
      the 99th cross-sectional percentile to mute extreme outliers such as pure-fee firms).
    - ``roa_signal.loc[y]`` = NetIncomeLoss_y / Assets_y (the profitability baseline for
      a head-to-head comparison with AT).
    - ``fwd_ret.loc[y]``    = calendar-year y+1 returns (one-year-plus reporting lag).

    If any required cache file is absent the function returns three empty DataFrames so
    the test-suite and the synthetic demo work offline.

    **Survivorship-bias warning**: the EDGAR panel covers only current S&P 500 members
    projected backwards.  All positive results should be treated as upper bounds.
    """
    paths = {c: os.path.join(cache_dir, f"_edgar_{c}.parquet") for c in CONCEPTS}
    r_path = os.path.join(cache_dir, "_edgar_yrret.parquet")

    if not all(os.path.exists(p) for p in paths.values()) or not os.path.exists(r_path):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rev_raw = pd.read_parquet(paths["Revenues"])
    assets_raw = pd.read_parquet(paths["Assets"])
    ni_raw = pd.read_parquet(paths["NetIncomeLoss"])
    yr_ret = pd.read_parquet(r_path)

    # Common ticker universe (tickers present in both Revenues and Assets)
    common = sorted(set(rev_raw.columns) & set(assets_raw.columns))
    rev = rev_raw.reindex(columns=common)
    assets = assets_raw.reindex(columns=common)
    ni = ni_raw.reindex(columns=common)

    # Asset Turnover = Revenues / Assets (assets must be positive)
    assets_pos = assets.where(assets > 0)
    at_raw = rev / assets_pos

    # Winsorise at the 99th cross-sectional percentile per year (handles pure-fee / financial
    # firms with anomalously high AT ratios that would dominate the signal)
    at_signal = at_raw.copy()
    for y in at_signal.index:
        row = at_signal.loc[y]
        cap = row.quantile(0.99)
        at_signal.loc[y] = row.clip(upper=cap)

    # ROA = NetIncomeLoss / Assets (profitability comparison leg)
    roa_signal = ni / assets_pos

    # Filter to signal window
    at_signal = at_signal.loc[signal_year_start:signal_year_end]
    roa_signal = roa_signal.loc[signal_year_start:signal_year_end]

    # Align: signal from fiscal year y → returns in calendar year y+1
    fwd = yr_ret.reindex(at_signal.index + 1)
    fwd.index = at_signal.index
    fwd.index.name = "year"

    return at_signal, roa_signal, fwd


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of any DataFrame, for the as-of stamp."""
    arr = df.to_numpy(dtype=float, na_value=0.0)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
