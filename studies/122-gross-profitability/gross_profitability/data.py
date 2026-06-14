"""Data layer for Study 122 (Gross-Profitability).

Two tapes, one schema — a (year × ticker) signal frame plus aligned next-year returns:

- ``synthetic_panel`` — a *deterministic, offline* generator. A tunable
  ``gpa_premium`` knob controls how strongly GrossProfit/Assets predicts next-year
  returns; ``gpa_premium = 0`` is the null hypothesis (no predictive power). This lets
  tests assert the long-short only outperforms when the premium is planted.
- ``fetch_panel`` — the real EDGAR fundamentals (GrossProfit and Assets), aligned to
  next-year CRSP-style annual returns, read from the desk's shared caches. No live
  network: ``fetch=False`` reads the staged parquets; ``fetch=True`` is a no-op here
  (the EDGAR cache is maintained centrally). Returns ``(signal, fwd_ret)`` or empty
  DataFrames if caches are absent.

No look-ahead: the signal for year *y* is built from fiscal year-*y* filings (reported
after the fiscal year ends — typically available mid-year for Dec-fiscal firms). We
conservatively lag by one full calendar year: the signal computed on year *y* data is
used to predict *year y+1* returns, which means entry is after year *y*'s filings are
all in (Jan 1 of year *y+1* at the earliest). This is the standard Novy-Marx lag
convention.

The panel is **survivorship-biased** (current S&P 500 membership projected backwards):
positive results should be read as **upper-bound** estimates.
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

    gpa_premium: float  # slope: how much GP/A z-score lifts next-year return

    @property
    def has_premium(self) -> bool:
        return self.gpa_premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 200,
    n_years: int = 18,
    gpa_premium: float = 0.06,
    base_ret: float = 0.10,
    ret_vol: float = 0.30,
    seed: int = 122,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A reproducible (year × ticker) gross-profitability panel with a known effect.

    Each firm-year draws a GP/Assets ratio from a skewed log-normal (reflecting the
    real distribution: GPA is bounded below at 0 and right-skewed). Next year's return
    follows ``base_ret + gpa_premium * z(GPA_t) + noise`` where ``z`` is the
    cross-sectional z-score of GPA in year *t*. ``gpa_premium = 0`` is the null.

    Returns ``(signal, fwd_ret, truth)`` — parallel (year × ticker) DataFrames where
    ``fwd_ret.loc[y]`` is the return that followed ``signal.loc[y]``.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2008, 2008 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # GPA is strictly positive and right-skewed: simulate as exp(Normal) capped plausibly.
    raw = rng.lognormal(mean=-0.5, sigma=0.7, size=(n_years, n_firms))
    # Cross-sectional z-score for signal construction.
    z = (raw - raw.mean(axis=1, keepdims=True)) / (raw.std(axis=1, keepdims=True) + 1e-9)
    # Next-year returns load on the z-scored signal.
    fwd = base_ret + gpa_premium * z + ret_vol * rng.standard_normal((n_years, n_firms))

    signal = pd.DataFrame(raw, index=pd.Index(years, name="year"), columns=firms)
    fwd_ret = pd.DataFrame(fwd, index=pd.Index(years, name="year"), columns=firms)
    return signal, fwd_ret, WorldTruth(gpa_premium)


# ---------------------------------------------------------------------------
# Real panel — EDGAR caches, shared desk infrastructure
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,  # noqa: ARG001 — present for API symmetry; EDGAR cache is read-only
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gross-profitability signal + aligned next-year returns from the shared EDGAR caches.

    Reads ``_edgar_GrossProfit.parquet``, ``_edgar_Assets.parquet``, and
    ``_edgar_yrret.parquet`` from *cache_dir*. Returns ``(signal, fwd_ret)`` where
    ``signal.loc[y]`` = GrossProfit_y / Assets_y for each ticker and ``fwd_ret.loc[y]``
    is the calendar return in year *y+1* (the one-year-lagged prediction).

    If the cache files are absent, returns ``(pd.DataFrame(), pd.DataFrame())`` rather
    than raising — the caller can check ``signal.empty``.

    **Survivorship caveat (naming it explicitly):** the universe is the current S&P 500
    membership projected backwards. Positive results should be read as upper-bound
    estimates, not generalisable population effects.
    """
    gp_p = os.path.join(cache_dir, "_edgar_GrossProfit.parquet")
    a_p = os.path.join(cache_dir, "_edgar_Assets.parquet")
    r_p = os.path.join(cache_dir, "_edgar_yrret.parquet")
    if not all(os.path.exists(p) for p in (gp_p, a_p, r_p)):
        return pd.DataFrame(), pd.DataFrame()

    gp = pd.read_parquet(gp_p)
    assets = pd.read_parquet(a_p)
    yr_ret = pd.read_parquet(r_p)

    # Align tickers across the two fundamental frames.
    common_tickers = gp.columns.intersection(assets.columns)
    gpa = gp[common_tickers] / assets[common_tickers]

    # Drop partial / placeholder years (2026 is ongoing) and early sparse years.
    gpa = gpa[gpa.index < 2026]

    # Align returns: signal in year y → return in year y+1.
    fwd = yr_ret.reindex(gpa.index + 1)
    fwd.index = gpa.index
    fwd.index.name = "year"

    return gpa, fwd


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a panel (for the as-of stamp in docs/results.md)."""
    arr = np.ascontiguousarray(df.fillna(0).to_numpy())
    h = hashlib.sha1(arr.tobytes())
    return h.hexdigest()[:12]
