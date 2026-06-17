"""Data layer for Study 232 (Mohanram G-score).

Two tapes, one schema -- a (year x ticker) signal frame plus aligned next-year returns:

- ``synthetic_panel`` -- a *deterministic, offline* generator. A tunable ``g_premium``
  knob controls how strongly the G-score predicts next-year returns; ``g_premium = 0``
  is the null hypothesis. This lets tests assert the long-short outperforms only when a
  premium is planted.
- ``fetch_panel`` -- the real EDGAR fundamentals, aligned to next-year calendar returns
  from the desk's shared caches. No live network: reads the staged parquets. Returns
  ``(signal, fwd_ret)`` or empty DataFrames if caches are absent.

No look-ahead: the signal for year *y* is built from fiscal year-*y* filings. We
conservatively lag by one full calendar year: the G-score computed on year *y* data
predicts *year y+1* returns, meaning entry is after year *y*'s filings are in
(Jan 1 of year *y+1* at the earliest). This is the standard Mohanram lag convention.

The panel is **survivorship-biased** (current S&P 500 membership projected backwards):
positive results should be read as **upper-bound** estimates; negative results are real
and concerning.

G-score components implemented (8 binary signals, score in {0..8}):
  G1 ROA > cross-sectional median (profitability level)
  G2 CFO/Assets > cross-sectional median (cash profitability)
  G3 Accruals (NI - CFO) / Assets < cross-sectional median (earnings quality)
  G4 Rolling ROA std < cross-sectional median (earnings stability)
  G5 Rolling CFO/Assets std < cross-sectional median (cash-flow stability)
  G6 Revenue growth positive (growth conservatism)
  G7 Asset-turnover growth positive (efficiency improvement)
  G8 ROA > 0 (absolute profitability)

Note: Mohanram (2005) also used R&D intensity and advertising intensity as G6/G7;
those EDGAR concepts are not cached on the desk. We substitute revenue-growth and
asset-turnover-growth, which capture the same "investment in future earnings" idea.
This substitution is documented explicitly: results may differ from the published paper.
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

# ---- Cache paths for required EDGAR concepts ---
_EDGAR_NI = "_edgar_NetIncomeLoss.parquet"
_EDGAR_ASSETS = "_edgar_Assets.parquet"
_EDGAR_CFO = "_edgar_NetCashProvidedByUsedInOperatingActivities.parquet"
_EDGAR_REV = "_edgar_Revenues.parquet"
_EDGAR_YRRET = "_edgar_yrret.parquet"


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    g_premium: float  # slope: how much G-score z-score lifts next-year return

    @property
    def has_premium(self) -> bool:
        return self.g_premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 200,
    n_years: int = 18,
    g_premium: float = 0.06,
    base_ret: float = 0.10,
    ret_vol: float = 0.30,
    seed: int = 232,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A reproducible (year x ticker) G-score panel with a known planted effect.

    Each firm-year draws a composite G-score (0-8 integer) from a discrete distribution
    resembling the empirical one. Next year's return follows
    ``base_ret + g_premium * z(G_t) + noise`` where ``z`` is the cross-sectional z-score
    of the G-score in year *t*. ``g_premium = 0`` is the null hypothesis.

    Returns ``(signal, fwd_ret, truth)`` -- parallel (year x ticker) DataFrames where
    ``fwd_ret.loc[y]`` is the return that followed ``signal.loc[y]``.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2007, 2007 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # G-score: 8 binary signals, sum 0-8. Model as Binomial(8, 0.5) + small firm effect.
    raw = rng.binomial(8, 0.5, size=(n_years, n_firms)).astype(float)
    # Cross-sectional z-score for return loading.
    z = (raw - raw.mean(axis=1, keepdims=True)) / (raw.std(axis=1, keepdims=True) + 1e-9)
    fwd = base_ret + g_premium * z + ret_vol * rng.standard_normal((n_years, n_firms))

    signal = pd.DataFrame(raw, index=pd.Index(years, name="year"), columns=firms)
    fwd_ret = pd.DataFrame(fwd, index=pd.Index(years, name="year"), columns=firms)
    return signal, fwd_ret, WorldTruth(g_premium)


# ---------------------------------------------------------------------------
# Real panel -- EDGAR caches, shared desk infrastructure
# ---------------------------------------------------------------------------
def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,  # noqa: ARG001 -- present for API symmetry; EDGAR cache is read-only
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mohanram G-score signal + aligned next-year returns from the shared EDGAR caches.

    Reads NetIncomeLoss, Assets, CFO, Revenues, and yrret from *cache_dir*.
    Returns ``(signal, fwd_ret)`` where ``signal.loc[y]`` = G-score in {0..8} for each
    ticker and ``fwd_ret.loc[y]`` is the calendar return in year *y+1*.

    G-score components (8 binary signals):
      G1  ROA > cross-sectional median
      G2  CFO/Assets > cross-sectional median
      G3  Accruals (NI-CFO)/Assets < cross-sectional median  (lower = better quality)
      G4  Rolling-3yr ROA std < cross-sectional median       (earnings stability)
      G5  Rolling-3yr CFO/A std < cross-sectional median     (cash-flow stability)
      G6  Revenue growth > 0                                  (proxy for growth conservatism)
      G7  Asset-turnover growth > 0                           (efficiency improvement)
      G8  ROA > 0                                             (absolute profitability)

    If any required cache file is absent, returns ``(pd.DataFrame(), pd.DataFrame())``.

    **Survivorship caveat:** the universe is the current S&P 500 membership projected
    backwards. Positive results are upper-bound estimates; negative results suggest the
    G-score as computed does not find genuinely superior growth stocks in large-cap survivors.
    """
    paths = {
        "ni": os.path.join(cache_dir, _EDGAR_NI),
        "assets": os.path.join(cache_dir, _EDGAR_ASSETS),
        "cfo": os.path.join(cache_dir, _EDGAR_CFO),
        "rev": os.path.join(cache_dir, _EDGAR_REV),
        "yrret": os.path.join(cache_dir, _EDGAR_YRRET),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        return pd.DataFrame(), pd.DataFrame()

    ni = pd.read_parquet(paths["ni"])
    assets = pd.read_parquet(paths["assets"])
    cfo = pd.read_parquet(paths["cfo"])
    rev = pd.read_parquet(paths["rev"])
    yrret = pd.read_parquet(paths["yrret"])

    # Common tickers across all four fundamental frames
    common = (
        ni.columns.intersection(assets.columns)
        .intersection(cfo.columns)
        .intersection(rev.columns)
    )

    # Drop partial / placeholder years (2026 is ongoing)
    yrs_sig = sorted([y for y in ni.index if y <= 2025])

    roa = ni.loc[yrs_sig, common] / assets.loc[yrs_sig, common]
    cfoa = cfo.loc[yrs_sig, common] / assets.loc[yrs_sig, common]
    accruals = (ni.loc[yrs_sig, common] - cfo.loc[yrs_sig, common]) / assets.loc[yrs_sig, common]
    rev_common = rev[common].reindex(yrs_sig)
    rev_growth = rev_common.pct_change()
    at_ratio = rev_common / assets.loc[yrs_sig, common]
    at_growth = at_ratio.pct_change()

    gscore = _build_gscore(roa, cfoa, accruals, rev_growth, at_growth)

    # Align returns: signal in year y -> return in year y+1
    fwd = yrret.reindex(gscore.index + 1)
    fwd.index = gscore.index
    fwd.index.name = "year"

    return gscore, fwd


def _build_gscore(
    roa: pd.DataFrame,
    cfoa: pd.DataFrame,
    accruals: pd.DataFrame,
    rev_growth: pd.DataFrame,
    at_growth: pd.DataFrame,
) -> pd.DataFrame:
    """Build the 8-component G-score panel (year x ticker), values in {0..8}."""
    roa_rolling_std = roa.rolling(3, min_periods=2).std()
    cfo_rolling_std = cfoa.rolling(3, min_periods=2).std()

    rows: dict[int, pd.Series] = {}
    for y in roa.index:
        r = roa.loc[y]
        c = cfoa.loc[y]
        acc = accruals.loc[y]
        rg = rev_growth.loc[y]
        atg = at_growth.loc[y]
        roa_std = roa_rolling_std.loc[y]
        cfo_std = cfo_rolling_std.loc[y]

        g1 = (r > r.median()).astype(float)             # ROA > median
        g2 = (c > c.median()).astype(float)              # CFO/A > median
        g3 = (acc < acc.median()).astype(float)          # accruals < median (lower = better)
        g4 = (roa_std < roa_std.median()).astype(float)  # earnings stability
        g5 = (cfo_std < cfo_std.median()).astype(float)  # cash-flow stability
        g6 = (rg > 0).astype(float)                     # revenue growth positive
        g7 = (atg > 0).astype(float)                    # asset-turnover improvement
        g8 = (r > 0).astype(float)                      # ROA positive

        score = (
            g1.fillna(0) + g2.fillna(0) + g3.fillna(0) + g4.fillna(0)
            + g5.fillna(0) + g6.fillna(0) + g7.fillna(0) + g8.fillna(0)
        )
        rows[int(y)] = score

    df = pd.DataFrame(rows).T.sort_index()
    df.index.name = "year"
    return df


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a panel (for the as-of stamp in docs/results.md)."""
    arr = np.ascontiguousarray(df.fillna(0).to_numpy())
    h = hashlib.sha1(arr.tobytes())
    return h.hexdigest()[:12]
