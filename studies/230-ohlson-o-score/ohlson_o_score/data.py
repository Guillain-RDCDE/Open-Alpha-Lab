"""Data layer for Study 230 (Ohlson O-score).

Two tapes, one contract (a year x ticker DataFrame of O-scores paired with a
next-year-return frame):

- ``synthetic_panel`` -- a **deterministic, offline** generator. A tunable
  ``o_premium`` knob plants a known relationship between O-score and forward
  return; ``o_premium = 0`` is the null (O tells you nothing). Note that the
  O-score is a *distress* score: higher O = more distressed. A *positive*
  o_premium means high-O (distressed) firms earn *more* (the risk-premium
  story). A *negative* o_premium means high-O firms earn *less* (the
  distress-puzzle story). ``o_premium = 0`` is the null.

- ``fetch_panel`` -- the real panel built from the desk's shared EDGAR caches
  (``_cache/_edgar_<Concept>.parquet``) and yfinance annual prices. **Cache-only
  by default.** Survivorship-bias is explicit: the panel covers *current* S&P
  500 members projected backwards, so results are upper bounds.

No look-ahead: the O-score for year *t* uses year-*t* fundamentals (10-K data
for fiscal year *t*) and the December month-end price. Because 10-Ks file in
Q1 of *t+1*, the signal is paired with the calendar year-*(t+1)* return -- a
clean one-year lag.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")

# EDGAR concepts needed for the Ohlson O-score
_CONCEPTS = [
    "Assets",
    "AssetsCurrent",
    "LiabilitiesCurrent",
    "Liabilities",
    "NetIncomeLoss",
    "NetCashProvidedByUsedInOperatingActivities",
]

# Ohlson (1980) O-score coefficients
# O = b0 + b_size*SIZE + b_tlta*TLTA + b_wcta*WCTA + b_clca*CLCA
#        + b_oeneg*OENEG + b_nita*NITA + b_futl*FUTL + b_intwo*INTWO + b_chin*CHIN
O_COEF = dict(
    intercept=-1.32,
    size=-0.407,
    tlta=6.03,
    wcta=-1.43,
    clca=0.076,
    oeneg=-1.72,
    nita=-2.37,
    futl=-1.83,
    intwo=0.285,
    chin=-0.521,
)

# O-score to bankruptcy probability via logistic transform: p = 1 / (1 + exp(-O))
# High O → high distress probability


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------

def synthetic_panel(
    n_firms: int = 200,
    n_years: int = 16,
    o_premium: float = 0.05,
    base_ret: float = 0.10,
    ret_vol: float = 0.30,
    seed: int = 230,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A synthetic firm panel with a known O-score -> return relationship.

    Each firm-year draws an O-score from a realistic distribution (roughly
    normal, centred around -1 with spread ~2). The next-year return is:

        ret = base_ret + o_premium * (O - median_O) / IQR_O + noise

    so high-O (distressed) firms outperform when ``o_premium > 0`` (the
    risk-premium story) and underperform when ``o_premium < 0`` (the distress
    puzzle). ``o_premium = 0`` is the null.

    Because the O-score is a *distress* measure (higher = worse), the hedge
    strategy sorts: low-O (safe) long, high-O (distressed) short. The
    ``hedge`` column is therefore ``ret_lo - ret_hi`` (safe minus distressed).

    Returns ``(o_scores, fwd_returns, truth)`` where truth is a dict of the
    planted parameters.
    """
    rng = np.random.default_rng(seed)
    years = pd.Index(range(2008, 2008 + n_years), name="year")
    tickers = [f"F{j:03d}" for j in range(n_firms)]

    # Draw O-scores: realistic range, centred around -1 (most S&P 500 firms
    # are not in distress, so the bulk of the distribution is negative)
    raw = rng.normal(-1.0, 2.0, size=(n_years, n_firms))
    o = pd.DataFrame(raw, index=years, columns=tickers)

    # Forward returns: loaded on O if o_premium != 0
    med = np.nanmedian(raw, axis=1, keepdims=True)
    iqr = (np.nanpercentile(raw, 75, axis=1, keepdims=True)
           - np.nanpercentile(raw, 25, axis=1, keepdims=True))
    iqr = np.where(iqr > 0, iqr, 1.0)
    o_norm = (raw - med) / iqr

    noise = rng.normal(0.0, ret_vol, size=(n_years, n_firms))
    fwd_raw = base_ret + o_premium * o_norm + noise
    fwd = pd.DataFrame(fwd_raw, index=years, columns=tickers)

    truth = dict(
        n_firms=n_firms, n_years=n_years, o_premium=o_premium,
        base_ret=base_ret, ret_vol=ret_vol, seed=seed,
        has_premium=o_premium != 0.0,
    )
    return o, fwd, truth


# ---------------------------------------------------------------------------
# Real panel -- EDGAR + yfinance, cache-first
# ---------------------------------------------------------------------------

def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    start_year: int = 2008,
    end_year: int = 2023,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ohlson O-score panel + aligned next-year returns, cache-first.

    Reads the desk's shared EDGAR concept caches (``_cache/_edgar_*.parquet``)
    and the annual-return frame (``_cache/_edgar_yrret.parquet``).

    **Survivorship bias is explicit and unavoidable**: the EDGAR caches contain
    only tickers that are *current* S&P 500 members, projected backwards. All
    results therefore represent an upper bound on what a live strategy could
    have earned.

    O-score for year *t* uses year-*t* 10-K data paired with the year-*(t+1)*
    return (a clean one-year lag; 10-Ks file in Q1 of *t+1*).

    Returns
    -------
    o_scores : DataFrame, shape (n_years, n_tickers)
        Ohlson O-score, year x ticker. Higher = more distressed.
    fwd_ret : DataFrame, same shape
        Calendar-year *(t+1)* return, aligned to year-*t* signal index.
    """
    # ---- load EDGAR concept frames -----------------------------------------
    paths = {c: os.path.join(cache_dir, f"_edgar_{c}.parquet") for c in _CONCEPTS}
    r_path = os.path.join(cache_dir, "_edgar_yrret.parquet")

    missing = [c for c, p in paths.items() if not os.path.exists(p)]
    if missing or not os.path.exists(r_path):
        return pd.DataFrame(), pd.DataFrame()

    panels = {c: pd.read_parquet(p) for c, p in paths.items()}
    yr_ret = pd.read_parquet(r_path)

    # ---- common tickers with all required concepts -------------------------
    common = sorted(
        set(panels["Assets"].columns)
        & set(panels["AssetsCurrent"].columns)
        & set(panels["LiabilitiesCurrent"].columns)
        & set(panels["Liabilities"].columns)
        & set(panels["NetIncomeLoss"].columns)
        & set(panels["NetCashProvidedByUsedInOperatingActivities"].columns)
    )

    # ---- year range --------------------------------------------------------
    years = list(range(start_year, end_year + 1))

    ta    = panels["Assets"].reindex(index=years, columns=common)
    ac    = panels["AssetsCurrent"].reindex(index=years, columns=common)
    lc    = panels["LiabilitiesCurrent"].reindex(index=years, columns=common)
    tl    = panels["Liabilities"].reindex(index=years, columns=common)
    ni    = panels["NetIncomeLoss"].reindex(index=years, columns=common)
    cfo   = panels["NetCashProvidedByUsedInOperatingActivities"].reindex(
                index=years, columns=common)

    # Also pull prior-year net income for INTWO and CHIN
    years_prev = list(range(start_year - 1, end_year))
    ni_prev = panels["NetIncomeLoss"].reindex(index=years_prev, columns=common)
    ni_prev.index = years  # align to current year

    # ---- compute O-score components ----------------------------------------
    # Valid mask: assets and liabilities strictly positive
    valid = (ta > 0) & (tl > 0)

    # SIZE = log(Total Assets) — using a constant GNP deflator is unnecessary
    # for cross-sectional sorting since all firms are deflated the same way.
    SIZE = np.log(ta.clip(lower=1.0)).where(valid)

    # TLTA = Total Liabilities / Total Assets
    TLTA = (tl / ta).where(valid)

    # WCTA = (Current Assets - Current Liabilities) / Total Assets
    WCTA = ((ac - lc) / ta).where(valid)

    # CLCA = Current Liabilities / Current Assets (undefined if AC = 0)
    CLCA = (lc / ac.replace(0, np.nan)).where(valid)

    # OENEG = 1 if Total Liabilities > Total Assets (technically insolvent)
    OENEG = (tl > ta).astype(float).where(valid)

    # NITA = Net Income / Total Assets
    NITA = (ni / ta).where(valid)

    # FUTL = Cash Flow from Operations / Total Liabilities
    FUTL = (cfo / tl).where(valid)

    # INTWO = 1 if net income negative in both current AND prior year
    INTWO = ((ni < 0) & (ni_prev < 0)).astype(float).where(valid)

    # CHIN = (NI_t - NI_{t-1}) / (|NI_t| + |NI_{t-1}|)
    denom_chin = (ni.abs() + ni_prev.abs()).replace(0, np.nan)
    CHIN = ((ni - ni_prev) / denom_chin).where(valid)

    # ---- assemble O-score --------------------------------------------------
    c = O_COEF
    o_scores = (
        c["intercept"]
        + c["size"]    * SIZE
        + c["tlta"]    * TLTA
        + c["wcta"]    * WCTA
        + c["clca"]    * CLCA
        + c["oeneg"]   * OENEG
        + c["nita"]    * NITA
        + c["futl"]    * FUTL
        + c["intwo"]   * INTWO
        + c["chin"]    * CHIN
    )

    # ---- forward returns: year t+1 return, aligned to signal year t --------
    fwd = yr_ret.reindex(index=[y + 1 for y in years], columns=common)
    fwd.index = years
    fwd.index.name = "year"

    return o_scores, fwd


def fingerprint(o: pd.DataFrame) -> str:
    """A short content fingerprint of the O-score panel (for the as-of stamp)."""
    arr = o.values.astype(float)
    flat = arr[np.isfinite(arr)]
    h = hashlib.sha1(np.ascontiguousarray(flat).tobytes())
    return h.hexdigest()[:12]
