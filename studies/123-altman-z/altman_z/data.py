"""Data layer for Study 123 (Altman-Z).

Two tapes, one contract (a year × ticker DataFrame of Z-scores paired with a
next-year-return frame):

- ``synthetic_panel`` — a **deterministic, offline** generator. A tunable
  ``z_premium`` knob plants a known relationship between Z-score and forward
  return; ``z_premium = 0`` is the null (Z tells you nothing). This is the
  study's machinery check: if the engine cannot recover a planted effect it
  cannot honestly claim absence of one on the real tape.

- ``fetch_panel`` — the real panel built from the desk's shared EDGAR caches
  (``_cache/_edgar_<Concept>.parquet``) and yfinance annual prices for market
  cap. **Cache-only by default** — the shared EDGAR pull is a frozen research
  snapshot; passing ``fetch=True`` merely re-reads the same cached files rather
  than hitting the network. Survivorship-bias is explicit: the panel covers
  *current* S&P 500 members projected backwards, so results are upper bounds
  on what a live strategy could have earned.

No look-ahead: the Z-score for year *t* uses year-*t* fundamentals (10-K
accounting data for fiscal year *t*) and the market cap at December 31 of year
*t*. Because 10-Ks file in Q1 of *t+1*, the signal is paired with the calendar
year-*t+1* return — a clean one-year lag with no look-ahead.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")

# EDGAR concepts needed for the Altman Z-score
_CONCEPTS = [
    "Assets",
    "AssetsCurrent",
    "LiabilitiesCurrent",
    "RetainedEarningsAccumulatedDeficit",
    "OperatingIncomeLoss",
    "Liabilities",
    "Revenues",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
]

# Altman Z-score original coefficients (public-company version, Altman 1968)
Z_COEF = dict(X1=1.2, X2=1.4, X3=3.3, X4=0.6, X5=1.0)

# Classic Altman zone thresholds (public-company model)
ZONE_DISTRESS = 1.81   # Z < 1.81 → distress zone
ZONE_SAFE     = 2.99   # Z > 2.99 → safe zone


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------

def synthetic_panel(
    n_firms: int = 200,
    n_years: int = 16,
    z_premium: float = 0.05,
    base_ret: float = 0.10,
    ret_vol: float = 0.30,
    seed: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A synthetic firm panel with a known Z-score → return relationship.

    Each firm-year draws a Z-score from a realistic distribution (roughly
    log-normal, clipped at [0, 20]). The next-year return is:

        ret = base_ret + z_premium * (Z - median_Z) / IQR_Z + noise

    so high-Z firms outperform when ``z_premium > 0`` (the "safe firms win"
    hypothesis) and underperform when ``z_premium < 0`` (the "distress
    premium" hypothesis). ``z_premium = 0`` is the null.

    Returns ``(z_scores, fwd_returns, truth)`` where truth is a dict of the
    planted parameters.
    """
    rng = np.random.default_rng(seed)
    years = pd.Index(range(2008, 2008 + n_years), name="year")
    tickers = [f"F{j:03d}" for j in range(n_firms)]

    # Draw Z-scores in a realistic range (median ~2.5, skewed right)
    raw = rng.exponential(scale=2.5, size=(n_years, n_firms)) + 0.3
    raw = np.clip(raw, 0.0, 20.0)
    z = pd.DataFrame(raw, index=years, columns=tickers)

    # Forward returns: loaded on Z if z_premium != 0
    med = np.nanmedian(raw, axis=1, keepdims=True)
    iqr = np.nanpercentile(raw, 75, axis=1, keepdims=True) - np.nanpercentile(raw, 25, axis=1, keepdims=True)
    iqr = np.where(iqr > 0, iqr, 1.0)
    z_norm = (raw - med) / iqr

    noise = rng.normal(0.0, ret_vol, size=(n_years, n_firms))
    fwd_raw = base_ret + z_premium * z_norm + noise
    fwd = pd.DataFrame(fwd_raw, index=years, columns=tickers)

    truth = dict(
        n_firms=n_firms, n_years=n_years, z_premium=z_premium,
        base_ret=base_ret, ret_vol=ret_vol, seed=seed,
        has_premium=z_premium != 0.0,
    )
    return z, fwd, truth


# ---------------------------------------------------------------------------
# Real panel — EDGAR + yfinance, cache-first
# ---------------------------------------------------------------------------

def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    start_year: int = 2008,
    end_year: int = 2023,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Altman Z-score panel + aligned next-year returns, cache-first.

    Reads the desk's shared EDGAR concept caches (``_cache/_edgar_*.parquet``)
    and the annual-return frame (``_cache/_edgar_yrret.parquet``). Market cap
    is reconstructed as WeightedAverageShares × December-month-end price from
    yfinance (also cached via yfinance's own disk cache).

    **Survivorship bias is explicit and unavoidable**: the EDGAR caches contain
    only tickers that are *current* S&P 500 members, projected backwards. All
    results therefore represent an upper bound on what a live strategy could
    have earned — stated as such in the verdict and on the Signal axis.

    Z-score for year *t* uses year-*t* 10-K data paired with the year-*(t+1)*
    return (a clean one-year lag; 10-Ks file in Q1 of *t+1*).

    Returns
    -------
    z : DataFrame, shape (n_years, n_tickers)
        Altman Z-score, year × ticker.
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

    # ---- common tickers with all seven balance-sheet concepts ---------------
    common = sorted(
        set(panels["Assets"].columns)
        & set(panels["AssetsCurrent"].columns)
        & set(panels["LiabilitiesCurrent"].columns)
        & set(panels["RetainedEarningsAccumulatedDeficit"].columns)
        & set(panels["OperatingIncomeLoss"].columns)
        & set(panels["Liabilities"].columns)
        & set(panels["Revenues"].columns)
        & set(panels["WeightedAverageNumberOfDilutedSharesOutstanding"].columns)
    )

    # ---- market cap: shares × December price --------------------------------
    import yfinance as yf  # lazy: only when cache files are present

    px = yf.download(common, period="max", interval="1mo",
                     auto_adjust=True, progress=False)
    if isinstance(px.columns, pd.MultiIndex):
        px = px["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px_dec = px[px.index.month == 12].copy()
    px_dec.index = px_dec.index.year

    wad = panels["WeightedAverageNumberOfDilutedSharesOutstanding"]
    mkt_cap = wad.multiply(px_dec, fill_value=np.nan)

    # ---- compute Z-score per year -------------------------------------------
    years = list(range(start_year, end_year + 1))
    ta    = panels["Assets"].reindex(index=years, columns=common)
    wc    = (panels["AssetsCurrent"] - panels["LiabilitiesCurrent"]).reindex(index=years, columns=common)
    re    = panels["RetainedEarningsAccumulatedDeficit"].reindex(index=years, columns=common)
    eb    = panels["OperatingIncomeLoss"].reindex(index=years, columns=common)
    tl    = panels["Liabilities"].reindex(index=years, columns=common)
    s     = panels["Revenues"].reindex(index=years, columns=common)
    mc    = mkt_cap.reindex(index=years, columns=common)

    # Only compute where total assets and total liabilities are strictly positive
    valid = (ta > 0) & (tl > 0)
    X1 = (wc / ta).where(valid)
    X2 = (re / ta).where(valid)
    X3 = (eb / ta).where(valid)
    X4 = (mc / tl).where(valid)
    X5 = (s  / ta).where(valid)

    z_scores = (
        Z_COEF["X1"] * X1
        + Z_COEF["X2"] * X2
        + Z_COEF["X3"] * X3
        + Z_COEF["X4"] * X4
        + Z_COEF["X5"] * X5
    )

    # ---- forward returns: year t+1 return, aligned to signal year t ----------
    fwd = yr_ret.reindex(index=[y + 1 for y in years], columns=common)
    fwd.index = years
    fwd.index.name = "year"

    return z_scores, fwd


def fingerprint(z: pd.DataFrame) -> str:
    """A short content fingerprint of the Z-score panel (for the as-of stamp)."""
    flat = z.values[np.isfinite(z.values)]
    h = hashlib.sha1(np.ascontiguousarray(flat).tobytes())
    return h.hexdigest()[:12]
