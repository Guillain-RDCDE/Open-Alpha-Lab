"""Data layer for Study 243 (Graham NCAV).

Two tapes, one contract (a year x ticker DataFrame of NCAV ratios paired with a
next-year-return frame):

- ``synthetic_panel`` — a **deterministic, offline** generator. A tunable
  ``ncav_premium`` knob plants a known relationship between the NCAV ratio and
  forward return; ``ncav_premium = 0`` is the null (NCAV tells you nothing).

- ``fetch_panel`` — the real panel built from the desk's shared EDGAR caches
  (``_cache/_edgar_AssetsCurrent.parquet``, ``_cache/_edgar_Liabilities.parquet``,
  ``_cache/_edgar_WeightedAverageDilutedShares.parquet``) and yfinance annual prices.
  **Cache-only by default.** Survivorship-bias is explicit: the panel covers
  *current* S&P 500 members projected backwards.

No look-ahead: the NCAV ratio for year *t* uses year-*t* balance-sheet data
(from the fiscal-year 10-K) and the market cap at December 31 of year *t*.
Because 10-Ks file in Q1 of *t+1*, the signal is paired with the calendar
year-*t+1* return — a clean one-year lag with no look-ahead.

KEY FINDING: In the S&P 500 large-cap universe, genuine Graham net-nets are
extremely rare (0-7 per year, zero in 2020-2023). The handful that appear are
asset-light tech/semiconductor firms (NVDA, AAPL, LRCX, MA) where high intangible
value means current assets < market cap on a per-share basis but the NCAV test is
satisfied — the *opposite* of what Graham intended. Results are dominated by
survivorship bias and a tiny, unrepresentative sample.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")

# EDGAR concepts needed for NCAV
_CONCEPTS = [
    "AssetsCurrent",
    "Liabilities",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
]

# Graham's threshold: price < 2/3 * NCAV/share, i.e. NCAV/MktCap > 3/2
# A looser screen: NCAV > 0 and NCAV/MktCap > 1 (price below NCAV/share)
GRAHAM_THRESHOLD = 1.0   # strict: NCAV/MktCap >= 1 (price <= NCAV/share)
GRAHAM_TWO_THIRDS = 2.0 / 3.0  # Graham's stated rule: price < 2/3 NCAV/share
                                # i.e. NCAV/MktCap >= 3/2


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------

def synthetic_panel(
    n_firms: int = 200,
    n_years: int = 16,
    ncav_premium: float = 0.05,
    base_ret: float = 0.10,
    ret_vol: float = 0.30,
    seed: int = 243,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A synthetic firm panel with a known NCAV ratio -> return relationship.

    Each firm-year draws an NCAV-to-market-cap ratio from a realistic
    distribution (mostly negative for large caps, a few positive). The
    next-year return is:

        ret = base_ret + ncav_premium * (ncav_ratio - median) / IQR + noise

    so high-NCAV-ratio firms (cheap relative to current assets) outperform
    when ``ncav_premium > 0``. ``ncav_premium = 0`` is the null.

    Returns ``(ncav_ratios, fwd_returns, truth)`` where truth is a dict of
    the planted parameters.
    """
    rng = np.random.default_rng(seed)
    years = pd.Index(range(2008, 2008 + n_years), name="year")
    tickers = [f"F{j:03d}" for j in range(n_firms)]

    # NCAV ratios: realistic large-cap distribution (mostly -1 to +1)
    raw = rng.normal(-0.3, 0.6, size=(n_years, n_firms))
    raw = np.clip(raw, -3.0, 3.0)
    ncav = pd.DataFrame(raw, index=years, columns=tickers)

    # Forward returns: loaded on ncav_ratio if ncav_premium != 0
    med = np.nanmedian(raw, axis=1, keepdims=True)
    iqr = (np.nanpercentile(raw, 75, axis=1, keepdims=True)
           - np.nanpercentile(raw, 25, axis=1, keepdims=True))
    iqr = np.where(iqr > 0, iqr, 1.0)
    ncav_norm = (raw - med) / iqr

    noise = rng.normal(0.0, ret_vol, size=(n_years, n_firms))
    fwd_raw = base_ret + ncav_premium * ncav_norm + noise
    fwd = pd.DataFrame(fwd_raw, index=years, columns=tickers)

    truth = dict(
        n_firms=n_firms, n_years=n_years, ncav_premium=ncav_premium,
        base_ret=base_ret, ret_vol=ret_vol, seed=seed,
        has_premium=ncav_premium != 0.0,
    )
    return ncav, fwd, truth


# ---------------------------------------------------------------------------
# Real panel — EDGAR + yfinance, cache-first
# ---------------------------------------------------------------------------

def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    start_year: int = 2008,
    end_year: int = 2023,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """NCAV ratio panel + aligned next-year returns, cache-first.

    Reads the desk's shared EDGAR concept caches and the annual-return frame.
    Market cap is reconstructed as WeightedAverageShares x December price.

    NCAV ratio = (AssetsCurrent - TotalLiabilities) / MarketCap

    A ratio > 1 means price is below NCAV/share (pure Graham net-net).
    A ratio > 3/2 means price is below 2/3 * NCAV/share (Graham's rule).

    **Survivorship bias is explicit and unavoidable**: the EDGAR caches contain
    only tickers that are *current* S&P 500 members, projected backwards.

    Returns
    -------
    ncav_ratio : DataFrame, shape (n_years, n_tickers)
        NCAV-to-market-cap ratio, year x ticker.
    fwd_ret : DataFrame, same shape
        Calendar-year *(t+1)* return, aligned to year-*t* signal index.
    """
    paths = {c: os.path.join(cache_dir, f"_edgar_{c}.parquet") for c in _CONCEPTS}
    r_path = os.path.join(cache_dir, "_edgar_yrret.parquet")

    missing = [c for c, p in paths.items() if not os.path.exists(p)]
    if missing or not os.path.exists(r_path):
        return pd.DataFrame(), pd.DataFrame()

    ac = pd.read_parquet(paths["AssetsCurrent"])
    tl = pd.read_parquet(paths["Liabilities"])
    wad = pd.read_parquet(paths["WeightedAverageNumberOfDilutedSharesOutstanding"])
    yr_ret = pd.read_parquet(r_path)

    # Common tickers with all three concepts and returns
    common = sorted(
        set(ac.columns)
        & set(tl.columns)
        & set(wad.columns)
        & set(yr_ret.columns)
    )

    import yfinance as yf  # lazy: only when cache files are present

    px = yf.download(common, period="max", interval="1mo",
                     auto_adjust=True, progress=False)
    if isinstance(px.columns, pd.MultiIndex):
        px = px["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px_dec = px[px.index.month == 12].copy()
    px_dec.index = px_dec.index.year

    years = list(range(start_year, end_year + 1))

    ac_r  = ac.reindex(index=years, columns=common)
    tl_r  = tl.reindex(index=years, columns=common)
    wad_r = wad.reindex(index=years, columns=common)
    px_r  = px_dec.reindex(index=years, columns=common)

    # Market cap: shares x December price
    mkt_cap = wad_r.multiply(px_r, fill_value=np.nan)

    # NCAV = CurrentAssets - TotalLiabilities
    ncav_total = ac_r - tl_r

    # NCAV ratio = NCAV / MktCap (> 1 means net-net)
    valid = mkt_cap > 0
    ncav_ratio = (ncav_total / mkt_cap).where(valid)

    # Forward returns: year t+1 return, aligned to signal year t
    fwd = yr_ret.reindex(index=[y + 1 for y in years], columns=common)
    fwd.index = years
    fwd.index.name = "year"

    return ncav_ratio, fwd


def fingerprint(ncav: pd.DataFrame) -> str:
    """A short content fingerprint of the NCAV ratio panel."""
    flat = ncav.values[np.isfinite(ncav.values)]
    h = hashlib.sha1(np.ascontiguousarray(flat).tobytes())
    return h.hexdigest()[:12]
