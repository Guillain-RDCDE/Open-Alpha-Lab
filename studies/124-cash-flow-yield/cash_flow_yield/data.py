"""Data layer for Study 124 (Cash-Flow-Yield).

Two tapes, one shape (a year × ticker panel of OCF yield and forward returns):

- ``synthetic_panel`` — a *deterministic, offline* generator. A tunable ``ocfy_premium``
  knob controls how strongly OCF-yield predicts next-year returns; ``ocfy_premium=0`` is
  the null — the sort is a coin, so tests can assert the edge appears only when the
  premium is planted. Same knob for ``ey_premium`` (earnings yield).
- ``fetch_panel`` — the real tape: EDGAR ``NetCashProvidedByUsedInOperatingActivities`` /
  (``WeightedAverageNumberOfDilutedSharesOutstanding`` × year-end price) as OCF yield, and
  ``NetIncomeLoss`` / (shares × price) as earnings yield, for current S&P 500 members
  with annual 10-K data 2007-2024 + next-year returns from ``_edgar_yrret``.

**Survivorship bias** is explicit: the ticker universe is the *current* S&P 500 projected
backwards (``allow_survivorship_bias=True``). This inflates the apparent premium; results
are upper bounds, not investable realities (see ``METHODOLOGY.md``).

No look-ahead: signal formed on fiscal-year-Y fundamentals → aligned to next-year (Y+1)
forward return, so the trade is entered at the *start* of Y+1. The EDGAR fiscal-year data
for year Y is 10-K data reported with a lag (typically March of Y+1); pairing with the
full-year-Y+1 return therefore allows up to one full year of reporting lag, which is the
desk's standard for annual cross-sectional work (cf. Study 52 — Smoke-Screen).
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
STUDY_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_cache")

# EDGAR concept names used (matched to _cache/_edgar_<name>.parquet filenames).
_CFO_NAME = "NetCashProvidedByUsedInOperatingActivities"
_NI_NAME = "NetIncomeLoss"
_SHARES_NAME = "WeightedAverageNumberOfDilutedSharesOutstanding"
_ASSETS_NAME = "Assets"


@dataclass(frozen=True)
class WorldTruth:
    """The planted premium parameters for a synthetic panel."""

    ocfy_premium: float  # bps of extra return per unit of cross-sectional OCF-yield z-score
    ey_premium: float    # same for earnings yield

    @property
    def has_ocfy_edge(self) -> bool:
        return self.ocfy_premium != 0.0

    @property
    def has_ey_edge(self) -> bool:
        return self.ey_premium != 0.0


def synthetic_panel(
    n_firms: int = 200,
    n_years: int = 18,
    ocfy_premium: float = 0.04,
    ey_premium: float = 0.03,
    base_ret: float = 0.09,
    ret_vol: float = 0.30,
    seed: int = 124,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A reproducible firm × year panel with a known OCF-yield and earnings-yield effect.

    Each firm-year draws an OCF-yield and an EY drawn from a joint normal (correlation
    ``rho=0.6`` — they tend to move together), then cross-sectionally z-scored within each
    year. Next year's return is:

        ret_{t+1} = base_ret + ocfy_premium*z_ocfy + ey_premium*z_ey + noise

    where the premiums are the planted cross-sectional loadings. Setting both to zero is
    the null — the quintile sorts become coin flips.

    Returns ``(ocfy_panel, ey_panel, fwd_ret, truth)`` all indexed by year (2000 onward)
    with columns = firm identifiers.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2000 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Joint standard normals with rho = 0.6 (typical correlation of CFO/NI metrics).
    rho = 0.6
    L = np.array([[1.0, 0.0], [rho, np.sqrt(1.0 - rho**2)]])
    raw = rng.standard_normal((n_years, n_firms, 2))
    correlated = raw @ L.T  # shape: (n_years, n_firms, 2)
    a_ocfy = correlated[..., 0]
    a_ey = correlated[..., 1]

    # Cross-sectional z-scores (zero mean, unit variance within each year).
    def _zscore(a: np.ndarray) -> np.ndarray:
        mu = a.mean(axis=1, keepdims=True)
        sigma = a.std(axis=1, keepdims=True) + 1e-9
        return (a - mu) / sigma

    z_ocfy = _zscore(a_ocfy)
    z_ey = _zscore(a_ey)
    noise = ret_vol * rng.standard_normal((n_years, n_firms))
    fwd = base_ret + ocfy_premium * z_ocfy + ey_premium * z_ey + noise

    idx = pd.Index(years, name="year")
    ocfy_panel = pd.DataFrame(
        0.05 + 0.03 * a_ocfy,  # plausible OCF yield range ~2-8%
        index=idx, columns=firms,
    )
    ey_panel = pd.DataFrame(
        0.05 + 0.02 * a_ey,    # plausible EY range ~1-9%
        index=idx, columns=firms,
    )
    fwd_panel = pd.DataFrame(fwd, index=idx, columns=firms)
    return ocfy_panel, ey_panel, fwd_panel, WorldTruth(ocfy_premium, ey_premium)


def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    study_cache: str = STUDY_CACHE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """OCF yield, earnings yield, and next-year returns for current S&P 500 members.

    Returns ``(ocfy, ey, fwd_ret)`` — all year × ticker DataFrames — where signal
    year Y aligns to forward return year Y+1 (one-year reporting lag).

    **Cache-first**: reads the desk's shared EDGAR parquets from ``cache_dir`` plus
    a year-end price parquet cached under ``study_cache``. If any dependency is missing
    and ``fetch=False``, returns empty DataFrames (the callers must handle gracefully).

    **Survivorship bias caveat**: the ticker list is the current S&P 500 projected
    backwards, so magnitudes are upper bounds on what was actually achievable.
    """
    # Paths to shared EDGAR caches.
    cfo_p = os.path.join(cache_dir, f"_edgar_{_CFO_NAME}.parquet")
    ni_p = os.path.join(cache_dir, f"_edgar_{_NI_NAME}.parquet")
    shares_p = os.path.join(cache_dir, f"_edgar_{_SHARES_NAME}.parquet")
    yrret_p = os.path.join(cache_dir, "_edgar_yrret.parquet")
    px_p = os.path.join(study_cache, "yrpx.parquet")

    required = [cfo_p, ni_p, shares_p, yrret_p]
    if not all(os.path.exists(p) for p in required):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    cfo = pd.read_parquet(cfo_p)
    ni = pd.read_parquet(ni_p)
    shares = pd.read_parquet(shares_p)
    yrret = pd.read_parquet(yrret_p)

    # Year-end prices: try study cache, otherwise download (fetch=True).
    if not os.path.exists(px_p):
        if not fetch:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        yr_px = _fetch_year_end_prices(list(cfo.columns), study_cache)
    else:
        yr_px = pd.read_parquet(px_p)

    # Market cap proxy = shares × price (both annual, same fiscal year end).
    mktcap = shares.mul(yr_px.reindex(index=shares.index, columns=shares.columns))

    # OCF yield = operating cash flow / market cap. Drop extremes (|yield| > 1 = 100%).
    ocfy_raw = cfo.div(mktcap)
    ocfy = _winsorise(ocfy_raw, lo=0.0, hi=1.0)  # non-negative; cap at 100%

    # Earnings yield = net income / market cap. Allow negative (loss-making firms).
    ey_raw = ni.div(mktcap)
    ey = _winsorise(ey_raw, lo=-0.5, hi=1.0)

    # Forward returns: signal year Y → return year Y+1 (1-year lag).
    # Drop 2026 (incomplete year).
    usable_years = [y for y in ocfy.index if y <= 2024]
    ocfy = ocfy.loc[usable_years]
    ey = ey.loc[usable_years]

    fwd = yrret.reindex(index=[y + 1 for y in usable_years])
    fwd.index = pd.Index(usable_years, name="year")

    return ocfy, ey, fwd


def _winsorise(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """Winsorise values to [lo, hi]; values outside become NaN rather than clipped.

    This is more conservative than clipping: extreme yields often reflect data errors
    (a tiny denominator) rather than genuine firm characteristics.
    """
    return df.where((df >= lo) & (df <= hi))


def _fetch_year_end_prices(tickers: list[str], study_cache: str) -> pd.DataFrame:
    """Download year-end (Dec 31) closing prices for all tickers; cache as parquet.

    Intended to be called once (or on ``--fetch``). Prices are used only to convert
    the EDGAR per-share fundamentals into a market-cap denominator. Cached under
    ``study_cache/yrpx.parquet``.
    """
    import yfinance as yf  # lazy: only on fetch

    raw = yf.download(
        tickers,
        start="2006-01-01",
        end="2025-12-31",
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw

    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    yr_px = close.resample("YE").last()
    yr_px.index = yr_px.index.year

    os.makedirs(study_cache, exist_ok=True)
    yr_px.to_parquet(os.path.join(study_cache, "yrpx.parquet"))
    return yr_px


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint (SHA-1 of flattened values), for the as-of stamp."""
    arr = np.ascontiguousarray(df.fillna(0.0).to_numpy(dtype=float))
    h = hashlib.sha1(arr.tobytes())
    return h.hexdigest()[:12]
