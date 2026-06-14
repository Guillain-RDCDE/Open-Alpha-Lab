"""Data layer for Study 141 (Turnover-Anomaly).

Two tapes, one shape (a year × ticker panel of turnover signals + forward returns):

- ``synthetic_panel`` — a *deterministic, offline* generator. A ``premium`` knob plants
  the exact effect under test: high-turnover stocks underperform (negative premium). Setting
  ``premium = 0`` is the null: turnover quintile carries no information. This is the study's
  null in a bottle and the positive control used in the test-suite.

- ``fetch_panel`` — the real tape, assembled from two shared caches:
    1. EDGAR WeightedAverageNumberOfDilutedSharesOutstanding (annual shares, 2007-2026)
       at ``_cache/_edgar_WeightedAverageNumberOfDilutedSharesOutstanding.parquet``.
    2. Annual trading volume from yfinance (daily OHLCV → annual sum), cached per-study
       under ``_cache/turnover_annual_volume.parquet``.
  Turnover = annual_volume / shares_outstanding. The signal for year y predicts calendar-year
  y+1 returns (lagged one year; the EDGAR data are annual 10-K filings, so they need time to
  be digested by the market). Returns from the shared ``_edgar_yrret.parquet``.

**Survivorship-bias caveat**: the EDGAR caches cover the *current* S&P 500 membership
projected backwards, so every firm in the panel survived to 2026. Positive results from the
real tape are upper bounds — the true live effect is weaker. This is named on the Signal axis
and in the results.

No look-ahead: ``signal.loc[y]`` is formed from fiscal year y data; ``fwd_ret.loc[y]`` is
the return in calendar year y+1, a conservative one-year lag.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
STUDY_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

SHARES_CONCEPT = "WeightedAverageNumberOfDilutedSharesOutstanding"
VOLUME_CACHE_FILE = "turnover_annual_volume.parquet"


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 150,
    n_years: int = 15,
    premium: float = -0.04,
    base_ret: float = 0.10,
    ret_vol: float = 0.28,
    seed: int = 141,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A firm × year panel with a known turnover-anomaly effect — deterministic given ``seed``.

    Each firm-year draws a random turnover score from a log-normal distribution (mimicking
    the right-skewed empirical distribution of share turnover). The signal for year y predicts
    next-year return as::

        base_ret + premium * z(turnover_rank) + noise

    where ``z(·)`` normalises ranks to zero mean / unit variance cross-sectionally.
    ``premium < 0`` is the Datar-Naik-Radcliffe claim: high turnover predicts underperformance.
    ``premium = 0`` is the null (quintile rank carries no information).

    Returns ``(signal_df, fwd_ret_df, truth)`` where ``signal_df`` is the (year × firm)
    turnover panel (higher = more turnover) and ``fwd_ret_df`` is next-year returns.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2009, 2009 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Log-normal turnover (annual trades / shares), typical range 0.2x–5x float
    raw_turnover = rng.lognormal(mean=0.0, sigma=0.8, size=(n_years, n_firms))

    # Cross-sectional z-score of the turnover signal
    mu_t = raw_turnover.mean(axis=1, keepdims=True)
    sd_t = raw_turnover.std(axis=1, keepdims=True) + 1e-9
    z = (raw_turnover - mu_t) / sd_t

    # Forward returns: higher turnover → lower return when premium < 0
    fwd = base_ret + premium * z + ret_vol * rng.standard_normal((n_years, n_firms))

    signal = pd.DataFrame(
        raw_turnover,
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
        "base_ret": base_ret,
        "ret_vol": ret_vol,
        "seed": seed,
        "has_effect": premium != 0.0,
    }
    return signal, fwd_ret, truth


# ---------------------------------------------------------------------------
# Real panel — EDGAR shares + yfinance volume
# ---------------------------------------------------------------------------
def _load_annual_volume(
    tickers: list[str],
    start_year: int = 2007,
    end_year: int = 2025,
    fetch: bool = False,
    study_cache: str = STUDY_CACHE,
) -> pd.DataFrame:
    """Annual trading volume (shares traded) per ticker, via yfinance or study cache.

    If ``fetch=False`` and the cache file does not exist, returns an empty DataFrame
    so the test-suite and synthetic demo work offline. With ``fetch=True`` the function
    downloads daily OHLCV from yfinance and aggregates to annual volume sums; the result
    is written to ``study_cache/turnover_annual_volume.parquet``.
    """
    vol_path = os.path.join(study_cache, VOLUME_CACHE_FILE)
    if not fetch:
        if not os.path.exists(vol_path):
            return pd.DataFrame()
        return pd.read_parquet(vol_path)

    import yfinance as yf  # lazy network import

    start = f"{start_year}-01-01"
    end = f"{end_year + 1}-01-01"

    # Download in batches of 50 to avoid Yahoo rate-limits
    batch_size = 50
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        raw = yf.download(
            batch,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            vol = raw["Volume"]
        else:
            vol = raw[["Volume"]]
        frames.append(vol)

    if not frames:
        return pd.DataFrame()

    daily_vol = pd.concat(frames, axis=1)
    daily_vol.index = pd.DatetimeIndex(daily_vol.index)
    # Resample to annual (calendar-year end), summing trading volume
    annual_vol = daily_vol.resample("YE").sum()
    annual_vol.index = annual_vol.index.year
    annual_vol.index.name = "year"

    os.makedirs(study_cache, exist_ok=True)
    annual_vol.to_parquet(vol_path)
    return annual_vol


def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    study_cache: str = STUDY_CACHE,
    fetch: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turnover signal + aligned next-year returns from the shared EDGAR + yfinance pulls.

    Reads the EDGAR shares-outstanding cache and the study-local annual-volume cache.
    Returns ``(signal, fwd_ret)`` where ``signal.loc[y]`` is the annual turnover
    (volume / shares) for fiscal year y (higher = more turnover, potentially worse alpha),
    and ``fwd_ret.loc[y]`` is calendar-year y+1 returns.

    If any required cache file is absent the function returns two empty DataFrames so the
    test-suite and the synthetic demo work offline.

    **Survivorship-bias warning**: the EDGAR panel covers only current S&P 500 members
    projected backwards. All results should be treated as upper bounds on any real effect.
    """
    shares_path = os.path.join(cache_dir, f"_edgar_{SHARES_CONCEPT}.parquet")
    yrret_path = os.path.join(cache_dir, "_edgar_yrret.parquet")

    if not os.path.exists(shares_path) or not os.path.exists(yrret_path):
        return pd.DataFrame(), pd.DataFrame()

    shares_raw = pd.read_parquet(shares_path)  # (year × ticker), annual shares
    yr_ret = pd.read_parquet(yrret_path)         # (year × ticker), annual returns

    # Load (or fetch) annual trading volume
    annual_vol = _load_annual_volume(
        tickers=list(shares_raw.columns),
        fetch=fetch,
        study_cache=study_cache,
    )
    if annual_vol.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Align on common tickers and years
    common_tickers = sorted(
        set(shares_raw.columns) & set(annual_vol.columns) & set(yr_ret.columns)
    )
    if not common_tickers:
        return pd.DataFrame(), pd.DataFrame()

    shares = shares_raw[common_tickers]
    vol = annual_vol[common_tickers]

    # Common years: shares are available 2007-2026; volume from yfinance 2007-2025
    common_years = sorted(set(shares.index) & set(vol.index))
    shares = shares.loc[common_years]
    vol = vol.loc[common_years]

    # Turnover = annual_volume / shares_outstanding
    # Both are in the same units (shares), so ratio is dimensionless.
    # Sanity filter: cap at 50x — values above this almost always reflect a
    # corporate-action unit error in EDGAR (e.g. MRK 2010 stored 3,120 shares
    # instead of 3,120,000,000) or a pre-split tiny-float artefact, not real
    # hyper-liquidity. The cap removes <2% of valid firm-years and does not
    # censor the highest-turnover quintile (Q5 median ~8x in sample).
    turnover = vol.divide(shares).replace([np.inf, -np.inf], np.nan)
    turnover = turnover.where((turnover >= 0.1) & (turnover <= 50))

    # Align: signal from year y → returns in year y+1
    fwd = yr_ret.reindex(index=[y + 1 for y in turnover.index])
    fwd.index = turnover.index  # re-label to signal year for easy join
    fwd.index.name = "year"
    fwd = fwd[common_tickers]

    return turnover, fwd


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint for any DataFrame, for the as-of stamp."""
    arr = df.to_numpy(dtype=float, na_value=0.0)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
