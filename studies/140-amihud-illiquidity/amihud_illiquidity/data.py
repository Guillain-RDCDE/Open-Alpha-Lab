"""Data layer for Study 140 (Amihud-Illiquidity).

Two tapes, one shape (a year x ticker panel of ILLIQ + forward returns):

- ``synthetic_panel`` — a *deterministic, offline* generator. A ``premium`` knob plants
  the exact effect Amihud (2002) claims: illiquid stocks outperform liquid ones by a known
  margin. ``premium = 0`` is the null: ILLIQ carries no information and the most-illiquid
  quintile should not beat the most-liquid one. This is the study's null in a bottle.

- ``fetch_panel`` — the real tape: daily OHLCV from Yahoo Finance (via yfinance) for the
  S&P 500 names in the shared EDGAR universe. ILLIQ for year y = mean( |ret_d| / DolVol_d )
  over all trading days d in year y. Forward return = calendar-year y+1 total return.

**Survivorship-bias caveat**: the universe is the *current* S&P 500 membership projected
backwards (via the shared EDGAR cache ticker set). This is a well-known large-cap panel;
the Amihud premium famously lives in small- and micro-cap stocks — on this universe the
premium is expected to be absent or muted, and any positive result is an upper bound.
This is named on the Signal axis.

No look-ahead: ILLIQ for year y sorts quintiles; the forward return is year y+1. The sort
is formed before the return window begins — no information from y+1 enters the sort.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
STUDY_CACHE = os.path.join(_HERE, "..", "_cache")

# Shared EDGAR cache — used only to establish the stock universe
_EDGAR_ASSETS = "_edgar_Assets.parquet"

# Study-local yfinance cache filenames
_PANEL_ILLIQ = "amihud_illiq_panel.parquet"
_PANEL_FWDRET = "amihud_fwdret_panel.parquet"


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 150,
    n_years: int = 15,
    premium: float = 0.04,   # annualised alpha per unit of z-scored log-ILLIQ
    base_ret: float = 0.10,
    ret_vol: float = 0.28,
    illiq_vol: float = 1.0,  # cross-sectional dispersion of log-ILLIQ
    seed: int = 140,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A firm x year panel with a known Amihud ILLIQ premium — deterministic given ``seed``.

    Each firm-year draws a log-ILLIQ from a normal distribution, then the next-year return
    is:

        base_ret + premium * z(log_ILLIQ) + noise

    where z(.) normalises log-ILLIQ cross-sectionally to zero mean / unit variance.
    ``premium = 0`` is the null (the illiquid quintile should beat the liquid one by zero).

    Returns ``(illiq_df, fwd_ret_df, truth)`` where ``illiq_df`` carries the raw ILLIQ
    measure (year x firm, higher = more illiquid) and ``fwd_ret_df`` is the aligned
    forward return panel.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2006, 2006 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Log-ILLIQ: cross-sectional variation + year-specific level shift
    log_illiq = rng.normal(0.0, illiq_vol, (n_years, n_firms))

    # Cross-sectional z-score
    mu_i = log_illiq.mean(axis=1, keepdims=True)
    sd_i = log_illiq.std(axis=1, keepdims=True) + 1e-9
    z = (log_illiq - mu_i) / sd_i

    # Forward returns: higher ILLIQ -> higher return when premium > 0
    fwd = base_ret + premium * z + ret_vol * rng.standard_normal((n_years, n_firms))

    illiq_df = pd.DataFrame(
        np.exp(log_illiq),   # raw ILLIQ (positive, right-skewed like the real thing)
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    fwd_ret_df = pd.DataFrame(
        fwd,
        index=pd.Index(years, name="year"),
        columns=firms,
    )
    truth = {
        "premium": premium,
        "n_firms": n_firms,
        "n_years": n_years,
        "seed": seed,
        "has_premium": premium != 0.0,
    }
    return illiq_df, fwd_ret_df, truth


# ---------------------------------------------------------------------------
# Real panel — Yahoo Finance daily OHLCV, cache-first
# ---------------------------------------------------------------------------
def fetch_panel(
    start_year: int = 2006,
    end_year: int = 2023,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    study_cache_dir: str | None = None,
    min_days: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Amihud ILLIQ panel + aligned next-year returns from Yahoo Finance.

    For each ticker in the shared EDGAR universe, downloads (or reads from cache) daily
    close and volume data. Computes for each year y:

        ILLIQ_y = mean_d( |r_d| / DolVol_d )

    where r_d = log daily return and DolVol_d = Close_d * Volume_d (dollar volume).
    Stocks with fewer than ``min_days`` trading days in a year are excluded for that year.

    Returns ``(illiq, fwd_ret)`` where:
    - ``illiq.loc[y]``     = ILLIQ for sort year y (higher = more illiquid)
    - ``fwd_ret.loc[y]``   = calendar-year y+1 total return (the forward window)

    Both DataFrames share the same (year x ticker) shape.

    **Survivorship-bias warning**: the ticker universe derives from the EDGAR cache
    (current S&P 500 members projected backwards). All results are upper bounds —
    the true live premium on large-caps is likely weaker or absent.

    If no cached panel exists and ``fetch=False``, returns two empty DataFrames (so
    the test-suite and synthetic demo work offline).
    """
    if study_cache_dir is None:
        study_cache_dir = STUDY_CACHE
    os.makedirs(study_cache_dir, exist_ok=True)

    illiq_path = os.path.join(study_cache_dir, _PANEL_ILLIQ)
    fwd_path = os.path.join(study_cache_dir, _PANEL_FWDRET)

    if not fetch:
        if os.path.exists(illiq_path) and os.path.exists(fwd_path):
            illiq = pd.read_parquet(illiq_path)
            fwd_ret = pd.read_parquet(fwd_path)
            return illiq, fwd_ret
        return pd.DataFrame(), pd.DataFrame()

    # --- build from raw Yahoo data ---
    edgar_path = os.path.join(cache_dir, _EDGAR_ASSETS)
    if not os.path.exists(edgar_path):
        raise FileNotFoundError(
            f"Shared EDGAR cache not found at {edgar_path}. "
            "Run from the repo root with the shared _cache/ intact."
        )
    assets_df = pd.read_parquet(edgar_path)
    tickers = sorted(assets_df.columns.tolist())

    import yfinance as yf  # lazy: only when fetching

    # Download all tickers at once for speed (daily, auto-adjusted)
    print(f"Fetching {len(tickers)} tickers from Yahoo Finance (daily, {start_year}-{end_year+1})...")
    raw = yf.download(
        tickers,
        start=f"{start_year}-01-01",
        end=f"{end_year + 2}-01-01",
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    # Handle MultiIndex columns: (field, ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
        volume = raw["Volume"].copy()
    else:
        raise RuntimeError("Expected MultiIndex columns from yfinance batch download.")

    close.index = pd.to_datetime(close.index)
    volume.index = pd.to_datetime(volume.index)

    # Compute per-year ILLIQ and calendar-year returns
    illiq_rows: dict[int, dict] = {}
    ret_rows: dict[int, dict] = {}

    years = range(start_year, end_year + 1)
    for ticker in tickers:
        if ticker not in close.columns:
            continue
        c = close[ticker].dropna()
        v = volume[ticker].dropna()
        if c.empty or v.empty:
            continue
        c, v = c.align(v, join="inner")

        for y in years:
            # ILLIQ sort year: compute over calendar year y
            c_y = c[c.index.year == y]
            v_y = v[v.index.year == y]
            if len(c_y) < min_days:
                continue
            log_ret = np.log(c_y / c_y.shift(1)).dropna()
            dv = (c_y * v_y).reindex(log_ret.index).dropna()
            if dv.empty or (dv <= 0).all():
                continue
            lr, dv = log_ret.align(dv, join="inner")
            if len(lr) < min_days - 1:
                continue
            illiq_val = (np.abs(lr) / dv).mean()
            if not np.isfinite(illiq_val) or illiq_val <= 0:
                continue
            if y not in illiq_rows:
                illiq_rows[y] = {}
            illiq_rows[y][ticker] = float(illiq_val)

            # Forward return: calendar year y+1
            c_nxt = c[c.index.year == (y + 1)]
            if len(c_nxt) < 50:
                continue
            fwd_ret_val = float(c_nxt.iloc[-1] / c_nxt.iloc[0] - 1.0)
            if not np.isfinite(fwd_ret_val):
                continue
            if y not in ret_rows:
                ret_rows[y] = {}
            ret_rows[y][ticker] = fwd_ret_val

    illiq = pd.DataFrame(illiq_rows).T.sort_index()
    illiq.index.name = "year"
    fwd_ret = pd.DataFrame(ret_rows).T.sort_index()
    fwd_ret.index.name = "year"

    # Align: keep only years where both illiq and fwd_ret are present
    common_years = sorted(set(illiq.index) & set(fwd_ret.index))
    illiq = illiq.reindex(common_years)
    fwd_ret = fwd_ret.reindex(common_years)

    illiq.to_parquet(illiq_path)
    fwd_ret.to_parquet(fwd_path)
    print(f"Cached panel: {len(common_years)} years, up to {len(tickers)} tickers.")
    return illiq, fwd_ret


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint for any DataFrame, for the as-of stamp."""
    arr = df.to_numpy(dtype=float, na_value=0.0)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
