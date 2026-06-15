"""Data layer for Study 167 (Hindenburg-Omen).

Two tapes, both daily:

- ``synthetic_breadth`` — a *deterministic, offline* generator.  Builds a fake panel of
  ``n_stocks`` assets with i.i.d. log-normal daily returns, so the fraction of 52-week
  highs and lows each day is purely noise.  An optional ``crash_signal_bps`` knob plants a
  negative drift on *Hindenburg days* so we can confirm the engine recovers signal when one
  is present — the null in a bottle.
- ``fetch_breadth`` / ``load_breadth`` — the real loader.  It reads the bulk-downloaded S&P
  500 constituent daily OHLC parquet (cached under the study's ``_cache/``), computes daily
  52-week-high and 52-week-low counts, derives the McClellan-oscillator sign, and assembles
  the Hindenburg signal.

The Hindenburg Omen fires when, on the same trading day, ALL of the following hold:

1. new 52-week highs  ≥  THRESHOLD * total issues
2. new 52-week lows   ≥  THRESHOLD * total issues
3. The daily NYSE/SPX 50-day moving average is *rising* (close > MA50)
4. The McClellan Oscillator is *negative* (advances < declines)
5. At least 252 trading days of price history exist (we need a full year)

The canonical threshold is 2.2 % (Miekka, 1995).  We also test 2.0 % and 2.5 % for
sensitivity.

Conditions 3–4 are approximated on the S&P 500 universe directly (the index serves for
the 50-day MA; advances/declines come from our constituent panel), which gives us a
self-contained breadth calculation without needing NYSE tape data.

No look-ahead: the signal is formed on the closes of day *t*; forward returns are measured
starting at the open of *t+1*.

NOTE ON SURVIVORSHIP BIAS: the constituent panel uses ``quantlab.universe.sp500_symbols``
with ``allow_survivorship_bias=True``, meaning it reflects *current* membership.  This
biases the panel toward survivors and likely *understates* the true fraction of 52-week
lows during crash episodes (removed losers would boost low-counts precisely during bad
markets).  This bias, which would push *against* the signal, is named on the Signal axis.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

# ---------------------------------------------------------------------------
# Omen parameters (canonical Miekka 1995 and sensitivity variants)
# ---------------------------------------------------------------------------
THRESHOLD_PCT = 2.2          # canonical: ≥ 2.2% of issues
THRESHOLD_VARIANTS = (2.0, 2.2, 2.5)
LOOKBACK_52W = 252           # trading days in one year
MA50_WINDOW = 50             # 50-day MA for the trend filter


# ---------------------------------------------------------------------------
# Synthetic tape — deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_breadth(
    n_stocks: int = 200,
    n_days: int = 504,          # ~2 trading years
    daily_vol: float = 0.015,
    drift_bps: float = 3.0,     # baseline daily drift (bps)
    crash_signal_bps: float = 0.0,   # extra negative drift on Hindenburg days (NULL = 0)
    threshold_pct: float = THRESHOLD_PCT,
    seed: int = 167,
) -> tuple[pd.DataFrame, dict]:
    """Deterministic breadth panel with a tunable Hindenburg-day penalty.

    Builds ``n_stocks`` price paths with i.i.d. log-normal daily returns.  After
    computing the Hindenburg signal on this panel we plant an additional
    ``crash_signal_bps`` of *negative* drift in the 30-day window following each
    Hindenburg day (a positive value means a forward crash).

    - ``crash_signal_bps = 0``   → the null; signal days have no predictive power.
    - ``crash_signal_bps > 0``   → a planted effect the engine can harvest.

    Returns ``(breadth, truth)`` where ``breadth`` is the same shape as the real
    loader output (columns: ``n_highs``, ``n_lows``, ``n_stocks``, ``frac_highs``,
    ``frac_lows``, ``ma50_rising``, ``mcclellan_neg``, ``hindenburg``,
    ``spy_ret``).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n_days, name="date")

    # --- price paths ---------------------------------------------------------
    daily_mu = drift_bps * 1e-4
    log_ret = daily_mu - 0.5 * daily_vol**2 + daily_vol * rng.standard_normal((n_days, n_stocks))
    prices = 100.0 * np.exp(np.cumsum(log_ret, axis=0))

    # --- 52-week highs / lows ------------------------------------------------
    n_highs_arr = np.zeros(n_days, dtype=int)
    n_lows_arr = np.zeros(n_days, dtype=int)
    for t in range(LOOKBACK_52W, n_days):
        window = prices[t - LOOKBACK_52W : t, :]     # 252 days up to (not including) t
        px_today = prices[t, :]
        n_highs_arr[t] = int((px_today >= window.max(axis=0)).sum())
        n_lows_arr[t] = int((px_today <= window.min(axis=0)).sum())

    # --- Index proxy (equal-weight) for MA50 and McClellan sign --------------
    idx_close = prices.mean(axis=1)
    ma50 = pd.Series(idx_close).rolling(MA50_WINDOW, min_periods=MA50_WINDOW).mean().to_numpy()
    ma50_rising = (idx_close > ma50).astype(int)

    # Advance/decline: stocks up today vs yesterday
    ret_today = np.diff(prices, axis=0, prepend=prices[:1, :])
    advances = (ret_today > 0).sum(axis=1)
    declines = (ret_today <= 0).sum(axis=1)
    mcclellan_neg = (advances < declines).astype(int)

    # --- Hindenburg signal ---------------------------------------------------
    th = threshold_pct / 100.0
    enough_history = np.arange(n_days) >= LOOKBACK_52W
    frac_highs = n_highs_arr / n_stocks
    frac_lows = n_lows_arr / n_stocks
    hindenburg = (
        (frac_highs >= th)
        & (frac_lows >= th)
        & (ma50_rising == 1)
        & (mcclellan_neg == 1)
        & enough_history
    ).astype(int)

    # --- index return (equal-weight) -----------------------------------------
    spy_ret = np.zeros(n_days)
    spy_ret[1:] = idx_close[1:] / idx_close[:-1] - 1.0

    # --- optional planted crash effect ----------------------------------------
    if crash_signal_bps != 0:
        penalty = -crash_signal_bps * 1e-4
        crash_days = np.where(hindenburg)[0]
        # Apply the penalty on the 1–30 trading days *after* each signal day
        for cd in crash_days:
            for fwd in range(1, 31):
                if cd + fwd < n_days:
                    spy_ret[cd + fwd] += penalty

    df = pd.DataFrame(
        {
            "n_highs": n_highs_arr,
            "n_lows": n_lows_arr,
            "n_stocks": n_stocks,
            "frac_highs": frac_highs,
            "frac_lows": frac_lows,
            "ma50_rising": ma50_rising,
            "mcclellan_neg": mcclellan_neg,
            "hindenburg": hindenburg,
            "spy_ret": spy_ret,
        },
        index=idx,
    )

    truth = {
        "n_stocks": n_stocks,
        "n_days": n_days,
        "daily_vol": daily_vol,
        "drift_bps": drift_bps,
        "crash_signal_bps": crash_signal_bps,
        "threshold_pct": threshold_pct,
        "n_signals": int(hindenburg.sum()),
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — cache-only by default
# ---------------------------------------------------------------------------
def _panel_cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "hindenburg_sp500_daily.parquet")


def _breadth_cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "hindenburg_breadth.parquet")


def _spy_cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "hindenburg_spy_daily.parquet")


def fetch_constituent_panel(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "2005-01-01",
) -> pd.DataFrame:
    """Download daily close prices for S&P 500 constituents; cache as parquet.

    Uses ``quantlab.universe.sp500_symbols`` with ``allow_survivorship_bias=True``
    (current membership — survivors only).  Network touched only with ``fetch=True``.
    Returns a wide DataFrame: index=date, columns=ticker.
    """
    path = _panel_cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached constituent panel at {path}. "
                "Call fetch_constituent_panel(fetch=True) once to populate."
            )
        return pd.read_parquet(path)

    import yfinance as yf  # lazy

    sys_path = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from quantlab.universe import sp500_symbols

    tickers = sp500_symbols(allow_survivorship_bias=True)
    # yfinance bulk download
    raw = yf.download(
        tickers, start=start, interval="1d", auto_adjust=True, progress=False
    )
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]].rename(columns={"Close": tickers[0]})

    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None)
    closes.index.name = "date"
    # drop days with < 50 tickers available (early data)
    closes = closes.loc[closes.notna().sum(axis=1) >= 50]
    os.makedirs(cache_dir, exist_ok=True)
    closes.to_parquet(path)
    return closes


def fetch_spy(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "2005-01-01",
) -> pd.DataFrame:
    """SPY daily OHLCV, cache-first.  Used for MA50 and forward-return measurements."""
    path = _spy_cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached SPY tape at {path}. "
                "Call fetch_spy(fetch=True) once."
            )
        return pd.read_parquet(path)

    import yfinance as yf  # lazy

    raw = yf.download("SPY", start=start, interval="1d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    raw.index.name = "date"
    raw = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    raw = raw[raw.index < pd.Timestamp.now().normalize()]
    os.makedirs(cache_dir, exist_ok=True)
    raw.to_parquet(path)
    return raw


def compute_breadth(
    panel: pd.DataFrame,
    spy: pd.DataFrame,
    threshold_pct: float = THRESHOLD_PCT,
    cache_dir: str = DEFAULT_CACHE,
    save: bool = True,
) -> pd.DataFrame:
    """Compute daily breadth statistics and the Hindenburg signal from the raw panel.

    Parameters
    ----------
    panel : wide DataFrame (date × ticker) of adjusted closing prices.
    spy   : SPY daily OHLCV (for the 50-day MA trend filter).
    threshold_pct : fraction-of-issues threshold (canonical 2.2%).

    Returns a DataFrame with:
    - ``n_highs``, ``n_lows``, ``n_stocks`` (issues actively traded that day)
    - ``frac_highs``, ``frac_lows``
    - ``ma50_rising`` (1 if SPY close > SPY MA50)
    - ``mcclellan_neg`` (1 if advances < declines across the constituent panel)
    - ``hindenburg`` (1 if all five conditions met)
    - ``spy_ret`` (SPY daily simple return, from adjusted close)
    - ``spy_close`` (SPY adjusted close)
    """
    idx = panel.index.intersection(spy.index)
    panel = panel.loc[idx].copy()
    spy_aligned = spy.loc[idx].copy()

    n_days = len(panel)
    n_highs_arr = np.zeros(n_days, dtype=int)
    n_lows_arr = np.zeros(n_days, dtype=int)
    n_stocks_arr = np.zeros(n_days, dtype=int)

    prices = panel.to_numpy(dtype=float)

    for t in range(LOOKBACK_52W, n_days):
        window = prices[max(0, t - LOOKBACK_52W) : t, :]
        px_today = prices[t, :]
        # only count stocks with valid data on both today and in the window
        valid = np.isfinite(px_today) & np.isfinite(window).any(axis=0)
        n_stocks_arr[t] = int(valid.sum())
        if n_stocks_arr[t] == 0:
            continue
        hi52 = np.nanmax(window[:, valid], axis=0)
        lo52 = np.nanmin(window[:, valid], axis=0)
        n_highs_arr[t] = int((px_today[valid] >= hi52).sum())
        n_lows_arr[t] = int((px_today[valid] <= lo52).sum())

    frac_highs = np.where(n_stocks_arr > 0, n_highs_arr / n_stocks_arr, np.nan)
    frac_lows = np.where(n_stocks_arr > 0, n_lows_arr / n_stocks_arr, np.nan)

    # --- MA50 trend filter on SPY --------------------------------------------
    spy_close = spy_aligned["close"].to_numpy(dtype=float)
    ma50 = pd.Series(spy_close).rolling(MA50_WINDOW, min_periods=MA50_WINDOW).mean().to_numpy()
    ma50_rising = (spy_close > ma50).astype(int)

    # --- McClellan proxy: advance/decline from the panel --------------------
    prices_df = panel.ffill()   # forward-fill single-day gaps
    ret_today = prices_df.pct_change().to_numpy(dtype=float)
    advances = np.nansum(ret_today > 0, axis=1)
    declines = np.nansum(ret_today <= 0, axis=1)
    mcclellan_neg = (advances < declines).astype(int)

    # --- SPY return ----------------------------------------------------------
    spy_ret = spy_aligned["close"].pct_change().to_numpy(dtype=float)

    # --- Hindenburg signal ---------------------------------------------------
    th = threshold_pct / 100.0
    enough_history = np.arange(n_days) >= LOOKBACK_52W
    hindenburg = (
        (frac_highs >= th)
        & (frac_lows >= th)
        & (ma50_rising == 1)
        & (mcclellan_neg == 1)
        & enough_history
    ).astype(int)

    df = pd.DataFrame(
        {
            "n_highs": n_highs_arr,
            "n_lows": n_lows_arr,
            "n_stocks": n_stocks_arr,
            "frac_highs": frac_highs,
            "frac_lows": frac_lows,
            "ma50_rising": ma50_rising,
            "mcclellan_neg": mcclellan_neg,
            "hindenburg": hindenburg,
            "spy_ret": spy_ret,
            "spy_close": spy_close,
        },
        index=panel.index,
    )

    if save:
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(_breadth_cache_path(cache_dir))

    return df


def load_breadth(
    cache_dir: str = DEFAULT_CACHE,
    threshold_pct: float = THRESHOLD_PCT,
    fetch: bool = False,
    start: str = "2005-01-01",
) -> pd.DataFrame:
    """Load (or build) the precomputed breadth frame.  Cache-only unless ``fetch=True``.

    On a cache miss with ``fetch=False`` raises ``FileNotFoundError``.
    With ``fetch=True`` downloads the constituent panel and SPY, computes breadth, and saves.
    """
    bpath = _breadth_cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(bpath):
            raise FileNotFoundError(
                f"No cached breadth frame at {bpath}. "
                "Call load_breadth(fetch=True) once to populate."
            )
        df = pd.read_parquet(bpath)
        # Re-apply threshold (the stored frame has frac_highs / frac_lows)
        th = threshold_pct / 100.0
        enough = (df.index.get_loc(df.index[LOOKBACK_52W]) <= np.arange(len(df))) if len(df) > LOOKBACK_52W else np.zeros(len(df), dtype=bool)
        # Simpler: recalculate hindenburg from stored fracs
        df["hindenburg"] = (
            (df["frac_highs"] >= th)
            & (df["frac_lows"] >= th)
            & (df["ma50_rising"] == 1)
            & (df["mcclellan_neg"] == 1)
            & df["frac_highs"].notna()
            & df["frac_lows"].notna()
            & (df["n_stocks"] > 0)
        ).astype(int)
        return df

    panel = fetch_constituent_panel(cache_dir=cache_dir, fetch=True, start=start)
    spy = fetch_spy(cache_dir=cache_dir, fetch=True, start=start)
    return compute_breadth(panel, spy, threshold_pct=threshold_pct, cache_dir=cache_dir)


def fingerprint(df: pd.DataFrame) -> str:
    """Short content fingerprint of the breadth frame (spy_ret column)."""
    col = "spy_ret" if "spy_ret" in df.columns else df.columns[0]
    arr = df[col].fillna(0.0).to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
