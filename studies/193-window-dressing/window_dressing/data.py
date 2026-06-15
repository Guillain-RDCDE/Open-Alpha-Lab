"""Data layer for Study 193 (Window-Dressing).

Two tapes, one shape (a daily return series with quarter-boundary labels):

- ``synthetic_daily`` — a *deterministic, offline* generator. A ``wd_effect`` knob
  plants a mean-return bump on the last ``window`` trading days of each calendar
  quarter and an equal-and-opposite reversal over the first ``window`` days of the
  next quarter. ``wd_effect = 0`` is the null — no window dressing — so tests can
  assert the strategy only works when the effect is genuinely planted.
- ``fetch_spy`` — real SPY or ^GSPC daily prices (``yfinance`` / shared repo cache),
  **cache-only** by default so the test-suite and the reproducible core never touch
  the network.

Quarter boundaries: calendar quarters end on Mar/Jun/Sep/Dec. The last ``window``
trading days of the quarter are the *pump* window; the first ``window`` trading days
of the following quarter are the *reversal* window.

No look-ahead: the window label is derived purely from the calendar date, which is
known before the open.  The return recorded is the same-day close-to-close log-return.

Reference: Carhart, Kaniel, Musto & Reed (2002), *Leaning for the Tape: Evidence of
Gaming Behavior in Equity Mutual Funds*, Journal of Finance 57(2), 661–693.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "_cache"))

# Shared repo-level paths for the two preferred sources.
_SPY_SHARED = os.path.join(REPO_CACHE, "SPY_total_return.parquet")
_GSPC_SHARED = os.path.join(REPO_CACHE, "^GSPC_split_only.parquet")

# Window (trading days) on each side of the quarter end.
DEFAULT_WINDOW = 5


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------
def _quarter_labels(idx: pd.DatetimeIndex, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """Label each business day as belonging to a pump, reversal, or neutral period.

    A trading day is in the *pump* window if it falls among the last ``window``
    trading days of a calendar quarter (Mar / Jun / Sep / Dec). It is in the
    *reversal* window if it falls among the first ``window`` trading days of the
    following quarter.

    Parameters
    ----------
    idx : DatetimeIndex
        Business-day index of the price series.
    window : int
        Number of trading days on each side of the quarter boundary.

    Returns
    -------
    DataFrame indexed by *idx* with boolean columns ``is_pump``, ``is_reversal``,
    and helper columns ``quarter_end_month`` (3/6/9/12) and ``seq_in_quarter``
    (business-day ordinal within the quarter, 1 = first).
    """
    # Assign each date to a calendar quarter end month (March = 1st quarter end, etc.)
    month = idx.month
    quarter_end_month = np.where(
        month <= 3, 3,
        np.where(month <= 6, 6,
                 np.where(month <= 9, 9, 12))
    )
    # Group by (year, quarter_end_month) and compute ordinal position within each group.
    df = pd.DataFrame(
        {"month": month, "year": idx.year, "qem": quarter_end_month},
        index=idx,
    )
    # Reverse ordinal within the quarter = distance from the quarter end.
    df["seq_rev"] = df.groupby(["year", "qem"]).cumcount(ascending=False) + 1
    # Forward ordinal within the quarter = distance from the quarter start.
    df["seq_fwd"] = df.groupby(["year", "qem"]).cumcount(ascending=True) + 1

    is_pump = df["seq_rev"] <= window
    is_reversal = df["seq_fwd"] <= window

    return pd.DataFrame(
        {
            "is_pump": is_pump.values,
            "is_reversal": is_reversal.values,
            "seq_rev": df["seq_rev"].values,
            "seq_fwd": df["seq_fwd"].values,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 30,
    annual_vol: float = 0.16,
    annual_drift: float = 0.08,
    wd_effect: float = 0.0,
    window: int = DEFAULT_WINDOW,
    seed: int = 193,
    start: str = "1993-01-04",
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily equity tape with a known window-dressing anomaly.

    Log-returns follow N(drift/252, (vol/sqrt(252))**2) on neutral days. On the
    last *window* trading days of each quarter a bump of ``+wd_effect * daily_sig``
    is added (the "pump"). On the first *window* trading days of the next quarter
    a bump of ``-wd_effect * daily_sig`` is applied (the "reversal").

    - ``wd_effect = 0``   → pure random walk; the calendar is a fair coin.
    - ``wd_effect > 0``   → end-of-quarter returns are inflated, early-next-quarter
                            returns are depressed (the classic Carhart et al. story).
    - ``wd_effect < 0``   → the reverse (end-of-quarter weakness, early strength).

    Returns ``(df, truth)`` where ``df`` has columns
    ``ret, is_pump, is_reversal, seq_rev, seq_fwd``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=int(n_years * 252))
    n = len(idx)
    daily_mu = annual_drift / 252.0
    daily_sig = annual_vol / np.sqrt(252.0)
    ret = rng.normal(daily_mu, daily_sig, n)

    labels = _quarter_labels(idx, window=window)
    # Plant the pump-and-dump.
    ret[labels["is_pump"].values] += wd_effect * daily_sig
    ret[labels["is_reversal"].values] -= wd_effect * daily_sig

    df = pd.DataFrame(
        {
            "ret": ret,
            "is_pump": labels["is_pump"].values,
            "is_reversal": labels["is_reversal"].values,
            "seq_rev": labels["seq_rev"].values,
            "seq_fwd": labels["seq_fwd"].values,
        },
        index=idx,
    )
    df.index.name = "date"

    truth = {
        "n_years": n_years,
        "annual_vol": annual_vol,
        "annual_drift": annual_drift,
        "wd_effect": wd_effect,
        "window": window,
        "n_days": n,
        "n_pump": int(labels["is_pump"].sum()),
        "n_reversal": int(labels["is_reversal"].sum()),
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — SPY or ^GSPC daily, cache-only by default
# ---------------------------------------------------------------------------
def _build_daily(raw: pd.DataFrame, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """Convert a raw OHLCV frame to a daily return frame with quarter-boundary labels."""
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    # Normalise column names: try 'Close' first (repo cache format), then 'close'.
    cols = {c.lower(): c for c in raw.columns}
    close_col = cols.get("close", None)
    if close_col is None:
        raise ValueError(f"No 'Close' column found; got {list(raw.columns)}")
    close = raw[close_col].copy()
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close.index.name = "date"
    close = close.sort_index()
    ret = np.log(close / close.shift(1))
    idx = close.index

    labels = _quarter_labels(idx, window=window)
    df = pd.DataFrame(
        {
            "close": close.values,
            "ret": ret.values,
            "is_pump": labels["is_pump"].values,
            "is_reversal": labels["is_reversal"].values,
            "seq_rev": labels["seq_rev"].values,
            "seq_fwd": labels["seq_fwd"].values,
        },
        index=idx,
    )
    return df.dropna(subset=["ret"])


def fetch_spy(
    start: str = "1993-02-01",
    window: int = DEFAULT_WINDOW,
    fetch: bool = False,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Daily SPY return frame with quarter-boundary labels; cache-only unless ``fetch=True``.

    Tries sources in priority order:
    1. Shared repo cache at ``_cache/SPY_total_return.parquet`` (preferred: longest history).
    2. Shared repo cache at ``_cache/^GSPC_split_only.parquet`` (fallback).
    3. Per-study local cache at ``studies/193-window-dressing/_cache/spy_daily.parquet``.
    4. Network via ``yfinance`` — only with an explicit ``fetch=True``.

    Columns: ``close, ret, is_pump, is_reversal, seq_rev, seq_fwd``.
    """
    # 1. Preferred shared cache — SPY total return since 1993.
    if os.path.exists(_SPY_SHARED):
        raw = pd.read_parquet(_SPY_SHARED)
        df = _build_daily(raw, window=window)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        return df

    # 2. Fallback — GSPC since 1928.
    if os.path.exists(_GSPC_SHARED):
        raw = pd.read_parquet(_GSPC_SHARED)
        df = _build_daily(raw, window=window)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        return df

    # 3. Per-study local cache.
    local_dir = cache_dir or DEFAULT_CACHE
    local = os.path.join(local_dir, "spy_daily.parquet")
    if os.path.exists(local):
        raw = pd.read_parquet(local)
        df = _build_daily(raw, window=window)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        return df

    # 4. Network — only with explicit fetch=True.
    if not fetch:
        raise FileNotFoundError(
            f"No SPY/GSPC cache found at {_SPY_SHARED}, {_GSPC_SHARED}, or {local}. "
            "Call fetch_spy(fetch=True) to populate the local cache."
        )

    import yfinance as yf  # lazy: only when we actually go to the network

    raw = yf.download(
        "SPY",
        start=start,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data for SPY")

    os.makedirs(local_dir, exist_ok=True)
    raw.to_parquet(local)

    df = _build_daily(raw, window=window)
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (ret column), for the as-of stamp."""
    arr = df["ret"].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
