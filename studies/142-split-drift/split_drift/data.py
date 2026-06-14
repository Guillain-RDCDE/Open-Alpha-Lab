"""Data layer for Study 142 (Split-Drift).

Two tapes:

- ``synthetic_events`` — a *deterministic, offline* generator.  Simulates a panel
  of stocks with a ``drift_bps`` knob that controls how much extra daily return is
  earned in the ``horizon`` days following a synthetic split event.  Setting
  ``drift_bps=0`` is the null: splits carry no directional signal, so a test can
  assert "the effect appears only when we plant it, and not otherwise."
- ``fetch_splits_panel`` — the real loader.  Uses ``yfinance`` ``.splits`` on a
  basket of large-cap US stocks to extract actual split *effective* dates, then
  downloads split-adjusted daily prices and assembles a per-event return panel.
  Cache-only by default (``fetch=False`` raises without cache); ``fetch=True`` hits
  Yahoo! and caches both the split calendar and daily price parquets.

**Important caveat (stated prominently here and in the notebooks):**
``yfinance`` returns the split *effective* date (the ex-split date), NOT the
*announcement* date.  The classic Ikenberry et al. (1996) anomaly is measured
from the *announcement* date (roughly 3–6 weeks before the effective date), so
our post-effective drift is a *different*, strictly weaker window.  We say so
loudly rather than pretend we are replicating the original study.

No look-ahead: signals are constructed as of the effective date close; the forward
window starts at ``t+1``.  Prices are split-adjusted by Yahoo!, so no manual
adjustment is needed.
"""

from __future__ import annotations

import hashlib
import os
from typing import Literal

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# ---------------------------------------------------------------------------
# A broad basket of large-cap US stocks with historical split activity
# ---------------------------------------------------------------------------
SPLIT_BASKET = [
    "AAPL", "TSLA", "NVDA", "AMZN", "GOOGL", "META", "MSFT", "NFLX",
    "AMD", "AVGO", "COST", "WMT", "UNH", "V", "MA", "JPM", "BAC",
    "XOM", "CVX", "HD", "LOW", "NKE", "SBUX", "DIS", "PFE", "JNJ",
    "MRK", "ABBV", "LLY", "BMY",
]

# Baseline basket: non-splitting stocks to measure the no-split drift (control arm).
# We use the same tickers over the same calendar windows but restrict to periods
# when they did NOT split.
BASELINE_TICKERS = SPLIT_BASKET  # same basket; strategy.py tags split vs no-split periods


# ---------------------------------------------------------------------------
# Synthetic event tape — deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_events(
    n_stocks: int = 20,
    n_events: int = 50,
    horizon: int = 63,          # ~3 months in trading days
    drift_bps: float = 0.0,     # extra daily drift in the post-split window (0 = null)
    base_vol_ann: float = 0.30,
    start: str = "2000-01-03",
    seed: int = 142,
) -> tuple[pd.DataFrame, dict]:
    """Synthetic per-event forward return panel with a planted post-split drift.

    Each "event" is a split effective date drawn at random from a simulated stock
    history.  For ``drift_bps > 0`` each post-event day earns ``drift_bps`` of
    extra daily return; for ``drift_bps = 0`` the post-event period is i.i.d. normal
    — a clean null.

    Returns ``(events, truth)`` where ``events`` is a DataFrame with one row per
    synthetic event:

    Columns: ``ticker, event_date, ret_1m, ret_3m, ret_6m, ret_12m, is_split``.

    - ``ret_1m`` through ``ret_12m`` are buy-and-hold returns from ``event_date + 1``
      to ``event_date + horizon_days``.
    - ``is_split = True`` for all synthetic events (this tape has no baseline arm,
      which is added in strategy.py).
    """
    rng = np.random.default_rng(seed)
    sig_d = base_vol_ann / np.sqrt(252)
    extra_daily = drift_bps * 1e-4

    dates = pd.bdate_range(start=start, periods=500, name="date")
    rows = []
    for stock_i in range(n_stocks):
        events_this = rng.integers(1, max(2, n_events // n_stocks + 1))
        # Sample event dates from the calendar (leave at least horizon days of room)
        ev_indices = rng.choice(
            len(dates) - horizon - 10, size=int(events_this), replace=False
        )
        for ei in sorted(ev_indices):
            ev_date = dates[ei]
            fwd = rng.normal(sig_d, sig_d, horizon)
            # Plant the drift in the post-event window
            fwd += extra_daily
            cum = np.cumprod(1.0 + fwd) - 1.0  # cumulative return
            # Horizon buckets: 21, 63, 126, 252 trading days
            def _r(h: int) -> float:
                return float(cum[min(h, horizon) - 1]) if h <= horizon else float("nan")

            rows.append(
                {
                    "ticker": f"SYN{stock_i:02d}",
                    "event_date": ev_date,
                    "ret_1m": _r(21),
                    "ret_3m": _r(63),
                    "ret_6m": _r(126),
                    "ret_12m": _r(252),
                    "split_ratio": float(rng.choice([2.0, 3.0, 4.0])),
                    "is_split": True,
                }
            )

    events = pd.DataFrame(rows).sort_values("event_date").reset_index(drop=True)
    truth = {
        "n_stocks": n_stocks,
        "n_events": len(events),
        "horizon": horizon,
        "drift_bps": drift_bps,
        "base_vol_ann": base_vol_ann,
        "seed": seed,
    }
    return events, truth


# ---------------------------------------------------------------------------
# Real loader — Yahoo splits + daily prices, cache-only by default
# ---------------------------------------------------------------------------
def _splits_cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"splits_142_{safe}.parquet")


def _prices_cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_142_{safe}_1d.parquet")


def fetch_splits_panel(
    tickers: list[str] | None = None,
    start: str = "2000-01-01",
    horizon_days: tuple[int, ...] = (21, 63, 126, 252),
    min_split_ratio: float = 1.5,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load split events and compute post-event forward returns.

    Parameters
    ----------
    tickers:
        List of tickers to search for splits.  Defaults to ``SPLIT_BASKET``.
    start:
        Earliest split effective date to include.
    horizon_days:
        Tuple of look-ahead horizons in trading days (default 21/63/126/252).
    min_split_ratio:
        Minimum split ratio to include (e.g. 1.5 keeps 3:2, 2:1, 4:1, …; filters
        out reverse splits and tiny round lots).
    fetch:
        ``False`` (default): read from cache; raises if no cache.
        ``True``: hit Yahoo! and write parquet files under ``cache_dir``.
    cache_dir:
        Directory for parquet caches.

    Returns
    -------
    events : DataFrame
        One row per split event.  Columns: ``ticker, event_date, split_ratio,
        ret_1m, ret_3m, ret_6m, ret_12m, is_split=True``.
    prices : DataFrame
        Daily adjusted close prices, wide format (columns = tickers, index = date).

    Notes
    -----
    - ``event_date`` is the *effective* (ex-split) date as reported by yfinance —
      NOT the announcement date.  The announcement precedes the effective date by
      roughly 3–6 weeks; the drift measured here (post-effective) is therefore an
      *upper bound* on what is achievable from public announcement-day signals.
    - Prices are *split-adjusted* by Yahoo! so forward returns need no manual
      adjustment.
    """
    if tickers is None:
        tickers = SPLIT_BASKET
    os.makedirs(cache_dir, exist_ok=True)

    import yfinance as yf  # lazy import — only used when data are needed

    # ---- 1. Collect split calendars ----------------------------------------
    all_splits: list[pd.DataFrame] = []
    prices_dict: dict[str, pd.Series] = {}

    for ticker in tickers:
        sp_path = _splits_cache_path(ticker, cache_dir)
        pr_path = _prices_cache_path(ticker, cache_dir)

        if fetch:
            tkr = yf.Ticker(ticker)
            # Split history
            sp = tkr.splits
            if isinstance(sp, pd.Series):
                sp = sp.reset_index()
                sp.columns = ["date", "ratio"]
                sp["ticker"] = ticker
            else:
                sp = pd.DataFrame(columns=["date", "ratio", "ticker"])
            sp.to_parquet(sp_path)

            # Daily adjusted close
            raw = yf.download(
                ticker, start=start, interval="1d", auto_adjust=True, progress=False
            )
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.rename(columns=str.lower)
            if not raw.empty and "close" in raw.columns:
                raw[["close"]].to_parquet(pr_path)
        else:
            if not os.path.exists(sp_path):
                raise FileNotFoundError(
                    f"No cached splits for {ticker} at {sp_path}. "
                    f"Call fetch_splits_panel(fetch=True) once."
                )
            if not os.path.exists(pr_path):
                raise FileNotFoundError(
                    f"No cached prices for {ticker} at {pr_path}. "
                    f"Call fetch_splits_panel(fetch=True) once."
                )
            sp = pd.read_parquet(sp_path)

        # Load prices
        if os.path.exists(pr_path):
            pr = pd.read_parquet(pr_path)
            if not pr.empty and "close" in pr.columns:
                close = pr["close"].copy()
                close.index = pd.DatetimeIndex(close.index).tz_localize(None)
                close.index.name = "date"
                prices_dict[ticker] = close

        # Filter splits by date and ratio
        if not sp.empty:
            sp["date"] = pd.to_datetime(sp["date"]).dt.tz_localize(None)
            sp = sp[(sp["date"] >= pd.Timestamp(start)) & (sp["ratio"] >= min_split_ratio)]
            if not sp.empty:
                all_splits.append(sp[["ticker", "date", "ratio"]])

    # ---- 2. Assemble price panel -------------------------------------------
    if prices_dict:
        prices = pd.DataFrame(prices_dict)
        prices.index = pd.DatetimeIndex(prices.index, name="date")
        prices = prices.sort_index()
    else:
        prices = pd.DataFrame()

    if not all_splits:
        return pd.DataFrame(
            columns=["ticker", "event_date", "split_ratio",
                     "ret_1m", "ret_3m", "ret_6m", "ret_12m", "is_split"]
        ), prices

    splits_df = pd.concat(all_splits, ignore_index=True).rename(
        columns={"date": "event_date", "ratio": "split_ratio"}
    )
    splits_df = splits_df.sort_values("event_date").reset_index(drop=True)

    # ---- 3. Compute forward returns ----------------------------------------
    rows = []
    for _, row in splits_df.iterrows():
        tkr = row["ticker"]
        ev = pd.Timestamp(row["event_date"])
        if tkr not in prices.columns:
            continue
        close_s = prices[tkr].dropna()
        idx = close_s.index
        # Find position of event date (or the next available trading day)
        pos_arr = np.searchsorted(idx, ev)
        if pos_arr >= len(idx):
            continue
        ev_pos = pos_arr  # index of event date close (or next available)
        fwd_returns: dict[str, float] = {}
        ev_close = close_s.iloc[ev_pos]
        if ev_close <= 0 or not np.isfinite(ev_close):
            continue
        for h, col in zip(horizon_days, ("ret_1m", "ret_3m", "ret_6m", "ret_12m")):
            end_pos = ev_pos + h
            if end_pos >= len(close_s):
                fwd_returns[col] = float("nan")
            else:
                fwd_returns[col] = float(close_s.iloc[end_pos] / ev_close - 1.0)

        rows.append(
            {
                "ticker": tkr,
                "event_date": ev,
                "split_ratio": float(row["split_ratio"]),
                **fwd_returns,
                "is_split": True,
            }
        )

    events = pd.DataFrame(rows).sort_values("event_date").reset_index(drop=True)
    return events, prices


def fingerprint(events: pd.DataFrame) -> str:
    """Short content fingerprint of the event panel (event dates), for as-of stamps."""
    arr = pd.to_datetime(events["event_date"]).astype(np.int64).to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
