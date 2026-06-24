"""Data layer for Study 423 (Force Index).

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date):

- ``synthetic_panel`` — a *deterministic, offline* generator.  Its ``planted_edge`` knob
  builds a tape where, by construction, *the sign of yesterday's price-times-volume flow
  predicts tomorrow's return*.  At ``planted_edge = 0`` the tape is a pure random walk with
  i.i.d. volume — a fair coin — so a test can assert the Force-Index timing rule beats
  buy-and-hold *only* when we plant the edge, and not otherwise.  This is the study's null
  in a bottle, and its positive control.
- ``load_real`` — the real Yahoo! daily tape (``yfinance``), cache-first so the test-suite
  and the reproducible core never touch the network.  Daily bars go back ~20+ years on
  the liquid ETFs we use, giving high statistical power compared with intraday studies.

Elder's Force Index is defined as::

    raw_FI(t) = (Close(t) - Close(t-1)) * Volume(t)
    FI_n(t)   = EMA_n( raw_FI )            # Elder smooths with an EMA

Elder (1993) uses **FI(2)** for short-term entries and **FI(13)** for the trend filter.
The folk rule under test: *the Force Index flags reversals* — go long when FI(13) crosses
**up** through zero (buying pressure has returned) and exit/short when it crosses **down**
(selling pressure dominates).  We turn it into a long/flat (and long/short) timing rule and
race its NET Sharpe against buy-and-hold, excess-vs-excess.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the signal is
formed on closes/volume up to *t*, and the position it implies earns the return of *t+1*).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (null + positive control)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 3000,
    planted_edge: float = 0.0,
    daily_vol: float = 0.011,
    start: str = "2008-01-02",
    seed: int = 423,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a tunable *Force-Index* edge knob.

    The mechanism the Force Index is supposed to harvest is **signed volume flow**: a big
    up-day on heavy volume (positive raw FI) is supposed to forecast continuation, a big
    down-day on heavy volume (negative raw FI) a decline.  We plant exactly that:

        sign_flow(t) = sign( (close(t)-close(t-1)) * volume(t) )
        r(t+1)       = planted_edge * daily_vol * sign_flow(t) + eps(t+1)

    so tomorrow's drift leans in the direction of today's signed dollar-volume flow with
    strength ``planted_edge``.

    - ``planted_edge = 0``  → a random walk with i.i.d. volume; FI carries no information,
                              the timing rule is a fair coin against buy-and-hold.
    - ``planted_edge > 0``  → the planted, harvestable Force-Index edge (positive control).

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)

    eps = rng.normal(0.0, daily_vol, n_days)
    # Volume is a positive, persistent process (log-AR(1)) so heavy days cluster.
    log_v = np.empty(n_days)
    log_v[0] = 0.0
    v_noise = rng.normal(0.0, 0.45, n_days)
    for i in range(1, n_days):
        log_v[i] = 0.85 * log_v[i - 1] + v_noise[i]
    volume = (5_000_000.0 * np.exp(log_v)).round()

    log_ret = np.empty(n_days)
    close = np.empty(n_days)
    close[0] = 100.0
    log_ret[0] = 0.0
    prev_close = close[0]
    for i in range(1, n_days):
        # signed flow known at i-1 (price change i-1 vs i-2, volume i-1)
        if i >= 2:
            flow = (close[i - 1] - close[i - 2]) * volume[i - 1]
            sflow = np.sign(flow)
        else:
            sflow = 0.0
        log_ret[i] = planted_edge * daily_vol * sflow + eps[i]
        close[i] = prev_close * np.exp(log_ret[i])
        prev_close = close[i]

    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": volume},
        index=pd.DatetimeIndex(cal, name="date"),
    )
    truth = {
        "planted_edge": planted_edge,
        "daily_vol": daily_vol,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"fi_{safe}_1d.parquet")


def load_real(
    ticker: str,
    period: str = "max",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 3,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-first.

    Reads the cached parquet under ``_cache/`` if present.  If absent (or ``fetch=True``),
    pulls from Yahoo via ``yfinance`` with a small backoff on rate-limits and caches the
    result, so every subsequent run is offline.  ``auto_adjust=True`` gives a
    total-return-adjusted close (splits + dividends) — labelled as such downstream.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path) and not fetch:
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        last_err = None
        bars = None
        for attempt in range(retries):
            try:
                raw = yf.download(
                    ticker, period=period, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is not None and not raw.empty:
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)
                    bars = raw.rename(columns=str.lower)[
                        ["open", "high", "low", "close", "volume"]
                    ]
                    bars.index.name = "date"
                    break
            except Exception as e:  # noqa: BLE001
                last_err = e
            time.sleep(1.5 * (attempt + 1))
        if bars is None or bars.empty:
            raise RuntimeError(
                f"yfinance returned no daily bars for {ticker} "
                f"(last error: {last_err})"
            )
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
