"""Data layer for Study 448 — Point & Figure price targets.

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a planted-edge knob.
  ``edge = 0`` is a pure geometric random walk: a P&F double-top buy carries **no**
  forecast power, and the count target is hit only as often as geometry alone allows
  (the honest null). ``edge > 0`` plants a literal "the count target gets hit" structure:
  whenever a fresh column of Xs prints, the next stretch of returns gets an upward drift
  *scaled to carry price toward the horizontal-count target* — so the harness can prove
  it would detect a real P&F effect if one existed. The knob is the positive control,
  never market evidence.

- ``load_real`` — the real Yahoo daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet under ``_cache/`` and only goes to the network on a cache miss (with a
  couple of back-off retries), then caches the result so re-runs are fully offline.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: a P&F column
and its signal are only knowable once the box-crossing close has printed, the target is
fixed at the signal close, and the hit/stop is resolved on strictly *subsequent* bars.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The instruments a P&F practitioner would chart: the broad US market (SPY), the index
# the method grew up on (the Dow, ^DJI), a high-beta single name with plenty of columns
# (AAPL), and a commodity proxy (GLD) for an instrument with different congestion texture.
DEFAULT_TICKERS = ("SPY", "^DJI", "AAPL", "GLD")


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 6000,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    box_frac: float = 0.02,
    seed: int = 448,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *planted* "P&F targets get hit" edge.

    The path is a geometric random walk. When ``edge > 0`` we plant the exact pattern
    the folklore claims: each time price climbs through a fresh up-box (a ~``box_frac``
    move above the running low), the following ``run`` bars get an **upward drift** so
    that the move continues in the breakout direction — a momentum/continuation that a
    horizontal-count target (which extrapolates the breakout) *can* harvest. With
    ``edge = 0`` the path is a pure random walk: the count target is hit only as often
    as geometry alone allows, and the detector must NOT manufacture a hit-rate edge.

    The drift is deliberately self-limiting (it decays over the run) so the planted tape
    stays a realistic price series rather than an exponential blow-up.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252.0)
    sessions = pd.bdate_range(start="2002-01-02", periods=n_days)

    rets = rng.normal(0.0, daily_vol, n_days)
    if edge > 0.0:
        # Plant continuation: track a running low; when price has risen ~box_frac above it,
        # arm an upward-drift "run" that carries the breakout further (the count's premise).
        log_p = np.cumsum(rets)
        run_left = 0
        ref_low = log_p[0]
        for t in range(1, n_days):
            if log_p[t] < ref_low:
                ref_low = log_p[t]
            elif (log_p[t] - ref_low) > box_frac and run_left == 0:
                run_left = 40                       # arm a continuation run
                ref_low = log_p[t]                  # reset reference at the breakout
            if run_left > 0:
                rets[t] += edge * daily_vol * (run_left / 40.0)
                run_left -= 1
            log_p[t] = log_p[t - 1] + rets[t]

    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, close.size)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick

    bars = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "box_frac": box_frac,
             "n_days": n_days, "seed": seed}
    return bars, truth


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily, cache-first
# --------------------------------------------------------------------------- #
def _safe(ticker: str) -> str:
    return ticker.replace("=", "").replace("^", "").replace("/", "")


def _cache_path(ticker: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"bars_{_safe(ticker)}_1d.parquet")


def load_real(
    ticker: str = "SPY",
    start: str = "2000-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = True,
) -> pd.DataFrame:
    """Real daily OHLC for ``ticker`` — cache-first.

    Reads the cached parquet if present; otherwise (and only if ``fetch=True``) pulls
    from yfinance with a couple of back-off retries and caches it. The reproducible core
    and tests run fully offline once the cache exists.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        bars = pd.read_parquet(path)
    elif fetch:
        import yfinance as yf  # lazy: only on a true cache miss

        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(ticker, start=start, end=end, interval="1d",
                                  auto_adjust=True, progress=False)
                if raw is not None and not raw.empty:
                    break
            except Exception:
                raw = None
            time.sleep(1.5 * (attempt + 1))
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)
    else:
        raise FileNotFoundError(
            f"No cached daily tape for {ticker} at {path}. "
            f"Call load_real({ticker!r}, fetch=True) once to populate the cache."
        )

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars.dropna()


def have_real(ticker: str = "SPY", cache_dir: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(_cache_path(ticker, cache_dir))


def fingerprint(bars: pd.DataFrame) -> str:
    """Short sha1 content fingerprint of a tape (close column) for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
