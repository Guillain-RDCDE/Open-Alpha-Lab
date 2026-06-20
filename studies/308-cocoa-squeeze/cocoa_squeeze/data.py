"""Data layer for Study 308 (Cocoa-Squeeze).

Two tapes, one shape (a tz-naive daily OHLCV frame):

- ``synthetic_blowoff`` — a *deterministic, offline* generator. A ``bubble`` knob plants
  the only thing a "fade the parabola" rule can possibly harvest: a slow super-exponential
  run-up followed by a mean-reverting crack. ``bubble = 0`` collapses the tape to a pure
  geometric random walk (the null — a fair coin for any momentum/reversion timer), so a
  test can assert the engine detects an edge *only* when a real blow-off is planted. This
  is the study's positive control and its null, in one bottle.
- ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only by default so
  the test-suite and the reproducible core never touch the network. The headline asset is
  cocoa front-month futures (``CC=F``); the synthetic control stands in when no cache and
  no network are available.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (one shift,
documented). The reproducible core and tests never reach the network.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_blowoff(
    n_days: int = 2500,
    bubble: float = 1.0,
    peak_frac: float = 0.72,
    daily_vol: float = 0.018,
    crash_reversion: float = 0.03,
    start: str = "2016-01-04",
    seed: int = 308,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape that optionally contains a planted blow-off.

    The price is a geometric random walk plus a deterministic *bubble* bump: a smooth,
    accelerating drift that peaks at ``peak_frac`` of the sample and then mean-reverts
    back toward the pre-bubble trend. ``bubble`` scales the amplitude of that bump.

    - ``bubble = 0``   → pure geometric random walk; no edge for any timer (the null).
    - ``bubble > 0``   → a parabola that runs up and then cracks — momentum works on the
      way up, mean-reversion works after the peak (what the rules try to harvest).

    The crack is modelled as a mild AR(1) pull of log-returns back toward the trend once
    the price sits far above its slow moving average (``crash_reversion`` sets the pull).

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters and the
    integer index of the planted peak.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)
    t = np.arange(n_days)

    # Deterministic bubble bump in log-price: a sharp run-up to ``peak`` then a faster
    # crack. The run-up is a smooth ramp; the unwind is steeper (a real blow-off shape).
    peak = int(round(peak_frac * n_days))
    up_width = max(n_days * 0.14, 40.0)     # how long the parabola takes to inflate
    dn_width = max(n_days * 0.07, 25.0)     # the crack is faster than the climb
    up = np.exp(-0.5 * (np.maximum(peak - t, 0) / up_width) ** 2)
    dn = np.exp(-0.5 * (np.maximum(t - peak, 0) / dn_width) ** 2)
    # bubble * 2.4 in log-space ≈ a ~11x parabola at bubble=1 (a genuine multi-bagger).
    bubble_log = bubble * 2.4 * np.where(t <= peak, up, dn)

    # Base random-walk log-returns with a tiny positive drift.
    base_ret = rng.normal(0.0002, daily_vol, n_days)
    log_base = np.cumsum(base_ret)

    log_price = np.log(2500.0) + log_base + bubble_log

    # A mild post-peak reversion pull: only AFTER the peak, nudge returns back toward trend,
    # so the crack carries exploitable short-term mean reversion without flattening the run-up.
    if crash_reversion > 0 and bubble > 0:
        slow = pd.Series(log_price).ewm(span=60, adjust=False).mean().to_numpy()
        stretch = log_price - slow
        pull = np.where(t > peak, crash_reversion * np.clip(stretch, 0, None), 0.0)
        log_price = log_price - pull

    close = np.exp(log_price)

    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1] * np.exp(rng.normal(0.0, daily_vol * 0.3, n_days - 1))
    wick = np.abs(rng.normal(0.0, daily_vol * 0.6, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(5_000, 80_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "bubble": bubble,
        "peak_frac": peak_frac,
        "peak_idx": peak,
        "daily_vol": daily_vol,
        "crash_reversion": crash_reversion,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_daily.parquet")


def fetch_daily(
    ticker: str = "CC=F",
    start: str = "2000-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker`` (default cocoa front-month ``CC=F``).

    Cache-only unless ``fetch=True``. The network is touched only on an explicit
    ``fetch=True``, then the result is cached as a parquet under ``_cache/``. Note: Yahoo's
    continuous-front-month commodity series is a *price-only* roll, not a total-return
    index — we label it as such everywhere and never call it a return stream.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_daily({ticker!r}, fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, start=start, interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
