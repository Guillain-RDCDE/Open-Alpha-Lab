"""Data layer for Study 418 (Money Flow Index — the volume-weighted RSI).

Two tapes, one shape (a tz-naive daily OHLCV frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge
  knob**.  The Money Flow Index is the volume-weighted cousin of the RSI: it is only
  meant to beat the plain RSI when *volume actually carries information about the next
  move*.  So the knob ``edge`` plants exactly that — a coupling between abnormal volume
  and the sign of the subsequent return (high-volume up-days tend to be followed by
  more up, etc.).  At ``edge = 0`` volume is pure noise sprinkled on a random walk, so
  the MFI can carry no information the RSI lacks; the test must then find no advantage.
  At ``edge > 0`` the MFI *should* pull ahead — the positive control that proves the
  harness can see a volume edge when one is really there.

- ``load_real`` / ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``),
  **cache-first**: a parquet under ``_cache/`` is read if present and the network is
  only touched on an explicit cache miss (with a small retry/backoff).  Daily history
  is long (20+ years) and free of the 60-day cap that affects sub-hourly bars.  We pull
  **auto-adjusted** OHLCV — note this makes ``close`` a total-return-ish split/dividend-
  adjusted series; ``volume`` is the raw traded share count.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: an indicator
is computed on bars up to *t*, the position is decided at the close of *t*, and it earns
the return of *t+1* (one ``shift``).
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
# Synthetic tape — the deterministic offline core (with a planted-edge knob)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 4000,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    drift: float = 0.0003,
    start: str = "2008-01-02",
    seed: int = 418,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape where *volume can carry information*.

    The price is a log random walk with daily drift ``drift`` and daily vol
    ``annual_vol / sqrt(252)``.  Each day also draws an i.i.d. **volume shock**
    ``z_t ~ N(0, 1)`` that scales the traded volume (``volume = base * exp(0.6 * z)``).

    The knob ``edge`` plants the one pattern the MFI is *supposed* to see better than
    the RSI: a **volume-conditioned overreaction**.  A down-move on *heavy* volume
    (large positive ``z_t`` on a negative day) is a panic flush that snaps back; the
    same down-move on *light* volume drifts.  Concretely the next day reverts in
    proportion to the volume shock that accompanied today's move:

        ``r_{t+1} -= edge * r_t * max(z_t, 0) * scale``

    The reversion is keyed to **volume** (``z_t``), so the volume-weighted MFI lines up
    its oversold reading with the high-volume flushes that actually bounce, while the
    plain RSI — blind to volume — fires on every oversold day equally, heavy or light.

    - ``edge = 0`` → volume is decorative noise; the MFI can hold no edge over the RSI.
    - ``edge > 0`` → a genuine volume-conditioned reversion the MFI *should* harvest
      better than the RSI (the positive control).

    Returns ``(bars, truth)`` with ``truth`` recording the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252.0)
    sessions = pd.bdate_range(start=start, periods=n_days)

    z = rng.standard_normal(n_days)                 # volume shocks
    eps = rng.normal(drift, daily_vol, size=n_days)  # base returns

    log_ret = eps.copy()
    if edge != 0.0:
        # volume-conditioned overreaction: today's move partially reverses tomorrow,
        # and the reversal is *stronger* the heavier today's volume shock (z>0).
        for t in range(1, n_days):
            heavy = max(z[t - 1], 0.0)
            log_ret[t] -= edge * eps[t - 1] * heavy

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    base_vol = 5_000_000.0
    volume = (base_vol * np.exp(0.6 * z)).round()

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": volume},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "edge": edge,
        "annual_vol": annual_vol,
        "drift": drift,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def fetch_daily(
    ticker: str,
    start: str = "2000-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 3,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-first, network only on a miss.

    If ``fetch=False`` (the default) the cached parquet is read and the network is
    never touched.  If the cache is missing and ``fetch=True``, we download from
    yfinance with a couple of retries + small backoff, then cache the result so all
    later runs are offline.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path) and not fetch:
        bars = pd.read_parquet(path)
    elif os.path.exists(path) and fetch:
        bars = pd.read_parquet(path)
    else:
        if not fetch:
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_daily({ticker!r}, fetch=True) once to populate the cache."
            )
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = None
        for k in range(retries):
            try:
                raw = yf.download(
                    ticker, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is not None and not raw.empty:
                    break
            except Exception:
                raw = None
            time.sleep(1.5 * (k + 1))
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars.dropna()


def load_real(ticker: str = "SPY", cache_dir: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Cache-first convenience loader for one ticker (default SPY)."""
    return fetch_daily(ticker, fetch=False, cache_dir=cache_dir)


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close + volume), for the as-of stamp."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    h.update(np.ascontiguousarray(bars["volume"].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]
