"""Data layer for Study 419 (Chaikin Money Flow).

Two tapes, one shape (a tz-naive daily OHLCV frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge
  knob**.  Chaikin Money Flow (CMF) is a *volume-flow* oscillator: it sums the
  Accumulation/Distribution money-flow volume over a trailing window and divides by
  total volume, giving a value in [-1, +1].  The folk claim is that **money flow leads
  price** — sustained accumulation (CMF > 0) precedes a rise, distribution (CMF < 0)
  precedes a fall.  That is *only* true if the sign of where-in-the-bar trading happened
  (weighted by volume) actually forecasts the next return.  So the knob ``edge`` plants
  exactly that — a coupling between the volume-weighted close-location of a bar and the
  sign of the *following* day's return.  At ``edge = 0`` the close-location is pure
  noise on a random walk, so CMF can carry no information; the test must then find none.
  At ``edge > 0`` CMF *should* lead price — the positive control proving the harness can
  see a money-flow edge when one is really there.

- ``load_real`` / ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``),
  **cache-first**: a parquet under ``_cache/`` is read if present and the network is
  only touched on an explicit cache miss (with a small retry/backoff).  Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.  We pull
  **auto-adjusted** OHLCV — note this makes ``close``/``high``/``low`` split/dividend-
  adjusted (total-return-ish); ``volume`` is the raw traded share count.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the indicator
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
    seed: int = 419,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape where *money flow can lead price*.

    The price is a log random walk with daily drift ``drift`` and daily vol
    ``annual_vol / sqrt(252)``.  Each bar's intrabar **close location** — how near the
    close sits to the bar's high vs its low — is drawn with a controllable bias, and a
    volume shock ``z_t`` scales the traded volume.

    The Money Flow Multiplier ``MFM = ((C-L) - (H-C)) / (H-L)`` is the per-bar building
    block of CMF: it is +1 when the close is at the high (pure accumulation), -1 at the
    low (pure distribution).  The knob ``edge`` plants the one pattern CMF is *supposed*
    to see: today's volume-weighted close-location forecasts tomorrow's return —

        ``r_{t+1} += edge * mfm_t * (volume_t / base_vol) * daily_vol``

    so a high-volume accumulation bar (close near the high, heavy volume) is followed by
    a positive drift, a high-volume distribution bar by a negative one.  This is exactly
    the "money flow leads price" story, keyed to volume-weighted close location.

    - ``edge = 0`` → close-location and volume are decorative noise; CMF holds no edge.
    - ``edge > 0`` → a genuine money-flow-leads-price effect CMF *should* harvest
      (the positive control).

    Returns ``(bars, truth)`` with ``truth`` recording the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252.0)
    sessions = pd.bdate_range(start=start, periods=n_days)

    z = rng.standard_normal(n_days)                       # volume shocks
    # Per-bar close-location bias in [-1, 1] (proxy for the money-flow multiplier).
    mfm = rng.uniform(-1.0, 1.0, size=n_days)
    base_ret = rng.normal(drift, daily_vol, size=n_days)  # base returns

    log_ret = base_ret.copy()
    if edge != 0.0:
        for t in range(1, n_days):
            vfac = np.exp(0.6 * z[t - 1])                 # volume factor of prior bar
            log_ret[t] += edge * mfm[t - 1] * vfac * daily_vol

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    # Build H/L so that the close sits at the planted location mfm within [L, H].
    rng_span = np.abs(rng.normal(0.0, daily_vol, close.size)) * close + 1e-9
    # close location frac f in [0,1] : mfm = 2f - 1  ->  f = (mfm+1)/2
    frac = (mfm + 1.0) / 2.0
    hi = close + (1.0 - frac) * rng_span
    lo = close - frac * rng_span
    hi = np.maximum.reduce([hi, open_, close])
    lo = np.minimum.reduce([lo, open_, close])
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

    If ``fetch=False`` (the default) the cached parquet is read and the network is never
    touched.  If the cache is missing and ``fetch=True`` we download from yfinance with a
    couple of retries + small backoff, then cache the result so all later runs are offline.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
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
