"""Data layer for Study 430 — Klinger Volume Oscillator.

The Klinger Oscillator needs **OHLCV** (it leans on volume and the high/low/close
"trend" of the typical price), so both tapes are full daily OHLCV frames, tz-naive
and calendar-date indexed.

Two tapes, one shape:

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge**
  knob. The edge is volume-*led* drift: when accumulation volume builds, the next few
  days carry a small positive drift (``edge = 0`` → pure random walk, a fair coin;
  ``edge > 0`` → a real volume-leads-price effect the Klinger rule can harvest). It
  returns ``(bars, truth)`` and is the positive control — the harness MUST find nothing
  at ``edge = 0`` and light up at large ``edge``.

- ``load_real`` / ``fetch_panel`` — the real Yahoo! daily OHLCV tape (``yfinance``),
  cache-first into ``_cache/`` so the test-suite and the reproducible core never touch
  the network. Daily history is long (decades) and free of the 60-day cap on intraday.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the
oscillator is computed on bars up to *t*, the position taken at *t+1* (one shift).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Headline instrument + a small robustness panel (all liquid, long-listed survivors).
HEADLINE = "SPY"
PANEL = ["SPY", "QQQ", "IWM", "DIA", "EEM", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 4000,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    start: str = "2007-01-03",
    seed: int = 430,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a known amount of *volume-led* drift.

    The construction encodes the Klinger thesis literally: a slow, persistent
    **accumulation state** ``a_t`` (an AR(1) latent) drives *both* the day's volume
    (high when |a| is large) *and*, when ``edge > 0``, a small **forward** drift in
    price proportional to ``a_{t-1}`` — i.e. volume *leads* price by a day. So:

    - ``edge = 0``  → volume swings around but carries **no** information about the next
      day's return: log-returns are a pure random walk, a fair coin. The Klinger timing
      rule must NOT beat buy-and-hold however the noise falls.
    - ``edge > 0``  → accumulation genuinely precedes price moves; the Klinger
      oscillator (which is, in essence, a smoothed signed-volume integrator) can detect
      the state and the long/flat rule should earn a real timing premium.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # Latent accumulation state: a slow AR(1) (half-life ~ a couple of weeks).
    phi = 0.93
    a = np.zeros(n_days)
    shock = rng.normal(0.0, 1.0, n_days)
    for i in range(1, n_days):
        a[i] = phi * a[i - 1] + np.sqrt(1.0 - phi**2) * shock[i]

    # Forward drift driven by *yesterday's* accumulation state (volume leads price).
    eps = rng.normal(0.0, daily_vol, n_days)
    drift = np.zeros(n_days)
    if edge != 0.0:
        drift[1:] = edge * daily_vol * a[:-1]
    log_ret = drift + eps

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    # Volume rises with the magnitude of the accumulation state (the thing Klinger
    # tries to read). Floor + state-proportional component + noise.
    base_vol = 6_000_000.0
    vol = base_vol * (1.0 + 0.6 * np.abs(a)) * rng.lognormal(0.0, 0.15, n_days)

    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "edge": edge,
        "annual_vol": annual_vol,
        "n_days": n_days,
        "phi": phi,
        "seed": seed,
    }
    return bars, truth


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily OHLCV, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def _download(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import time

    import yfinance as yf  # lazy: only when we actually go to the network

    last_err = None
    for attempt in range(3):
        try:
            raw = yf.download(
                ticker, start=start, end=end, interval="1d",
                auto_adjust=True, progress=False, threads=False,
            )
            if raw is not None and not raw.empty:
                return raw
        except Exception as exc:  # pragma: no cover - network path
            last_err = exc
        time.sleep(1.5 * (attempt + 1))
    if last_err is not None:  # pragma: no cover
        raise last_err
    raise RuntimeError(f"yfinance returned no daily bars for {ticker}")


def fetch_panel(
    ticker: str = HEADLINE,
    start: str = "2005-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-first.

    Reads the cached parquet when present; only goes to the network on an explicit
    cache miss with ``fetch=True`` (with a small retry/backoff), then caches the result.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        bars = pd.read_parquet(path)
    elif fetch:
        raw = _download(ticker, start, end)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)
    else:
        raise FileNotFoundError(
            f"No cached daily tape for {ticker} at {path}. "
            f"Call fetch_panel({ticker!r}, fetch=True) once to populate the cache."
        )

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars.dropna()


def load_real(
    ticker: str = HEADLINE,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Convenience: cached daily OHLCV for ``ticker`` (cache-only, no network)."""
    return fetch_panel(ticker, fetch=False, cache_dir=cache_dir)


def have_real(ticker: str = HEADLINE, cache_dir: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(_cache_path(ticker, cache_dir))


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
