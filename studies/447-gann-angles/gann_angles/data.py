"""Data layer for Study 447 — Gann Angles (the 1x1 fixed-slope line).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a planted-edge knob.
  ``edge = 0`` is a pure geometric random walk: a Gann 1x1 line drawn from any pivot
  carries **no** forecast power (the honest null). ``edge > 0`` plants a literal
  "line-respecting" structure — the price is nudged back toward a moving 1x1 reference
  whenever it strays, so the harness can *prove* it would detect a real Gann effect if
  one existed. The knob is the positive control, never market evidence.

- ``load_real`` — the real Yahoo daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet under ``_cache/`` and only goes to the network on a cache miss (with a
  couple of back-off retries), then caches the result so re-runs are fully offline.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the 1x1 line
is anchored at confirmed *past* pivots, the regime is read at the close of *t*, and the
position earns the return of *t+1* (one documented execution lag).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The instruments a Gann practitioner would point a 1x1 line at: the broad US market
# (SPY), the index Gann himself charted (the Dow, ^DJI), and two high-beta single names
# whose big swings give the angle plenty of pivots to anchor on.
DEFAULT_TICKERS = ("SPY", "^DJI", "AAPL", "GLD")


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 4000,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    leg_len: int = 120,
    seed: int = 447,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *planted* Gann-1x1-respecting edge.

    The market alternates **up-legs and down-legs** of ~``leg_len`` bars each. When
    ``edge > 0`` the regime is genuinely persistent and trendy: an up-leg drifts at
    ``+edge`` per day, a down-leg at ``-edge`` per day, so the 1x1 line drawn from each
    leg-starting **pivot low** sits *under* price during the up-leg (close above the line)
    and price falls below it during the down-leg — exactly the support/resistance Gann
    claimed, and a structure a line-timing rule **can** harvest while buy-and-hold (which
    eats the down-legs) cannot. With ``edge = 0`` the legs have zero drift: a pure random
    walk segmented only cosmetically, where no angle rule beats holding the tape and the
    detector must NOT manufacture significance.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252.0)
    sessions = pd.bdate_range(start="2008-01-02", periods=n_days)

    # alternating regime: +1 up-leg, -1 down-leg, each ~leg_len bars (jittered)
    regime = np.empty(n_days)
    t, sign = 0, 1
    while t < n_days:
        L = max(20, int(rng.normal(leg_len, leg_len * 0.25)))
        regime[t:t + L] = sign
        t += L
        sign = -sign

    rets = rng.normal(0.0, daily_vol, n_days) + edge * regime
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
    truth = {"edge": edge, "annual_vol": annual_vol, "leg_len": leg_len,
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
