"""Data layer for Study 439 (Linear-Regression-Channel).

Two tapes, one shape (a tz-naive daily OHLCV frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a planted-edge knob.
  ``edge`` injects a slow, persistent trend component into log-returns that is *only*
  visible through a positive rolling-OLS slope — exactly the structure a regression-slope
  timing rule is supposed to harvest. At ``edge = 0`` the log-return series is a pure
  random walk (a fair coin), so the harness must find nothing; at ``edge > 0`` a faithful
  detector must light up. This is a machinery proof, never market evidence.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), cache-first by default so the
  test-suite and the reproducible core never touch the network. Daily history is long
  (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the slope/SMA
are computed on closes up to *t*, and the position they imply earns the return of *t+1*
(one ``shift``, applied once).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The small real panel the study ships a pinned run for.
PANEL = ["SPY", "QQQ", "AAPL", "MSFT", "JPM", "XLE"]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (with planted-edge knob)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 4000,
    edge: float = 0.0,
    annual_vol: float = 0.18,
    regime_len: int = 250,
    start: str = "2010-01-04",
    seed: int = 439,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a *known* amount of slope-harvestable trend.

    The price path is built from log-returns ``r_t = mu_t + eps_t`` where ``eps_t`` is
    i.i.d. normal (daily sigma = ``annual_vol / sqrt(252)``) and ``mu_t`` is a slow,
    sign-persistent drift that flips every ``regime_len`` bars. ``edge`` scales the drift:

    - ``edge = 0`` → ``mu_t = 0`` everywhere: a pure martingale. A regression-slope rule
      that is just chasing noise cannot beat buy-and-hold here.
    - ``edge > 0`` → persistent trends a *forward-looking* slope can ride: when the slope
      is positive the next-day drift is (on average) positive, and vice-versa. A faithful
      slope detector must extract this.

    The drift is built to be *detectable only with a lag-respecting slope*: it is the sign
    of a smooth regime variable, so a same-bar slope of past prices is genuinely predictive
    of the next bar's drift (no look-ahead is needed for the planted edge to be real).

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # A slow square-wave regime that flips sign every regime_len bars.
    n_reg = int(np.ceil(n_days / regime_len))
    signs = rng.choice([-1.0, 1.0], size=n_reg)
    regime = np.repeat(signs, regime_len)[:n_days]
    # Drift per bar: edge (annualised) split across the trading year, signed by regime.
    mu = edge / 252.0 * regime

    eps = rng.normal(0.0, daily_vol, n_days)
    log_ret = mu + eps

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(50_000, 500_000, close.size).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "edge": edge,
        "annual_vol": annual_vol,
        "regime_len": regime_len,
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


def load_real(
    ticker: str = "SPY",
    start: str = "2005-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-first (network only on ``fetch=True``).

    On a cache miss with ``fetch=True`` it pulls from yfinance with a couple of retries
    and a small backoff, then writes a parquet under ``_cache/`` so re-runs are offline.
    Auto-adjusted (total-return) closes — labeled as such everywhere downstream.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call load_real({ticker!r}, fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import time

        import yfinance as yf  # lazy: only when we actually go to the network

        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(
                    ticker, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is not None and not raw.empty:
                    break
            except Exception:  # noqa: BLE001 — retry transient network/rate-limit errors
                pass
            time.sleep(2.0 * (attempt + 1))
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
    return bars


def have_real(ticker: str = "SPY", cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff the cached real tape for ``ticker`` is present (no network)."""
    return os.path.exists(_cache_path(ticker, cache_dir))


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
