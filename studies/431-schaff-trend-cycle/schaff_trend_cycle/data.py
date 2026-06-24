"""Data layer for Study 431 (Schaff Trend Cycle).

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date):

- ``synthetic_panel`` — a *deterministic, offline* generator with a planted-edge knob.
  ``trend`` plants a slow regime-switching drift (the only structure a trend-timing
  oscillator can possibly harvest): the series spends stretches in an up regime and
  stretches in a down regime. ``trend = 0`` is a pure random walk — a fair coin — so the
  harness can assert the STC timing rule beats buy-and-hold *only* when a tradeable trend
  is actually planted, and not otherwise. This is the study's null in a bottle, and the
  positive control that proves the engine can bank a real edge.
- ``load_real`` — the real Yahoo! daily tape (``yfinance``), cache-first so the test-suite
  and the reproducible core never touch the network. Daily bars go back decades, giving
  high statistical power compared with intraday studies.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the indicator is
computed on closes up to bar *t*; the position is held over bar *t+1*'s return, i.e. one
documented execution lag).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The headline real-tape instrument and a tiny robustness panel.
TICKERS = ["SPY", "QQQ", "IWM", "EFA", "AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (null + positive control)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 4000,
    trend: float = 0.0,
    daily_vol: float = 0.011,
    regime_p: float = 0.008,
    start: str = "2008-01-02",
    seed: int = 431,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a tunable *trend* (planted-edge) knob.

    The log return is ``r_t = trend * s_t + eps_t`` where ``s_t in {+1, -1}`` is a slow
    two-state regime that flips with probability ``regime_p`` each bar (mean dwell time
    ``1 / regime_p`` bars), and ``eps_t`` is i.i.d. normal with standard deviation
    ``daily_vol``. A persistent regime drift is exactly the structure a trend-following
    oscillator (STC, MACD) is designed to ride.

    - ``trend = 0``   → a martingale; no regime drift, so any timing rule is a fair coin
                        and *cannot* beat buy-and-hold on a risk-adjusted basis.
    - ``trend > 0``   → tradeable persistent trends the STC long/flat rule can harvest.

    Bars are stamped on consecutive business days. Returns ``(bars, truth)`` where
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)

    # Slow two-state regime via persistent sign flips.
    flips = rng.random(n_days) < regime_p
    state = np.empty(n_days, dtype=float)
    s = 1.0
    for i in range(n_days):
        if flips[i]:
            s = -s
        state[i] = s

    eps = rng.normal(0.0, daily_vol, n_days)
    log_ret = trend * state + eps

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 50_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(cal, name="date"),
    )
    truth = {
        "trend": trend,
        "daily_vol": daily_vol,
        "regime_p": regime_p,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"stc_{safe}_1d.parquet")


def _download(ticker: str, period: str, retries: int = 3) -> pd.DataFrame:
    import yfinance as yf  # lazy: only when we actually go to the network

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                ticker, period=period, interval="1d",
                auto_adjust=True, progress=False,
            )
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                bars = raw.rename(columns=str.lower)[
                    ["open", "high", "low", "close", "volume"]
                ]
                bars.index.name = "date"
                return bars
        except Exception as exc:  # pragma: no cover - network path
            last_err = exc
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(
        f"yfinance returned no daily bars for {ticker} after {retries} tries"
        + (f" (last error: {last_err})" if last_err else "")
    )


def load_real(
    ticker: str = "SPY",
    period: str = "max",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-first.

    Reads the cached parquet under ``_cache/`` if present. On a cache miss (or
    ``fetch=True``) it pulls from Yahoo with a small retry/backoff, then caches the
    parquet so re-runs are fully offline. ``auto_adjust=True`` → total-return-adjusted
    closes (dividends + splits folded in); we label this explicitly as TR everywhere.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path) and not fetch:
        bars = pd.read_parquet(path)
    else:
        bars = _download(ticker, period)
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    return bars


def have_real(ticker: str = "SPY", cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff the cached real tape for ``ticker`` exists (offline-safe check)."""
    return os.path.exists(_cache_path(ticker, cache_dir))


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
