"""Data layer for Study 428 (Stochastic RSI).

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date):

- ``synthetic_panel`` — a *deterministic, offline* generator with a planted-edge knob.
  ``mean_rev = 0`` is a pure random walk (a fair coin); ``mean_rev > 0`` plants short-term
  mean-reversion — the only structure the Stochastic-RSI overbought/oversold rule can
  possibly harvest. This is the study's null in a bottle plus its positive control.
- ``load_real`` — the real Yahoo! daily tape (``yfinance``), cache-first by default so the
  reproducible core and tests never touch the network unless a cache miss forces a fetch.

The Stochastic RSI (Chande & Kroll, 1994) applies the Stochastic-oscillator transform to
the RSI series instead of to price::

    RSI(n)            = Wilder's Relative Strength Index over n bars
    StochRSI(t)       = (RSI_t - min(RSI, k)) / (max(RSI, k) - min(RSI, k))
    %K                = SMA_smoothK(StochRSI)        (scaled to 0..100)
    %D                = SMA_smoothD(%K)

The folk rule: StochRSI %K below 20 = "oversold" (buy / go long), above 80 = "overbought"
(exit / go short). The pitch is that the double-normalisation makes it "more sensitive" than
plain RSI — i.e. it should *add* signal over RSI alone. That is the claim this study tests.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (signals are
formed on closes up to *t*; positions earn the return of *t+1*).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The real-tape basket: SPY is the headline; a small panel cross-checks robustness.
PANEL = ["SPY", "QQQ", "IWM", "DIA", "EFA"]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (null + positive control)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 3000,
    mean_rev: float = 0.0,
    daily_vol: float = 0.011,
    drift: float = 0.0003,
    start: str = "2008-01-02",
    seed: int = 428,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a tunable mean-reversion knob.

    Log returns follow a bar-level AR(1) around a drift::

        r_t = drift + (-mean_rev) * (r_{t-1} - drift) + eps_t

    where ``eps_t`` is i.i.d. normal with standard deviation ``daily_vol``. A *negative*
    serial correlation (positive ``mean_rev``) is the structure an oversold-buy /
    overbought-exit oscillator is designed to harvest.

    - ``mean_rev = 0``   → a drifting martingale; the StochRSI signal is a fair coin.
    - ``mean_rev > 0``   → mean-reverting tape the oversold-buy rule can ride (positive control).
    - ``mean_rev < 0``   → trending tape the mean-reversion rule is on the *wrong* side of.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters. The drift is
    kept small and positive so the long-flat rule has a realistic equity-like backdrop.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)

    eps = rng.normal(0.0, daily_vol, n_days)
    log_ret = np.empty(n_days)
    prev = drift
    for i in range(n_days):
        r = drift + (-mean_rev) * (prev - drift) + eps[i]
        log_ret[i] = r
        prev = r

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 80_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(cal, name="date"),
    )
    truth = {
        "mean_rev": mean_rev,
        "daily_vol": daily_vol,
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
    return os.path.join(cache_dir, f"srsi_{safe}_1d.parquet")


def load_real(
    ticker: str = "SPY",
    period: str = "max",
    cache_dir: str = DEFAULT_CACHE,
    force_fetch: bool = False,
    retries: int = 3,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker`` — cache-first.

    Reads the cached parquet if present (offline). On a cache miss (or ``force_fetch``)
    it goes to Yahoo via ``yfinance``, retrying a couple of times with a small backoff if
    rate-limited, then caches the parquet so re-runs are offline. Auto-adjusted (total
    return) bars.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path) and not force_fetch:
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
                    break
            except Exception as exc:  # pragma: no cover - network path
                last_err = exc
            time.sleep(1.5 * (attempt + 1))
        if bars is None or bars.empty:
            raise RuntimeError(
                f"yfinance returned no daily bars for {ticker} ({last_err})"
            )
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    return bars.dropna()


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
