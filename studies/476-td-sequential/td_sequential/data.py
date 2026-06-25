"""Data layer for Study 476 (TD Sequential — DeMark 9-13 exhaustion count).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  TD Sequential's defining geometry is a run of nine consecutive closes each below the close
  four bars earlier (a TD Buy Setup); the believers' claim is that *once that exhaustion count
  completes, price reverts upward*. We plant exactly that: with ``edge > 0`` the path receives
  a real upward bounce **conditioned on the down-streak count** — the longer the run of "close
  below close-4" bars, the stronger the upward pull, so a completed-setup entry harvests a real
  reversal; with ``edge = 0`` the log-return series is a pure random walk and the setup-9 entry
  is a fair coin. This is the positive control — a harness that cannot bank the planted bounce
  proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the setup/countdown
are read on the close of *t* (each rung uses only closes/lows at or before *t*), and the trade
is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs TD-Sequential proponents run on: the broad tape, big-cap tech, small caps,
# the Dow proxy and gold. Daily, liquid, long history. Reused so the study runs offline.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    start: str = "2010-01-04",
    seed: int = 476,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of TD-exhaustion mean reversion.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that we plant a TD-respecting force: we track the running count of consecutive
    bars whose close is below the close four bars earlier (the TD Buy-Setup streak). When a
    **completed setup** (a 9-count, or a sell-side 9-count) is detected, we inject a real upward
    (resp. downward) bounce into the *next several bars* — an exhaustion reversal conditioned
    exactly on the count completing. The pull is applied *after* completion so it does not stop
    the streak from reaching 9 first. At ``edge = 0`` the tape is a pure martingale and a
    setup-9 entry is a fair coin; at ``edge > 0`` the completed setup is followed by a real
    reversal the detector should bank.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    log_p = np.zeros(n_days)
    log_p[0] = np.log(100.0)
    eps = rng.normal(0.0, daily_vol, n_days)

    down_streak = 0  # consecutive "close < close[-4]"
    up_streak = 0    # consecutive "close > close[-4]"
    bounce = 0       # remaining bars of a planted post-completion reversal (signed)
    bounce_dir = 0.0
    for i in range(1, n_days):
        pull = 0.0
        if edge > 0.0 and bounce > 0:
            pull = bounce_dir * edge * daily_vol  # post-completion exhaustion reversal
            bounce -= 1
        log_p[i] = log_p[i - 1] + eps[i] + pull
        if i >= 4:
            if log_p[i] < log_p[i - 4]:
                down_streak += 1
                up_streak = 0
            elif log_p[i] > log_p[i - 4]:
                up_streak += 1
                down_streak = 0
            else:
                down_streak = up_streak = 0
            # arm a reversal the bar AFTER a setup completes (9 lower / 9 higher closes)
            if edge > 0.0:
                if down_streak == 9:
                    bounce, bounce_dir = 12, +1.0
                elif up_streak == 9:
                    bounce, bounce_dir = 12, -1.0

    close = np.exp(log_p)
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "n_days": n_days, "seed": seed}
    return bars, truth


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def load_real(
    ticker: str = "SPY",
    start: str = "2005-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    allow_fetch: bool = True,
) -> pd.DataFrame:
    """Real daily OHLC for ``ticker``; **cache-first** (network only on a cache miss).

    Reads a cached parquet if present. Otherwise — and only if ``allow_fetch`` — downloads
    from yfinance (with a couple of retries + back-off on rate limits) and caches the parquet,
    so every subsequent call is fully offline.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        bars = pd.read_parquet(path)
    elif allow_fetch:
        bars = _download(ticker, start, end)
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)
    else:
        raise FileNotFoundError(
            f"No cached daily tape for {ticker} at {path}. "
            f"Call load_real({ticker!r}) once (network) to populate the cache."
        )

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars[["open", "high", "low", "close"]]


def _download(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf  # lazy: only on a real cache miss

    last_err = None
    for attempt in range(3):
        try:
            raw = yf.download(ticker, start=start, end=end, interval="1d",
                              auto_adjust=True, progress=False)
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                bars = raw.rename(columns=str.lower)[["open", "high", "low", "close"]]
                bars.index.name = "date"
                return bars
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"yfinance returned no daily bars for {ticker}: {last_err}")


def have_real(tickers: list[str] | None = None, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every cached parquet for ``tickers`` is present (offline-safe check)."""
    tickers = tickers or DEFAULT_TICKERS
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in tickers)


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
