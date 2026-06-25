"""Data layer for Study 478 (Percentage Price Oscillator).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  The PPO buys when the fast/slow EMA spread crosses above its own 9-EMA signal line — i.e.
  it is a **momentum / trend-following** crossover. The believers' claim is that a bullish
  crossover is followed by *continuation* (a real upward drift after the cross). We plant
  exactly that: with ``edge > 0`` the path is given short bursts of positive momentum that
  *begin near* a PPO-up-cross, so a crossover entry harvests a real continuation; with
  ``edge = 0`` the log-return series is a pure random walk and the crossover is a fair coin.
  This is the positive control — a harness that cannot bank the planted continuation proves
  nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the PPO and its
signal are read on the close of *t* (a crossover needs *t* and *t-1*), and the trade is
entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs PPO/MACD proponents trade: the broad tape, big-cap tech, small caps, and a
# couple of cross-asset charts. Daily, liquid, long history. (Same 5 as the desk's idiom so
# the study runs fully offline from the cached parquets.)
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    start: str = "2010-01-04",
    seed: int = 478,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of post-crossover momentum.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that we plant a PPO-respecting force: we keep the *same* fast/slow/signal EMA
    machinery the strategy uses, and whenever the PPO crosses **above** its signal line we
    inject a small persistent positive drift for the following window (a continuation), scaled
    by ``edge`` (and a symmetric negative drift after a bearish cross). At ``edge = 0`` the
    tape is a pure martingale and a crossover entry is a fair coin; at ``edge > 0`` a bullish
    crossover is followed by a real continuation the detector should bank.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    af, asl, asig = 2.0 / (fast + 1), 2.0 / (slow + 1), 2.0 / (signal + 1)
    close = np.empty(n_days)
    log_p = np.log(100.0)
    ema_f = ema_s = log_p
    ppo = 0.0
    sig = 0.0
    prev_diff = 0.0           # PPO - signal one step back
    boost = 0.0               # remaining planted drift to apply this bar
    boost_left = 0            # bars of planted drift remaining
    hold = 20                 # how long a planted continuation lasts

    for i in range(n_days):
        eps = rng.normal(0.0, daily_vol)
        drift = boost if boost_left > 0 else 0.0
        log_p += eps + drift
        if boost_left > 0:
            boost_left -= 1
        close[i] = np.exp(log_p)

        # update EMAs on the (log) price, then PPO and its signal — exactly the rule's geometry
        ema_f += af * (log_p - ema_f)
        ema_s += asl * (log_p - ema_s)
        ppo_new = 100.0 * (ema_f - ema_s) / abs(ema_s) if ema_s != 0 else 0.0
        sig_new = sig + asig * (ppo_new - sig)
        diff = ppo_new - sig_new
        # a fresh crossover this bar?
        if edge > 0.0 and i > slow + signal:
            if prev_diff <= 0.0 < diff:          # bullish cross -> plant a continuation up
                boost = edge * daily_vol
                boost_left = hold
            elif prev_diff >= 0.0 > diff:        # bearish cross -> plant a continuation down
                boost = -edge * daily_vol
                boost_left = hold
        ppo, sig, prev_diff = ppo_new, sig_new, diff

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
    truth = {"edge": edge, "annual_vol": annual_vol, "fast": fast, "slow": slow,
             "signal": signal, "n_days": n_days, "seed": seed}
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
