"""Data layer for Study 471 (QQE — Quantitative Qualitative Estimation).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**
  specific to the QQE rule. QQE fires a long when the smoothed RSI (RSI MA) crosses *above*
  its ATR-of-RSI trailing band — the believers read this as a **momentum ignition** that is
  followed by continuation. We plant exactly that: with ``edge > 0`` the path receives a short
  burst of positive drift right after a genuine QQE band-cross, so a QQE-cross entry harvests a
  real continuation; with ``edge = 0`` the log-return series is a pure random walk and the
  cross is a fair coin. This is the positive control — a harness that cannot bank the planted
  continuation proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the QQE band is
built from a causal Wilder smoothing of RSI, the band-cross is detected on the close of *t*,
and the trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs QQE proponents trade on: the broad tape, big-cap tech, small caps, and a
# couple of cross-asset charts. Daily, liquid, long history. (Reused from the desk basket so
# the study runs fully offline against the shared cache.)
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    rsi_len: int = 14,
    start: str = "2010-01-04",
    seed: int = 471,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of QQE-cross continuation.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that we plant a QQE-respecting force: we keep a *causal* Wilder-smoothed RSI and
    its ATR-of-RSI trailing band, and whenever the smoothed RSI crosses *above* the band (the
    QQE long trigger) we inject a short burst of positive drift over the following few bars,
    proportional to ``edge``. At ``edge = 0`` the tape is a pure martingale and a QQE-cross
    entry is a fair coin; at ``edge > 0`` a band-cross is followed by a real upward continuation
    that the detector should bank.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    log_p = np.log(100.0)
    prev_p = np.exp(log_p)

    # Causal Wilder RSI state (avg gain / avg loss) and the QQE smoothing/band state.
    alpha = 1.0 / rsi_len
    avg_gain = 0.0
    avg_loss = 0.0
    rsi_ma = 50.0          # smoothed RSI (RSI MA)
    rsi_ma_prev = 50.0
    atr_rsi = 0.0          # ATR of the smoothed RSI (band half-width driver)
    long_band = 0.0
    short_band = 100.0
    prev_long_band = 0.0
    prev_short_band = 100.0
    trend = 1
    qqe_factor = 4.236     # Wilder's QQE multiplier (the canonical default)

    boost = np.zeros(n_days)   # planted continuation drift queue
    started = False

    for i in range(n_days):
        # --- planted continuation drift from any earlier band-cross
        pull = boost[i] if edge > 0.0 else 0.0

        eps = rng.normal(0.0, daily_vol)
        log_p += eps + pull
        cur_p = np.exp(log_p)
        close[i] = cur_p

        # --- update causal Wilder RSI from the realised price change
        change = cur_p - prev_p
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        if not started:
            avg_gain, avg_loss = gain, loss
            started = True
        else:
            avg_gain = (1 - alpha) * avg_gain + alpha * gain
            avg_loss = (1 - alpha) * avg_loss + alpha * loss
        rs = avg_gain / avg_loss if avg_loss > 0 else (np.inf if avg_gain > 0 else 1.0)
        rsi = 100.0 - 100.0 / (1.0 + rs) if np.isfinite(rs) else 100.0

        # --- QQE smoothing of RSI + ATR-of-RSI trailing band (Wilder smoothing)
        rsi_ma_prev = rsi_ma
        rsi_ma = (1 - alpha) * rsi_ma + alpha * rsi          # smoothed RSI
        dar = abs(rsi_ma - rsi_ma_prev)
        atr_rsi = (1 - alpha) * atr_rsi + alpha * dar        # ATR of smoothed RSI
        delta = atr_rsi * qqe_factor

        prev_long_band = long_band
        prev_short_band = short_band
        new_long = rsi_ma - delta
        new_short = rsi_ma + delta
        # trailing dual bands (mirror of strategy.qqe_bands)
        if rsi_ma_prev > prev_long_band and rsi_ma > prev_long_band:
            long_band = max(prev_long_band, new_long)
        else:
            long_band = new_long
        if rsi_ma_prev < prev_short_band and rsi_ma < prev_short_band:
            short_band = min(prev_short_band, new_short)
        else:
            short_band = new_short
        prev_trend = trend
        if rsi_ma > prev_short_band:
            trend = 1
        elif rsi_ma < prev_long_band:
            trend = -1

        # --- QQE long trigger: trailing stop flips up (RSI MA crosses above the stop)
        crossed_up = (prev_trend == -1) and (trend == 1)
        if edge > 0.0 and crossed_up and i + 1 < n_days:
            horizon = 6
            for j in range(i + 1, min(i + 1 + horizon, n_days)):
                boost[j] += edge * daily_vol * 2.5

        prev_p = cur_p

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
    truth = {"edge": edge, "annual_vol": annual_vol, "rsi_len": rsi_len,
             "n_days": n_days, "seed": seed}
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
