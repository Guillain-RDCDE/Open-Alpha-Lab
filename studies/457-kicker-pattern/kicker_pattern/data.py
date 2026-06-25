"""Data layer for Study 457 (Kicker-Pattern).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  A kicker is a gap-and-reverse marubozu pair: an opposite-colour pair of near-bodyless-wick
  candles separated by a gap in the *new* direction. The believers' claim is that the kicker
  marks a violent, reliable turn — price keeps going the kicker's way. We plant exactly that:
  with ``edge > 0`` we sprinkle genuine kicker formations into the path and pull the subsequent
  days in the kicker's direction, so a kicker-direction entry harvests a real continuation; with
  ``edge = 0`` the log-return series is a pure random walk and any kicker that happens to form
  by chance is a fair coin. This is the positive control — a harness that cannot bank the planted
  continuation proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: a kicker is
*completed* on the close of the second marubozu (bar ``t``), and the trade is entered at
``t+1``'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs kicker-pattern proponents draw on: the broad tape, big-cap tech, small
# caps, and a couple of cross-asset charts. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    start: str = "2010-01-04",
    seed: int = 457,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of kicker-continuation.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that, with probability ``0.03`` per day we *plant a kicker formation*: we
    manufacture two opposite, near-marubozu candles with a gap in the new direction. The
    formation is **always** planted (so the detector has a sample to test at every edge); the
    *continuation* is what the edge controls. With ``edge = 0`` the days after a planted kicker
    are a pure random walk, so the kicker-direction entry is a **fair coin** (t ≈ 0, win ≈ 50%
    — no false positive). With ``edge > 0`` we push the following ``DRIFT_DAYS`` sessions in the
    kicker's direction proportional to ``edge``, so a real continuation follows each kicker,
    which the detector should bank.

    We build *clean marubozu candles directly* (open near one extreme, close near the other,
    tiny wicks) so the planted formations are detectable; the surrounding bars are ordinary
    walk bars with realistic wicks. Returns ``(bars, truth)``; ``truth`` records the planted
    parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    DRIFT_DAYS = 6
    GAP = 0.8 * daily_vol          # gap size in log units (small: no artificial displacement)
    BODY = 1.2 * daily_vol         # marubozu body in log units

    open_ = np.empty(n_days)
    high = np.empty(n_days)
    low = np.empty(n_days)
    close = np.empty(n_days)

    log_p = np.log(100.0)
    drift_left = 0          # remaining sessions of planted continuation
    drift_dir = 0
    i = 0
    while i < n_days:
        # planted continuation push (after a planted kicker)
        push = 0.0
        if drift_left > 0:
            push = drift_dir * edge * daily_vol * 2.0
            drift_left -= 1

        # decide whether to PLANT a kicker pair starting here (needs room for 2 bars + drift).
        # The FORMATION is always planted (the detector needs a sample at every edge); only the
        # continuation push depends on edge. At edge=0 the post-kicker path is a fair coin.
        plant = (drift_left == 0 and i + 1 < n_days and rng.random() < 0.03)

        if plant:
            kdir = 1 if rng.random() < 0.5 else -1
            # --- bar i: marubozu in the OPPOSITE direction (the "prior" candle) ---
            o0 = np.exp(log_p)
            log_c0 = log_p - kdir * BODY               # opposite-colour body
            c0 = np.exp(log_c0)
            open_[i] = o0
            close[i] = c0
            wick0 = abs(rng.normal(0, daily_vol * 0.1)) * o0
            high[i] = max(o0, c0) + wick0
            low[i] = min(o0, c0) - wick0
            # --- bar i+1: GAP in the new (kicker) direction, then marubozu in kicker dir ---
            # gap relative to bar i's OPEN (kicker definition gaps past the prior open)
            log_o1 = log_p + kdir * GAP                # gap up (bull) / down (bear) past o0
            o1 = np.exp(log_o1)
            log_c1 = log_o1 + kdir * BODY              # kicker-colour body
            c1 = np.exp(log_c1)
            open_[i + 1] = o1
            close[i + 1] = c1
            wick1 = abs(rng.normal(0, daily_vol * 0.1)) * o1
            high[i + 1] = max(o1, c1) + wick1
            low[i + 1] = min(o1, c1) - wick1
            log_p = log_c1
            drift_dir = kdir
            drift_left = DRIFT_DAYS
            i += 2
            continue

        # ordinary walk bar
        eps = rng.normal(0.0, daily_vol)
        o = np.exp(log_p)
        log_p += eps + push
        c = np.exp(log_p)
        wick = abs(rng.normal(0.0, daily_vol * 0.5)) * c
        open_[i] = o
        close[i] = c
        high[i] = max(o, c) + wick
        low[i] = min(o, c) - wick
        i += 1

    bars = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "n_days": n_days, "seed": seed,
             "drift_days": DRIFT_DAYS}
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
