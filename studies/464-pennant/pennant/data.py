"""Data layer for Study 464 (Pennant).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  A pennant is a steep **pole** (a strong directional thrust), then a tight **converging
  consolidation** (a small symmetrical triangle on shrinking range), then a **breakout** in
  the pole's direction. The believers' claim is that the breakout *continues the prior thrust*
  — the post-breakout move runs in the pole direction. We plant exactly that: with ``edge > 0``
  the generator periodically lays down a pole, a shrinking-range pause, and then a genuine
  continuation drift **in the pole's direction**, so a breakout-in-pole-direction entry harvests
  a real continuation; with ``edge = 0`` the path is a pure random walk and a "breakout" after a
  consolidation is a fair coin (50/50 up or down). This is the positive control — a harness that
  cannot bank the planted continuation proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the pole and the
converging triangle are read on bars up to and including *t*, the breakout is detected on the
close of *t*, and the trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs pennant proponents draw on: the broad tape, big-cap tech, small caps, and a
# couple of cross-asset charts. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    pole_len: int = 8,
    pause_len: int = 12,
    cont_len: int = 20,
    start: str = "2010-01-04",
    seed: int = 464,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of pennant continuation.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that, with ``edge > 0`` the generator periodically lays down a textbook pennant:
    (1) a steep **pole** — ``pole_len`` bars of strong drift in a randomly chosen direction;
    (2) a **converging pause** — ``pause_len`` bars whose intraday range shrinks toward a point
    while the close oscillates with damped amplitude (a symmetrical triangle); (3) a
    **continuation** — ``cont_len`` bars of drift *in the pole's direction*, proportional to
    ``edge``. At ``edge = 0`` none of this is planted: the path is a pure martingale and any
    detected breakout is a fair coin. At ``edge > 0`` a breakout in the pole direction is
    followed by a real continuation that the detector should bank.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    wick_amp = np.full(n_days, daily_vol * 0.5)   # per-bar wick scale (shrinks during a pause)
    log_p = np.log(100.0)
    cycle = pole_len + pause_len + cont_len
    pole_dir = 0      # +1 / -1 during a planted pennant, 0 otherwise
    phase = "walk"    # walk | pole | pause | cont
    phase_i = 0
    next_event = rng.integers(20, 40)  # bars until the first planted pole (edge>0 only)

    for i in range(n_days):
        eps = rng.normal(0.0, daily_vol)
        drift = 0.0
        if edge > 0.0:
            if phase == "walk":
                if i >= next_event:
                    phase, phase_i = "pole", 0
                    pole_dir = 1 if rng.random() < 0.5 else -1
            if phase == "pole":
                drift = pole_dir * edge * daily_vol * 6.0   # steep thrust
                phase_i += 1
                if phase_i >= pole_len:
                    phase, phase_i = "pause", 0
            elif phase == "pause":
                # damped close-to-close move AND damped wick -> a converging triangle
                shrink = max(1.0 - phase_i / max(pause_len - 1, 1), 0.12)
                eps = rng.normal(0.0, daily_vol * 0.45 * shrink)
                wick_amp[i] = daily_vol * 0.5 * shrink
                phase_i += 1
                if phase_i >= pause_len:
                    phase, phase_i = "cont", 0
            elif phase == "cont":
                # first continuation bar is a clean breakout thrust, then a steady run
                kick = 4.0 if phase_i == 0 else 1.5
                drift = pole_dir * edge * daily_vol * kick
                phase_i += 1
                if phase_i >= cont_len:
                    phase = "walk"
                    next_event = i + int(rng.integers(20, 40))
        log_p += eps + drift
        close[i] = np.exp(log_p)

    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, 1.0, close.size)) * wick_amp * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "pole_len": pole_len,
             "pause_len": pause_len, "cont_len": cont_len, "cycle": cycle,
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
