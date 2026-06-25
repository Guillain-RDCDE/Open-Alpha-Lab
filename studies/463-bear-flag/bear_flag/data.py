"""Data layer for Study 463 (Bear-Flag).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  A bear flag is a sharp drop (the *pole*) followed by a small up-sloping consolidation (the
  *flag*) and then a *breakdown* that "continues the drop". The believers' claim is that the
  flag **forecasts continuation** — the second leg down follows the breakout below the flag.
  We plant exactly that: with ``edge > 0`` every time the path traces pole→up-flag→breakdown,
  the next stretch is given a real downward push, so a short on the breakdown harvests a real
  continuation; with ``edge = 0`` the log-return series is a pure random walk and the
  breakdown is a fair coin. This is the positive control — a harness that cannot bank the
  planted continuation proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the pole and the
flag are read on the close of *t* (the flag's geometry uses only bars up to *t*), the
breakdown is detected on the close of *t*, and the short is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs bear-flag proponents trade: the broad tape, big-cap tech, small caps, and a
# couple of cross-asset charts. Daily, liquid, long history. (Reused so it runs offline.)
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    pole_drop: float = 0.06,
    flag_len: int = 6,
    start: str = "2010-01-04",
    seed: int = 463,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of bear-flag continuation.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that we plant a bear-flag-respecting force: periodically (on a schedule) we carve
    a **pole** (a sharp multi-bar drop of ``pole_drop`` in log units), then an **up-sloping
    flag** (``flag_len`` bars drifting gently *up* against the pole), and then — when ``edge >
    0`` — a real **continuation** down (a downward pull proportional to ``edge`` applied to the
    bars right after the flag). At ``edge = 0`` the pole/flag are still carved but no
    continuation is planted, so a breakdown short is a fair coin; at ``edge > 0`` the breakdown
    is followed by a real second leg down that the detector should bank.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    log_p = np.log(100.0)

    # Schedule of bear-flag episodes: every ~45 bars carve pole(4) + flag(flag_len) + cont(8).
    period = 45
    pole_len = 4
    cont_len = 8
    # episode phase for bar i: -1 idle, else step within the carved structure
    pole_step = pole_drop / pole_len           # per-bar pole drop (log)
    flag_up = (pole_drop * 0.4) / flag_len     # per-bar up-drift of the flag (retraces ~40%)

    for i in range(n_days):
        phase = i % period
        pull = 0.0
        if phase < pole_len:                                   # POLE — sharp drop
            pull = -pole_step
        elif phase < pole_len + flag_len:                      # FLAG — gentle up-slope
            pull = +flag_up
        elif phase < pole_len + flag_len + cont_len:           # CONTINUATION — planted leg down
            if edge > 0.0:
                pull = -edge * pole_step
        eps = rng.normal(0.0, daily_vol)
        log_p += eps + pull
        close[i] = np.exp(log_p)

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
    truth = {"edge": edge, "annual_vol": annual_vol, "pole_drop": pole_drop,
             "flag_len": flag_len, "n_days": n_days, "seed": seed}
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
