"""Data layer for Study 488 (FRAMA — Fractal Adaptive Moving Average).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  FRAMA is a moving average whose smoothing constant adapts to the **fractal dimension** of
  recent price: in a clean trend (low fractal dimension) it speeds up and hugs price; in chop
  (high fractal dimension) it slows down and flattens. The believers' claim is that this
  adaptive smoothing lets a ``price > FRAMA`` (or FRAMA-cross) long **catch trends sooner and
  sit out chop**, banking real momentum. We plant exactly that: with ``edge > 0`` the path
  contains *persistent runs* — a slowly-switching trend regime whose drift the price follows —
  so that being long while ``price > FRAMA`` harvests genuine continuation; with ``edge = 0``
  the log-return series is a pure random walk and ``price > FRAMA`` is a fair coin. This is the
  positive control — a harness that cannot bank the planted momentum proves nothing by finding
  nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the FRAMA is a
strictly causal recursion (each bar uses only bars up to ``t``), the ``price > FRAMA`` state is
read on the close of *t*, and the trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs FRAMA proponents apply it to: the broad tape, big-cap tech, small caps, and a
# couple of cross-asset charts. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    frama_n: int = 16,
    flip_p: float = 0.01,
    start: str = "2010-01-04",
    seed: int = 488,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of trend-following edge.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that we plant a regime-switching **drift** that FRAMA is designed to ride: a
    hidden state flips slowly between an up-trend and a down-trend, and while a trend persists a
    small ``edge``-scaled drift is added each day. In a trend the path is smooth (low fractal
    dimension) — exactly the regime FRAMA speeds up for — so a ``price > FRAMA`` long banks the
    continuation. At ``edge = 0`` there is no drift, the tape is a pure martingale, and the
    ``price > FRAMA`` state is a fair coin; at ``edge > 0`` the long harvests a real trend.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    log_p = np.log(100.0)
    # Hidden trend state: +1 up, -1 down. Flips with small daily probability => long persistent
    # runs (smooth, low-fractal-dimension regimes). The planted drift is edge*daily_vol*state.
    state = 1
    # ``flip_p`` per-bar regime-switch probability => persistent (smooth, low-fractal-D) runs.
    for i in range(n_days):
        if rng.random() < flip_p:
            state = -state
        drift = edge * daily_vol * state if edge > 0.0 else 0.0
        eps = rng.normal(0.0, daily_vol)
        log_p += drift + eps
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
    truth = {"edge": edge, "annual_vol": annual_vol, "frama_n": frama_n,
             "flip_p": flip_p, "n_days": n_days, "seed": seed}
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
