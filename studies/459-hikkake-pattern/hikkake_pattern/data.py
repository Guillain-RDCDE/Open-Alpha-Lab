"""Data layer for Study 459 (Hikkake pattern — the false-breakout trap).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  A hikkake is an **inside bar** followed by a *false breakout* one way and then a snap back
  *through* the inside-bar range — a trap that supposedly forecasts a move in the reversal
  direction. We plant exactly that: with ``edge > 0`` every time the path pokes just outside a
  recent inside-bar range and then closes back inside it, we add a small drift **in the
  reversal direction** for the following few bars, so the hikkake entry harvests a real move;
  with ``edge = 0`` the log-return series is a pure random walk and the trap is a fair coin.
  This is the positive control — a harness that cannot bank the planted reversal proves nothing
  by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the inside bar and
the false breakout are confirmed on the close of *t* (the trap is complete by *t*), and the
trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs price-action / candlestick traders watch: the broad tape, big-cap tech, small
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
    seed: int = 459,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of hikkake-trap reversal.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that we plant a hikkake-respecting force: we track the most recent *inside bar*
    (a bar whose high/low sit inside the prior bar's range). Whenever the close pokes just
    **outside** that inside bar's range and then snaps **back inside** it — the textbook
    false-breakout trap — we add a small drift in the **reversal** direction (up after a false
    downside break, down after a false upside break) for the next few bars. At ``edge = 0`` the
    tape is a pure martingale and the trap is a fair coin; at ``edge > 0`` a completed hikkake
    is followed by a real move the detector should bank.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    high = np.empty(n_days)
    low = np.empty(n_days)
    open_ = np.empty(n_days)

    log_p = np.log(100.0)
    prev_close = np.exp(log_p)
    prev_hi = prev_close
    prev_lo = prev_close

    # rolling memory of the most recent inside bar's range (in price units) and the planted
    # reversal pull that a completed trap leaves behind for a few bars.
    inside_hi = np.nan
    inside_lo = np.nan
    pull_dir = 0.0      # +1 long, -1 short, 0 none
    pull_left = 0       # bars of planted drift remaining

    for i in range(n_days):
        pull = 0.0
        if edge > 0.0 and pull_left > 0:
            pull = pull_dir * edge * daily_vol * 8.0
            pull_left -= 1
        eps = rng.normal(0.0, daily_vol)
        log_p += eps + pull
        c = np.exp(log_p)
        o = prev_close
        wick = abs(rng.normal(0.0, daily_vol * 0.5)) * c
        hi = max(o, c) + wick
        lo = min(o, c) - wick

        open_[i] = o
        close[i] = c
        high[i] = hi
        low[i] = lo

        # detect a completed hikkake to arm the planted pull (uses only past/current bars).
        if edge > 0.0 and not np.isnan(inside_hi):
            # a false UPSIDE break that snapped back inside -> plant a DOWN move
            if (hi > inside_hi) and (c < inside_hi) and (c > inside_lo):
                pull_dir = -1.0
                pull_left = 4
                inside_hi = np.nan
            # a false DOWNSIDE break that snapped back inside -> plant an UP move
            elif (lo < inside_lo) and (c > inside_lo) and (c < inside_hi):
                pull_dir = +1.0
                pull_left = 4
                inside_hi = np.nan

        # update the "most recent inside bar" memory: is THIS bar inside the PRIOR bar?
        if (hi <= prev_hi) and (lo >= prev_lo):
            inside_hi = hi
            inside_lo = lo

        prev_close, prev_hi, prev_lo = c, hi, lo

    bars = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
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
