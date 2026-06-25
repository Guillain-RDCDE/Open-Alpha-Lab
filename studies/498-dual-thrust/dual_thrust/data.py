"""Data layer for Study 498 (Dual Thrust opening-range breakout).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  Dual Thrust is a breakout system: it buys when price clears ``open + k1*Range`` (an N-day
  high/low/close span). The believers' claim is that a break of the upper trigger is followed
  by *continuation* — the day keeps trending in the breakout direction. We plant exactly that:
  with ``edge > 0`` the path gets a forward momentum kick *on the days the close exceeds the
  upper trigger* (and a symmetric down-kick when it breaks the lower trigger), so a breakout
  entry harvests a real continuation; with ``edge = 0`` the log-return series is a pure random
  walk and the breakout is a fair coin. This is the positive control — a harness that cannot
  bank the planted continuation proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the Range uses the
*prior* N bars (high/low/close known by yesterday's close), the breakout is read on the close
of *t*, and the trade is entered at *t+1*'s close (one documented lag).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs Dual-Thrust breakout proponents trade: the broad tape, big-cap tech, small
# caps, the Dow, and a cross-asset chart. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    lookback: int = 5,
    start: str = "2010-01-04",
    seed: int = 498,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of breakout continuation.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that we plant a Dual-Thrust-respecting force: we keep a rolling ``Range`` from the
    last ``lookback`` bars (the span of HH-LC and HC-LL, Chalek's definition) and the trigger
    bands ``open ± k*Range``. Whenever the running close pierces the **upper** trigger we inject
    a small upward continuation kick on the *next* bar proportional to ``edge`` (and a symmetric
    downward kick when it pierces the **lower** trigger). At ``edge = 0`` the tape is a pure
    martingale and a breakout is a fair coin; at ``edge > 0`` an upper-trigger break is followed
    by a real up-move that the breakout entry should bank.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)
    k = 0.5  # symmetric trigger coefficient used for the planted force

    close = np.empty(n_days)
    open_ = np.empty(n_days)
    hi = np.empty(n_days)
    lo = np.empty(n_days)

    log_p = np.log(100.0)
    prev_close = np.exp(log_p)
    # rolling rings of recent highs/lows/closes for the Range
    hh = [prev_close] * lookback
    ll = [prev_close] * lookback
    cc = [prev_close] * lookback
    kick = 0.0  # carried continuation force from a prior-bar breakout

    for i in range(n_days):
        o = prev_close
        open_[i] = o
        # Range from the trailing window (Chalek): max(HH-LC, HC-LL)
        rng_val = max(max(hh) - min(cc), max(cc) - min(ll))
        rng_log = np.log1p(rng_val / o) if o > 0 else 0.0
        buy_line = o * np.exp(k * rng_log)
        sell_line = o * np.exp(-k * rng_log)

        eps = rng.normal(0.0, daily_vol)
        log_p += eps + kick
        c = np.exp(log_p)

        # detect breakout on this bar's close -> plant continuation for the NEXT bar
        kick = 0.0
        if edge > 0.0:
            if c > buy_line:
                kick = edge * daily_vol
            elif c < sell_line:
                kick = -edge * daily_vol

        wick = abs(rng.normal(0.0, daily_vol * 0.5)) * c
        hi[i] = max(o, c) + wick
        lo[i] = min(o, c) - wick
        close[i] = c

        hh = hh[1:] + [hi[i]]
        ll = ll[1:] + [lo[i]]
        cc = cc[1:] + [c]
        prev_close = c

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "lookback": lookback,
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
