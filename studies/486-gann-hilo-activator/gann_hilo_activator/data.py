"""Data layer for Study 486 (Gann Hi-Lo Activator).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  The Gann Hi-Lo Activator is a trailing stop-and-reverse line: when price closes **above** it
  the rule flips long and the activator trails the SMA of recent lows. The believers' claim is
  that the **up-flip forecasts trend** — once price flips above the activator, the move
  *continues*. We plant exactly that: with ``edge > 0`` the path is given a persistent upward
  *continuation* drift for a few days right after an up-flip (and a symmetric downward push
  after a down-flip), so a flip-entry harvests a real trend; with ``edge = 0`` the log-return
  series is a pure random walk and the flip is a fair coin. This is the positive control — a
  harness that cannot bank the planted trend proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the activator at
bar *t* uses only highs/lows through *t* (the SMA is shifted one bar so the flip is read on the
close of *t*), and the trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs Gann-tool proponents draw on: the broad tape, big-cap tech, small caps, and a
# couple of cross-asset charts. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    period: int = 10,
    start: str = "2010-01-04",
    seed: int = 486,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of Gann-flip trend-continuation.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that we plant a *flip-respecting* force: we keep a rolling Gann Hi-Lo Activator
    (period ``period``) on the path so far, and whenever the close has just *flipped above* the
    activator we inject a small persistent **upward** continuation drift for the next handful of
    bars (and a symmetric downward drift right after a flip below). At ``edge = 0`` the tape is a
    pure martingale and a flip-entry is a fair coin; at ``edge > 0`` an up-flip is followed by a
    real trend the detector should bank.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)

    close = np.empty(n_days)
    high = np.empty(n_days)
    low = np.empty(n_days)
    log_p = np.log(100.0)
    prev_close = np.exp(log_p)

    # rolling buffers for the SMA-of-highs / SMA-of-lows activator
    hi_buf: list[float] = []
    lo_buf: list[float] = []
    regime = 1            # +1 long (trail SMA of lows), -1 short (trail SMA of highs)
    activator = np.nan
    continuation = 0      # remaining days of planted post-flip drift
    cont_sign = 0

    for i in range(n_days):
        # planted continuation drift bleeds off over a few bars after a flip
        pull = 0.0
        if edge > 0.0 and continuation > 0:
            pull = cont_sign * edge * daily_vol
            continuation -= 1

        eps = rng.normal(0.0, daily_vol)
        log_p += eps + pull
        c = np.exp(log_p)
        # a plausible OHLC bar around the open (= prev close) and the new close
        o = prev_close
        wick = abs(rng.normal(0.0, daily_vol * 0.5)) * c
        h = max(o, c) + wick
        lo_ = min(o, c) - wick

        close[i] = c
        high[i] = h
        low[i] = lo_

        # update the activator from bars strictly before the *next* flip check (no look-ahead:
        # we use highs/lows through bar i to decide the regime that price at i is compared to)
        if not np.isnan(activator):
            new_regime = regime
            if regime == 1 and c < activator:
                new_regime = -1
            elif regime == -1 and c > activator:
                new_regime = +1
            if new_regime != regime and edge > 0.0:
                continuation = 20
                cont_sign = +1 if new_regime == 1 else -1
            regime = new_regime

        hi_buf.append(h)
        lo_buf.append(lo_)
        if len(hi_buf) > period:
            hi_buf.pop(0)
            lo_buf.pop(0)
        if len(hi_buf) == period:
            sma_hi = float(np.mean(hi_buf))
            sma_lo = float(np.mean(lo_buf))
            activator = sma_lo if regime == 1 else sma_hi

        prev_close = c

    bars = pd.DataFrame(
        {"open": np.concatenate([[100.0], close[:-1]]), "high": high, "low": low, "close": close},
        index=pd.DatetimeIndex(pd.bdate_range(start=start, periods=n_days), name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "period": period,
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
