"""Data layer for Study 496 (Three-Line-Break).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  A Three-Line-Break chart reverses colour only after the close breaks the extremes of the
  three most-recent opposite lines; the believers' claim is that an **up-reversal forecasts a
  real new up-trend** (and a down-reversal a down-trend). We plant exactly that: with
  ``edge > 0`` the path is given a short burst of positive drift for a few sessions right after
  a genuine 3-line up-break (and negative drift after a down-break), so a TLB up-reversal entry
  harvests a real continuation; with ``edge = 0`` the log-return series is a pure random walk
  and the reversal is a fair coin. This is the positive control — a harness that cannot bank the
  planted continuation proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: a TLB reversal is
confirmed on the close of *t* (the break is a function of *past* lines only), and the trade is
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

# Indices / ETFs the TLB folklore is drawn on: the broad tape, big-cap tech, small caps, and a
# couple of cross-asset charts. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    n_lines: int = 3,
    start: str = "2010-01-04",
    seed: int = 496,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of TLB-reversal predictiveness.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that we plant a Three-Line-Break-respecting force: we keep a rolling TLB state
    (the colour and the extremes of the last ``n_lines`` lines). When the close *breaks* the
    three latest down-lines to flip the TLB to **up** (a genuine 3-line up-reversal), we inject a
    small positive drift for the next handful of sessions, proportional to ``edge`` (and a
    symmetric negative drift after a 3-line **down**-reversal). At ``edge = 0`` the tape is a pure
    martingale and a reversal is a fair coin; at ``edge > 0`` a reversal is followed by a real
    continuation the detector should bank.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    log_p = np.log(100.0)

    # Rolling TLB state on the synthetic closes themselves, computed causally as we walk forward.
    # ``lines`` holds (top, bottom) of each emitted line; ``color`` is +1 up / -1 down / 0 unset.
    lines: list[tuple[float, float]] = []
    color = 0
    boost = 0  # remaining sessions of planted continuation drift
    boost_sign = 0.0
    boost_len = 20  # the planted continuation lasts ~one H=20 window after a reversal

    for i in range(n_days):
        drift = 0.0
        if edge > 0.0 and boost > 0:
            drift = boost_sign * edge * daily_vol
            boost -= 1
        eps = rng.normal(0.0, daily_vol)
        log_p += eps + drift
        c = np.exp(log_p)
        close[i] = c

        # --- update the causal TLB state on close c (drives the planted boost) ---------------
        if not lines:
            lines.append((c, c))
            color = 0
        else:
            cur_top, cur_bot = lines[-1]
            if c > cur_top:  # extend / flip up
                if color < 0:
                    # need to break the highs of the last n_lines down-lines
                    ref = max(t for t, _ in lines[-n_lines:])
                    if c > ref:
                        if color < 0:  # genuine up-reversal
                            boost, boost_sign = boost_len, +1.0
                        color = +1
                else:
                    color = +1
                lines.append((c, cur_top))
            elif c < cur_bot:  # extend / flip down
                if color > 0:
                    ref = min(b for _, b in lines[-n_lines:])
                    if c < ref:
                        if color > 0:  # genuine down-reversal
                            boost, boost_sign = boost_len, -1.0
                        color = -1
                else:
                    color = -1
                lines.append((cur_bot, c))
            # else: inside the prior line -> no new line, state unchanged

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
    truth = {"edge": edge, "annual_vol": annual_vol, "n_lines": n_lines,
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
