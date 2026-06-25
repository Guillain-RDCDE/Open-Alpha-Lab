"""Data layer for Study 468 (Gartley / AB=CD harmonic patterns).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  A harmonic pattern is a five-point zig-zag X-A-B-C-D of confirmed pivots whose *retracement
  ratios* sit on the Fibonacci grid (Gartley: B = 0.618·XA, D = 0.786·XA). The believers' claim
  is that when such ratios are satisfied, **point D is a reversal** (price bounces up). We plant
  exactly that: with ``edge > 0`` we splice in clean XABCD zig-zags that *do* match the Gartley
  grid and *are* followed by a real upward bounce at D, so the harmonic detector harvests a
  genuine effect; with ``edge = 0`` the path is a pure random walk and any "pattern" the detector
  finds is a coincidence whose D-point is a fair coin. This is the positive control — a harness
  that cannot bank the planted bounce proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: every pivot is
*confirmed* (a fractal needs ``k`` bars on each side), the XABCD pattern is only recognised once
its D pivot is confirmed, the completion is read on the close of *t*, and the trade is entered at
*t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs harmonic-pattern traders draw on: the broad tape, big-cap tech, small caps,
# and a couple of cross-asset charts. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 4000,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    pivot_k: int = 4,
    start: str = "2010-01-04",
    seed: int = 468,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of Gartley-D-point reversal.

    The baseline price path is a random walk in log-returns with daily sigma
    ``annual_vol/sqrt(252)``. When ``edge > 0`` we *splice in* clean Gartley zig-zags: at random
    spacing we lay down an X-A-B-C-D leg structure whose retracement ratios match the Gartley grid
    (B = 0.618·XA, C between 0.382–0.886·AB, D = 0.786·XA — a bullish Gartley completing at a
    swing low), and immediately after D we inject a real upward drift of size ``edge`` over the
    following ~15 bars (the "reversal"). At ``edge = 0`` no zig-zags are spliced and the tape is a
    pure martingale, so a D-point entry is a fair coin; at ``edge > 0`` the harmonic detector that
    correctly identifies the completed Gartley banks the planted bounce.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # Build the log-price path bar by bar. Between planted patterns the path takes very small,
    # nearly-flat steps so it does NOT manufacture spurious fractal pivots that would corrupt the
    # clean XABCD windows; inside a pattern we lay down the zig-zag legs explicitly.
    log_p = np.zeros(n_days)
    log_p[0] = np.log(100.0)
    quiet_vol = daily_vol * 0.12  # near-flat between patterns: keeps stray pivots out

    planted = []
    i = 1
    while i < n_days:
        if edge > 0.0 and i < n_days - 140 and (not planted or i - planted[-1] > 40):
            # leg lengths >= pivot_k+1 on each side so every turn is a strict fractal (k-confirmed)
            la = int(rng.integers(pivot_k + 2, pivot_k + 7))
            lb = int(rng.integers(pivot_k + 2, pivot_k + 7))
            lc = int(rng.integers(pivot_k + 2, pivot_k + 7))
            ld = int(rng.integers(pivot_k + 2, pivot_k + 7))
            xa = 0.10
            ab = 0.618 * xa                                # B retraces 0.618 of XA  (Gartley)
            bc = rng.uniform(0.45, 0.80) * ab              # C retraces 0.45-0.80 of AB
            cd = (0.786 * xa) - (ab - bc)                   # so A-D = 0.786*XA (Gartley D)
            base = log_p[i - 1]
            yX = base
            yA = yX + xa
            yB = yA - ab
            yC = yB + bc
            yD = yC - cd                                   # bullish Gartley: D is a swing low

            def _ramp(start_y, end_y, length):
                nonlocal i
                for j in range(length):
                    if i >= n_days:
                        return
                    frac = (j + 1) / length
                    log_p[i] = start_y + frac * (end_y - start_y) + rng.normal(0.0, quiet_vol)
                    i += 1

            _ramp(yX, yA, la)
            _ramp(yA, yB, lb)
            _ramp(yB, yC, lc)
            _ramp(yC, yD, ld)
            dpos = i - 1                                    # bar of D (the swing low)
            planted.append(dpos)
            # the planted reversal: a clean upward drift over the ~20 bars after D
            recov = 20
            for j in range(recov):
                if i >= n_days:
                    break
                frac = (j + 1) / recov
                log_p[i] = yD + frac * (edge * xa) + rng.normal(0.0, quiet_vol)
                i += 1
        else:
            # quiet near-flat segment (pure low-vol noise, no planted structure)
            step = rng.normal(0.0, quiet_vol if edge > 0.0 else daily_vol)
            log_p[i] = log_p[i - 1] + step
            i += 1

    close = np.exp(log_p)
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
    truth = {"edge": edge, "annual_vol": annual_vol, "pivot_k": pivot_k,
             "n_days": n_days, "seed": seed, "n_planted": len(planted)}
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
