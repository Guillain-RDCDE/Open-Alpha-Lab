"""Data layer for Study 461 (Descending Triangle).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  A descending triangle is flat support + descending highs; the believers' claim is that
  the **break below the flat support resolves DOWN** (a bearish continuation). We plant
  exactly that: with ``edge > 0`` the path is built so that, once price closes below a
  horizontal support level it has repeatedly bounced off, it is given a real *downward*
  drift (a genuine break-down momentum) that a short-the-break entry can bank; with
  ``edge = 0`` the support break is a fair coin (continuation up is as likely as down).
  This is the positive control — a harness that cannot bank the planted break-down proves
  nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the triangle is
anchored on pivots that are *confirmed* (a fractal needs ``k`` bars on each side), the
support break is detected on the close of *t*, and the short is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs descending-triangle proponents draw on: the broad tape, big-cap tech, small
# caps, and a couple of cross-asset charts. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    pivot_k: int = 5,
    start: str = "2010-01-04",
    seed: int = 461,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of break-down continuation.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that we plant the descending-triangle effect: we maintain a slowly-updated
    horizontal **support** level that price repeatedly bounces off (a soft floor), while the
    local **highs descend** (a downward squeeze). Once the close finally breaks *below* that
    support, we switch on a transient downward drift proportional to ``edge`` for a short
    window — a genuine bearish continuation the short-the-break rule should bank. At
    ``edge = 0`` the support floor is symmetric (no post-break drift) so a break is a fair
    coin; at ``edge > 0`` the break is followed by a real decline.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    log_p = np.log(100.0)
    # A horizontal support that re-forms in regimes, with a descending ceiling above it. The
    # planted force: a soft floor reflecting price up off support, plus a downward kick once
    # price has *closed through* the support (the break-down continuation).
    support = log_p - 0.05            # the flat floor, in log units
    ceiling_slope = -daily_vol * 0.20  # descending highs (squeeze toward support)
    ceiling = log_p + 0.07
    broken_for = 0                   # bars remaining in an active break-down drift
    regime_len = 220
    for i in range(n_days):
        if i % regime_len == 0 and i > 0:
            # re-seed a fresh triangle one notch lower (so supports keep forming)
            support = log_p - 0.05
            ceiling = log_p + 0.07
            broken_for = 0
        ceiling += ceiling_slope
        ceiling = max(ceiling, support + 0.015)  # converge but never cross

        pull = 0.0
        if edge > 0.0:
            if broken_for > 0:
                # active break-down: a real downward continuation drift
                pull = -edge * daily_vol * 4.0
                broken_for -= 1
            elif log_p < support:
                # just closed below support -> trigger the break-down window
                broken_for = 8
                pull = -edge * daily_vol * 4.0
            elif log_p < support + 0.008:
                # soft floor: bounce up off support (the repeated touches)
                pull = edge * (support + 0.008 - log_p) * 1.5
            if log_p > ceiling:
                # respect the descending ceiling (fade down)
                pull += -edge * (log_p - ceiling)
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
    truth = {"edge": edge, "annual_vol": annual_vol, "pivot_k": pivot_k,
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
