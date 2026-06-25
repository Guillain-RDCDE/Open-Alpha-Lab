"""Data layer for Study 482 (VWMA-Crossover).

Two tapes, one shape (a tz-naive daily OHLCV frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**
  tuned to THIS indicator. A volume-weighted MA differs from a plain SMA only when volume
  correlates with *where the price is going*. So the planted effect is exactly that: with
  ``edge > 0`` the informative up-moves arrive on **high-volume** bars (a volume-led drift
  pulse), so a VWMA leans into the move *earlier* than the equal-weighted SMA and a VWMA
  golden-cross banks a real forward bounce that the plain-SMA cross does not. With
  ``edge = 0`` volume is statistically **independent** of returns: the VWMA and SMA crosses
  are interchangeable and the VWMA cross is a fair coin. This is the positive control — a
  harness that cannot bank the planted volume-led pulse proves nothing by finding nothing on
  the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and carries the **volume** column the VWMA needs.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the moving
averages are causal (trailing windows only), the cross is read on the close of *t*, and the
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

# The same 5 liquid, long-history ETFs the desk reuses for single-instrument trend studies:
# broad tape, big-cap tech, small caps, the Dow, and gold. Daily, deep volume.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    fast: int = 10,
    slow: int = 30,
    start: str = "2010-01-04",
    seed: int = 482,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a *known* amount of volume-led drift.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    Volume is a lognormal series. The planted edge is **volume-led drift**: with ``edge > 0`` a
    fraction of bars carry both an *above-median* volume spike and a positive drift pulse, so
    the informative up-moves land on heavy-volume days. A volume-weighted MA over-weights those
    bars and therefore turns up *before* the equal-weighted SMA — a VWMA golden-cross then
    precedes a real forward bounce. At ``edge = 0`` volume is independent of returns, so the
    VWMA cross and the SMA cross are statistically interchangeable (a fair coin).

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # base log-volume: lognormal around a slowly-wandering level
    base_logv = 16.0 + np.cumsum(rng.normal(0.0, 0.01, n_days))
    logv = base_logv + rng.normal(0.0, 0.35, n_days)

    # volume-led pulses: on a subset of days, a heavy-volume bar carries a drift kick
    pulse = rng.random(n_days) < 0.06          # ~6% of bars are "informed" days
    eps = rng.normal(0.0, daily_vol, n_days)
    drift_kick = np.zeros(n_days)
    if edge > 0.0:
        # heavy volume on pulse days, and the pulse day's return is pulled upward
        logv = logv + pulse * (1.2 + 0.4 * rng.random(n_days))
        # the drift kick decays over the next few bars so the VWMA-led cross can bank it
        kick = edge * daily_vol * 6.0
        for i in np.flatnonzero(pulse):
            for j, w in enumerate((1.0, 0.6, 0.35, 0.2)):
                if i + j < n_days:
                    drift_kick[i + j] += kick * w

    log_ret = eps + drift_kick
    log_p = np.log(100.0) + np.cumsum(log_ret)
    close = np.exp(log_p)
    vol = np.exp(logv)

    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "fast": fast, "slow": slow,
             "n_days": n_days, "seed": seed, "pulse_frac": float(pulse.mean())}
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
    """Real daily OHLCV for ``ticker``; **cache-first** (network only on a cache miss).

    Reads a cached parquet if present. Otherwise — and only if ``allow_fetch`` — downloads
    from yfinance (with a couple of retries + back-off on rate limits) and caches the parquet,
    so every subsequent call is fully offline. The frame carries the **volume** column the
    VWMA needs.
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
    return bars[["open", "high", "low", "close", "volume"]]


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
                bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
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
