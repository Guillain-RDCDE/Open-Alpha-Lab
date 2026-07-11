"""Data layer for Study 675 (Qstick).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  Qstick reads each bar's body — close minus open — smoothed over N days; the believers' claim
  is that when smoothed Qstick *turns positive* (buying pressure has taken over) price
  subsequently rises. We plant exactly that: with ``edge > 0`` we pull the *next* day's intraday
  (open-to-close) return in the direction of the recent smoothed body, so a "Qstick crosses up
  through zero" entry genuinely harvests a forward bounce; with ``edge = 0`` the log-return
  series is a pure random walk and the cross is a fair coin. This is the positive control — a
  harness that cannot bank a planted lead proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: smoothed Qstick is a
trailing average, a zero-cross is read on the close of *t*, and the trade is entered at
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

# Indices / ETFs a Qstick chartist would run this on: the broad tape, big-cap tech, small
# caps, and a couple of cross-asset charts. Daily, liquid, long history — the same basket used
# by sibling study 473-balance-of-power for a direct, apples-to-apples comparison.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 4000,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    smooth: int = 8,
    phi: float = 0.95,
    start: str = "2010-01-04",
    seed: int = 675,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of Qstick-leads-price structure.

    Each session's intraday (open-to-close) return is ``z_t + idiosyncratic noise``, where
    ``z_t`` is a **latent, mean-reverting buying-pressure factor** — a stationary AR(1),
    ``z_t = phi * z_{t-1} + innovation``, generated *independently* of price (no self-referential
    feedback, which would make the path explosive). ``edge`` sets the stationary standard
    deviation of ``z`` as a multiple of the daily idiosyncratic vol: at ``edge = 0``, ``z`` is
    identically zero and every session's body is pure noise — a fair coin, no forecastability.
    At ``edge > 0``, ``z`` persists across days (``phi = 0.95`` gives a multi-week half-life), so
    a smoothed *average* of recent noisy bodies is a genuine (noisy) estimate of the current
    state, and that state keeps driving future bodies — exactly the believers' claim that
    "buying pressure persists". A zero-cross entry on the smoothed body should bank it when
    ``edge > 0`` and see nothing when ``edge = 0``.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    z_sigma = edge * daily_vol
    innov_sigma = z_sigma * np.sqrt(max(1.0 - phi * phi, 0.0))
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    open_ = np.empty(n_days)
    hi = np.empty(n_days)
    lo = np.empty(n_days)

    prev_close = 100.0
    z = 0.0
    for i in range(n_days):
        if z_sigma > 0.0:
            z = phi * z + rng.normal(0.0, innov_sigma)
        overnight = rng.normal(0.0, daily_vol * 0.5)
        o = prev_close * np.exp(overnight)
        intraday = z + rng.normal(0.0, daily_vol * 0.7)
        c = o * np.exp(intraday)
        wick = abs(rng.normal(0.0, daily_vol * 0.4)) * c
        h = max(o, c) + wick
        l = min(o, c) - wick
        close[i], open_[i], hi[i], lo[i] = c, o, h, l
        prev_close = c

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "smooth": smooth, "phi": phi,
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
