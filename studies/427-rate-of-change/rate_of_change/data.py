"""Data layer for Study 427 (Rate of Change).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator. A trend-persistence
  knob (``edge``) injects an AR(1) drift component into the log-return series, so
  a momentum/ROC timing rule has something real to harvest. At ``edge = 0`` the
  log-return series is a pure random walk (plus a constant equity drift) — the
  forecastable structure ROC needs is absent — so the harness can assert "ROC
  beats buy-and-hold only when we plant persistence, and not otherwise." Returns
  ``(data, truth_dict)``.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it
  reads the cached parquet if present and only touches the network on an explicit
  cache miss (with a couple of retries + backoff), then caches the parquet so all
  re-runs are offline. Daily history is long (30+ years on SPY) and free of the
  60-day cap that hits sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the
ROC signal is formed on the close of day *t*, the position is held over day
*t+1*'s return (one ``shift``, applied once).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The headline instrument plus a small confirmation panel.
TICKERS = ["SPY", "QQQ", "DIA", "IWM", "EFA"]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (positive control)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 6000,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    regime_days: int = 80,
    persistence: float = 0.0,
    start: str = "2000-01-03",
    seed: int = 427,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a **persistent trend regime** knob.

    This is the positive control for a *timing* rule, so the planted structure is
    not just autocorrelation — it is a hidden up/down **regime** that lasts long
    enough for a momentum filter to ride it. A latent state ``z_t`` follows a slow
    random walk and the daily log-return is::

        r_t = edge * tanh(z_t) * daily_amp + eps_t

    so when ``z_t`` is high the asset *trends up* and when it is low it *trends
    down* — and the regime persists for ~``regime_days``. The ``edge`` knob is the
    only forecastable structure:

    - ``edge = 0``   → returns are i.i.d. zero-mean noise; there is **no** trend to
      ride. A ROC long/flat rule cannot beat buy-and-hold except by luck, and it
      pays costs to try — so the harness must NOT manufacture a positive delta.
    - ``edge > 0``   → genuine persistent up/down regimes. Stepping aside when ROC
      turns negative avoids the down-regime losses, so ROC *should* add value over
      always-being-long — the harness must light that up.

    (``persistence`` is an alias of ``edge``; if both given, ``edge`` wins.)
    Returns ``(bars, truth)`` recording the planted parameters.
    """
    if edge == 0.0 and persistence != 0.0:
        edge = persistence
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252.0)
    daily_amp = 0.6 / np.sqrt(252.0)  # ~60%/yr swing between full up- and down-regime
    sessions = pd.bdate_range(start=start, periods=n_days)

    # Latent regime: an Ornstein-Uhlenbeck-ish slow walk, mean-reverting to 0,
    # whose half-life is ~regime_days so trends last weeks-to-months.
    theta = 1.0 / regime_days
    z = np.empty(n_days)
    z[0] = 0.0
    z_shock = rng.normal(0.0, 1.0, n_days)
    for i in range(1, n_days):
        z[i] = (1.0 - theta) * z[i - 1] + np.sqrt(2.0 * theta) * z_shock[i]

    eps = rng.normal(0.0, daily_vol, n_days)
    log_ret = edge * np.tanh(z) * daily_amp + eps

    close = 100.0 * np.exp(np.cumsum(log_ret))
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
    truth = {
        "edge": edge,
        "regime_days": regime_days,
        "annual_vol": annual_vol,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def have_real(ticker: str = "SPY", cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff the cached daily tape for ``ticker`` is present (offline-ready)."""
    return os.path.exists(_cache_path(ticker, cache_dir))


def load_real(
    ticker: str = "SPY",
    start: str = "1993-01-29",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    allow_fetch: bool = True,
) -> pd.DataFrame:
    """Real daily OHLC for ``ticker`` — **cache-first**.

    Reads the cached parquet if present (offline). On a cache miss, and only if
    ``allow_fetch`` is True, it downloads from yfinance (a couple of retries with
    a small backoff), caches the parquet, and returns it. Daily SPY history goes
    back to its 1993 inception, giving 30+ years of signal.

    ``auto_adjust=True`` → **total-return** closes (splits + dividends folded in),
    which is the correct series for a Sharpe race against a held index.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        bars = pd.read_parquet(path)
    else:
        if not allow_fetch:
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call load_real({ticker!r}) once with network access to populate it."
            )
        import yfinance as yf  # lazy: only on a real cache miss

        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(
                    ticker, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is not None and not raw.empty:
                    break
            except Exception:
                raw = None
            time.sleep(2.0 * (attempt + 1))
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        cols = [c for c in ["open", "high", "low", "close"] if c in raw.rename(columns=str.lower).columns]
        bars = raw.rename(columns=str.lower)[cols]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars.dropna(how="all")


def load_panel(
    tickers: list[str] | None = None,
    cache_dir: str = DEFAULT_CACHE,
    allow_fetch: bool = True,
) -> dict[str, pd.DataFrame]:
    """Cache-first loader for the small confirmation panel."""
    tickers = tickers or TICKERS
    out = {}
    for t in tickers:
        try:
            out[t] = load_real(t, cache_dir=cache_dir, allow_fetch=allow_fetch)
        except Exception:
            continue
    return out


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
