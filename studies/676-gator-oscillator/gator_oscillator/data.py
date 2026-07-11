"""Data layer for Study 676 — Gator Oscillator.

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date):

* **Real tape.** Daily OHLCV (yfinance, ``auto_adjust=True``, no key) for a fixed basket
  of long-listed, liquid US large-caps + **SPY**, cached as one parquet per ticker under
  ``_cache/``. Cache-first: the network is touched only on an explicit cache miss, then
  every re-run is offline. This is a **survivors** basket (every name still trades in
  2026) — named on the Signal axis, same discipline as sibling studies 402/421.

* **Synthetic.** ``synthetic_panel`` — a deterministic, fixed-seed generator with a
  **planted trend-persistence knob** (``edge``), identical in spirit to
  [421-williams-alligator](../../421-williams-alligator/)'s positive control: multi-week
  directional regimes whose strength scales with ``edge``. ``edge = 0`` is a fair-coin
  random walk (the Gator must NOT manufacture a directional edge); ``edge > 0`` plants
  genuine multi-week trends a trend-following "wake" signal should be able to ride.

Pure numpy + pandas + stdlib for the offline path. ``fetch_one`` (network) only runs on
a cache miss and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252
AS_OF = "2026-06-30"          # last complete calendar month at publication (2026-07-10)
START = "2000-01-03"

# A transparent, fixed basket of long-listed, liquid US large-caps + the market proxy
# (SPY) — the same shape of basket used by sibling studies 402/421 for long, clean daily
# OHLC history and sector spread. This is a *survivors* basket (every name still trades
# in 2026); a fixed surviving-names basket cannot include firms delisted after a bad
# run, which is named explicitly on the Signal axis wherever this basket is used.
BASKET = [
    "SPY", "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "IBM",
    "CVX", "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT",
    "MMM", "HON", "UNH", "ORCL", "PEP", "ABT", "TXN", "COST", "LOW", "GE",
]


# --------------------------------------------------------------------------- #
# Real tape — cache-first yfinance
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str = DEFAULT_CACHE) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"gator_{safe}_1d.parquet")


def fetch_one(ticker: str, start: str = START, end: str | None = None,
              cache_dir: str = DEFAULT_CACHE, retries: int = 3) -> pd.DataFrame:
    """Download one ticker's daily OHLCV and cache it (network; cache-miss path only).

    Retries a couple of times with a small backoff on a transient yfinance rate-limit,
    then caches the parquet so every re-run is offline. ``auto_adjust=True`` gives
    split/dividend-adjusted total-return closes.
    """
    import time

    import yfinance as yf

    last_err: Exception | None = None
    raw = None
    for attempt in range(retries):
        try:
            raw = yf.download(ticker, start=start, end=end, interval="1d",
                              auto_adjust=True, progress=False)
            if raw is not None and len(raw) > 0:
                break
        except Exception as exc:  # pragma: no cover - network path
            last_err = exc
        time.sleep(1.5 * (attempt + 1))
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"yfinance returned no daily bars for {ticker}: {last_err}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    bars = bars.dropna(subset=["close"])
    os.makedirs(cache_dir, exist_ok=True)
    bars.to_parquet(_cache_path(ticker, cache_dir))
    return bars


def load_real(ticker: str, start: str = START, end: str | None = None,
              cache_dir: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Cache-first single-ticker loader: read the parquet if present, else fetch+cache."""
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        bars = pd.read_parquet(path)
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        return bars
    return fetch_one(ticker, start=start, end=end, cache_dir=cache_dir)


def have_real(tickers: list[str] = BASKET, cache_dir: str = DEFAULT_CACHE) -> bool:
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in tickers)


def load_panel(tickers: list[str] = BASKET, start: str = START, asof: str = AS_OF,
              cache_dir: str = DEFAULT_CACHE) -> dict[str, pd.DataFrame]:
    """Cache-first basket loader: dict[ticker] -> OHLCV frame sliced to [start, asof]."""
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        bars = load_real(t, start=start, cache_dir=cache_dir)
        bars = bars[(bars.index >= start) & (bars.index <= asof)]
        out[t] = bars
    return out


def fetch(tickers: list[str] = BASKET, start: str = START, cache_dir: str = DEFAULT_CACHE
          ) -> None:
    """Populate the cache for the whole basket (network; run once)."""
    for t in tickers:
        if not os.path.exists(_cache_path(t, cache_dir)):
            fetch_one(t, start=start, cache_dir=cache_dir)


def fingerprint(panel: dict[str, pd.DataFrame]) -> str:
    """A short content fingerprint of the whole panel's close columns, for the as-of stamp."""
    h = hashlib.sha1()
    for t in sorted(panel):
        h.update(t.encode())
        h.update(np.ascontiguousarray(panel[t]["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic world — planted multi-week trend persistence (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(n_days: int = 4000, edge: float = 0.0, daily_vol: float = 0.011,
                    start: str = "2005-01-03", seed: int = 676,
                    ) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a tunable trend-persistence knob.

    Built from multi-week directional regimes (runs of ~40 trading days, each carrying a
    constant drift of the same sign) — the sustained trends a "wake and follow" rule is
    meant to catch:

        log_ret_t = mu + ``edge`` * regime_drift_t + daily_vol * eps_t

    ``edge = 0`` collapses the regime drift to zero: returns are i.i.d. noise around a
    tiny constant drift, so a Gator-wake timing rule must NOT reliably predict the sign
    of the forward move. ``edge > 0`` plants genuine, harvestable multi-week trends.

    Business-day index, span well under the ~250-year pandas ns-timestamp ceiling.
    Returns ``(bars, truth)`` with the planted parameters recorded.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)

    edge = float(max(edge, 0.0))
    mu = 0.0002
    regime_len = 40
    regime_drift_mag = 0.0016

    regime_drift = np.zeros(n_days)
    i = 0
    while i < n_days:
        L = max(5, int(rng.normal(regime_len, regime_len * 0.4)))
        sign = rng.choice([-1.0, 1.0])
        regime_drift[i:i + L] = sign * regime_drift_mag
        i += L

    eps = rng.normal(0.0, daily_vol, n_days)
    log_ret = mu + edge * regime_drift + eps

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 80_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(cal, name="date"),
    )
    truth = {"edge": edge, "daily_vol": daily_vol, "n_days": n_days, "seed": seed}
    return bars, truth


def synthetic_multi_panel(seed: int = 676, n_series: int = 20, edge: float = 0.0,
                          n_days: int = 3000, start: str = "2005-01-03",
                          ) -> dict[str, pd.DataFrame]:
    """A panel of ``n_series`` independent synthetic tapes sharing one ``edge`` and one
    macro ``seed`` (each series gets its own derived sub-seed) — the synthetic analogue
    of :data:`BASKET`, giving the wake-event detector enough pooled events to compute a
    meaningful HAC/Welch t, exactly like the real 30-name basket does.
    """
    panel = {}
    for i in range(n_series):
        bars, _ = synthetic_panel(n_days=n_days, edge=edge, start=start,
                                  seed=seed * 97 + i * 13 + 1)
        panel[f"SYN{i}"] = bars
    return panel
