"""Data layer for Study 405 — Doji Reversal.

A doji is the candlestick of *indecision*: open and close finish at (almost) the same
price, leaving a tiny real body with wicks on either side. The folk recipe says a doji
marks the moment buyers and sellers reach a stand-off — and therefore *precedes a
reversal* of the prevailing move. This module produces two tapes, both as a tz-naive
daily OHLCV frame indexed by date:

* ``synthetic_panel(...)`` — a *deterministic, offline* generator returning
  ``(data, truth)``.  A single ``edge`` knob plants the only structure a doji-reversal
  rule could possibly harvest: a real, signed mean-reversion that fires **specifically
  on the sessions we stamp as dojis**.  ``edge = 0`` is a pure random walk whose doji
  bars carry no forward information — the null in a bottle.  ``edge > 0`` plants a
  genuine post-doji reversal so the harness's positive control can light up.  A
  synthetic positive control proves the *machinery* can find an edge; it never backs a
  REAL stamp on the tape.

* ``load_real(cache_dir=DEFAULT_CACHE)`` — the real Yahoo! daily tape (``yfinance``),
  **cache-first**: it reads cached parquets and only touches the network on a cache
  miss (with a couple of retries + backoff), then caches the parquet so re-runs are
  offline.  Daily bars go back ~25 years, giving high statistical power.

The basket is a fixed set of ~28 long-listed, liquid US large-caps **+ SPY** — a
*survivors* basket (all still trading in 2026).  Survivorship is named on the Signal
axis: a fixed surviving-names basket cannot include firms that delisted, a mild bias we
flag wherever the stamp appears (its direction here is ambiguous — it neither obviously
helps nor hurts a same-direction reversal count).

No look-ahead is baked in here.  The doji is detected on bar *t*'s OHLC; forward returns
are measured strictly from *t+1*'s close onward (see ``strategy.py``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# A transparent, fixed basket of large, long-listed US large-caps with deep, clean OHLC
# histories on yfinance, plus SPY as the index reference.  Chosen for long history +
# sector spread.  This is a *survivors* basket — survivorship is named on the Signal axis.
BASKET = [
    "SPY",
    "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "IBM", "CVX",
    "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT", "MMM",
    "HON", "UNH", "ORCL", "PEP", "ABT", "TXN", "COST", "GS",
]


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"doji_{safe}_1d.parquet")


def fetch_one(ticker: str, period: str = "25y", cache_dir: str = DEFAULT_CACHE,
              fetch: bool = False, retries: int = 3) -> pd.DataFrame:
    """Daily OHLCV for one ``ticker``; cache-first.

    With ``fetch=False`` (default) this reads the cached parquet and never touches the
    network — on a cache miss it raises so the offline core stays offline.  With
    ``fetch=True`` it downloads (retrying a couple of times with backoff on a transient
    failure) and writes the parquet cache.  Returns a tz-naive daily OHLCV frame.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call load_real(fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = None
        for attempt in range(retries):
            try:
                raw = yf.download(ticker, period=period, interval="1d",
                                  auto_adjust=True, progress=False)
                if raw is not None and not raw.empty:
                    break
            except Exception:
                raw = None
            time.sleep(1.5 * (attempt + 1))
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    return bars.dropna(how="any")


def load_real(cache_dir: str = DEFAULT_CACHE, fetch: bool = False,
              tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Cache-first dict ``{ticker: daily OHLCV frame}`` for the basket.

    With ``fetch=False`` (default) reads cached parquets only.  With ``fetch=True`` warms
    the cache from yfinance.  Tickers with no usable bars are skipped.
    """
    tickers = tickers or BASKET
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            out[t] = fetch_one(t, cache_dir=cache_dir, fetch=fetch)
        except (FileNotFoundError, RuntimeError):
            if not fetch:
                raise
    return out


def have_real(cache_dir: str = DEFAULT_CACHE, tickers: list[str] | None = None) -> bool:
    tickers = tickers or BASKET
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in tickers)


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 28, n_days: int = 3000, edge: float = 0.0,
                    seed: int = 405, daily_vol: float = 0.013,
                    doji_rate: float = 0.06,
                    start: str = "2005-01-03") -> tuple[dict, dict]:
    """Deterministic OHLCV panel with a PLANTED post-doji reversal knob.

    Each name is a daily random walk.  On a fraction ``doji_rate`` of sessions we *force a
    doji*: the close is pulled back to (almost) the open, leaving a tiny real body with
    symmetric wicks — exactly the bar the detector keys on.  When ``edge != 0``, each
    forced-doji session injects a **signed reversal** into the next few days proportional
    to ``edge`` and *opposite* the move that led into the doji — the precise pattern the
    folk recipe claims.  When ``edge = 0`` the doji bars carry no forward information, so a
    forward-return test on them must NOT manufacture significance however the noise falls.

    Returns ``(data, truth)`` where ``data`` is ``{name: OHLCV frame}`` (same shape as
    ``load_real``) and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]
    H = 5  # horizon over which a planted reversal plays out

    data: dict[str, pd.DataFrame] = {}
    for name in names:
        ret = rng.normal(0.0002, daily_vol, n_days)
        is_doji = rng.random(n_days) < doji_rate
        # forced-doji sessions: close returns ~to open (small body)
        for i in range(2, n_days - H):
            if not is_doji[i]:
                continue
            prior = ret[i - 2:i].sum()  # the move leading into the doji
            ret[i] = rng.normal(0.0, daily_vol * 0.1)  # tiny body -> doji
            if edge != 0.0:
                # planted reversal: push the next H days against the prior move
                ret[i + 1:i + 1 + H] += -edge * np.sign(prior) / H
        close = 100.0 * np.exp(np.cumsum(ret))
        open_ = np.empty_like(close)
        open_[0] = 100.0
        open_[1:] = close[:-1]
        # wicks: doji bars get long symmetric wicks; others a modest wick
        body = np.abs(close - open_)
        wick = np.where(is_doji, daily_vol * 1.2, daily_vol * 0.4) * close
        wick = np.abs(rng.normal(wick, daily_vol * 0.2 * close))
        hi = np.maximum(open_, close) + wick
        lo = np.minimum(open_, close) - wick
        vol = rng.integers(1_000_000, 50_000_000, n_days).astype(float)
        data[name] = pd.DataFrame(
            {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
            index=pd.DatetimeIndex(cal, name="date"),
        )
        _ = body  # body retained for clarity; wick sizing uses doji flags
    truth = {"edge": edge, "n_names": n_names, "n_days": n_days,
             "doji_rate": doji_rate, "seed": seed, "horizon": H}
    return data, truth


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]


def panel_fingerprint(panel: dict[str, pd.DataFrame]) -> str:
    """One fingerprint over the whole basket (concatenated closes, ticker-sorted)."""
    h = hashlib.sha1()
    for t in sorted(panel):
        h.update(t.encode())
        h.update(np.ascontiguousarray(panel[t]["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
