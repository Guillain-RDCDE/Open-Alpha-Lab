"""Data layer for Study 685 — Tri-Star Doji.

The **tri-star** is candlestick lore's rarest reversal claim (Nison, *Japanese
Candlestick Charting Techniques*): three dojis in a row — three consecutive sessions
where open and close finish at (almost) the same price — supposedly mark a *major*
trend exhaustion. Because a single doji is already a minority of sessions (~5-10% of
days on a 10%-of-range cut), *three in a row* is combinatorially rare: at an
independent per-bar doji rate of ``p`` the naive expectation is ``p**3`` of all
3-bar windows. To have any chance of collecting more than a handful of these events
we need **many names, many years** — a single ticker's ~25-year daily tape (~6,300
bars) essentially never produces one.

This module produces two tapes, both a tz-naive daily OHLCV frame indexed by date:

* ``load_real(cache_dir=DEFAULT_CACHE)`` — the real Yahoo! daily tape (``yfinance``),
  **cache-first**: reads cached parquets and only touches the network on a cache miss
  (with retries + backoff), then caches so re-runs are offline. A broad basket of
  60 long-listed, liquid US large-caps + SPY, ~25 years each, to maximize the tiny
  event count a rare 3-bar pattern can produce.

* ``synthetic_panel(...)`` — a deterministic, offline generator returning
  ``(data, truth)``. A single ``edge`` knob plants the only structure a tri-star rule
  could possibly harvest: forced 3-bar doji blocks at a controlled rate, each followed
  by a genuine, signed reversal *opposite* the move that led into the block. ``edge=0``
  is a pure random walk whose tri-star blocks carry no forward information — the null
  in a bottle. A synthetic positive control proves the machinery can find a planted
  edge; it never backs a REAL stamp on the tape.

No look-ahead is baked in here: the tri-star is detected on bars *t-2, t-1, t*'s OHLC;
forward returns are measured strictly from *t+1*'s open onward (see ``strategy.py``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# A broad, transparent, fixed basket of long-listed US large-caps with deep, clean
# OHLC histories on yfinance, plus SPY as the index reference. 60 names spanning every
# major sector — chosen for long history + sector spread, to give a combinatorially
# rare 3-bar pattern the best honest chance of a usable sample. This is a *survivors*
# basket — survivorship is named on the Signal axis.
BASKET = [
    "SPY",
    "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "IBM", "CVX",
    "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT", "MMM",
    "HON", "UNH", "ORCL", "PEP", "ABT", "TXN", "COST", "GS", "AXP", "BAC",
    "C", "WFC", "MS", "T", "VZ", "NKE", "LOW", "UPS", "FDX", "EMR",
    "ETN", "ITW", "APD", "ECL", "PPG", "NEM", "DE", "GD", "LMT", "NOC",
    "RTX", "ADP", "PAYX", "SYK", "MDT", "BDX", "CL", "KMB", "GIS", "SO",
]

AS_OF = "2026-06-30"  # last complete calendar month at publication (2026-07-10)


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"tsd_{safe}_1d.parquet")


def fetch_one(ticker: str, period: str = "25y", cache_dir: str = DEFAULT_CACHE,
              fetch: bool = False, retries: int = 3) -> pd.DataFrame:
    """Daily OHLCV for one ``ticker``; cache-first.

    With ``fetch=False`` (default) this reads the cached parquet and never touches the
    network — on a cache miss it raises so the offline core stays offline. With
    ``fetch=True`` it downloads (retrying a couple of times with backoff on a
    transient failure) and writes the parquet cache. Returns a tz-naive daily OHLCV
    frame sliced to :data:`AS_OF` (no partial-month drift).
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
    bars = bars.dropna(how="any")
    return bars.loc[bars.index <= pd.Timestamp(AS_OF)]


def load_real(cache_dir: str = DEFAULT_CACHE, fetch: bool = False,
              tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Cache-first dict ``{ticker: daily OHLCV frame}`` for the basket.

    With ``fetch=False`` (default) reads cached parquets only. With ``fetch=True``
    warms the cache from yfinance. Tickers with no usable bars are skipped.
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
def synthetic_panel(n_names: int = 40, n_days: int = 3500, edge: float = 0.0,
                    seed: int = 685, daily_vol: float = 0.013,
                    tri_rate: float = 0.004,
                    start: str = "2005-01-03") -> tuple[dict, dict]:
    """Deterministic OHLCV panel with a PLANTED post-tri-star reversal knob.

    Each name is a daily random walk. On a fraction ``tri_rate`` of *eligible*
    sessions we force a **3-bar doji block**: three consecutive sessions whose close
    is pulled back to (almost) the open, each leaving a tiny real body with symmetric
    wicks — exactly the shape the detector keys on. When ``edge != 0``, each forced
    tri-star block injects a **signed reversal** into the next few days proportional
    to ``edge`` and *opposite* the move that led into the block — the precise "major
    reversal" the folk claim describes. When ``edge = 0`` the tri-star blocks carry no
    forward information, so a forward-return test on them must NOT manufacture
    significance however the noise falls.

    Returns ``(data, truth)`` where ``data`` is ``{name: OHLCV frame}`` (same shape as
    :func:`load_real`) and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]
    H = 20  # horizon over which a planted reversal plays out ("major" -> longer than 1 bar)

    data: dict[str, pd.DataFrame] = {}
    for name in names:
        ret = rng.normal(0.0002, daily_vol, n_days)
        is_tri_start = rng.random(n_days) < tri_rate
        is_doji = np.zeros(n_days, dtype=bool)
        for i in range(5, n_days - H - 3):
            if not is_tri_start[i]:
                continue
            if is_doji[i - 1] or is_doji[i - 2]:
                continue  # don't overlap two forced blocks
            prior = ret[i - 5:i].sum()  # the move leading into the tri-star block
            for k in range(3):
                is_doji[i + k] = True
                ret[i + k] = rng.normal(0.0, daily_vol * 0.08)  # tiny body -> doji
            if edge != 0.0:
                # planted reversal: push the next H days against the prior move
                ret[i + 3:i + 3 + H] += -edge * np.sign(prior) / H
        close = 100.0 * np.exp(np.cumsum(ret))
        open_ = np.empty_like(close)
        open_[0] = 100.0
        open_[1:] = close[:-1]
        wick = np.where(is_doji, daily_vol * 1.1, daily_vol * 0.4) * close
        wick = np.abs(rng.normal(wick, daily_vol * 0.2 * close))
        hi = np.maximum(open_, close) + wick
        lo = np.minimum(open_, close) - wick
        vol = rng.integers(1_000_000, 50_000_000, n_days).astype(float)
        data[name] = pd.DataFrame(
            {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
            index=pd.DatetimeIndex(cal, name="date"),
        )
    truth = {"edge": edge, "n_names": n_names, "n_days": n_days,
             "tri_rate": tri_rate, "seed": seed, "horizon": H}
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
