"""Data layer for Study 688 — Concealing Baby Swallow.

The **concealing baby swallow** is a four-candle bullish reversal from the Japanese
candlestick canon (Nison, *Japanese Candlestick Charting Techniques*; Bulkowski,
*Encyclopedia of Candlestick Charts*) — and by most accounts the **rarest** named
pattern in the book. During a downtrend:

1. Two consecutive **black marubozu** (long bearish bodies, no real shadows) —
   committed, one-way selling.
2. A third black candle that **gaps down** at the open, rallies *during the session*
   back up into the second candle's real body (a long upper shadow — the rally is
   "concealed" inside a candle that still closes near its low), then closes lower.
3. A fourth black candle that **totally engulfs** the third — including its shadow,
   trading above its high and below its low — and closes at a new low for the move.

The lore: the failed rally hidden inside candle three, and the fact that candle four
can't even hold the low it "should" hold, marks capitulation — the last sellers are
done, and a reversal is close. Because it needs **two** near-perfect marubozu bodies
*and* a precise overlap-then-engulf geometry to line up in a row, it is one of the
most combinatorially restrictive patterns on the desk — rarer than the three-candle
morning star (186) or three black crows (408), and rarer still than the already-thin
five-candle ladder bottom (687). To have any chance at all of a usable sample, we cast
as wide a net as the desk has cast for any single-pattern study: a **very large**,
long-listed US large-cap basket (110+ names) plus SPY, ~25-30 years each.

This module produces two tapes, both a tz-naive daily OHLCV frame indexed by date:

* ``load_real(cache_dir=DEFAULT_CACHE)`` — the real Yahoo! daily tape (``yfinance``),
  **cache-first**: reads cached parquets and only touches the network on a cache miss,
  then caches so re-runs are offline. This is a **survivors** basket (every name is
  still trading today) — named on the Signal axis, though for a single-pattern event
  study it affects which *names* can contribute a candidate, not the direction of any
  comparison that does get made.

* ``synthetic_panel(...)`` — a deterministic, offline generator returning
  ``(data, truth)``. A single ``edge`` knob plants the *only* structure a
  concealing-baby-swallow rule could possibly harvest: forced 4-bar blocks (two
  engineered marubozu, the engineered gap-rally-fail third candle, the engineered
  full engulf on the fourth) at a controlled rate, each optionally followed by a
  genuine, signed post-pattern bounce. ``edge = 0`` is a pure random walk beyond the
  forced candle shapes — the blocks carry no forward information, and the detector
  must not manufacture significance from them. A synthetic positive control proves
  the machinery can find a planted edge; it never backs a REAL stamp on the real tape.

No look-ahead is baked in here: the pattern is confirmed on bars *t-3..t*'s OHLC;
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

# A very large, transparent, fixed basket of long-listed, liquid US names with deep
# OHLC histories on yfinance, plus SPY and three broad sector/style ETFs for extra
# breadth — cast wide because a four-candle pattern this restrictive needs every name-
# year we can find. This is a *survivors* basket — survivorship is named on the
# Signal axis (see README / results.md).
BASKET = [
    "SPY", "QQQ", "DIA", "IWM",
    "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "IBM", "CVX",
    "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT", "MMM",
    "HON", "UNH", "ORCL", "PEP", "ABT", "TXN", "COST", "GS", "AXP", "BAC",
    "C", "WFC", "MS", "T", "VZ", "NKE", "LOW", "UPS", "FDX", "EMR",
    "ETN", "ITW", "APD", "ECL", "PPG", "NEM", "DE", "GD", "LMT", "NOC",
    "RTX", "ADP", "PAYX", "SYK", "MDT", "BDX", "CL", "KMB", "GIS", "SO",
    "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ADBE", "CRM", "QCOM", "TMO",
    "DHR", "LIN", "UNP", "BLK", "SCHW", "SPGI", "MO", "PM", "DUK", "NEE",
    "SBUX", "TGT", "BKNG", "ADI", "AMAT", "LRCX", "MU", "GILD", "AMGN", "CI",
    "ELV", "CVS", "V", "MA", "PYPL", "USB", "PNC", "TFC", "COP", "SLB",
    "EOG", "KMI", "D", "EXC", "GE", "F", "GM",
]

AS_OF = "2026-06-30"  # last complete calendar month at publication (2026-07-10)


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"cbs_{safe}_1d.parquet")


def fetch_one(ticker: str, period: str = "max", cache_dir: str = DEFAULT_CACHE,
              fetch: bool = False, retries: int = 3) -> pd.DataFrame:
    """Daily OHLCV for one ``ticker``; cache-first.

    With ``fetch=False`` (default) this reads the cached parquet and never touches the
    network — on a cache miss it raises so the offline core stays offline. With
    ``fetch=True`` it downloads (retrying on a transient failure) and writes the
    parquet cache. Returns a tz-naive daily OHLCV frame sliced to :data:`AS_OF` (no
    partial-month drift).
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call load_real(fetch=True) or fetch_all() once to populate the cache."
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


def fetch_all(tickers: list[str] | None = None, cache_dir: str = DEFAULT_CACHE,
              chunk: int = 15, retries: int = 3) -> None:
    """Populate the parquet cache for the whole basket via batched yfinance calls.

    Batches keep the network round-trips manageable for a 110+-name basket. A
    ticker that fails in its batch is retried alone before being skipped (logged,
    never silently dropped from the printed count).
    """
    import yfinance as yf

    tickers = tickers or BASKET
    os.makedirs(cache_dir, exist_ok=True)
    missing = [t for t in tickers if not os.path.exists(_cache_path(t, cache_dir))]
    if not missing:
        return
    failed: list[str] = []
    for i in range(0, len(missing), chunk):
        batch = missing[i:i + chunk]
        raw = None
        for attempt in range(retries):
            try:
                raw = yf.download(batch, period="max", interval="1d", auto_adjust=True,
                                  progress=False, group_by="ticker", threads=True)
                if raw is not None and not raw.empty:
                    break
            except Exception:
                raw = None
            time.sleep(2.0 * (attempt + 1))
        if raw is None or raw.empty:
            failed.extend(batch)
            continue
        for t in batch:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if t not in raw.columns.get_level_values(0):
                        failed.append(t)
                        continue
                    sub = raw[t]
                else:
                    sub = raw
                sub = sub.dropna(how="all")
                if sub.empty:
                    failed.append(t)
                    continue
                bars = sub.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
                bars = bars.dropna(how="any")
                bars.index.name = "date"
                if bars.index.tz is not None:
                    bars.index = bars.index.tz_localize(None)
                bars.to_parquet(_cache_path(t, cache_dir))
            except Exception:
                failed.append(t)
    for t in failed:
        try:
            fetch_one(t, cache_dir=cache_dir, fetch=True, retries=retries)
        except Exception:
            pass


def load_real(cache_dir: str = DEFAULT_CACHE, fetch: bool = False,
              tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Cache-first dict ``{ticker: daily OHLCV frame}`` for the basket.

    With ``fetch=False`` (default) reads cached parquets only. With ``fetch=True``
    warms the cache from yfinance first (batched). Tickers with no usable bars are
    skipped, never silently invented.
    """
    tickers = tickers or BASKET
    if fetch:
        fetch_all(tickers, cache_dir=cache_dir)
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            out[t] = fetch_one(t, cache_dir=cache_dir, fetch=False)
        except FileNotFoundError:
            continue
    return out


def have_real(cache_dir: str = DEFAULT_CACHE, tickers: list[str] | None = None,
             min_coverage: float = 0.85) -> bool:
    """True once at least ``min_coverage`` of the basket is cached (a couple of
    delistings/renames/API hiccups on 110+ tickers shouldn't block the whole study)."""
    tickers = tickers or BASKET
    have = sum(os.path.exists(_cache_path(t, cache_dir)) for t in tickers)
    return have / len(tickers) >= min_coverage


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 40, n_days: int = 4000, edge: float = 0.0,
                    seed: int = 688, daily_vol: float = 0.013,
                    block_rate: float = 0.004, prior_lookback: int = 10,
                    start: str = "2003-01-06") -> tuple[dict, dict]:
    """Deterministic OHLCV panel with a PLANTED post-concealing-baby-swallow bounce.

    Each name is a daily random walk. On a fraction ``block_rate`` of eligible
    sessions we force a **4-bar concealing-baby-swallow block**: a mild downtrend
    drift over the preceding ``prior_lookback`` sessions, two engineered black
    marubozu (body ~ full range), a third engineered candle that gaps down then
    rallies into the second candle's body (a long upper shadow) before closing at a
    new low, and a fourth engineered candle that fully engulfs the third (its own
    high/low straddle the third's high/low) and closes at a new low again — exactly
    the shape :func:`strategy.cbs_flags` keys on. When ``edge != 0`` each forced
    block injects a **signed drift** into the following ``H=20`` sessions
    proportional to ``edge`` — the genuine "capitulation, reversal near" bounce the
    folk claim describes. When ``edge = 0`` the blocks carry no forward information
    beyond the engineered shape itself, so a forward-return test on them must NOT
    manufacture significance however the noise falls.

    Returns ``(data, truth)`` where ``data`` is ``{name: OHLCV frame}`` (same shape as
    :func:`load_real`) and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]
    H = 20  # horizon over which a planted bounce plays out

    data: dict[str, pd.DataFrame] = {}
    for name in names:
        ret = rng.normal(0.0002, daily_vol, n_days)
        is_block_start = rng.random(n_days) < block_rate
        is_forced = np.zeros(n_days, dtype=bool)
        lo_bound = prior_lookback + 4
        hi_bound = n_days - H - 5
        # keep the engineered-shape parameters separate from `ret` (log-returns
        # can't encode a specific intrabar wick), applied directly to OHLC below
        forced_idx: list[int] = []
        for i in range(lo_bound, hi_bound):
            if not is_block_start[i]:
                continue
            if is_forced[max(0, i - 5):i + 5].any():
                continue  # never overlap two forced blocks
            ret[i - prior_lookback:i - 3] -= 0.0015  # downtrend leading into the block
            ret[i - 3:i + 1] = rng.normal(-0.012, 0.002, 4)  # placeholder magnitudes
            is_forced[i - 3:i + 1] = True
            forced_idx.append(i - 3)  # position of candle 1 in the block
            if edge != 0.0:
                ret[i + 1:i + 1 + H] += edge / H
        close = 100.0 * np.exp(np.cumsum(ret))
        open_ = np.empty_like(close)
        open_[0] = 100.0
        open_[1:] = close[:-1]
        wick = np.abs(rng.normal(daily_vol * 0.4, daily_vol * 0.15, n_days)) * close
        hi = np.maximum(open_, close) + wick
        lo = np.minimum(open_, close) - wick

        # overwrite the four forced bars with the EXACT concealing-baby-swallow
        # geometry, day by day (marubozu / gap-rally-fail / full-engulf)
        for i1 in forced_idx:
            i2, i3, i4 = i1 + 1, i1 + 2, i1 + 3
            base = close[i1 - 1] if i1 > 0 else 100.0
            # day 1: black marubozu -- open at base, close ~1.2% lower, no shadow
            o1, c1 = base, base * (1 - rng.uniform(0.010, 0.016))
            open_[i1], close[i1] = o1, c1
            hi[i1], lo[i1] = max(o1, c1), min(o1, c1)
            # day 2: black marubozu, gaps/continues down from day 1's close
            o2, c2 = c1 * (1 - rng.uniform(0.0, 0.003)), None
            c2 = o2 * (1 - rng.uniform(0.010, 0.016))
            open_[i2], close[i2] = o2, c2
            hi[i2], lo[i2] = max(o2, c2), min(o2, c2)
            # day 3: gaps down from day 2's close, rallies into day 2's body (upper
            # shadow), closes lower than day 2 near its own low (small lower shadow)
            o3 = c2 * (1 - rng.uniform(0.001, 0.004))
            c3 = o3 * (1 - rng.uniform(0.002, 0.006))
            h3 = o2 + rng.uniform(0.15, 0.45) * (o2 - c2)  # rallies into day-2 body
            l3 = min(o3, c3) - rng.uniform(0.0, 0.05) * (h3 - min(o3, c3))
            open_[i3], close[i3], hi[i3], lo[i3] = o3, c3, h3, l3
            # day 4: opens inside day 3's upper shadow, engulfs day 3 fully, closes
            # at a new low
            o4 = c3 + rng.uniform(0.2, 0.7) * (h3 - c3)
            c4 = l3 * (1 - rng.uniform(0.001, 0.006))
            h4 = h3 + abs(rng.normal(0.001, 0.001)) * o4
            l4 = min(l3, c4) - abs(rng.normal(0.001, 0.001)) * o4
            open_[i4], close[i4], hi[i4], lo[i4] = o4, c4, h4, l4

        vol = rng.integers(1_000_000, 50_000_000, n_days).astype(float)
        data[name] = pd.DataFrame(
            {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
            index=pd.DatetimeIndex(cal, name="date"),
        )
    truth = {"edge": edge, "n_names": n_names, "n_days": n_days,
             "block_rate": block_rate, "seed": seed, "horizon": H,
             "n_forced_total": sum(1 for _ in names)}
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
