"""Data layer for Study 690 — Three Stars in the South.

The **three stars in the south** is one of candlestick lore's more obscure bullish
reversal claims (Nison, *Japanese Candlestick Charting Techniques*): in a downtrend,
three consecutive **black** (bearish) candles print, each with a **shrinking** real
body/range and a **rising low** — sellers pushing, but with visibly less and less
conviction each session — and the third candle is a small near-marubozu that never
even tests the second candle's low. The story: the selling is running out of ammunition
one session at a time, and the exhaustion itself is the reversal signal.

Because it needs **three** independent, low-probability shape conditions to line up in
a row (shrinking body/range AND a rising low, three times, inside an active downtrend),
it is at least as combinatorially rare as the desk's other rare multi-bar patterns
(ladder bottom, tri-star doji). To have any chance of a usable sample we pool a broad,
long-listed basket — the same idiom, and the same fixed 61-name basket, as siblings
685-tri-star-doji and 687-ladder-bottom (chosen for long history + sector spread, not
cherry-picked for this study).

This module produces two tapes, both a tz-naive daily OHLCV frame indexed by date:

* ``load_real(cache_dir=DEFAULT_CACHE)`` — the real Yahoo! daily tape (``yfinance``),
  **cache-first**: reads cached parquets and only touches the network on a cache miss
  (with retries + backoff), then caches so re-runs are offline. SPY + 60 long-listed,
  liquid US large-caps, ~25 years each. This is a **survivors** basket (all still
  trading) — named on the Signal axis, though for a single-pattern event study it
  affects which *names* contribute events, not the direction of the comparison.

* ``synthetic_panel(...)`` — a deterministic, offline generator returning
  ``(data, truth)``. A single ``edge`` knob plants the only structure a three-stars
  rule could possibly harvest: forced 3-bar shrinking-body / rising-low blocks, planted
  only where the plain random walk is **already** in a downtrend on its own (no
  artificial drift is injected to manufacture one), at a controlled rate, each
  optionally followed by a genuine, signed post-pattern bounce. ``edge = 0`` is a pure
  random walk beyond the forced candle shapes — the blocks carry no forward information,
  and the detector must not manufacture significance from them. A synthetic positive
  control proves the machinery can find a planted edge; it never backs a REAL stamp on
  the real tape.

No look-ahead is baked in here: the pattern is confirmed on bars *t-2..t*'s OHLC;
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

# A broad, transparent, fixed basket of long-listed US large-caps with deep, clean OHLC
# histories on yfinance, plus SPY as the index reference — the same basket idiom siblings
# 685 (tri-star doji) and 687 (ladder bottom) use, chosen for the same reason: a rare
# multi-bar pattern needs many names, many years, for any usable sample at all. This is a
# *survivors* basket — survivorship is named on the Signal axis.
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
    return os.path.join(cache_dir, f"tss_{safe}_1d.parquet")


def fetch_one(ticker: str, period: str = "25y", cache_dir: str = DEFAULT_CACHE,
              fetch: bool = False, retries: int = 3) -> pd.DataFrame:
    """Daily OHLCV for one ``ticker``; cache-first.

    With ``fetch=False`` (default) this reads the cached parquet and never touches the
    network — on a cache miss it raises so the offline core stays offline. With
    ``fetch=True`` it downloads (retrying a couple of times with backoff on a transient
    failure) and writes the parquet cache. Returns a tz-naive daily OHLCV frame sliced
    to :data:`AS_OF` (no partial-month drift).
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
def synthetic_panel(n_names: int = 60, n_days: int = 3500, edge: float = 0.0,
                    seed: int = 690, daily_vol: float = 0.013,
                    star_rate: float = 0.008, prior_lookback: int = 10,
                    start: str = "2005-01-03") -> tuple[dict, dict]:
    """Deterministic OHLCV panel with a PLANTED post-pattern bounce (knob ``edge``).

    Each name is a daily random walk. On a fraction ``star_rate`` of *candidate* sessions
    that already sit in a **naturally occurring** downtrend (the plain random walk, with
    no drift injected to manufacture one) we force a **3-bar "three stars" block**: three
    engineered bearish sessions with **shrinking magnitude** and **rising lows** —
    exactly the shape :func:`strategy.three_stars_flags` keys on. Requiring the downtrend
    to already be there (rather than drifting the price down artificially over the
    preceding sessions) keeps star events and the downtrend-matched base rate drawing
    from the *same kind* of prior history, so their forward returns are directly
    comparable and the null is honest. When ``edge != 0``, each forced block injects a
    **signed drift** into the following ``H=20`` sessions proportional to ``edge`` — the
    genuine "selling exhaustion ends the downtrend" bounce the folk claim describes. When
    ``edge = 0`` the blocks carry no forward information beyond the engineered shape
    itself, so a forward-return test on them must NOT manufacture significance however
    the noise falls.

    Returns ``(data, truth)`` where ``data`` is ``{name: OHLCV frame}`` (same shape as
    :func:`load_real`) and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]
    H = 20  # horizon over which a planted bounce plays out

    data: dict[str, pd.DataFrame] = {}
    active_masks: dict[str, np.ndarray] = {}
    for name in names:
        ret = rng.normal(0.0002, daily_vol, n_days)
        # the UNFORCED baseline path decides eligibility: a candidate is only turned
        # into a three-stars block if the plain random walk already put it in a
        # downtrend on its own -- no artificial drift is injected anywhere else on the
        # tape, so star events and the base rate are drawn from the same population.
        close0 = 100.0 * np.exp(np.cumsum(ret))
        is_block_start = rng.random(n_days) < star_rate
        # marks the three FORCED star sessions of every accepted block. Used below (as
        # the "active mask") so a base-rate candidate -- or a DIFFERENT star event --
        # whose own forward-return window overlaps somebody else's forced star days
        # isn't scored on a return that partly belongs to a different, unrelated block.
        # A candidate never overlaps its OWN block this way: its forward window starts
        # the session *after* its own three forced days end.
        is_forced = np.zeros(n_days, dtype=bool)
        lo_bound = prior_lookback + 5
        hi_bound = n_days - H - 6
        blocks = []  # (block_start_index, mags) for the second geometry pass below
        for i in range(lo_bound, hi_bound):
            if not is_block_start[i]:
                continue
            if is_forced[max(0, i - 5):i + 5].any():
                continue  # never overlap two forced blocks
            if not (close0[i - 2] < close0[i - 2 - prior_lookback]):
                continue  # only plant on a session that is ALREADY in a downtrend
            # three engineered, shrinking-magnitude bearish sessions (the "stars")
            mags = np.sort(rng.uniform(0.006, 0.018, 3))[::-1]  # descending magnitudes
            for k in range(3):
                ret[i + k] = -mags[k]
                is_forced[i + k] = True
            blocks.append((i, mags))
            if edge != 0.0:
                ret[i + 3:i + 3 + H] += edge / H
        close = 100.0 * np.exp(np.cumsum(ret))
        open_ = np.empty_like(close)
        open_[0] = 100.0
        open_[1:] = close[:-1]
        wick = np.abs(rng.normal(daily_vol * 0.3, daily_vol * 0.1, n_days)) * close
        hi = np.maximum(open_, close) + wick
        lo = np.minimum(open_, close) - wick

        # Second pass: on the forced star blocks, replace the generic wick with an
        # EXPLICIT geometry that guarantees the claimed shape -- shrinking range AND a
        # strictly rising low across the three stars -- rather than hoping a generic
        # random wick happens to produce it. Star 1 gets a generous "hammer" lower
        # shadow (selling met by some buying); each following star's shadow shrinks by
        # at least that day's price drop plus a small margin, so the LOW still rises
        # even while the close keeps drifting down; star 3's shadow (and upper wick)
        # is deliberately tiny -- a near-marubozu, exactly the strict cut's third star.
        for i, mags in blocks:
            shadow_prev = body_lo_prev = None
            for k in range(3):
                idx = i + k
                body_lo = min(open_[idx], close[idx])
                body_hi = max(open_[idx], close[idx])
                if k == 0:
                    shadow = 2.2 * mags[0] * close[idx]
                else:
                    price_drop = body_lo_prev - body_lo  # positive: price keeps sinking
                    margin = 0.08 * mags[2] * close[idx]
                    shadow = max(shadow_prev - price_drop - margin,
                                 0.02 * mags[2] * close[idx])
                upper = 0.12 * mags[k] * close[idx] * (0.3 if k == 2 else 1.0)
                lo[idx] = body_lo - shadow
                hi[idx] = body_hi + upper
                shadow_prev, body_lo_prev = shadow, body_lo

        vol = rng.integers(1_000_000, 50_000_000, n_days).astype(float)
        data[name] = pd.DataFrame(
            {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
            index=pd.DatetimeIndex(cal, name="date"),
        )
        active_masks[name] = is_forced
    truth = {"edge": edge, "n_names": n_names, "n_days": n_days,
             "star_rate": star_rate, "seed": seed, "horizon": H, "active_masks": active_masks}
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
