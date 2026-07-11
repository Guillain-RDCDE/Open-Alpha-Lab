"""Data layer for Study 689 — the Upside Gap Two Crows candlestick pattern.

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date), so the detector and
the inference never care whether the bars are real or planted:

* **Real tape.** Daily OHLCV (yfinance, ``auto_adjust=True``, no key) for a fixed basket
  of long-listed, liquid US large-caps + **SPY**, cached as one parquet per ticker under
  ``_cache/``. Cache-first: the network is touched only on an explicit cache miss (then
  the bars are cached so every re-run is offline). This is a **survivors** basket (all
  still trading) — survivorship is named on the Signal axis. Same 30-name basket the
  desk's other candlestick-pattern studies (408-three-black-crows, 683-evening-star) use,
  for a like-for-like comparison.

* **Synthetic.** ``synthetic_panel`` — a deterministic, fixed-seed generator with a
  **planted-edge knob**. With ``edge = 0`` the post-pattern path is pure noise (a fair
  coin: the pattern must NOT manufacture an edge); with ``edge > 0`` the bars are nudged
  so that the day(s) *after* a real upside-gap-two-crows keep falling — the positive
  control proves the harness *can* bank a planted continuation.

Pure numpy + pandas + stdlib for the offline path. ``fetch_one`` (network) is only used
on a cache miss and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# A transparent, fixed basket of large, long-listed, liquid US large-caps + the market
# proxy (SPY). Chosen for long, clean daily OHLC history on yfinance and sector spread —
# the same 30-name basket the desk's sibling candlestick studies (408-three-black-crows,
# 683-evening-star) use, for a like-for-like comparison. This is a *survivors* basket
# (every name still trades in 2026) — named on the Signal axis: a fixed surviving-names
# basket cannot include firms that were delisted after a bad run, a mild bias we reason
# about explicitly in docs/results.md.
BASKET = [
    "SPY", "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "IBM",
    "CVX", "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT",
    "MMM", "HON", "UNH", "ORCL", "PEP", "ABT", "TXN", "COST", "LOW", "GE",
]


# --------------------------------------------------------------------------- #
# Real tape — cache-first yfinance
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"ugtc_{safe}_1d.parquet")


def fetch_one(ticker: str, start: str = "2005-01-01", end: str | None = None,
              cache_dir: str = DEFAULT_CACHE, retries: int = 3) -> pd.DataFrame:
    """Download one ticker's daily OHLCV and cache it (network; cache miss only).

    Retries a couple of times with a small backoff if yfinance rate-limits, then caches
    the parquet so every re-run is offline. Returns a tz-naive OHLCV frame.
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
    os.makedirs(cache_dir, exist_ok=True)
    bars.to_parquet(_cache_path(ticker, cache_dir))
    return bars


def load_one(ticker: str, cache_dir: str = DEFAULT_CACHE,
             allow_fetch: bool = True) -> pd.DataFrame:
    """One ticker's daily OHLCV — cache-first; fetch on a miss if ``allow_fetch``."""
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        bars = pd.read_parquet(path)
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        bars.index = pd.DatetimeIndex(bars.index, name="date")
        return bars
    if not allow_fetch:
        raise FileNotFoundError(
            f"No cached daily tape for {ticker} at {path}. "
            f"Call data.load_one({ticker!r}) with network once to populate the cache."
        )
    return fetch_one(ticker, cache_dir=cache_dir)


def have_real(basket: list[str] | None = None, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every basket name has a cached parquet (offline-ready)."""
    basket = basket or BASKET
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in basket)


def load_real(cache_dir: str = DEFAULT_CACHE, basket: list[str] | None = None,
              allow_fetch: bool = True, asof: str | None = None
              ) -> dict[str, pd.DataFrame]:
    """Load the whole basket as a dict ticker -> daily OHLCV frame (cache-first).

    ``asof`` (if given) trims every ticker's tape to rows on or before that date, so a
    headline run stays pinned even as the cache is refreshed with newer sessions.
    """
    basket = basket or BASKET
    out: dict[str, pd.DataFrame] = {}
    for t in basket:
        try:
            bars = load_one(t, cache_dir=cache_dir, allow_fetch=allow_fetch)
            if asof is not None:
                bars = bars[bars.index <= pd.Timestamp(asof)]
            out[t] = bars
        except Exception:
            continue
    return out


def fingerprint(panel: dict[str, pd.DataFrame]) -> str:
    """Short sha1[:12] content fingerprint of the basket (closes), for the as-of stamp."""
    h = hashlib.sha1()
    for t in sorted(panel):
        h.update(t.encode())
        h.update(np.ascontiguousarray(panel[t]["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic positive control — deterministic, planted-edge knob
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 30, n_days: int = 2200, edge: float = 0.0,
                    seed: int = 689, daily_vol: float = 0.013,
                    start: str = "2010-01-04") -> tuple[dict[str, pd.DataFrame], dict]:
    """Deterministic OHLCV panel with a PLANTED post-pattern crash (knob ``edge``).

    Each name is a random walk in log-price with a mild upward drift (so an occasional
    bullish bar0 followed by two down bars — the pattern's raw material — actually
    occurs) and a modest overnight-gap component (so genuine price gaps form sometimes).
    We build proper OHLC bars: the open gaps from the previous close, the close is the
    random-walk close, and wicks extend the high/low. After each bar we test whether the
    last three bars formed a textbook **upside gap two crows** (see
    ``strategy.is_upside_gap_two_crows``); if ``edge != 0`` we add an *extra* drift of
    ``-edge`` to the **next** day's return — i.e. a real day-after continuation down in
    the direction the bearish pattern predicts.

    - ``edge = 0`` -> pure random walk: any post-pattern move is luck, the detector must
      NOT manufacture significance however the noise falls.
    - ``edge > 0`` -> a genuine planted crash the detector must light up on (signed short,
      so the signed forward return is POSITIVE).

    Returns ``(panel dict, truth dict)`` — the panel matches ``load_real`` in shape.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)
    panel: dict[str, pd.DataFrame] = {}
    n_planted = 0
    for k in range(n_names):
        eps = rng.normal(0.0004, daily_vol, n_days)   # mild up-drift feeds bullish bar0s
        gap = rng.normal(0.0, daily_vol * 0.6, n_days)  # overnight gap component
        close = np.empty(n_days)
        open_ = np.empty(n_days)
        hi = np.empty(n_days)
        lo = np.empty(n_days)
        wick = np.abs(rng.normal(0.0, daily_vol * 0.35, n_days))
        prev_close = 100.0
        pending = 0      # -1 if an upside-gap-two-crows just formed (applies to next bar)
        for i in range(n_days):
            r = eps[i] + (edge * pending if edge != 0.0 else 0.0)
            if pending != 0:
                n_planted += 1
            o = prev_close * np.exp(gap[i])     # open gaps overnight from prior close
            c = o * np.exp(r)                   # then drifts intraday to the close
            close[i] = c
            open_[i] = o
            w = wick[i] * max(o, c)
            hi[i] = max(o, c) + w
            lo[i] = min(o, c) - w
            # detect upside gap two crows on bars (i-2, i-1, i) -> set pending for i+1
            pending = 0
            if i >= 2:
                trip = _ugtc_from_arrays(open_, close, hi, lo, i)
                if trip == -1:
                    pending = -1
            prev_close = c
        vol = rng.integers(1_000_000, 40_000_000, n_days).astype(float)
        panel[f"N{k:02d}"] = pd.DataFrame(
            {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
            index=pd.DatetimeIndex(cal, name="date"),
        )
    truth = {"edge": edge, "n_names": n_names, "n_days": n_days, "seed": seed,
             "n_planted_days": int(n_planted)}
    return panel, truth


def _ugtc_from_arrays(o: np.ndarray, c: np.ndarray, h: np.ndarray, l: np.ndarray,
                      i: int, true_range_gap: bool = False) -> int:
    """Inline upside-gap-two-crows test on bars (i-2, i-1, i); -1 if present else 0.

    Mirrors ``strategy.is_upside_gap_two_crows`` so the synthetic generator stays causal
    and self-contained (no circular import at module build time). Same default (loose,
    body-level gap) definition as the real-tape basic detector.
    """
    o0, o1, o2 = o[i - 2], o[i - 1], o[i]
    c0, c1, c2 = c[i - 2], c[i - 1], c[i]
    h0 = h[i - 2]
    l1 = l[i - 1]
    # 1. bar0: bullish white body
    if not (c0 > o0):
        return 0
    # 2. gap up bar0 -> bar1
    if true_range_gap:
        if not (l1 > h0):
            return 0
    else:
        if not (min(o1, c1) > max(o0, c0)):
            return 0
    # 3. bar1: bearish black body
    if not (c1 < o1):
        return 0
    # 4. bar2: bearish black body, engulfing bar1's body from above
    if not (c2 < o2):
        return 0
    if not (o2 >= o1 and c2 <= c1):
        return 0
    # 5. gap not fully filled: bar2 close remains above bar0 close
    if not (c2 > c0):
        return 0
    return -1
