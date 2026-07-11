"""Data layer for Study 694 — the Matching Low candlestick pattern.

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date), so the detector and
the inference never care whether the bars are real or planted:

* **Real tape.** Daily OHLCV (yfinance, ``auto_adjust=True``, no key) for a fixed basket
  of long-listed, liquid US large-caps + **SPY**, cached as one parquet per ticker under
  ``_cache/``. Cache-first: the network is touched only on an explicit cache miss (then
  the bars are cached so every re-run is offline). This is the **same 30-name basket**
  the desk's other candlestick-pattern studies use (408-three-black-crows,
  683-evening-star, 689-upside-gap-two-crows, 693-tasuki-gap) for a like-for-like
  comparison. A **survivors** basket (all still trading) — survivorship is named on the
  Signal axis.

* **Synthetic.** ``synthetic_panel`` — a deterministic, fixed-seed generator with a
  **planted-reversal knob**. With ``edge = 0`` the post-pattern path is pure noise (the
  detector must NOT manufacture a reversal edge); with ``edge > 0`` the bars after a
  genuine matching low are nudged upward for a few sessions — the positive control
  proves the harness *can* bank a planted reversal.

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

# The SAME 30-name basket the desk's sibling candlestick-pattern studies use
# (408-three-black-crows, 683-evening-star, 689-upside-gap-two-crows, 693-tasuki-gap) —
# long-listed, liquid US large-caps + the market proxy (SPY), chosen for long, clean
# daily OHLC history on yfinance and sector spread. This is a *survivors* basket (every
# name still trades in 2026) — named on the Signal axis: a fixed surviving-names basket
# cannot include firms that were delisted after a decline that never reversed, a mild
# bias reasoned about explicitly in docs/results.md.
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
    return os.path.join(cache_dir, f"ml_{safe}_1d.parquet")


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
# Synthetic positive control — deterministic, planted-reversal knob
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 30, n_days: int = 2200, edge: float = 0.0,
                    seed: int = 694, daily_vol: float = 0.013,
                    start: str = "2010-01-04", pull_days: int = 3,
                    tol: float = 0.0015) -> tuple[dict[str, pd.DataFrame], dict]:
    """Deterministic OHLCV panel with a PLANTED post-pattern REVERSAL (knob ``edge``).

    Each name is a random walk in log-price with a mild drift and a wick component — no
    overnight gap (the matching low, unlike the tasuki gap, needs no gap; the open of
    bar ``i`` is simply the previous close). After each bar we test whether the last two
    bars form a matching low (see ``strategy.matching_low_signal``, same ``tol``); if
    ``edge != 0`` we add an *extra* upward drift of ``edge`` to the next ``pull_days``
    bars' returns — a genuine bullish reversal the detector should bank.

    - ``edge = 0`` -> pure random walk: any post-pattern move is luck, the detector must
      NOT manufacture significance however the noise falls.
    - ``edge > 0`` -> a genuine planted reversal the detector must light up on (the
      forward return after a matching low is nudged positive).

    Returns ``(panel dict, truth dict)`` — the panel matches ``load_real`` in shape.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)
    panel: dict[str, pd.DataFrame] = {}
    n_planted = 0
    for k in range(n_names):
        eps = rng.normal(0.0001, daily_vol, n_days)
        close = np.empty(n_days)
        open_ = np.empty(n_days)
        hi = np.empty(n_days)
        lo = np.empty(n_days)
        wick = np.abs(rng.normal(0.0, daily_vol * 0.35, n_days))
        prev_close = 100.0
        pull_left = 0
        for i in range(n_days):
            r = eps[i] + (edge if (edge != 0.0 and pull_left > 0) else 0.0)
            if pull_left > 0:
                pull_left -= 1
                n_planted += 1
            o = prev_close                      # no overnight gap: open = prior close
            c = o * np.exp(r)
            open_[i] = o
            close[i] = c
            w = wick[i] * max(o, c)
            hi[i] = max(o, c) + w
            lo[i] = min(o, c) - w

            # detect a matching low on bars (i-1, i) -> arm the planted pull for i+1..
            if i >= 1 and edge != 0.0:
                if _matching_low_from_arrays(open_, close, i, tol=tol):
                    pull_left = pull_days

            prev_close = c
        vol = rng.integers(1_000_000, 40_000_000, n_days).astype(float)
        panel[f"N{k:02d}"] = pd.DataFrame(
            {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
            index=pd.DatetimeIndex(cal, name="date"),
        )
    truth = {"edge": edge, "n_names": n_names, "n_days": n_days, "seed": seed,
             "pull_days": pull_days, "n_planted_days": int(n_planted)}
    return panel, truth


def _matching_low_from_arrays(o: np.ndarray, c: np.ndarray, i: int,
                              tol: float = 0.0015) -> bool:
    """Inline matching-low test on bars (i-1, i); mirrors ``strategy.matching_low_signal``
    so the synthetic generator stays causal and self-contained (no circular import at
    module build time)."""
    o0, o1 = o[i - 1], o[i]
    c0, c1 = c[i - 1], c[i]
    down0 = c0 < o0
    down1 = c1 < o1
    if not (down0 and down1):
        return False
    denom = abs(c0) if abs(c0) > 1e-9 else 1e-9
    return abs(c1 - c0) / denom <= tol
