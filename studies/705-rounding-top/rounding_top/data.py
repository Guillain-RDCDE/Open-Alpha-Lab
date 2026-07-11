"""Data layer for Study 705 — Rounding Top (dome distribution).

Two tapes, one shape (a tz-naive daily OHLC frame per ticker, columns
``open/high/low/close/volume``):

* **Real tape.** Daily adjusted OHLC for SPY plus a fixed large-cap basket
  (``yfinance``, no key). Cache-first: each ticker is stored as a parquet under
  ``_cache/`` and the offline core / notebooks never touch the network once the
  cache exists. Daily bars go back ~20+ years, giving plenty of windows to scan.

* **Synthetic.** A deterministic, fixed-seed generator that *plants* clean
  rounding-top figures (a smooth dome-shaped distribution followed by a confirmed
  breakdown below the rim/support) into an otherwise random-walk tape, with a
  controllable post-breakdown drift ``edge`` (negative = decline, the claimed
  direction). ``edge = 0`` plants the *shape* but no continuation — the positive
  control's null (the detector must NOT manufacture a decline edge from a shape
  with no follow-through); ``edge < 0`` plants a real decline the detector must
  recover.

Pure numpy + pandas + stdlib for the offline path. ``fetch_panel`` (network) is
only used to build the cache and is never imported by the offline cells.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# A transparent, fixed basket: SPY (the index) + long-listed US large-caps with deep,
# clean daily histories on yfinance. This is a *survivors* basket (all still trading) —
# survivorship is named on the Signal axis: a fixed surviving-names basket cannot include
# firms that were delisted/collapsed after a failed dome (bankruptcy, buyout at a
# discount, etc.), which mildly *deflates* any "rounding top predicts a decline" finding
# by dropping some of the worst confirmed outcomes — a bias AGAINST the bearish claim,
# named explicitly (the mirror of 416, where survivorship flatters the bullish claim).
BASKET = [
    "SPY", "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "IBM",
    "CVX", "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT",
    "MMM", "UNH", "ORCL", "PEP", "ABT", "TXN", "COST", "LOW", "AMGN", "GS",
]


# --------------------------------------------------------------------------- #
# Real tape — cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"rt_{safe}_1d.parquet")


def fetch_panel(tickers: list[str] | None = None, start: str = "2004-01-01",
                end: str | None = None, cache_dir: str = DEFAULT_CACHE,
                retries: int = 3, pause: float = 1.5) -> dict[str, pd.DataFrame]:
    """Download daily OHLC for ``tickers`` and cache each as a parquet.

    Network-only; used once to build the cache. Never imported by the offline cells.
    Retries a couple of times with a small backoff on rate-limit hiccups, then caches
    so all re-runs are offline. Returns the dict of fetched frames.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    tickers = tickers or BASKET
    os.makedirs(cache_dir, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        raw = None
        for attempt in range(retries):
            try:
                raw = yf.download(tk, start=start, end=end, interval="1d",
                                  auto_adjust=True, progress=False)
                if raw is not None and not raw.empty:
                    break
            except Exception:
                raw = None
            time.sleep(pause * (attempt + 1))
        if raw is None or raw.empty:
            print(f"  ! no bars for {tk} (skipped)")
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index = pd.DatetimeIndex(bars.index).tz_localize(None)
        bars.index.name = "date"
        bars = bars.dropna()
        bars.to_parquet(path)
        out[tk] = bars
        print(f"  cached {tk}: {len(bars)} bars -> {path}")
    return out


def have_real(tickers: list[str] | None = None, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff at least SPY plus a handful of basket names are cached."""
    tickers = tickers or BASKET
    present = [tk for tk in tickers if os.path.exists(_cache_path(tk, cache_dir))]
    return ("SPY" in present) and len(present) >= 5


def load_real(cache_dir: str = DEFAULT_CACHE,
              tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Cache-first load of the real OHLC panel as a dict ``{ticker: bars}``.

    Reads only the cached parquets — no network. Missing names are silently skipped
    (the panel is whatever was cached by ``fetch_panel``).
    """
    tickers = tickers or BASKET
    out: dict[str, pd.DataFrame] = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            continue
        bars = pd.read_parquet(path)
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        bars.index = pd.DatetimeIndex(bars.index, name="date")
        out[tk] = bars.dropna()
    return out


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 30, n_days: int = 3000, edge: float = 0.0,
                    seed: int = 705, daily_vol: float = 0.012,
                    n_patterns: int = 8, base_len: int = 90,
                    drift_len: int = 20, depth: float = 0.18,
                    start: str = "2008-01-02") -> tuple[dict[str, pd.DataFrame], dict]:
    """Deterministic OHLC panel with PLANTED rounding-top figures + a decline knob.

    Each name is a daily random walk into which ``n_patterns`` clean **rounding
    tops** are stamped: over ``base_len`` bars the log-price follows a smooth,
    inverted-U dome (a parabola of total ``depth``, peak in the middle) that
    returns to its rim/support, immediately followed by a small breakdown leg
    below the rim. If ``edge != 0`` the ``drift_len`` bars *after* the breakdown
    get an extra per-bar drift of ``edge / drift_len`` — a genuine post-breakdown
    decline the detector should harvest. Use a negative ``edge`` for the claimed
    (bearish) direction.

    The honest null is ``edge = 0``: the *shape* is present (so the detector
    fires) but there is **no** continuation, so a faithful pipeline must NOT find
    a forward decline edge.

    Returns ``({ticker: bars}, truth)`` where ``truth`` records the planted
    parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    panel: dict[str, pd.DataFrame] = {}

    for k in range(n_names):
        r = rng.normal(0.0002, daily_vol, n_days)
        # place patterns on a coarse grid, jittered, leaving room for base + drift
        slots = np.linspace(150, n_days - base_len - drift_len - 50, n_patterns)
        for s in slots:
            t0 = int(s + rng.integers(-30, 30))
            if t0 < 50 or t0 + base_len + drift_len >= n_days:
                continue
            # smooth dome over bars [t0 .. t0+base_len-1]: the top rises from the
            # support, domes over and returns to JUST above the support; a flat
            # consolidation ``pad`` sits at the support, then the final bar nudges
            # a hair below the left rim/support to trigger the confirmed-breakdown
            # cross. All post-breakdown follow-through is governed by ``edge`` so
            # edge=0 is a genuine null (shape present, no continuation), not a
            # planted artefact.
            #
            # Anchoring matters here: the detector's "left support" reference for
            # a breakdown confirmed at bar t is close[t-base_len+1] — i.e. bar
            # ``t0`` itself once the window lines up on the plant — so the level
            # path below is defined over base_len POINTS for bars t0..t0+base_len-1
            # with level[0] = 0 = that exact reference (bar t0's return, r[t0], is
            # left untouched — it IS the anchor, not part of the shape's delta).
            # ``pad`` is intentionally wide (not just a short flat lip): a short
            # pad leaves the dome's ascending/descending flank sitting inside the
            # detector's sliding lookback window for many bars *after* the plant
            # ends, so a later window can reference a still-elevated interior
            # dome level as "support" and mistake ordinary drift for a second,
            # spurious breakdown — a look-ahead-flavoured artefact that leaks the
            # dome's own decline into what should be a clean forward window and
            # inflates the null. A wide flat pad (support-level, not elevated)
            # for most of the trailing window removes that stale-reference bias
            # (verified across 20 seeds in the module's own test harness).
            pad = 33
            dome_len = base_len - pad - 1
            u = np.arange(dome_len + 1)
            dome = -depth * (2.0 * u / dome_len - 1.0) ** 2 + depth  # dome[0]=0=anchor
            dome[-1] = 0.006                                          # finish just over support
            level = np.concatenate([dome, np.zeros(pad)])             # flat pad at support
            level[-1] = -0.004                                        # final close just under support
            seg = np.diff(level)                                      # length base_len - 1
            r[t0 + 1:t0 + base_len] = seg + rng.normal(0.0, daily_vol * 0.18, base_len - 1)
            # continuation region begins the bar AFTER the breakdown close
            bd = t0 + base_len
            if edge != 0.0:
                r[bd:bd + drift_len] += edge / drift_len
        close = 100.0 * np.exp(np.cumsum(r))
        open_ = np.empty_like(close)
        open_[0] = 100.0
        open_[1:] = close[:-1]
        wick = np.abs(rng.normal(0.0, daily_vol * 0.4, n_days)) * close
        hi = np.maximum(open_, close) + wick
        lo = np.minimum(open_, close) - wick
        vol = rng.integers(1_000_000, 30_000_000, n_days).astype(float)
        panel[f"N{k:02d}"] = pd.DataFrame(
            {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
            index=pd.DatetimeIndex(idx, name="date"))

    truth = {"edge": edge, "n_names": n_names, "n_patterns": n_patterns,
             "base_len": base_len, "drift_len": drift_len, "depth": depth, "seed": seed}
    return panel, truth


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #
def fingerprint(panel: dict[str, pd.DataFrame]) -> str:
    """A short content fingerprint of a panel (sorted tickers + close columns)."""
    h = hashlib.sha1()
    for tk in sorted(panel):
        h.update(tk.encode())
        h.update(np.ascontiguousarray(panel[tk]["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
