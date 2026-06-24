"""Data layer for Study 416 — Rounding Bottom (saucer base).

Two tapes, one shape (a tz-naive daily OHLC frame per ticker, columns
``open/high/low/close/volume``):

* **Real tape.** Daily adjusted OHLC for SPY plus a fixed large-cap basket
  (``yfinance``, no key). Cache-first: each ticker is stored as a parquet under
  ``_cache/`` and the offline core / notebooks never touch the network once the
  cache exists. Daily bars go back ~20+ years, giving plenty of windows to scan.

* **Synthetic.** A deterministic, fixed-seed generator that *plants* clean
  rounding-bottom figures (a smooth U-shaped basin followed by a breakout) into an
  otherwise random-walk tape, with a controllable post-breakout drift ``edge``.
  ``edge = 0`` plants the *shape* but no continuation — the positive control's null
  (the detector must NOT manufacture an edge from a shape with no follow-through);
  ``edge > 0`` plants a real continuation the detector must recover.

Pure numpy + pandas + stdlib for the offline path. ``fetch_panel`` (network) is only
used to build the cache and is never imported by the offline cells.
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
# firms that delisted after a failed base, which mildly flatters any "base works" finding.
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
    return os.path.join(cache_dir, f"rb_{safe}_1d.parquet")


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
    return ("SPY" in [tk for tk in present]) and len(present) >= 5


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
                    seed: int = 416, daily_vol: float = 0.012,
                    n_patterns: int = 8, base_len: int = 90,
                    drift_len: int = 20, depth: float = 0.18,
                    start: str = "2008-01-02") -> tuple[dict[str, pd.DataFrame], dict]:
    """Deterministic OHLC panel with PLANTED rounding-bottom figures + a drift knob.

    Each name is a daily random walk into which ``n_patterns`` clean **rounding bottoms**
    are stamped: over ``base_len`` bars the log-price follows a smooth U (a parabola of
    total ``depth``) that returns the price to its rim, immediately followed by a small
    breakout leg above the rim. If ``edge != 0`` the ``drift_len`` bars *after* the
    breakout get an extra per-bar drift of ``edge / drift_len`` — a genuine
    post-breakout continuation the detector should harvest.

    The honest null is ``edge = 0``: the *shape* is present (so the detector fires) but
    there is **no** continuation, so a faithful pipeline must NOT find a forward edge.

    Returns ``({ticker: bars}, truth)`` where ``truth`` records the planted parameters.
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
            # smooth U over [0, base_len]: the basin returns to the rim and then nudges
            # just above it on the final bar so the close crosses the left rim (the
            # confirmed-breakout trigger). The rim-cross itself is a tiny, fixed nudge —
            # ALL post-breakout follow-through is governed by ``edge`` so that edge=0 is a
            # genuine null (shape present, no continuation), not a planted artefact.
            # The basin spans the first ``base_len - pad`` bars and recovers to JUST below
            # the rim; a short flat consolidation ``pad`` sits at the rim; the final bar
            # nudges a hair above the left rim to trigger the confirmed-breakout cross.
            # All post-breakout follow-through is governed by ``edge`` so edge=0 is a true
            # null (shape present, recovery finished, no continuation in the forward window).
            pad = 8
            basin_len = base_len - pad
            u = np.arange(basin_len + 1)
            basin = depth * (2.0 * u / basin_len - 1.0) ** 2 - depth   # rim≈0, trough=-depth
            basin[-1] = -0.006                                          # finish just under the rim
            shape = np.concatenate([basin, np.zeros(pad)])             # flat pad at the rim
            shape[-1] = 0.004                                          # final close just over rim
            seg = np.diff(shape)                                        # length base_len
            r[t0:t0 + base_len] = seg + rng.normal(0.0, daily_vol * 0.18, base_len)
            # continuation region begins the bar AFTER the breakout close
            bo = t0 + base_len
            if edge != 0.0:
                r[bo:bo + drift_len] += edge / drift_len
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
