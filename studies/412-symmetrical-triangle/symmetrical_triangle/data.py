"""Data layer for Study 412 — Symmetrical Triangle.

Two tapes, one shape (a timezone-naive daily OHLCV frame indexed by date):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge
  knob**. It builds price paths that contain genuine symmetrical-triangle figures
  (a converging range that contracts toward an apex) and then plants a *continuation*
  drift after the breakout proportional to ``edge``:

    * ``edge = 0``   → the post-breakout path is a pure random walk; a confirmed
      breakout is a fair coin. The detector must NOT manufacture significance.
    * ``edge > 0``   → the breakout *continues* in its direction (a real follow-through
      edge the timing rule can harvest). The detector must light up.

  This is the positive control: a pipeline that cannot bank a planted continuation
  edge proves nothing by finding nothing on the real tape.

- ``fetch_panel`` / ``load_real`` — the real Yahoo! daily OHLCV tape (``yfinance``),
  **cache-first** so the test-suite and the reproducible core never touch the network.
  We use SPY plus a fixed large-cap basket, 2005-onward, giving 15-20 years of bars
  and hundreds of detected figures across names — a meaningful power budget.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the figure
is detected and the breakout confirmed on the close of bar *t*; the position is entered
at *t+1*'s open and forward returns measured from there.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# A fixed, transparent basket: SPY (the market) + long-listed US large-caps with deep,
# clean daily histories on yfinance. This is a *survivors* basket (all still trading);
# survivorship is named on the Signal axis — a surviving-names panel cannot capture
# firms that delisted, which mildly tilts continuation patterns up.
BASKET = [
    "SPY", "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "IBM",
    "CVX", "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT",
    "MMM", "HON", "UNH", "ORCL", "PEP", "ABT", "TXN", "COST", "LOW", "GS",
]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_names: int = 30,
    n_days: int = 2400,
    edge: float = 0.0,
    seed: int = 412,
    daily_vol: float = 0.012,
    n_triangles: int = 6,
    start: str = "2010-01-04",
) -> tuple[dict[str, pd.DataFrame], dict]:
    """A reproducible multi-name daily OHLCV panel with planted symmetrical triangles.

    Each name is a log random walk into which we splice ``n_triangles`` symmetrical
    triangles: a window where the *range* (distance between successive highs and lows)
    contracts linearly toward an apex while the close oscillates inside the shrinking
    envelope. At the apex the price breaks out in a (random) direction; if ``edge`` !=
    0 the ``H``-day path after the breakout receives an extra daily drift of
    ``edge * breakout_sign / 60`` — a genuine post-breakout *continuation*.

    - ``edge = 0`` → breakouts are followed by pure noise (the honest null).
    - ``edge > 0`` → breakouts continue (the planted, detectable edge).

    Returns ``(panel, truth)`` where ``panel`` maps name -> OHLCV frame (same shape as
    :func:`load_real`) and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)
    H = 60
    panel: dict[str, pd.DataFrame] = {}

    for k in range(n_names):
        ret = rng.normal(0.0002, daily_vol, size=n_days)
        # carve out non-overlapping triangle windows
        starts = sorted(rng.choice(np.arange(150, n_days - 200, 250),
                                   size=min(n_triangles, max(1, (n_days - 350) // 250)),
                                   replace=False))
        for s0 in starts:
            width = int(rng.integers(40, 80))          # bars to apex
            apex = s0 + width
            if apex + H + 5 >= n_days:
                continue
            amp = daily_vol * rng.uniform(4.0, 7.0)    # starting half-range
            # inside the triangle: oscillation whose amplitude shrinks to ~0 at apex
            for j in range(width):
                frac = 1.0 - j / width
                osc = np.sin(j * np.pi / 6.0) * amp * frac
                ret[s0 + j] = osc - (ret[s0 + j - 1] if j else 0.0) * 0.0 + rng.normal(0, daily_vol * 0.3)
            # breakout direction + planted continuation
            bsign = 1.0 if rng.random() < 0.5 else -1.0
            ret[apex] += bsign * amp * 1.5             # the breakout bar
            if edge != 0.0:
                ret[apex + 1:apex + 1 + H] += edge * bsign / H
        close = 100.0 * np.exp(np.cumsum(ret))
        prev = np.empty(n_days); prev[0] = 100.0; prev[1:] = close[:-1]
        of = rng.uniform(0.1, 0.9, n_days)
        open_ = prev * np.exp(of * ret)
        wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
        high = np.maximum(open_, close) + wick
        low = np.minimum(open_, close) - wick
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        vol = rng.integers(1_000_000, 5_000_000, n_days).astype(float)
        panel[f"N{k:02d}"] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
            index=pd.DatetimeIndex(sessions, name="date"),
        )

    truth = {"edge": edge, "n_names": n_names, "n_days": n_days,
             "n_triangles": n_triangles, "seed": seed}
    return panel, truth


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily OHLCV, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def fetch_daily(ticker: str, start: str = "2005-01-01", end: str | None = None,
                fetch: bool = False, cache_dir: str = DEFAULT_CACHE,
                retries: int = 3) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-first, network only on ``fetch=True``.

    Returns a tz-naive frame indexed by date with columns
    ``open, high, low, close, volume`` (lowercase). On a network fetch the parquet is
    cached so re-runs are fully offline. Retries a couple of times with backoff if
    yfinance rate-limits.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_daily({ticker!r}, fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf
        raw = None
        for attempt in range(retries):
            try:
                raw = yf.download(ticker, start=start, end=end, interval="1d",
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
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars


def fetch_panel(tickers: list[str] | None = None, start: str = "2005-01-01",
                end: str | None = None, cache_dir: str = DEFAULT_CACHE) -> dict[str, pd.DataFrame]:
    """Download (and cache) the whole basket. Network-only; used once to build the cache."""
    tickers = tickers or BASKET
    panel = {}
    for tk in tickers:
        try:
            panel[tk] = fetch_daily(tk, start=start, end=end, fetch=True, cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {tk}: {exc}")
    return panel


def have_real(tickers: list[str] | None = None, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every basket cache parquet exists (offline-ready)."""
    tickers = tickers or BASKET
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_real(cache_dir: str = DEFAULT_CACHE,
              tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Cache-first load of the basket -> {ticker: OHLCV frame}. Offline once cached."""
    tickers = tickers or BASKET
    panel = {}
    for tk in tickers:
        p = _cache_path(tk, cache_dir)
        if os.path.exists(p):
            panel[tk] = fetch_daily(tk, fetch=False, cache_dir=cache_dir)
    return panel


def fingerprint(panel: dict[str, pd.DataFrame]) -> str:
    """Short content fingerprint over the concatenated close columns (the as-of stamp)."""
    h = hashlib.sha1()
    for tk in sorted(panel):
        h.update(tk.encode())
        h.update(np.ascontiguousarray(panel[tk]["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
