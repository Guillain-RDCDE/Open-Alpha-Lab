"""Data layer for Study 407 — the Dark Cloud Cover / Piercing Line twins.

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date), so the detector and
the inference never care whether the bars are real or planted:

* **Real tape.** Daily OHLCV (yfinance, ``auto_adjust=True``, no key) for a fixed basket
  of long-listed, liquid US large-caps + **SPY**, cached as one parquet per ticker under
  ``_cache/``. Cache-first: the network is touched only on an explicit cache miss (then
  the bars are cached so every re-run is offline). This is a **survivors** basket (all
  still trading) — survivorship is named on the Signal axis.

* **Synthetic.** ``synthetic_panel`` — a deterministic, fixed-seed generator with a
  **planted-edge knob**. With ``edge = 0`` the post-pattern path is pure noise (a fair
  coin: the twins must NOT manufacture an edge); with ``edge > 0`` the bars are nudged so
  that the day *after* a Piercing Line drifts up (and after a Dark Cloud Cover, down) —
  the positive control proves the harness *can* bank a planted reversal.

Pure numpy + pandas + stdlib for the offline path. ``fetch_one`` (network) is only used
on a cache miss and is never imported by the notebooks' offline cells.

NB: this study deliberately re-uses Study 402's 30-name basket (same liquid large-caps +
SPY) so the candlestick teardowns are directly comparable; the cached parquets carry the
``dcp_`` prefix but are the same total-return-adjusted closes.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# A transparent, fixed basket of large, long-listed, liquid US large-caps + the market
# proxy (SPY). Chosen for long, clean daily OHLC history on yfinance and sector spread.
# This is a *survivors* basket (every name still trades in 2026) — named on the Signal
# axis: a fixed surviving-names basket cannot include firms that were delisted after a
# bad run, a mild bias we reason about explicitly in docs/results.md.
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
    return os.path.join(cache_dir, f"dcp_{safe}_1d.parquet")


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
              allow_fetch: bool = True) -> dict[str, pd.DataFrame]:
    """Load the whole basket as a dict ticker -> daily OHLCV frame (cache-first)."""
    basket = basket or BASKET
    out: dict[str, pd.DataFrame] = {}
    for t in basket:
        try:
            out[t] = load_one(t, cache_dir=cache_dir, allow_fetch=allow_fetch)
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
                    seed: int = 407, daily_vol: float = 0.013,
                    start: str = "2010-01-04") -> tuple[dict[str, pd.DataFrame], dict]:
    """Deterministic OHLCV panel with a PLANTED post-twin reversal (knob ``edge``).

    Each name is a random walk in log-price. We build proper OHLC bars: the open is the
    previous close, the close is the random-walk close, and symmetric wicks extend the
    high/low. After each bar we test whether it formed a textbook **Dark Cloud Cover** or
    **Piercing Line** (see ``strategy.is_dark_cloud_or_piercing``); if ``edge != 0`` we
    add an *extra* drift of ``edge * direction`` to the **next** day's return — i.e. a
    real day-after reversal in the direction the pattern predicts (up after a Piercing,
    down after a Dark Cloud).

    - ``edge = 0`` → pure random walk: any post-pattern move is luck, the detector must
      NOT manufacture significance however the noise falls.
    - ``edge > 0`` → a genuine planted reversal the detector must light up on.

    Returns ``(panel dict, truth dict)`` — the panel matches ``load_real`` in shape.
    """
    from .strategy import is_dark_cloud_or_piercing

    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)
    panel: dict[str, pd.DataFrame] = {}
    n_planted = 0
    for k in range(n_names):
        eps = rng.normal(0.0002, daily_vol, n_days)
        log_ret = eps.copy()
        close = np.empty(n_days)
        open_ = np.empty(n_days)
        hi = np.empty(n_days)
        lo = np.empty(n_days)
        wick = np.abs(rng.normal(0.0, daily_vol * 0.4, n_days))
        # Overnight gaps so the twins (which REQUIRE an open beyond the prior extreme) can
        # form on the synthetic — a pure open==prev_close walk would never trigger them.
        gap = rng.normal(0.0, daily_vol * 1.2, n_days)
        prev_close = 100.0
        # direction memory of the just-formed twin (applies to the next bar)
        pending = 0
        for i in range(n_days):
            r = log_ret[i] + (edge * pending if edge != 0.0 else 0.0)
            if pending != 0:
                n_planted += 1
            o = prev_close * np.exp(gap[i])
            c = o * np.exp(r)
            close[i] = c
            open_[i] = o
            w = wick[i] * max(o, c)
            hi[i] = max(o, c) + w
            lo[i] = min(o, c) - w
            # detect the twin on the bar just formed (uses i-1, i) -> set pending for i+1
            pending = 0
            if i >= 1:
                pending = is_dark_cloud_or_piercing(
                    open_[i - 1], hi[i - 1], lo[i - 1], close[i - 1], o, hi[i], lo[i], c)
            prev_close = c
        vol = rng.integers(1_000_000, 40_000_000, n_days).astype(float)
        panel[f"N{k:02d}"] = pd.DataFrame(
            {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
            index=pd.DatetimeIndex(cal, name="date"),
        )
    truth = {"edge": edge, "n_names": n_names, "n_days": n_days, "seed": seed,
             "n_planted_days": int(n_planted)}
    return panel, truth
