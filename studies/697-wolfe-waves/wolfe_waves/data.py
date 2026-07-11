"""Data layer for Study 697 — Wolfe Waves.

Two tapes, one shape (a tz-naive daily OHLC frame indexed by date):

* ``load_real`` / ``load_basket`` — real daily OHLC for SPY + a basket of broad indices/ETFs
  (yfinance, no key), cache-first into ``_cache/`` so the reproducible core and notebooks run
  offline once cached. Decades of daily bars give the pattern room to occur without us having
  to cherry-pick a chart.

* ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  ``edge = 0`` is a pure random walk (a fair coin), so the Wolfe-wave detector must NOT
  manufacture significance. ``edge > 0`` plants the one structure the pattern needs to be
  true: a converging 5-point wedge (1-2-3-4-5) whose point 5 throws over the 1-3 trendline,
  followed by a real snap back toward the 1-4 (EPA) line. With the planted edge the detector
  must light up — the positive control that proves the harness can detect the effect it plants.

Wolfe's own criteria were never published as a single canonical rule set — every trading-
education site states them slightly differently. We encode the one rule everybody agrees on
(point 5 pierces the 1-3 trendline) plus the channel/convergence checks most sources add, and
say so plainly in ``strategy.py`` and ``docs/references.md``.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# SPY = the headline tape; the rest are the broad indices/ETFs a Wolfe-wave chartist would draw
# wedges on (deep daily history, no exotic single names). Same basket family as sibling
# 445-elliott-wave for comparability.
TICKERS = ("SPY", "QQQ", "DIA", "IWM", "^GSPC", "^IXIC", "^DJI", "GLD")

AS_OF = "2026-06-30"   # last complete calendar month at publication (2026-07-10)


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"ww_{safe}_1d.parquet")


def load_real(ticker: str = "SPY", period: str = "max",
              cache_dir: str = DEFAULT_CACHE, fetch: bool = False,
              retries: int = 3, asof: str = AS_OF) -> pd.DataFrame:
    """Real daily OHLC for ``ticker``, sliced to ``asof``; cache-first.

    Returns a tz-naive daily OHLC frame (open/high/low/close) indexed by date. The cache is a
    parquet under ``_cache/``; once present, every re-run is offline. On a cache miss (or
    ``fetch=True``) it pulls from yfinance with a small retry/backoff and writes the parquet.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path) and not fetch:
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
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    bars = bars.dropna().sort_index()
    if asof:
        bars = bars[bars.index <= pd.Timestamp(asof)]
    return bars


def have_real(ticker: str = "SPY", cache_dir: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(_cache_path(ticker, cache_dir))


def load_basket(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE,
                fetch: bool = False, asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """Load every cached ticker we can (cache-first). Returns {ticker: OHLC frame}."""
    out = {}
    for tk in tickers:
        if fetch or have_real(tk, cache_dir):
            try:
                out[tk] = load_real(tk, cache_dir=cache_dir, fetch=fetch, asof=asof)
            except Exception:
                continue
    return out


def fetch_all(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> None:
    """Force a network fetch + cache write for every ticker (used once on a cache miss)."""
    for tk in tickers:
        load_real(tk, cache_dir=cache_dir, fetch=True, asof=None)


# --------------------------------------------------------------------------- #
# Synthetic positive control — deterministic, planted-edge knob
# --------------------------------------------------------------------------- #
def synthetic_panel(n_days: int = 6000, edge: float = 0.0, seed: int = 697,
                    daily_vol: float = 0.011, start: str = "2000-01-03"
                    ) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a planted Wolfe-wave edge knob.

    Built by direct construction of the log-price path (not by nudging per-bar drift), so the
    planted geometry is exact: five clean anchor points at fixed bar spacing, piecewise-linear
    in log-price between them with noise far too small to fragment a leg into spurious extra
    ZigZag pivots, followed by a genuinely quiet (flat, noise-only) gap before the next planted
    wedge — so any detected "point 5" is confirmed on quiet noise, never on the run-up to the
    *next* planted wedge.

    - ``edge = 0`` -> the wedge shape still forms (geometry alone: point 5 throws past the 1-3
      line), but there is NO extra reversal drift after point 5 — a pure random walk follows,
      and the detector must NOT manufacture an edge from geometry alone.
    - ``edge > 0`` -> plants a real post-point-5 reversal of size ``edge`` (a log-return) toward
      the 1-4 (EPA) line, over the following ``rev_len`` bars — precisely the structure the
      pattern needs to be true.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)
    mu = 0.00015
    leg = 12                                    # bars per anchor-to-anchor leg
    rs = (0.08, -0.11, 0.07, -0.12)              # log-return legs 1->2, 2->3, 3->4, 4->5
    noise_std = daily_vol * 0.05                 # far below any leg's per-bar move (safe margin)
    rev_len = 16

    logp = np.empty(n_days)
    logp[0] = 0.0
    i = 0
    next_wedge = 80

    def fill_leg(lo, hi, lvl0, lvl1):
        """Piecewise-linear log-price from lvl0 (bar lo) to lvl1 (bar hi), + tiny noise."""
        span_bars = hi - lo
        for b in range(lo + 1, hi + 1):
            frac = (b - lo) / span_bars
            logp[b] = lvl0 + (lvl1 - lvl0) * frac + rng.normal(0.0, noise_std)

    while i < n_days - 1:
        if i == next_wedge and i + 4 * leg + rev_len + 260 < n_days:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            # sign=+1: descending 1-3-5 (a "low" throw-under) -> detector dir=+1 (reversal UP).
            # sign=-1: ascending 1-3-5 (a "high" throw-over) -> detector dir=-1 (reversal DOWN).
            lvl = logp[i]
            anchors = [lvl]
            for r in rs:
                lvl = lvl + sign * np.log1p(r)
                anchors.append(lvl)
            for k in range(4):
                fill_leg(i + k * leg, i + (k + 1) * leg, anchors[k], anchors[k + 1])
            i5 = i + 4 * leg
            end = i5
            if edge != 0.0:
                rev_lvl = anchors[4] + sign * np.log1p(edge)
                fill_leg(i5, i5 + rev_len, anchors[4], rev_lvl)
                end = i5 + rev_len
            i = end
            gap = int(rng.integers(180, 300))
            next_wedge = i + gap
            continue
        logp[i + 1] = logp[i] + rng.normal(mu, daily_vol)
        i += 1

    close = 100.0 * np.exp(logp)
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi_ = np.maximum(open_, close) + wick
    lo_ = np.minimum(open_, close) - wick

    bars = pd.DataFrame({"open": open_, "high": hi_, "low": lo_, "close": close},
                        index=pd.DatetimeIndex(cal, name="date"))
    truth = {"edge": edge, "n_days": n_days, "seed": seed, "daily_vol": daily_vol}
    return bars, truth


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #
def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
