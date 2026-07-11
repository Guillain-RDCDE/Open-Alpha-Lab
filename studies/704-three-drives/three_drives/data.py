"""Data layer for Study 704 — Three Drives.

Two tapes, one shape (a tz-naive daily OHLC frame indexed by date), mirroring sibling
697-wolfe-waves so the two Fibonacci five-point patterns are directly comparable:

* ``load_real`` / ``load_basket`` — real daily OHLC for SPY + a basket of broad indices/ETFs
  (yfinance, no key), cache-first into ``_cache/`` so the reproducible core and notebooks run
  offline once cached.

* ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**. The
  "Three Drives" pattern is five labelled turning points (1-2-3-4-5, an implicit start "0"
  before point 1) forming three drives (0->1, 2->3, 4->5) and two corrections (1->2, 3->4): each
  correction retraces ~61.8% of the prior drive, each drive extends the prior correction by
  ~1.27x (both point estimates, comfortably inside the wider 0.382-0.886 / 1.13-2.618 detection
  bands in ``strategy.py``) — genuinely Fibonacci-proportioned by construction here. ``edge = 0``
  builds the exact Fibonacci geometry but lets price continue as a pure random walk past point 5 (no
  reversal), so the detector must NOT manufacture significance from geometry alone; ``edge > 0``
  plants a real reversal of size ``edge`` after point 5 — the positive control that proves the
  harness can detect the effect it plants.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Same broad-index/ETF basket family as siblings 445-elliott-wave / 697-wolfe-waves — deep daily
# history, no exotic single names, no cross-sectional survivorship.
TICKERS = ("SPY", "QQQ", "DIA", "IWM", "^GSPC", "^IXIC", "^DJI", "GLD")

AS_OF = "2026-06-30"   # last complete calendar month at publication (2026-07-10)


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"td_{safe}_1d.parquet")


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
def synthetic_panel(n_days: int = 6000, edge: float = 0.0, seed: int = 704,
                    daily_vol: float = 0.011, start: str = "2000-01-03",
                    xa: float = 0.10, corr_ratio: float = 0.618, ext_ratio: float = 1.27,
                    ) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a planted, exactly-Fibonacci Three-Drives geometry.

    Every ~200 bars we splice a clean 5-leg structure (point0 -> point1 -> point2 -> point3 ->
    point4 -> point5) whose ratios are *exactly* on the folklore's grid: each correction retraces
    ``corr_ratio`` (0.618) of the prior drive, each drive extends the prior correction by
    ``ext_ratio`` (1.27) — built by direct anchor construction (piecewise-linear in log-price
    between exact anchors, not accumulated per-bar noise, so the geometry never fragments into
    spurious extra pivots), followed by a genuinely quiet gap before the next planted structure.

    - ``edge = 0`` -> the Fibonacci geometry still forms (three real, ratio-perfect drives), but
      NO extra reversal drift follows point 5 — a pure random walk continues, so the detector
      must NOT manufacture a reversal edge from geometry alone.
    - ``edge > 0`` -> plants a real post-point-5 reversal of size ``edge`` (a log-return) over the
      following ``rev_len`` bars, in the direction opposite the three drives — precisely the
      structure the pattern needs to be true.

    Direction alternates randomly: "three drives up" (sign=+1, reversal claim is DOWN) or "three
    drives down" (sign=-1, reversal claim is UP).

    Business-day index, span far below the ~250-year pandas ns-timestamp trap.
    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)
    mu = 0.00015
    leg = 12                                    # bars per anchor-to-anchor leg
    corr1 = corr_ratio * xa
    drive2 = ext_ratio * corr1
    corr2 = corr_ratio * drive2
    drive3 = ext_ratio * corr2
    rs = (xa, corr1, drive2, corr2, drive3)      # simple-return magnitudes, 5 legs
    leg_signs = np.array([+1, -1, +1, -1, +1])
    noise_std = daily_vol * 0.05                 # far below any leg's per-bar move (safe margin)
    rev_len = 16

    logp = np.empty(n_days)
    logp[0] = 0.0
    i = 0
    next_wedge = 80
    n_planted = 0

    def fill_leg(lo, hi, lvl0, lvl1):
        """Piecewise-linear log-price from lvl0 (bar lo) to lvl1 (bar hi), + tiny noise."""
        span_bars = hi - lo
        for b in range(lo + 1, hi + 1):
            frac = (b - lo) / span_bars
            logp[b] = lvl0 + (lvl1 - lvl0) * frac + rng.normal(0.0, noise_std)

    while i < n_days - 1:
        if i == next_wedge and i + 5 * leg + rev_len + 260 < n_days:
            sign = 1.0 if rng.random() < 0.5 else -1.0   # +1 = three drives UP; -1 = DOWN
            lvl = logp[i]
            anchors = [lvl]
            for k, r in enumerate(rs):
                lvl = lvl + sign * leg_signs[k] * np.log1p(r)
                anchors.append(lvl)
            for k in range(5):
                fill_leg(i + k * leg, i + (k + 1) * leg, anchors[k], anchors[k + 1])
            i5 = i + 5 * leg
            end = i5
            n_planted += 1
            if edge != 0.0:
                rev_lvl = anchors[5] - sign * np.log1p(edge)   # reversal OPPOSES the drives
                fill_leg(i5, i5 + rev_len, anchors[5], rev_lvl)
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
    truth = {"edge": edge, "n_days": n_days, "seed": seed, "daily_vol": daily_vol,
             "n_planted": n_planted, "corr_ratio": corr_ratio, "ext_ratio": ext_ratio}
    return bars, truth


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #
def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
