"""Data layer for Study 445 — Elliott Wave.

Two tapes, one shape (a tz-naive daily OHLC frame indexed by date):

* ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  ``edge = 0`` is a pure random walk (a fair coin), so the wave-3 detector must NOT manufacture
  significance. ``edge > 0`` plants the one structure Elliott's theory needs to be true: after a
  textbook impulse 1-2 (a swing up, then a Fibonacci-sized pullback), the next leg ("wave 3")
  is given an *extra* drift in the impulse direction. With the planted edge the detector must
  light up — the positive control that proves the harness can detect the effect it plants.

* ``load_real`` — real daily OHLC for SPY + a basket of indices/ETFs (yfinance, no key),
  cache-first into ``_cache/`` so the reproducible core and notebooks run offline once cached.
  Daily bars give decades of swings — far more statistical power than an intraday study.

Elliott Wave is *irreducibly subjective*: the "right" wave count is only knowable after the
fact and two analysts disagree on the same chart. There is therefore no single falsifiable
rule. The tightest mechanical version proponents accept is a **ZigZag** swing detector (the
standard EW labeling tool) plus **Fibonacci retracement** filters. We encode exactly that in
``strategy.py`` and test the one forward-looking prediction it makes.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# SPY = the headline tape; the rest are deep-history broad indices/ETFs proponents draw waves
# on (the "relevant indices" the brief asks for). All survivors — but these are *price index*
# tapes, not single-name baskets, so there is no cross-sectional survivorship sort here.
TICKERS = ("SPY", "QQQ", "DIA", "IWM", "^GSPC", "^IXIC", "^DJI", "GLD")


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"ew_{safe}_1d.parquet")


def load_real(ticker: str = "SPY", period: str = "max",
              cache_dir: str = DEFAULT_CACHE, fetch: bool = False,
              retries: int = 3) -> pd.DataFrame:
    """Real daily OHLC for ``ticker``; cache-first (network only on a cache miss or ``fetch``).

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
    return bars.dropna().sort_index()


def have_real(ticker: str = "SPY", cache_dir: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(_cache_path(ticker, cache_dir))


def load_basket(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE,
                fetch: bool = False) -> dict[str, pd.DataFrame]:
    """Load every cached ticker we can (cache-first). Returns {ticker: OHLC frame}."""
    out = {}
    for tk in tickers:
        if fetch or have_real(tk, cache_dir):
            try:
                out[tk] = load_real(tk, cache_dir=cache_dir, fetch=fetch)
            except Exception:
                continue
    return out


# --------------------------------------------------------------------------- #
# Synthetic positive control — deterministic, planted-edge knob
# --------------------------------------------------------------------------- #
def synthetic_panel(n_days: int = 6000, edge: float = 0.0, seed: int = 445,
                    daily_vol: float = 0.011, start: str = "2000-01-03"
                    ) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a planted Elliott-wave-3 edge knob.

    The price is a random walk in log space. On top of it we periodically build a **textbook
    impulse**: a swing up (wave 1), a Fibonacci-sized pullback of ~50% (wave 2), and then —
    *only if* ``edge != 0`` — an extra drift of ``edge`` spread over the next leg (wave 3),
    in the impulse direction. This is precisely the structure Elliott's theory needs:
    after a clean 1-2, wave 3 extends.

    - ``edge = 0`` → pure random walk; the wave-3 detector must NOT find significance.
    - ``edge > 0`` → planted wave-3 drift the ZigZag+Fibonacci rule can harvest.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)
    # Full-vol random-walk baseline — at edge=0 this is ALL there is (a fair coin), and the
    # ZigZag picks up only the swings noise happens to throw up. The detector must NOT find a
    # wave-3 edge in it.
    eps = rng.normal(0.00020, daily_vol, n_days)

    # The planted Elliott structure is added ONLY when edge != 0. Each impulse is a clean
    # wave 1 (big swing), a textbook ~50% Fibonacci wave-2 pullback, then a wave-3 extension
    # of size `edge` in the impulse direction — exactly what the theory needs to be true.
    if edge != 0.0:
        i = 60
        w1_days, w2_days, w3_len = 14, 8, 18
        w1_total = 0.16                  # ~16% wave-1 swing -> clears the 5% ZigZag with margin
        w2_frac = 0.50                   # textbook 50% retracement
        while i < n_days - 120:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            eps[i:i + w1_days] += sign * (w1_total / w1_days)
            i2 = i + w1_days
            eps[i2:i2 + w2_days] -= sign * (w1_total * w2_frac / w2_days)
            i3 = i2 + w2_days
            eps[i3:i3 + w3_len] += sign * edge / w3_len
            i = i3 + w3_len + int(rng.integers(20, 50))

    close = 100.0 * np.exp(np.cumsum(eps))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick

    bars = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close},
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
