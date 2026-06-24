"""Data layer for Study 433 (Kaufman Adaptive Moving Average).

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**
  (``edge``).  The edge is a slow, regime-switching trend component: when ``edge > 0`` the
  tape spends long stretches drifting up or down (persistent trends a fast/slow adaptive
  filter can ride), interrupted by choppy noise stretches where a fixed SMA would whipsaw.
  ``edge = 0`` is a pure random walk — a fair coin — so a test can assert that KAMA's
  volatility-adaptation only adds value *over a fixed SMA* when there is something for the
  adaptation to exploit, and not otherwise.  This is the study's null in a bottle.
- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first** so the
  test-suite and the reproducible core never touch the network unless a cache file is
  missing.  Daily bars go back decades, giving high statistical power.

Kaufman's Adaptive Moving Average (KAMA, Perry Kaufman, *Smarter Trading*, 1995) is an
EMA whose smoothing constant *adapts to the efficiency of the price move*.  The Efficiency
Ratio over an ``er_period`` window is::

    ER(t) = |price(t) - price(t - n)| / sum_{i=1..n} |price(t-i+1) - price(t-i)|

ER is 1 when the move is a clean straight line (all change in one direction) and ~0 when
the path is pure chop (lots of motion, little net travel).  The adaptive smoothing constant
interpolates between a fast EMA (when ER ~ 1) and a slow EMA (when ER ~ 0)::

    SC(t) = [ ER(t) * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1) ]^2
    KAMA(t) = KAMA(t-1) + SC(t) * (price(t) - KAMA(t-1))

The folk pitch: because KAMA *tightens up in trends and goes quiet in chop*, a price-cross
timing rule on KAMA should beat the same rule on a fixed SMA — fewer whipsaws, faster
trend entries.  We test that "it is better than a fixed SMA" claim head-on.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the indicator
is formed on closes up to *t*, the position is held over *t+1*'s return: one shift).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (planted-edge knob)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 6000,
    edge: float = 0.0,
    daily_vol: float = 0.011,
    trend_strength: float = 0.0012,
    regime_len: int = 120,
    start: str = "2014-01-02",
    seed: int = 433,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a tunable *regime-switching* edge knob.

    This is the study's **positive control**: it builds the one environment where a
    volatility-adaptive filter *should* beat a fixed SMA.  When ``edge > 0`` the tape
    alternates two hidden regimes every ``regime_len`` bars:

    - **trend** blocks — a strong, clean signed drift of magnitude ``edge * trend_strength``
      with *low* noise (efficiency ratio ≈ 1).  KAMA tightens to a fast EMA and rides them.
    - **chop** blocks — a fast near-zero oscillation (efficiency ratio ≈ 0) that crosses a
      moving average repeatedly.  KAMA *freezes* and holds its position; a fixed SMA flips
      back and forth, paying costs on every whipsaw.

    Under non-zero costs the adaptive filter's whipsaw avoidance is what lets it beat the
    SMA — so a test can confirm the harness recovers a positive KAMA−SMA edge when the
    structure is present, and reads ≈ 0 (``edge = 0`` → pure random walk) when it is not.

    - ``edge = 0``   → a martingale; KAMA and SMA timing are both fair coins.
    - ``edge > 0``   → regime tape the adaptive filter can ride (and dodge chop on).

    Bars are stamped on consecutive business days.  Returns ``(data, truth)`` where
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)

    if edge <= 0.0:
        log_ret = rng.normal(0.0, daily_vol, n_days)
    else:
        log_ret = np.zeros(n_days)
        i = 0
        while i < n_days:
            block = min(regime_len, n_days - i)
            if rng.random() < 0.55:  # trend block
                sign = 1.0 if rng.random() < 0.5 else -1.0
                log_ret[i : i + block] = (
                    sign * edge * trend_strength
                    + rng.normal(0.0, daily_vol * 0.3, block)
                )
            else:  # chop block — fast oscillation near zero (whipsaws an SMA)
                base = rng.normal(0.0, daily_vol * 0.4, block)
                osc = 0.004 * np.sin(np.arange(block) * 1.6)
                log_ret[i : i + block] = (
                    np.diff(np.concatenate([[0.0], osc])) + base
                )
            i += block

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 50_000_000, n_days).astype(float)

    data = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(cal, name="date"),
    )
    truth = {
        "edge": edge,
        "daily_vol": daily_vol,
        "trend_strength": trend_strength,
        "regime_len": regime_len,
        "n_days": n_days,
        "seed": seed,
    }
    return data, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"kama_{safe}_1d.parquet")


def load_real(
    ticker: str = "SPY",
    period: str = "max",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 3,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; **cache-first**.

    If a cached parquet exists it is read offline.  On a cache *miss* (or ``fetch=True``)
    the network is touched via ``yfinance`` with a small back-off retry, and the result is
    cached so every subsequent run is offline.  ``auto_adjust=True`` → split- and
    dividend-adjusted closes (total-return-ish; labelled as such downstream).
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path) and not fetch:
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        last_err: Exception | None = None
        raw = None
        for attempt in range(retries):
            try:
                raw = yf.download(
                    ticker, period=period, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is not None and not raw.empty:
                    break
            except Exception as exc:  # pragma: no cover - network path
                last_err = exc
            time.sleep(1.5 * (attempt + 1))
        if raw is None or raw.empty:
            raise RuntimeError(
                f"yfinance returned no daily bars for {ticker}"
                + (f" (last error: {last_err})" if last_err else "")
            )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
