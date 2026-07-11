"""Data layer for Study 674 — VIDYA (Chande's Variable Index Dynamic Average).

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date):

* **Real tape.** Daily total-return bars (`auto_adjust=True`) for a five-ticker basket
  (SPY, QQQ, AAPL, MSFT, XLE — the same liquid basket as sibling studies 672/673) via
  ``yfinance`` (no key), **cache-first**: a cached parquet under the study's own
  ``_cache/`` is read offline; the network is touched only on a cache miss or an
  explicit ``fetch=True``.

* **Synthetic world.** A deterministic, seeded tape with a TUNABLE planted trend
  (regime-switching drift, knob ``edge``). ``edge = 0`` is a pure random walk — a fair
  coin for any moving-average timing rule; ``edge > 0`` plants a slow, persistent trend
  a trend-follower can ride. This is the study's faithful-engine / power check: the
  harness must NOT manufacture an edge from the null and must recover a real one when
  planted.

Tushar Chande's VIDYA (*"Adapting Moving Averages to Market Volatility"*, Technical
Analysis of Stocks & Commodities, March 1992) is an EMA whose smoothing constant is
scaled by his own Chande Momentum Oscillator (CMO)::

    VI(t)    = |CMO(t, cmo_period)| / 100                       (volatility index in [0,1])
    alpha    = 2 / (period + 1)                                  (nominal EMA smoothing const)
    VIDYA(t) = alpha * VI(t) * P(t) + (1 - alpha * VI(t)) * VIDYA(t-1)

The claim: because CMO reads near 100 in a clean directional run and near 0 in chop,
VIDYA "speeds up" (tracks price like a fast EMA) in trending/volatile markets and
"slows down" (freezes like a very slow EMA, almost flat) in quiet/choppy ones — beating
a fixed-window MA. We test that mechanism directly (does VI actually correlate with
realized volatility and with trend strength, and does a deterministic step response
show faster catch-up?) before turning it into a trading rule.

Pure numpy + pandas + stdlib on the offline path. ``fetch_all()``/``load_real(fetch=True)``
(network) run once to build the cache and are never imported by the notebooks' offline
cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "XLE"]
AS_OF = "2026-06-30"  # last complete calendar month at publication (2026-07-10)


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str = DEFAULT_CACHE) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"vidya_{safe}_1d.parquet")


def load_real(
    ticker: str = "SPY",
    period: str = "max",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    asof: str | None = AS_OF,
    retries: int = 3,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``, sliced to <= ``asof``; cache-first.

    Cached parquet is read offline unless ``fetch=True`` or the cache is missing, in
    which case ``yfinance`` is hit once (small back-off retry) and the raw pull is
    cached so every later run is offline. ``auto_adjust=True`` folds splits/dividends
    into the close (total-return bars).
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path) and not fetch:
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only touched on an actual network pull

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
    if asof:
        bars = bars[bars.index <= pd.Timestamp(asof)]
    return bars


def have_real(cache_dir: str = DEFAULT_CACHE, tickers: list[str] = TICKERS) -> bool:
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in tickers)


def fetch_all(cache_dir: str = DEFAULT_CACHE, tickers: list[str] = TICKERS) -> None:
    """Warm the cache for every basket ticker. Network; once."""
    for t in tickers:
        load_real(t, fetch=True, cache_dir=cache_dir, asof=None)


# --------------------------------------------------------------------------- #
# Synthetic world — planted trend (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 4000,
    edge: float = 0.0,
    daily_vol: float = 0.011,
    regime_len: int = 60,
    start: str = "2014-01-02",
    seed: int = 674,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a tunable, slow regime-switching trend.

    The only structure any moving-average trend rule (VIDYA included) can possibly
    harvest is *persistent* drift. The drift sign flips every ``regime_len`` bars; its
    magnitude is ``edge * daily_vol``.

    - ``edge = 0``   -> pure martingale; any MA-crossover rule is a fair coin (the null).
    - ``edge > 0``   -> persistent trending regimes a trend rule can ride (positive control).

    Business-day index, span far below the ~250-year pandas ns-timestamp trap. Returns
    ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)

    n_regimes = int(np.ceil(n_days / regime_len))
    regime_sign = rng.choice([-1.0, 1.0], size=n_regimes)
    drift = np.repeat(regime_sign, regime_len)[:n_days] * (edge * daily_vol)

    eps = rng.normal(0.0, daily_vol, n_days)
    log_ret = drift + eps

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 50_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(cal, name="date"),
    )
    truth = {"edge": edge, "daily_vol": daily_vol, "regime_len": regime_len,
             "n_days": n_days, "seed": seed}
    return bars, truth
