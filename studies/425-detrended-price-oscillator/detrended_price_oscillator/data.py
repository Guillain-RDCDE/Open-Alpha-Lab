"""Data layer for Study 425 — the Detrended Price Oscillator (DPO).

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-cycle knob**
  (``edge``).  The DPO is, by construction, a band-pass filter: it removes the trend (the
  ``n``-period SMA) so that what is left is whatever oscillates *faster* than the window.
  The only structure such a detector can possibly harvest is a genuine short-period cycle
  in the price.  ``edge = 0`` plants nothing (pure random walk → the DPO timing rule is a
  coin); ``edge > 0`` injects a sinusoidal cycle of a chosen period whose phase the DPO
  can, in principle, lock onto.  This is the study's null — and its positive control — in
  a bottle.
- ``load_real`` / ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), **cache-first**
  so the test-suite and the reproducible core never touch the network.  Daily bars go back
  decades, giving high statistical power compared with intraday studies.

The DPO (Steven Achelis / standard charting definition) is::

    DPO(t) = Close(t - (n//2 + 1)) - SMA_n(Close)(t)

i.e. the price *displaced back* by ``n//2 + 1`` bars minus the ``n``-bar simple moving
average.  The displacement is the whole point: it lines the detrended series up under the
*centre* of the averaging window, which is why the textbook DPO is **centered and therefore
peeks at data the SMA already used** — a fact we handle explicitly in ``strategy.py`` (the
tradable rule never uses a value it could not have known at the bar it trades on).

Pure numpy + pandas + stdlib for the offline path.  ``fetch_daily`` (network) is only used
to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core / positive control
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 2500,
    edge: float = 0.0,
    cycle_period: int = 20,
    daily_vol: float = 0.011,
    drift: float = 0.0003,
    start: str = "2014-01-02",
    seed: int = 425,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a tunable planted-cycle knob.

    The log price is a random walk with drift plus, when ``edge != 0``, a deterministic
    **sinusoidal cycle** of period ``cycle_period`` whose amplitude scales with ``edge``::

        log_price(t) = drift*t + cumsum(eps_t) + edge * sin(2*pi*t / cycle_period)

    The cycle is exactly the structure a band-pass detrender such as the DPO is built to
    isolate: a component oscillating faster than the ``n``-bar trend window.

    - ``edge = 0``  → a pure random walk with drift; the DPO timing rule is a fair coin.
    - ``edge > 0``  → a planted cycle the DPO can, in principle, lock onto and trade.

    The amplitude is in *log-return* units per bar of the sinusoid's derivative envelope;
    a small ``edge`` (e.g. 0.02) is a faint but real cycle.  Bars are stamped on consecutive
    business days.  Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)

    t = np.arange(n_days)
    eps = rng.normal(0.0, daily_vol, n_days)
    log_price = drift * t + np.cumsum(eps)
    if edge != 0.0:
        log_price = log_price + edge * np.sin(2.0 * np.pi * t / float(cycle_period))

    close = 100.0 * np.exp(log_price)
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    # Intrabar wicks: a small symmetric deviation around the open-close range.
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 50_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(cal, name="date"),
    )
    truth = {
        "edge": edge,
        "cycle_period": cycle_period,
        "daily_vol": daily_vol,
        "drift": drift,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"dpo_{safe}_1d.parquet")


def fetch_daily(
    ticker: str,
    period: str = "max",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-first, network only on ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as a
    parquet under ``_cache/``).  On a couple of retries with a small backoff if the vendor
    rate-limits.  Yahoo's daily history goes back decades, giving thousands of bars per
    instrument.  Returns split- and dividend-**adjusted** closes (total-return-ish via
    ``auto_adjust=True``); we label this everywhere as adjusted / total-return-adjusted.
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
        import time

        import yfinance as yf  # lazy: only when we actually go to the network

        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(
                    ticker, period=period, interval="1d",
                    auto_adjust=True, progress=False,
                )
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
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    # Ensure tz-naive DatetimeIndex for consistent downstream handling.
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    return bars


def load_real(
    ticker: str = "SPY",
    cache_dir: str = DEFAULT_CACHE,
    end: str | None = None,
) -> pd.DataFrame:
    """Cache-first convenience loader for one real daily tape.

    Reads the cached parquet (raising if absent — populate it once with ``fetch_daily``).
    If ``end`` is given, the tape is truncated to ``<= end`` so a stamped run can pin an
    explicit as-of date and drop any partial trailing bar.
    """
    bars = fetch_daily(ticker, fetch=False, cache_dir=cache_dir)
    if end is not None:
        bars = bars.loc[bars.index <= pd.Timestamp(end)]
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
