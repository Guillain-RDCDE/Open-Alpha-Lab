"""Data layer for Study 137 (Mansfield-RS).

Two tapes, one shape (a weekly OHLCV frame per ticker):

- ``synthetic_weekly`` — a *deterministic, offline* generator.  A week-level AR(1) drift
  knob (``rs_momentum``) controls how strongly the relative-strength line (stock vs
  benchmark) trends.  When ``rs_momentum = 0`` the stock's excess return is i.i.d. white
  noise — a Stage-2 label is a random coin.  When ``rs_momentum > 0`` the RS line
  genuinely slopes up, so a stock flagged as Stage 2 is likelier to outperform going
  forward.  This is the study's null in a bottle — we can ask "does the filter add
  anything beyond the trend component it selects for?".
- ``fetch_weekly`` — the real Yahoo! daily tape, resampled to weekly OHLCV and cached as
  parquet.  The basket covers SPY (the benchmark) plus a representative set of large-cap
  US equities.

Mansfield Relative Strength is defined as:
    MRS_t = (stock_price_t / stock_SMA30w_t) / (benchmark_price_t / benchmark_SMA30w_t) − 1
A stock is in Weinstein Stage 2 when:
    1. Its price is above its own rising 30-week SMA (``sma30w_slope > 0``).
    2. Its Mansfield RS is positive (MRS > 0), i.e. the stock outperforms the benchmark
       relative to their respective 30-week baselines.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (signals are
formed on closes up to week *t*, positions entered at the *next* week's open).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_WEEKS_PER_YEAR = 52
BENCHMARK = "SPY"

# A representative large-cap basket (Weinstein tested broad universe; we use a
# survivorship-biased proxy of well-known large caps available throughout our history).
DEFAULT_TICKERS = [
    "SPY",   # the benchmark itself (must always be first)
    "AAPL", "MSFT", "JNJ", "JPM", "XOM",
    "PG", "KO", "WMT", "HD", "BAC",
    "PFE", "CVX", "MRK", "ABBV", "AMZN",
]


# ---------------------------------------------------------------------------
# Synthetic weekly tapes — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_weekly(
    n_weeks: int = 520,
    rs_momentum: float = 0.0,
    bar_vol_bps: float = 200.0,
    n_stocks: int = 10,
    start: str = "2010-01-04",
    seed: int = 137,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """A reproducible set of synthetic weekly OHLCV tapes with a known RS structure.

    The benchmark (key ``"SPY"``) follows a simple random walk.  Each stock follows
    an AR(1) excess-return process:

        excess_t = rs_momentum * excess_{t-1} + eps_t

    where ``eps_t`` is i.i.d. normal.  The stock's total return is the benchmark
    return plus the excess return.  ``rs_momentum`` is the *only* forecastable
    structure in the tape:

    - ``rs_momentum = 0``  → each stock's excess return is white noise; Stage-2
      membership is a random label; the filter adds nothing.
    - ``rs_momentum > 0``  → RS-line momentum exists; Stage-2 stocks genuinely
      tend to outperform going forward.
    - ``rs_momentum < 0``  → RS-line mean-reverts; Stage-2 stocks will
      underperform in the near term (wrong side).

    Returns ``(tapes, truth)`` where ``tapes`` is a dict mapping each ticker to a
    weekly OHLCV DataFrame, and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_weeks, freq="W-FRI")

    def _make_ohlcv(log_rets: np.ndarray) -> pd.DataFrame:
        close = 100.0 * np.exp(np.cumsum(log_rets))
        open_ = np.empty_like(close)
        open_[0] = 100.0
        open_[1:] = close[:-1]
        wick = np.abs(rng.normal(0.0, bar_vol_bps * 0.3e-4, close.size)) * close
        hi = np.maximum(open_, close) + wick
        lo = np.minimum(open_, close) - wick
        vol = rng.integers(100_000, 1_000_000, close.size).astype(float)
        return pd.DataFrame(
            {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
            index=sessions,
        )

    # Benchmark: a gentle long-run upward drift (Sharpe ≈ 0.5 annualised)
    bm_rets = rng.normal(0.07 / TRADING_WEEKS_PER_YEAR, bar_vol_bps * 1e-4, n_weeks)
    tapes: dict[str, pd.DataFrame] = {"SPY": _make_ohlcv(bm_rets)}

    ticker_names = [f"STOCK{i:02d}" for i in range(n_stocks)]
    for tick in ticker_names:
        excess = np.empty(n_weeks)
        prev = 0.0
        eps = rng.normal(0.0, bar_vol_bps * 1.2e-4, n_weeks)  # stocks slightly more vol
        for t in range(n_weeks):
            e = rs_momentum * prev + eps[t]
            excess[t] = e
            prev = e
        stock_rets = bm_rets + excess
        tapes[tick] = _make_ohlcv(stock_rets)

    truth = {
        "rs_momentum": rs_momentum,
        "bar_vol_bps": bar_vol_bps,
        "n_weeks": n_weeks,
        "n_stocks": n_stocks,
        "seed": seed,
        "tickers": ticker_names,
    }
    return tapes, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, resampled to weekly, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "").replace(".", "_")
    return os.path.join(cache_dir, f"bars_{safe}_1w.parquet")


def fetch_weekly(
    ticker: str,
    period: str = "25y",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real weekly OHLCV for ``ticker``, derived from daily bars; cache-only unless ``fetch=True``.

    We download daily data and resample to weekly (Friday close) ourselves: Yahoo's
    native weekly download occasionally returns Monday-anchored weeks that don't
    align cleanly with the Weinstein 30-week baseline, so we standardise in code.
    The result is cached as a parquet under ``_cache/``.

    The 30-week SMA needs ~30 bars of warmup; 25 years of weekly data gives ~1,300
    observations — ample for meaningful inference.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached weekly tape for {ticker} at {path}. "
                f"Call fetch_weekly({ticker!r}, fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, period=period, interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        raw.index.name = "ts"
        # Resample to weekly bars anchored on Friday close
        bars = raw.resample("W-FRI").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["close"])
        bars.index.name = "ts"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    return bars


def fetch_basket(
    tickers: list[str] = DEFAULT_TICKERS,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> dict[str, pd.DataFrame]:
    """Load (or fetch) weekly OHLCV tapes for the whole basket."""
    return {t: fetch_weekly(t, fetch=fetch, cache_dir=cache_dir) for t in tickers}


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
