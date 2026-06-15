"""Data layer for Study 202 (Fifty-Two-Week-Low).

Two tapes, one shape (a tz-naive daily close frame indexed by date):

- ``synthetic_panel`` — a *deterministic, offline* generator.  A ``reversion`` knob
  controls how much a stock that has fallen hard over 52 weeks tends to mean-revert
  going forward.  ``reversion = 0`` is a martingale — the low-proximity rank carries
  no forward information; ``reversion > 0`` plants contrarian bounce structure the
  strategy can harvest.  This is the study's null in a bottle.

- ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only by default
  so the test-suite and the reproducible core never touch the network.  Daily bars go
  back decades and give high statistical power.  We use a representative basket of
  S&P 500 large-cap names and cover at least 10 years.

The 52-week-low proximity signal:
    proximity(t) = (close(t) - low_52w(t)) / (high_52w(t) - low_52w(t) + eps)

where low_52w and high_52w are the trailing 252-day lowest low and highest high.
A proximity of ~0 means the stock is near its 52-week low (the contrarian "buy").
A proximity of ~1 means it is near its 52-week high (momentum territory).

Survivorship note: the basket is fixed to tickers still trading as of 2026; this
*biases toward survivors* and is named as such on the Signal axis throughout.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (signals
are formed on closes up to *t*, positions entered at the *next* bar's open at *t+1*).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Representative S&P 500 large-cap basket — survivorship-biased (all still trade).
# Covers a range of sectors to avoid sector-concentration artefacts.
SP500_BASKET = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "JPM", "JNJ", "XOM", "PG",
    "V", "MA", "HD", "UNH", "CVX",
    "MRK", "ABBV", "PEP", "KO", "WMT",
]


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 20,
    n_days: int = 600,
    reversion: float = 0.0,
    daily_vol: float = 0.015,
    start: str = "2022-01-03",
    seed: int = 202,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily close panel with a tunable contrarian-reversion knob.

    Each stock follows an AR(1) log-return process:
        r_{i,t} = -reversion * r_{i,t-1} + eps_{i,t}

    where eps_{i,t} is i.i.d. Normal(0, daily_vol).  The *negative* AR(1)
    coefficient (positive ``reversion``) is what the 52-week-low signal is
    designed to harvest: stocks that have fallen sharply (proximity near 0)
    should bounce back.

    - ``reversion = 0``  → a martingale; the low-proximity rank is a coin flip.
    - ``reversion > 0``  → short-term mean reversion the strategy can ride.
    - ``reversion < 0``  → momentum; the strategy bets the wrong way.

    Returns ``(panel, truth)`` where ``panel`` is a DataFrame of shape
    (n_days, n_stocks) with columns as ticker names ('S00' ... 'S19'), indexed by
    business-day dates.  ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)
    tickers = [f"S{i:02d}" for i in range(n_stocks)]

    closes = np.empty((n_days, n_stocks))
    for j in range(n_stocks):
        eps = rng.normal(0.0, daily_vol, n_days)
        log_ret = np.empty(n_days)
        prev = 0.0
        for i in range(n_days):
            r = -reversion * prev + eps[i]
            log_ret[i] = r
            prev = r
        closes[:, j] = 100.0 * np.exp(np.cumsum(log_ret))

    panel = pd.DataFrame(closes, index=pd.DatetimeIndex(cal, name="date"), columns=tickers)
    truth = {
        "n_stocks": n_stocks,
        "n_days": n_days,
        "reversion": reversion,
        "daily_vol": daily_vol,
        "seed": seed,
    }
    return panel, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"ftw52l_{safe}_1d.parquet")


def fetch_daily(
    ticker: str,
    period: str = "15y",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached
    as a parquet under ``_cache/``).  Yahoo's daily history goes back well over a
    decade for S&P 500 names, giving ~3,750 bars for a 15-year window.
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
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, period=period, interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
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


def load_panel(
    tickers: list[str] = SP500_BASKET,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Load cached daily close prices for all tickers into a wide panel.

    Returns a DataFrame of shape (n_days, n_tickers) with columns = tickers,
    indexed by date.  Only dates common to all tickers are kept (inner join).
    Missing values within a ticker's history are forward-filled (up to 5 days).
    """
    frames = {}
    for t in tickers:
        bars = fetch_daily(t, fetch=False, cache_dir=cache_dir)
        frames[t] = bars["close"]

    panel = pd.DataFrame(frames).sort_index()
    panel = panel.ffill(limit=5)   # fill short gaps (dividends / data holes)
    panel = panel.dropna()         # require all tickers to be present
    return panel


def fingerprint(panel: pd.DataFrame) -> str:
    """A short content fingerprint of the close panel, for the as-of stamp."""
    arr = np.ascontiguousarray(panel.to_numpy(dtype=float))
    h = hashlib.sha1(arr.tobytes())
    return h.hexdigest()[:12]
