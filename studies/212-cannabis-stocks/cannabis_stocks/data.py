"""Data layer for Study 212 (Cannabis Stocks).

Two tapes:

- ``synthetic_daily`` — a *deterministic, offline* generator. Simulates SPY and a
  cannabis thematic series where cannabis is defined relative to SPY with a configurable
  beta and an alpha (positive for a genuine edge, negative for the wealth-shredder thesis).
  At ``beta=0`` cannabis is uncorrelated with SPY — the null. At ``alpha_ann < 0`` and
  ``bubble_factor > 0`` it models the post-2019 blow-up-and-bust pattern.

- ``fetch_prices`` — daily adjusted closes from Yahoo! Finance (``yfinance``), cache-only
  by default. ``fetch=False`` raises ``FileNotFoundError`` if the parquet is absent, so the
  test-suite and offline core never touch the network.

Key tickers:
  - MSOS (AdvisorShares Pure US Cannabis ETF) — inception 2020-09-02
  - MJ (ETFMG Alternative Harvest ETF) — inception 2017-11-08
  - TLRY (Tilray Brands) — inception 2018-07-19
  - CGC (Canopy Growth) — inception 2018-05-24 (US listing)
  - CRON (Cronos Group) — inception 2018-02-27 (US listing)
  - SPY (S&P 500 benchmark) — inception 1993-01-29

No look-ahead is baked in here — that discipline lives in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# Ticker inception dates (US exchanges)
TICKER_START = {
    "MSOS": "2020-09-02",
    "MJ":   "2017-11-08",
    "TLRY": "2018-07-19",
    "CGC":  "2018-05-24",
    "CRON": "2018-02-27",
    "SPY":  "2018-03-01",  # common window start aligned to CRON listing
}

# Common analysis window: MSOS IPO is the shortest, but we also run MJ from its start
MJ_START = "2017-11-08"
MSOS_START = "2020-09-02"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 8,
    spy_drift_ann: float = 0.10,   # annualised SPY drift
    spy_vol_ann: float = 0.18,     # annualised SPY volatility
    beta: float = 1.5,             # cannabis beta to SPY
    alpha_ann: float = -0.25,      # annualised alpha (negative = wealth destruction)
    idio_vol_ann: float = 0.40,    # cannabis idiosyncratic volatility (very high sector)
    bubble_factor: float = 0.0,    # extra bubble-and-bust variance injection (0 = none)
    start: str = "2018-01-02",
    seed: int = 212,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible pair of daily (SPY-like, cannabis-like) price series.

    Cannabis log-returns follow:
        r_can = alpha_daily + beta * r_spy + idio_noise + bubble_noise

    ``alpha_ann < 0`` represents sustained capital destruction relative to the market.
    ``bubble_factor > 0`` adds a random large-variance component simulating the 2019-2020
    blow-up and bust cycle. At ``beta=0``, cannabis is uncorrelated with SPY — the null.

    Returns ``(prices, truth)`` where ``prices`` has columns ``spy`` and ``cannabis``,
    indexed by business dates starting from ``start``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    dates = pd.bdate_range(start=start, periods=n_days)

    # Daily moments
    mu_spy = spy_drift_ann / TRADING_DAYS_PER_YEAR
    sig_spy = spy_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    alpha_daily = alpha_ann / TRADING_DAYS_PER_YEAR
    sig_idio = idio_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)

    r_spy = rng.normal(mu_spy, sig_spy, n_days)

    idio = rng.normal(0.0, sig_idio, n_days)

    # Optional bubble noise: infrequent large shocks both ways
    bubble_noise = np.zeros(n_days)
    if bubble_factor > 0:
        n_shocks = max(1, int(n_years * 4))  # ~4 large shocks per year on average
        shock_days = rng.choice(n_days, size=n_shocks, replace=False)
        shock_sizes = rng.normal(0.0, bubble_factor / np.sqrt(TRADING_DAYS_PER_YEAR), n_shocks)
        bubble_noise[shock_days] = shock_sizes

    r_cannabis = alpha_daily + beta * r_spy + idio + bubble_noise

    spy = 100.0 * np.exp(np.cumsum(r_spy))
    cannabis = 100.0 * np.exp(np.cumsum(r_cannabis))

    prices = pd.DataFrame(
        {"spy": spy, "cannabis": cannabis},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "beta": beta,
        "alpha_ann": alpha_ann,
        "spy_drift_ann": spy_drift_ann,
        "spy_vol_ann": spy_vol_ann,
        "idio_vol_ann": idio_vol_ann,
        "bubble_factor": bubble_factor,
        "n_years": n_years,
        "n_days": n_days,
        "seed": seed,
        "planted_beta": float(beta),
        "planted_alpha_ann": float(alpha_ann),
    }
    return prices, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo! Finance daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch_prices(
    ticker: str,
    start: str = "2017-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Daily adjusted close prices for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``). Returns a frame with a single ``close`` column indexed
    by date. Uses ``auto_adjust=True`` (split- and dividend-adjusted total-return close).
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily prices for {ticker} at {path}. "
                f"Call fetch_prices({ticker!r}, fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, start=start, end=end, interval="1d",
            auto_adjust=True, progress=False,
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        df = raw[["close"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(path)

    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.dropna(subset=["close"])
    return df


def load_multi(
    tickers: list[str] | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    start: str = MJ_START,
) -> pd.DataFrame:
    """Load multiple tickers as adjusted closes, aligned on the common date intersection.

    Returns a DataFrame with one column per ticker (lower-cased), index = date.
    SPY is always included as the benchmark. Tickers with fewer than 200 trading days
    of overlap are excluded with a warning.

    Default universe: MJ, MSOS, TLRY, CGC, CRON, SPY.
    """
    if tickers is None:
        tickers = ["MJ", "MSOS", "TLRY", "CGC", "CRON", "SPY"]

    frames = {}
    for t in tickers:
        try:
            df = fetch_prices(t, start=start, fetch=fetch, cache_dir=cache_dir)
            frames[t.lower()] = df["close"].rename(t.lower())
        except FileNotFoundError:
            raise

    prices = pd.concat(frames.values(), axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    # Drop rows where any ticker is missing (outer-joined initially, now inner)
    prices = prices.dropna()
    return prices


def load_pair(
    cannabis_ticker: str = "MSOS",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    start: str = MSOS_START,
) -> pd.DataFrame:
    """Load a single cannabis ticker vs SPY, aligned on common dates from ``start``.

    Returns a DataFrame with columns ``spy`` and ``cannabis``, index = date. This
    is the primary pair used for the main OLS/alpha analysis.
    """
    spy = fetch_prices("SPY", start=start, fetch=fetch, cache_dir=cache_dir)
    can = fetch_prices(cannabis_ticker, start=start, fetch=fetch, cache_dir=cache_dir)
    prices = spy.rename(columns={"close": "spy"}).join(
        can.rename(columns={"close": "cannabis"}), how="inner"
    )
    return prices.dropna()


def fingerprint(prices: pd.DataFrame, col: str = "spy") -> str:
    """A short content fingerprint of a price series, for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(prices[col].to_numpy()).tobytes())
    return h.hexdigest()[:12]
