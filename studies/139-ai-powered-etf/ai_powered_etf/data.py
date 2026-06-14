"""Data layer for Study 139 (AI-Powered-ETF).

Two tapes, one shape (a daily adjusted-close price frame):

- ``synthetic_daily`` — a *deterministic, offline* generator. An alpha knob lets us
  plant a known excess-return-vs-market so tests can assert the analysis recovers it
  when present, and reads zero when absent. ``alpha_ann=0`` is a fund that tracks the
  market perfectly (after beta scaling) — the fair null. ``alpha_ann<0`` mirrors the
  hypothesis that the AI fund *underperforms* after fees, which is what the real tape
  shows. This is the study's null in a bottle.

- ``fetch_daily`` — the real Yahoo! daily adjusted-close tape (AIEQ, SPY, IVV), cache-
  only by default so the test-suite and the reproducible core never touch the network.
  AIEQ launched 18 Oct 2017; Yahoo data covers that date forward.

No look-ahead is baked in here: returns are computed as close_{t}/close_{t-1} - 1, and
strategy.py uses them as-is (no forward information).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

AIEQ_LAUNCH = "2017-10-18"   # First trading day for AIEQ
TICKERS = ["AIEQ", "SPY", "IVV"]

# Expense ratios (annual): AIEQ ~0.75%, SPY ~0.0945%, IVV ~0.03%
ER_AIEQ = 0.0075
ER_SPY = 0.000945
ER_IVV = 0.0003


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 500,
    alpha_ann: float = 0.0,
    beta: float = 1.05,
    market_vol_ann: float = 0.16,
    idio_vol_ann: float = 0.08,
    start: str = "2020-01-02",
    seed: int = 139,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily price frame mimicking AIEQ vs SPY.

    The market (SPY-like) follows a random walk with annual vol ``market_vol_ann``.
    The AI fund (AIEQ-like) is generated as:

        r_fund_t = alpha_daily + beta * r_market_t + eps_t

    where ``alpha_daily = (1 + alpha_ann)^(1/252) - 1`` and ``eps_t`` is i.i.d.
    normal of annual vol ``idio_vol_ann / sqrt(252)``. ``alpha_ann=0`` is a fair null
    where the fund adds no alpha; ``alpha_ann=-0.044`` replicates the real-tape result
    (~-4.4%/yr Jensen alpha). The ``beta`` default (1.05) matches the empirical OLS
    estimate from the real tape.

    Returns ``(prices, truth)`` where ``prices`` is a DataFrame with columns
    ``['aieq', 'spy', 'ivv']`` (IVV tracks SPY almost exactly here) and ``truth``
    records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    # Market (SPY proxy)
    sigma_mkt_daily = market_vol_ann / np.sqrt(252)
    mu_mkt_daily = 0.07 / 252   # roughly 7%/yr drift in SPY
    r_mkt = rng.normal(mu_mkt_daily, sigma_mkt_daily, n_days)

    # IVV almost identical to SPY (tiny tracking error)
    r_ivv = r_mkt + rng.normal(0, 0.0002, n_days)  # 2bp daily TE

    # AI fund
    alpha_daily = (1.0 + alpha_ann) ** (1.0 / 252) - 1.0
    sigma_idio = idio_vol_ann / np.sqrt(252)
    r_fund = alpha_daily + beta * r_mkt + rng.normal(0, sigma_idio, n_days)

    # Price series (starts at 100)
    spy_px = 100.0 * np.cumprod(1.0 + r_mkt)
    ivv_px = 100.0 * np.cumprod(1.0 + r_ivv)
    aieq_px = 100.0 * np.cumprod(1.0 + r_fund)

    prices = pd.DataFrame(
        {"aieq": aieq_px, "spy": spy_px, "ivv": ivv_px},
        index=idx,
    )
    prices.index.name = "Date"

    truth = {
        "alpha_ann": alpha_ann,
        "alpha_daily": alpha_daily,
        "beta": beta,
        "market_vol_ann": market_vol_ann,
        "idio_vol_ann": idio_vol_ann,
        "n_days": n_days,
        "seed": seed,
    }
    return prices, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "").lower()
    return os.path.join(cache_dir, f"{safe}_daily.parquet")


def fetch_daily(
    ticker: str,
    start: str = AIEQ_LAUNCH,
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily adjusted-close OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``). The tape starts from AIEQ's launch date (2017-10-18)
    so all series are aligned on the same window. ``auto_adjust=True`` ensures total-
    return pricing (dividends reinvested), so we are comparing like with like — the
    expense-ratio drag is already baked into the market prices that Yahoo records.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_daily({ticker!r}, fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, start=start, end=end, interval="1d",
            auto_adjust=True, progress=False,
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)
        df.index.name = "Date"
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(path)

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def load_prices(
    tickers: list[str] | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Load adjusted-close prices for AIEQ, SPY (and optionally IVV), aligned on
    common trading days. Returns a DataFrame of close prices, one column per ticker.

    This is the entry point for strategy.py and the notebooks.
    """
    if tickers is None:
        tickers = TICKERS
    frames = {}
    for t in tickers:
        df = fetch_daily(t, fetch=fetch, cache_dir=cache_dir)
        col = "close" if "close" in df.columns else "Close"
        frames[t] = df[col]
    prices = pd.DataFrame(frames).dropna()
    prices.index.name = "Date"
    return prices


def fingerprint(series: pd.Series) -> str:
    """A short content fingerprint of a price series, for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(series.to_numpy()).tobytes())
    return h.hexdigest()[:12]
