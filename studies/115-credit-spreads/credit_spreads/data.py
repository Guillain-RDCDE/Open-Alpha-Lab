"""Data layer for Study 115 (Credit-Spreads).

Two tapes, one shape (a tz-aware daily price frame for three ETFs):

- ``synthetic_daily`` — a *deterministic, offline* generator.  A ``spread_signal``
  knob controls how strongly widening credit-spread precedes negative equity returns.
  ``spread_signal=0`` is a pure random walk — there is no lead-lag — so a test can
  assert the strategy beats random *only* when we plant the relationship.  This is the
  study's null in a bottle.
- ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only by default so
  the test-suite and the reproducible core never touch the network.

Three tickers drive the study:

- **HYG** (iShares iBoxx $ High Yield Corporate Bond ETF) — proxy for high-yield credit.
- **LQD** (iShares iBoxx $ Investment Grade Corporate Bond ETF) — proxy for IG credit.
- **IEF** (iShares 7-10 Year Treasury Bond ETF) — risk-free duration baseline.
- **SPY** (SPDR S&P 500 ETF) — the equity leg.

**Credit-spread proxy (ETF-based).** Because FRED OAS data is not accessible in this
environment, we approximate the credit-risk premium two ways:

1. ``hyg_ief_spread`` — rolling 20-day return of HYG minus IEF (HYG underperformance
   relative to duration-matched Treasuries signals credit stress).
2. ``hyg_lqd_rs`` — rolling 20-day relative strength of HYG vs LQD (HY vs IG, isolating
   the credit-risk component from pure duration).

A widening spread (more negative value) is interpreted as rising credit risk.  No
look-ahead is baked in here — that discipline lives in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TICKERS = ["HYG", "LQD", "IEF", "SPY"]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 500,
    spread_signal: float = 0.0,
    equity_vol: float = 0.012,
    spread_vol: float = 0.008,
    start: str = "2020-01-02",
    seed: int = 115,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV-like frame for HYG, LQD, IEF, SPY.

    The DGP plants a tunable credit-spread lead-lag:

    - ``spread_signal = 0``  → spread changes and equity returns are independent
      (null model; the strategy is a fair coin).
    - ``spread_signal > 0``  → widening spread (negative HYG-IEF return) predicts
      negative SPY return ``spread_signal`` periods forward.  The larger the value,
      the stronger the planted predictability.
    - ``spread_signal < 0``  → inverse relationship (credit tightening → equity down).

    Returns ``(prices, truth)`` where ``prices`` has columns
    ``[hyg, lqd, ief, spy]`` indexed on business days (tz-naive, daily).
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)

    # --- Generate independent log-returns for each series ---
    # SPY: a slight upward drift plus equity vol
    spy_r = rng.normal(0.0003, equity_vol, n_days)
    # IEF: treasury bond, low vol
    ief_r = rng.normal(0.0001, spread_vol * 0.6, n_days)
    # HYG: like IEF but with credit-spread noise on top
    hyg_extra = rng.normal(0.0, spread_vol * 0.8, n_days)
    hyg_r = ief_r + hyg_extra
    # LQD: between HYG and IEF in risk
    lqd_extra = rng.normal(0.0, spread_vol * 0.4, n_days)
    lqd_r = ief_r + lqd_extra

    # --- Plant the lead-lag: spread change → future equity return ---
    if spread_signal != 0.0:
        # Credit-spread proxy: HYG underperformance vs IEF (negative = stress)
        spread_chg = hyg_r - ief_r
        # A widening spread (negative spread_chg) adds a negative impulse to future SPY
        # We use a 1-period lead so no look-ahead: spread_chg[t] predicts spy_r[t+1]
        signal = np.roll(spread_chg, 1)  # shifted one day forward
        signal[0] = 0.0
        spy_r = spy_r + spread_signal * signal  # positive spread_chg (calm) → positive SPY; negative (stress) → negative SPY

    # Build cumulative price series from 100
    def _prices(r):
        return 100.0 * np.exp(np.cumsum(r))

    prices = pd.DataFrame(
        {
            "hyg": _prices(hyg_r),
            "lqd": _prices(lqd_r),
            "ief": _prices(ief_r),
            "spy": _prices(spy_r),
        },
        index=dates,
    )
    truth = {
        "spread_signal": spread_signal,
        "equity_vol": equity_vol,
        "spread_vol": spread_vol,
        "n_days": n_days,
        "seed": seed,
    }
    return prices, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_daily.parquet")


def fetch_daily(
    ticker: str,
    start: str = "2010-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily adjusted-close prices for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``).  The daily history is long (2010+), giving us enough
    data to test the credit-spread signal across multiple credit cycles, including the
    2015-16 HY stress, the 2018 Q4 sell-off, the 2020 COVID crash, and the 2022
    inflation sell-off.

    Returns a DataFrame with column ``close`` and a tz-naive DatetimeIndex.
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
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df.index.name = "date"
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(path)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    return df


def load_all(
    start: str = "2010-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Load all four tickers and return a single aligned DataFrame of adjusted closes.

    Columns: ``hyg``, ``lqd``, ``ief``, ``spy``.  Rows are aligned on their common
    trading days (inner join).  Missing values are forward-filled within each series
    (rare ETF non-trading days), then the intersection is returned.
    """
    frames = {}
    for t in TICKERS:
        df = fetch_daily(t, start=start, end=end, fetch=fetch, cache_dir=cache_dir)
        frames[t.lower()] = df["close"]
    prices = pd.DataFrame(frames).dropna(how="all").ffill().dropna()
    return prices


def fingerprint(prices: pd.DataFrame) -> str:
    """A short content fingerprint of a prices frame (spy close), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(prices["spy"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
