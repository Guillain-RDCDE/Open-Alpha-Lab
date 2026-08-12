"""Data layer for Study 912 — Gold + Trend (a 200-day trend overlay on gold).

Two tapes, one shape (a date-indexed daily total-return close frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the gold ETFs (GLD, IAU), a cash proxy
  (BIL, the 1-3 month T-bill ETF), and two diversification benchmarks (SPY, TLT).
  ``fetch`` touches the network and caches parquet under ``_cache/`` (retry up to
  4×); ``load_prices`` reads that cache **offline** and never imports yfinance.
  The whole test-suite runs with NO cache present (synthetic-only), so CI is green
  on a fresh checkout where ``_cache/`` is git-ignored and absent.

- ``synthetic_daily`` — a *deterministic, offline* generator. A two-state Markov
  bull/bear price tape plus a cash accrual leg. The ``signal_strength`` knob blends
  the world toward a flat-vol null: at ``signal_strength=0`` the 200-day trend filter
  has nothing to read (the null); at ``signal_strength=1`` the bear regime is a real,
  detectable dead-decade the filter can duck. Seed is fixed → tests are deterministic.

Gold is the poster child for a *volatile diversifier with long dead decades*
(1980-2001, 2012-2018). The question this study asks: does a mechanical 200-day trend
filter — hold gold only above its 200-day moving average, else sit in T-bills — turn
gold into a **drawdown-managed diversifier** with a better excess-of-cash Sharpe and
much shallower drawdowns than buy-and-hold gold?

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the signal
formed on data through day ``t`` is acted on at day ``t+1``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# The liquid ETFs this study pulls. GLD/IAU = gold; BIL = 1-3M T-bill (cash proxy);
# SPY/TLT = the diversification benchmarks a gold sleeve sits next to.
TICKERS = ("GLD", "IAU", "BIL", "SPY", "TLT")

# Study-wide as-of: the last COMPLETE calendar month at build time (drop the partial
# current month so the sample never creeps between reruns).
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2004-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the cache. Uses ``auto_adjust=True`` so the
    ``close`` column is split- and dividend-adjusted total return — essential for a
    multi-decade buy-and-hold comparison (GLD pays no dividend, but BIL/SPY/TLT do,
    and the cash leg's yield is the whole point of the excess-of-cash race).
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(
                    tk, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned no data for {tk}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        df = raw[["close"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.dropna(subset=["close"])
        df.to_parquet(_cache_path(tk, cache_dir))
        out[tk] = df
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's parquet is present in the cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read cached daily total-return closes OFFLINE into one aligned close frame.

    Returns a frame indexed by date with one column per ticker (the adjusted close),
    sliced to ``asof`` so the sample never creeps. Raises ``FileNotFoundError`` if any
    ticker is missing — the offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call gold_trend.data.fetch() once to populate the cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    return df


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted "dead decade")
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_years: int = 30,
    drift_bull: float = 0.14,          # annualised bull-market return
    drift_bear: float = -0.18,         # annualised dead-decade return (a slow grind down)
    vol_bull: float = 0.13,            # annualised bull vol
    vol_bear: float = 0.14,            # annualised dead-decade vol (low — a grind, not a crash)
    p_bull_to_bear: float = 0.02,      # monthly prob of entering a dead decade (~50mo bull)
    p_bear_to_bull: float = 0.035,     # monthly prob of leaving it (~28mo grind)
    signal_strength: float = 1.0,      # 0 = flat-vol null, 1 = full regime
    start: str = "1994-01-03",
    seed: int = 912,
    cash_rate_ann: float = 0.03,       # cash yield credited when out of the asset
) -> tuple[pd.DataFrame, dict]:
    """A daily gold-like price tape with a two-state Markov bull / dead-decade regime.

    The ``signal_strength`` knob blends the two-regime world toward a flat-vol null:

    - ``signal_strength = 1`` → full regime separation; the 200-day trend filter can
      see the dead decade coming (via the sustained downtrend) and move to cash.
    - ``signal_strength = 0`` → drift and vol collapse to a single state; the trend
      filter carries no net information (the null — the overlay must NOT win).

    Returns ``(prices, truth)`` where ``prices`` has columns ``asset`` (the gold-like
    total-return close) and ``cash`` (a cash accrual index), plus ``truth`` recording
    the planted parameters and the realised bear fraction. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = n_years * TRADING_DAYS_PER_YEAR
    # OOB-safe: bdate_range with n <= 10000 daily bars stays well inside pandas' ns range.
    dates = pd.bdate_range(start=start, periods=n_days)

    # Blend regime params toward the null at signal_strength=0.
    stationary_bear = p_bull_to_bear / (p_bull_to_bear + p_bear_to_bull)
    avg_vol = (1 - stationary_bear) * vol_bull + stationary_bear * vol_bear
    avg_drift = (1 - stationary_bear) * drift_bull + stationary_bear * drift_bear
    vol_bull_eff = (1 - signal_strength) * avg_vol + signal_strength * vol_bull
    vol_bear_eff = (1 - signal_strength) * avg_vol + signal_strength * vol_bear
    drift_bull_eff = (1 - signal_strength) * avg_drift + signal_strength * drift_bull
    drift_bear_eff = (1 - signal_strength) * avg_drift + signal_strength * drift_bear

    d_bull = drift_bull_eff / TRADING_DAYS_PER_YEAR
    d_bear = drift_bear_eff / TRADING_DAYS_PER_YEAR
    s_bull = vol_bull_eff / np.sqrt(TRADING_DAYS_PER_YEAR)
    s_bear = vol_bear_eff / np.sqrt(TRADING_DAYS_PER_YEAR)

    MONTH = 21
    regime = np.zeros(n_days, dtype=int)   # 0 = bull, 1 = dead decade
    state = 0
    for i in range(n_days):
        if i % MONTH == 0 and i > 0:
            if state == 0:
                state = 1 if rng.random() < p_bull_to_bear else 0
            else:
                state = 0 if rng.random() < p_bear_to_bull else 1
        regime[i] = state

    log_ret = np.where(
        regime == 0,
        rng.normal(d_bull, s_bull, n_days),
        rng.normal(d_bear, s_bear, n_days),
    )
    asset = 100.0 * np.exp(np.cumsum(log_ret))
    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash_idx = np.cumprod(np.full(n_days, cash_daily))

    prices = pd.DataFrame(
        {"asset": asset, "cash": cash_idx},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": signal_strength,
        "drift_bull": drift_bull,
        "drift_bear": drift_bear,
        "vol_bull": vol_bull,
        "vol_bear": vol_bear,
        "p_bull_to_bear": p_bull_to_bear,
        "p_bear_to_bull": p_bear_to_bull,
        "n_years": n_years,
        "n_days": n_days,
        "seed": seed,
        "cash_rate_ann": cash_rate_ann,
        "bear_frac": float(regime.mean()),
        "vol_bull_eff": float(vol_bull_eff),
        "vol_bear_eff": float(vol_bear_eff),
    }
    return prices, truth
