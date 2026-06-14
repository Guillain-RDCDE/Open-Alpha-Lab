"""Data layer for Study 147 (FX-Momentum).

Two tapes, one logical shape (a daily panel of G10 FX vs USD log-returns, monthly rebalance):

- ``synthetic_panel`` — a *deterministic, offline* generator. A ``momentum_signal``
  knob injects cross-sectional return persistence: with ``momentum_signal > 0`` the
  currency with the strongest past return continues to outperform; at
  ``momentum_signal = 0`` returns are i.i.d. across currencies and time — a fair
  shuffled baseline where the momentum portfolio earns zero. This is the study's null
  in a bottle: any test that claims a signal must beat this.

- ``fetch_panel`` — the real G10 FX vs USD daily panel via yfinance spot rates.
  Cache-only by default (``fetch=False`` raises FileNotFoundError if no cache exists;
  ``fetch=True`` hits Yahoo and caches a parquet). Nine pairs: EURUSD, GBPUSD, JPY,
  AUDUSD, USDCAD, USDCHF, NZDUSD, USDNOK, USDSEK — all quoted as USD per foreign
  currency (inverted where necessary so positive log-return = USD appreciation of
  foreign ccy). USD-quoted meaning: positive = foreign ccy appreciated vs USD.

No look-ahead is baked in here: returns used to form the signal at month-end t are
log-returns through t; the position is entered at the start of t+1.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

# G10 FX tickers on Yahoo Finance.
# Pairs quoted as USD per 1 foreign unit (EURUSD, GBPUSD, AUDUSD, NZDUSD).
# USD-base pairs (USDCAD, USDCHF, USDJPY, USDNOK, USDSEK) are inverted so all
# series represent: positive log-return = foreign currency appreciated vs USD.
G10_TICKERS: list[str] = [
    "EURUSD=X",   # EUR/USD — positive = EUR stronger
    "GBPUSD=X",   # GBP/USD — positive = GBP stronger
    "JPY=X",      # USD/JPY on Yahoo — INVERTED to JPY/USD
    "AUDUSD=X",   # AUD/USD
    "USDCAD=X",   # USD/CAD — INVERTED to CAD/USD
    "USDCHF=X",   # USD/CHF — INVERTED to CHF/USD
    "NZDUSD=X",   # NZD/USD
    "USDNOK=X",   # USD/NOK — INVERTED to NOK/USD
    "USDSEK=X",   # USD/SEK — INVERTED to SEK/USD
]
# Tickers that need sign inversion (they are quoted as USD per unit of USD, i.e.,
# USD is the price currency; we want foreign ccy vs USD).
_INVERT: set[str] = {"JPY=X", "USDCAD=X", "USDCHF=X", "USDNOK=X", "USDSEK=X"}

# Short names for display (preserving order of G10_TICKERS).
CCY_NAMES: list[str] = ["EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "NOK", "SEK"]

LOOKBACK_MONTHS: int = 12   # 12-1 momentum: past 12 months, skip most recent 1
SKIP_MONTHS: int = 1        # skip the most recent month (reversal bias)
TOP_N: int = 3              # long the top-3 strongest currencies
BOT_N: int = 3              # short the bottom-3 weakest currencies


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_currencies: int = 9,
    n_months: int = 240,
    momentum_signal: float = 0.0,
    base_vol: float = 0.03,
    seed: int = 147,
    start: str = "2004-01-01",
) -> tuple[pd.DataFrame, dict]:
    """A reproducible monthly FX return panel with tunable cross-sectional momentum.

    Monthly log-returns for ``n_currencies`` are drawn i.i.d. normal(0, ``base_vol``).
    When ``momentum_signal > 0`` the rank of the prior 12-month cumulative return is
    used to inject a cross-sectional trend: the top half of the ranking gets a small
    positive lift and the bottom half a matching negative lift, proportional to
    ``momentum_signal``. At ``momentum_signal = 0`` there is no forecastable structure
    and the momentum portfolio earns zero on average — the null.

    Returns ``(returns_df, truth)`` where ``returns_df`` is a monthly DataFrame with
    currencies as columns (labelled C0..C{n_currencies-1}), index = month-end dates;
    ``truth`` records the planted parameters.

    No look-ahead: the signal is always computed on returns up to month t; the
    portfolio enters at month t+1.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_months, freq="BME", name="date")
    ccy_cols = [f"C{i}" for i in range(n_currencies)]

    # Base i.i.d. monthly returns (log).
    raw = rng.normal(0.0, base_vol, size=(n_months, n_currencies))

    if momentum_signal != 0.0:
        # At each month t, rank the trailing 12-month return (excluding current month)
        # and inject a proportional premium.  We plant the signal on the PAST and
        # apply it to the FUTURE (month t+1 in the strategy), so there is no look-ahead
        # inside the data generator; the strategy layer respects the same shift.
        lookback = 12
        for t in range(lookback, n_months):
            past = raw[t - lookback : t, :].sum(axis=0)  # 12-month cum log ret
            ranks = past.argsort().argsort()  # 0 = weakest
            # Normalise ranks to [-0.5, +0.5] and scale by momentum_signal.
            premium = (ranks / (n_currencies - 1) - 0.5) * momentum_signal
            raw[t] += premium

    returns_df = pd.DataFrame(raw, index=idx, columns=ccy_cols)
    truth = {
        "n_currencies": n_currencies,
        "n_months": n_months,
        "momentum_signal": momentum_signal,
        "base_vol": base_vol,
        "seed": seed,
    }
    return returns_df, truth


# ---------------------------------------------------------------------------
# Real tape — G10 FX daily panel, Yahoo Finance, cache-first
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "fx_momentum_g10_daily.parquet")


def fetch_panel(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    start: str = "2000-01-01",
) -> pd.DataFrame:
    """G10 FX vs USD daily log-return panel; cache-only unless ``fetch=True``.

    Returns a DataFrame with columns = CCY_NAMES (EUR, GBP, JPY, AUD, CAD, CHF, NZD,
    NOK, SEK), index = calendar date, values = daily log-returns expressed as
    *foreign currency vs USD* (positive = foreign ccy appreciated). USD-base pairs
    (JPY, CAD, CHF, NOK, SEK) are sign-inverted so the convention is uniform.

    Network is touched only with ``fetch=True`` (caches a parquet under ``_cache/``).
    Call once to populate the cache, then use ``fetch=False`` (the default) so tests
    and the reproducible core stay offline.
    """
    path = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached G10 FX panel at {path}. "
                "Call fetch_panel(fetch=True) once to populate the cache, or run "
                "examples/verify.py --fetch."
            )
        daily = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we hit the network

        raw = yf.download(
            G10_TICKERS, start=start, auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError("yfinance returned no FX data")

        # Extract Close prices; handle MultiIndex columns.
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[[c for c in raw.columns if "close" in str(c).lower()]]
            closes.columns = G10_TICKERS

        # Rename columns to CCY_NAMES.
        closes = closes.rename(columns=dict(zip(G10_TICKERS, CCY_NAMES)))

        # Compute log-returns and invert USD-base pairs.
        log_ret = np.log(closes).diff()
        for ticker, name in zip(G10_TICKERS, CCY_NAMES):
            if ticker in _INVERT:
                log_ret[name] = -log_ret[name]

        daily = log_ret.dropna(how="all")
        daily.index = pd.DatetimeIndex(daily.index).tz_localize(None)
        daily.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        daily.to_parquet(path)

    daily.index = pd.DatetimeIndex(daily.index).tz_localize(None)
    daily.index.name = "date"
    daily = daily[daily.index >= start]
    return daily


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily log-returns to calendar-month totals.

    Sum log-returns within each calendar month so cumulative log-returns are
    additive.  The result is indexed by month-end date (period end).
    """
    monthly = daily.resample("ME").sum()
    # Drop months with fewer than 10 trading days of data (start/end boundary).
    counts = daily.resample("ME").count()
    monthly = monthly[counts.min(axis=1) >= 10]
    monthly.index.name = "date"
    return monthly


def fingerprint(returns: pd.DataFrame) -> str:
    """A short content fingerprint of the return panel (row-sum), for the as-of stamp."""
    arr = returns.sum(axis=1).to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
