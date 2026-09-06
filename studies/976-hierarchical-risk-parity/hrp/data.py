"""Data layer for Study 976 — The Family Tree.

Three panels: eleven sectors, forty single names, and a ten-sleeve multi-asset book.
The single-name panel is the one HRP's claim is about — a thinly estimated covariance matrix
whose inverse is unstable — and the multi-asset panel is where the *clustering* has the most
obvious economic content (bonds hang together, equities hang together).

Two tapes, one shape:

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``), cached as parquet in the **shared** desk cache
  ``studies/_cache``. ``fetch`` is the only thing here that touches the network;
  ``load_prices`` reads the cache offline and never imports yfinance, so the test-suite
  and CI stay green on a fresh checkout where ``_cache/`` is git-ignored and absent.
- ``synthetic_panel`` — a deterministic, offline generator with a ``signal_strength``
  knob: at ``1.0`` the effect this study hunts is **planted** at a known size, at ``0.0``
  the panel is the matching null. Every test runs on it; nothing in the suite needs a
  network or a cache.

The sample is pinned at ``AS_OF`` so a rerun months later cannot quietly grow the window.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache — most of these tapes are already sitting there.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# The same two cross-sections the shrinkage study uses, so the two results are
# comparable: a benign sector sleeve and a wide single-name panel where the covariance
# matrix is thinly estimated. HRP's claim is specifically about the second case.
SECTORS = ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY")
NAMES = ("AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "IBM", "INTC", "QCOM", "TXN",
         "JPM", "BAC", "WFC", "GS", "MS", "AXP", "C", "PNC", "USB", "MET",
         "JNJ", "MRK", "ABT", "AMGN", "DHR", "BDX", "LH", "ZBH",
         "XOM", "CVX", "COP", "SLB", "EOG", "OXY",
         "PG", "KO", "MMM", "GE", "T", "VZ")
MULTI = ("SPY", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "DBC")
BENCH = "SPY"
CASH = "BIL"
TICKERS = SECTORS + NAMES + MULTI + (CASH,)

# Study-wide as-of: the sample is sliced here so reruns cannot creep.
AS_OF = "2026-06-30"
START = "1999-01-01"


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return closes, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(tickers=TICKERS, start: str = START, end: str | None = None,
          cache_dir: str = DEFAULT_CACHE, retries: int = 4) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so the
    ``close`` column is split- **and** dividend-adjusted: a price-only tape would bias
    every comparison here toward whichever leg happens to yield less.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(tk, start=start, end=end, interval="1d",
                                  auto_adjust=True, progress=False)
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
    """True iff every ticker's parquet is present in the shared cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE,
                asof: str = AS_OF) -> pd.DataFrame:
    """Read cached daily total-return closes OFFLINE into one aligned close panel.

    Returns a date-indexed frame with one column per ticker, sliced to ``asof``. A
    ticker that lists late is simply NaN before its inception — never back-filled.
    Raises ``FileNotFoundError`` if any ticker is missing: the offline core and the
    test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call hrp.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)]


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]




# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_assets: int = 6,
    n_years: int = 20,
    market_vol: float = 0.16,           # annualised common-factor vol
    idio_vol: float = 0.14,             # annualised idiosyncratic vol
    alpha_vol: float = 0.06,            # annualised sd of the persistent expected-return leg
    alpha_halflife_days: float = 63.0,  # persistence of that leg (~a quarter)
    signal_strength: float = 1.0,       # 0 = pure-noise null, 1 = fully planted
    start: str = "2004-01-02",
    seed: int = 976,
    cash_rate_ann: float = 0.02,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """A daily total-return panel with a *planted, slowly-decaying* signal.

    Each asset's daily log return is ``beta * market + a_i(t) + idio``, where ``a_i(t)``
    is an AR(1) expected-return component with half-life ``alpha_halflife_days``.
    ``signal_strength`` scales that component: ``0.0`` switches it off entirely and the
    panel becomes a market factor plus pure idiosyncratic noise — the null under which
    nothing in this study may find anything.

    Returns ``(prices, cash, truth)``: a date-by-asset close panel (``A0..A{n-1}``), a
    cash accrual index on the same index, and the planted parameters. Deterministic
    given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    dates = pd.bdate_range(start=start, periods=n_days)

    mkt_d = market_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    idio_d = idio_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    alpha_d = alpha_vol / np.sqrt(TRADING_DAYS_PER_YEAR) * float(signal_strength)

    phi = float(np.exp(-np.log(2.0) / alpha_halflife_days))
    shock_sd = alpha_d * np.sqrt(1.0 - phi ** 2)

    market = rng.normal(0.06 / TRADING_DAYS_PER_YEAR, mkt_d, n_days)
    betas = rng.uniform(0.8, 1.2, n_assets)

    alpha = np.zeros((n_days, n_assets))
    a = rng.normal(0.0, alpha_d, n_assets) if alpha_d > 0 else np.zeros(n_assets)
    shocks = rng.normal(0.0, 1.0, (n_days, n_assets))
    for t in range(n_days):
        a = phi * a + shock_sd * shocks[t]
        alpha[t] = a

    idio = rng.normal(0.0, idio_d, (n_days, n_assets))
    log_ret = market[:, None] * betas[None, :] + alpha + idio
    px = 100.0 * np.exp(np.cumsum(log_ret, axis=0))

    prices = pd.DataFrame(px, index=pd.DatetimeIndex(dates, name="date"),
                          columns=[f"A{i}" for i in range(n_assets)])
    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash = pd.Series(np.cumprod(np.full(n_days, cash_daily)), index=prices.index, name="cash")
    truth = {
        "n_assets": n_assets, "n_years": n_years, "n_days": n_days, "seed": seed,
        "signal_strength": float(signal_strength),
        "alpha_vol_eff": float(alpha_d * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "alpha_halflife_days": alpha_halflife_days, "market_vol": market_vol,
        "idio_vol": idio_vol, "cash_rate_ann": cash_rate_ann, "phi": phi,
    }
    return prices, cash, truth


def synthetic_daily(n_years: int = 20, signal_strength: float = 1.0,
                    seed: int = 976) -> tuple[pd.DataFrame, dict]:
    """Single-name convenience wrapper: one asset of the panel plus the cash leg."""
    prices, cash, truth = synthetic_panel(
        n_years=n_years, signal_strength=signal_strength, seed=seed
    )
    return pd.DataFrame({"asset": prices.iloc[:, 0], "cash": cash}), truth

