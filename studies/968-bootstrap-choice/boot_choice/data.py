"""Data layer for Study 968 — Which Bootstrap.

The real tapes here are not where the answer comes from — coverage cannot be measured
on data whose true mean is unknown. They are used for the *second* question: how much the
choice of bootstrap changes the interval a desk would actually publish.

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

# Four tapes with deliberately different dependence structures: an index (vol clustering,
# almost no return autocorrelation), long bonds, gold, and bitcoin (the fat-tail stress test).
TICKERS = ("SPY", "TLT", "GLD", "BTC-USD")

# Study-wide as-of: the sample is sliced here so reruns cannot creep.
AS_OF = "2026-06-30"
START = "1993-01-01"


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
                f"Call boot_choice.data.fetch() once to populate the shared cache."
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
def synthetic_returns(
    n_years: int = 10,
    mu_ann: float = 0.06,
    vol_ann: float = 0.16,
    ar1: float = 0.0,                 # serial correlation in returns
    garch_alpha: float = 0.08,        # volatility clustering
    garch_beta: float = 0.90,
    df_t: float = 6.0,                # fat tails
    signal_strength: float = 1.0,     # scales BOTH dependence knobs; 0 = i.i.d. null
    start: str = "2004-01-02",
    seed: int = 968,
) -> tuple[pd.Series, dict]:
    """Daily returns with a **known mean and Sharpe** and controllable dependence.

    Three dials, because the bootstrap literature's disagreements are entirely about which of
    them is present:

    - ``ar1`` — serial correlation in the returns themselves (a trending or mean-reverting
      tape). This is what block bootstraps exist for.
    - ``garch_alpha`` / ``garch_beta`` — volatility clustering with no autocorrelation in the
      returns. Common in markets, and the case where intuition most often goes wrong.
    - ``df_t`` — fat tails, which inflate the sampling variance of any variance-based
      statistic (and therefore of a Sharpe ratio).

    ``signal_strength`` scales the AR(1) and the ARCH term together: at 0 the process is
    i.i.d. Student-t, the null in which every bootstrap must deliver its nominal coverage.

    The population mean is ``mu_ann / TRADING_DAYS_PER_YEAR`` **exactly** and the population
    Sharpe is returned in ``truth`` — that is what makes coverage measurable at all.
    """
    rng = np.random.default_rng(seed)
    n = int(n_years * TRADING_DAYS_PER_YEAR)
    dates = pd.bdate_range(start=start, periods=n)

    a = float(garch_alpha) * float(signal_strength)
    b = float(garch_beta)
    phi = float(ar1) * float(signal_strength)
    sd_d = vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    omega = sd_d ** 2 * max(1.0 - a - b, 1e-6)

    z = rng.standard_t(df_t, n) / np.sqrt(df_t / (df_t - 2.0))
    e = np.empty(n)
    v = sd_d ** 2
    for t in range(n):
        v = omega + a * (e[t - 1] ** 2 if t else sd_d ** 2) + b * v
        e[t] = np.sqrt(v) * z[t]

    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    # Scale by the POPULATION factor, never by the realised sample moments. Normalising a
    # draw to its own mean and standard deviation is the classic way to build a coverage
    # study that reports 100% coverage for every method: it removes exactly the sampling
    # variation the interval is supposed to describe. An AR(1) with innovation variance
    # sd^2 has population variance sd^2/(1-phi^2), so the factor below is analytic.
    x = x * np.sqrt(max(1.0 - phi ** 2, 1e-9))
    mu_d = mu_ann / TRADING_DAYS_PER_YEAR
    r = pd.Series(mu_d + x, index=pd.DatetimeIndex(dates, name="date"), name="ret")

    truth = {"n_days": n, "n_years": n_years, "seed": seed, "mu_daily": float(mu_d),
             "mu_ann": mu_ann, "vol_ann": vol_ann, "ar1": phi, "garch_alpha": a,
             "garch_beta": b, "df_t": df_t, "signal_strength": float(signal_strength),
             "sharpe_ann": float(mu_d / sd_d * np.sqrt(TRADING_DAYS_PER_YEAR))}
    return r, truth


def synthetic_panel(n_assets: int = 3, n_years: int = 10, signal_strength: float = 1.0,
                    seed: int = 968, cash_rate_ann: float = 0.02):
    """Panel wrapper (prices, cash, truth) so the shared data-layer tests apply unchanged."""
    cols = {}
    for i in range(n_assets):
        r, truth = synthetic_returns(n_years=n_years, signal_strength=signal_strength,
                                     ar1=0.10, seed=seed + i)
        cols[f"A{i}"] = 100.0 * (1.0 + r).cumprod()
    prices = pd.DataFrame(cols)
    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash = pd.Series(np.cumprod(np.full(len(prices), cash_daily)), index=prices.index,
                     name="cash")
    truth = {"n_assets": n_assets, "n_years": n_years, "n_days": len(prices), "seed": seed,
             "signal_strength": float(signal_strength),
             "alpha_vol_eff": 0.05 * float(signal_strength),
             "cash_rate_ann": cash_rate_ann}
    return prices, cash, truth


def synthetic_daily(n_years: int = 10, signal_strength: float = 1.0, seed: int = 968):
    """Single-name convenience wrapper, matching the desk's shared shape."""
    prices, cash, truth = synthetic_panel(n_assets=1, n_years=n_years,
                                          signal_strength=signal_strength, seed=seed)
    return pd.DataFrame({"asset": prices.iloc[:, 0], "cash": cash}), truth

