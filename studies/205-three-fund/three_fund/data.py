"""Data layer for Study 205 (Three-Fund).

Two tapes, one purpose:

- ``synthetic_prices`` — a *deterministic, offline* generator. Simulates correlated
  log-normal price paths for three stylised assets (equity, international, bond) with
  controllable correlation and drift parameters. An ``intl_alpha`` knob lets us dial
  the international sleeve's excess return above zero (the null is ``intl_alpha=0``,
  meaning diversification alone, no extra return). This lets us confirm that the engine
  correctly attributes value to international when one exists, and correctly finds none
  when it doesn't.

- ``load_real`` — daily adjusted-close prices for the canonical three-fund basket
  (VTI, VXUS, BND) plus the two baselines (SPY for US-only, a 60/40 mix of SPY+BND),
  fetched via yfinance and cached as parquet. Cache-only by default.

No look-ahead: all rebalancing weights are fixed (the three-fund is a static allocation),
so there is no forward-looking signal to guard against. The study uses annual rebalancing
to keep turnover costs realistic.

Tickers used:
  - VTI  — Vanguard Total Stock Market ETF (US total market; launched 2001-05-24)
  - VXUS — Vanguard Total International Stock ETF (ex-US; launched 2011-01-26)
  - BND  — Vanguard Total Bond Market ETF (US total bond; launched 2007-04-10)
  - EFA  — iShares MSCI EAFE ETF (developed international; surrogate back to 2001-08-27)
  - SPY  — SPDR S&P 500 ETF (US baseline)

The common start date is determined by the youngest ETF with sufficient history.
VXUS launched 2011-01-26; we use 2011-02-01 as the study start.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# ---------------------------------------------------------------------------
# Canonical tickers used in this study
# ---------------------------------------------------------------------------
THREE_FUND_TICKERS = ["VTI", "VXUS", "BND"]   # the three-fund itself
BASELINE_TICKERS   = ["SPY", "BND"]             # 60/40 baseline components
ALL_TICKERS        = ["VTI", "VXUS", "BND", "SPY"]

# Common universe start: VXUS launched 2011-01-26; use 2011-02-01 to skip partial month.
UNIVERSE_START = "2011-02-01"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_prices(
    n_days: int = 2520,            # ~10 years of trading days
    vol_eq: float = 0.16,          # annual vol for equity (US and intl)
    vol_bond: float = 0.04,        # annual vol for bond
    drift_eq: float = 0.07,        # annual log-drift for US equity
    intl_alpha: float = 0.0,       # additional annual log-drift for international
    corr_us_intl: float = 0.80,    # US vs international equity correlation
    corr_eq_bond: float = -0.10,   # equity vs bond correlation
    start: str = "2011-02-01",
    seed: int = 205,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV-style price matrix for three stylised assets.

    Three assets: US equity (VTI-like), international equity (VXUS-like), total bond
    (BND-like). Log returns follow a correlated multivariate normal with the planted
    drift and correlation structure:

    - ``intl_alpha = 0``   → international equity earns the same log-drift as US;
                             any diversification benefit is pure correlation, not alpha.
    - ``intl_alpha > 0``   → international earns more; the three-fund benefits.
    - ``intl_alpha < 0``   → international is a drag.

    The SPY-like price is set equal to US equity (same process). Returns
    ``(prices, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)

    # Daily scaling
    dt = 1.0 / 252.0
    daily_vol_eq = vol_eq * np.sqrt(dt)
    daily_vol_bond = vol_bond * np.sqrt(dt)
    mu_us   = (drift_eq - 0.5 * vol_eq**2) * dt
    mu_intl = (drift_eq + intl_alpha - 0.5 * vol_eq**2) * dt
    mu_bond = (0.04 - 0.5 * vol_bond**2) * dt  # fixed 4% bond drift

    # Correlation matrix: [US, INTL, BOND]
    cmat = np.array([
        [1.0,           corr_us_intl,  corr_eq_bond],
        [corr_us_intl,  1.0,           corr_eq_bond],
        [corr_eq_bond,  corr_eq_bond,  1.0         ],
    ])
    vols = np.array([daily_vol_eq, daily_vol_eq, daily_vol_bond])
    cov = np.outer(vols, vols) * cmat

    drifts = np.array([mu_us, mu_intl, mu_bond])

    # Draw log returns: shape (n_days, 3)
    log_ret = rng.multivariate_normal(drifts, cov, size=n_days)

    index = pd.bdate_range(start=start, periods=n_days)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(log_ret, axis=0)),
        index=index,
        columns=["VTI", "VXUS", "BND"],
    )
    prices.index.name = "date"
    # SPY is the same process as VTI (both track US total market)
    prices["SPY"] = prices["VTI"].copy()

    truth = {
        "n_days": n_days,
        "vol_eq": vol_eq,
        "vol_bond": vol_bond,
        "drift_eq": drift_eq,
        "intl_alpha": intl_alpha,
        "corr_us_intl": corr_us_intl,
        "corr_eq_bond": corr_eq_bond,
        "seed": seed,
    }
    return prices, truth


# ---------------------------------------------------------------------------
# Real tape — yfinance daily adjusted closes; cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "three_fund_daily.parquet")


def load_real(
    tickers: list[str] | None = None,
    start: str = UNIVERSE_START,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Daily adjusted-close prices for the three-fund basket and baselines.

    Returns a date-indexed DataFrame of adjusted-close prices, one column per ticker.
    The study uses VTI, VXUS, BND and SPY; the common start is ``UNIVERSE_START``
    (2011-02-01, the month after VXUS launched). Network is touched only on
    ``fetch=True`` (result cached as parquet under ``_cache/``).
    """
    if tickers is None:
        tickers = ALL_TICKERS

    path = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached three-fund prices at {path}. "
                f"Call load_real(fetch=True) once to populate the cache."
            )
        prices = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when going to the network

        raw = yf.download(
            tickers,
            start=start,
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            raise RuntimeError("yfinance returned no data for three-fund tickers")
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
        prices = prices.dropna(how="all")
        os.makedirs(cache_dir, exist_ok=True)
        prices.to_parquet(path)

    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.index.name = "date"
    prices = prices[tickers].dropna()
    prices = prices[prices.index >= pd.Timestamp(start)]
    return prices


def fingerprint(prices: pd.DataFrame) -> str:
    """A short content fingerprint of the price matrix, for the as-of stamp."""
    arr = np.ascontiguousarray(prices.to_numpy().ravel())
    h = hashlib.sha1(arr.tobytes())
    return h.hexdigest()[:12]
