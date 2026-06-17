"""Data layer for Study 237 (Residual-Momentum).

Two tapes, one shape (a month x ticker panel of prices and market returns):

- ``synthetic_panel`` -- a *deterministic, offline* generator. A ``premium``
  knob plants the exact cross-sectional momentum effect on *residual* returns
  that Blitz, Huij & Martens (2011) claim: past residual winners earn more
  than past residual losers by a known amount. ``premium = 0`` is the null:
  the residual rank carries no information.

- ``fetch_monthly`` -- the real Yahoo! monthly close panel for a representative
  large-cap universe plus the SPY market proxy (via yfinance). Cache lives
  under ``_cache/resmo_monthly.parquet`` and ``_cache/resmo_spy.parquet``.

**Survivorship-bias caveat**: the ticker list is a fixed large-cap basket
(current survivors). Results from the real tape are upper bounds.

No look-ahead: the formation window for month t uses data strictly up to
month t-1. Residuals are computed from a rolling 36-month OLS of each
stock's excess return on the market excess return.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")

MONTHLY_CACHE = os.path.join(DEFAULT_CACHE, "resmo_monthly.parquet")
SPY_CACHE = os.path.join(DEFAULT_CACHE, "resmo_spy.parquet")

# Fixed large-cap basket (survivorship-biased; consistent with other studies)
LARGE_CAP_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
    "UNH", "JNJ", "JPM", "V", "PG", "MA", "HD", "XOM", "CVX", "MRK",
    "ABBV", "PEP", "KO", "AVGO", "COST", "WMT", "LLY", "TMO", "MCD",
    "CSCO", "ACN", "ABT", "CRM", "DHR", "NEE", "TXN", "NFLX", "QCOM",
    "AMD", "ORCL", "INTC", "CMCSA", "ADBE", "PFE", "T", "DIS", "RTX",
    "HON", "BAC", "GE", "CAT", "GS", "IBM", "AXP", "SPGI", "BLK", "SCHW",
    "MMM", "DE", "LMT", "BA", "GD", "NOC", "EMR", "ETN", "PLD", "AMT",
    "SHW", "APD", "ECL", "NUE", "FCX", "NEM", "VALE", "RIO", "CLF", "AA",
]


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 150,
    n_months: int = 300,
    premium: float = 0.04,
    base_ret: float = 0.008,
    ret_vol: float = 0.07,
    market_vol: float = 0.04,
    beta_spread: float = 0.6,
    formation: int = 12,
    seed: int = 237,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A firm x month panel with a known residual-momentum effect.

    Each firm has a randomly-drawn beta drawn from U[0.5, 1.5]. The market
    return is a random walk with ``market_vol`` volatility. Each firm's
    monthly return is::

        r_{i,t} = beta_i * mkt_t + idio_{i,t}

    where ``idio`` is i.i.d. N(0, ret_vol).

    From month ``formation+1`` onward, an extra term is added to reward firms
    whose trailing ``formation``-month *residual* return (idiosyncratic, after
    subtracting beta_i * mkt) is high::

        extra_{i,t} = premium * z(residual_rank_{i,t-1})

    ``premium = 0`` is the null: residual rank carries no information.

    Returns ``(price_df, mkt_df, truth)`` where:
    - ``price_df`` is (month x firm) cumulative prices.
    - ``mkt_df`` is a (month,) Series of market returns.
    - ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Draw firm betas once
    betas = rng.uniform(0.5, 1.5, n_firms)  # shape (n_firms,)

    # Market returns (random walk)
    mkt_ret = rng.normal(base_ret, market_vol, n_months)  # shape (n_months,)

    # Idiosyncratic returns (i.i.d. noise)
    idio = rng.normal(0.0, ret_vol, (n_months, n_firms))  # shape (n_months, n_firms)

    # Base returns: systematic + idiosyncratic
    base_returns = mkt_ret[:, None] * betas[None, :] + idio  # (n_months, n_firms)

    # Build iteratively so planted premium propagates
    all_returns = np.zeros((n_months, n_firms))
    all_returns[:formation] = base_returns[:formation]

    for t in range(formation, n_months):
        # Rolling residual over the prior formation months
        # Residual = actual return - beta_i * market (using true betas for simplicity)
        window_ret = all_returns[t - formation:t]       # (formation, n_firms)
        window_mkt = mkt_ret[t - formation:t]           # (formation,)
        resid_cumsum = (window_ret - window_mkt[:, None] * betas[None, :]).sum(axis=0)  # (n_firms,)

        # Cross-sectional z-score of residuals -> momentum signal
        mu_r = resid_cumsum.mean()
        sd_r = resid_cumsum.std() + 1e-9
        z = (resid_cumsum - mu_r) / sd_r

        extra = premium * z  # reward high-residual firms
        all_returns[t] = base_returns[t] + extra

    # Build cumulative price series from returns
    cum_log = np.cumsum(all_returns, axis=0)
    price_arr = 100.0 * np.exp(cum_log)
    price_df = pd.DataFrame(price_arr, index=months, columns=firms)
    mkt_series = pd.Series(mkt_ret, index=months, name="market")

    truth = {
        "premium": premium,
        "base_ret": base_ret,
        "ret_vol": ret_vol,
        "market_vol": market_vol,
        "n_firms": n_firms,
        "n_months": n_months,
        "formation": formation,
        "seed": seed,
    }
    return price_df, mkt_series, truth


# ---------------------------------------------------------------------------
# Real tape -- Yahoo monthly closes, cache-only by default
# ---------------------------------------------------------------------------
def fetch_monthly(
    fetch: bool = False,
    cache_path: str = MONTHLY_CACHE,
    spy_cache: str = SPY_CACHE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Monthly adjusted close panel for large-cap names + SPY proxy.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached). Returns ``(prices, spy)`` where:
    - ``prices`` is a (month x ticker) frame of adjusted monthly closes.
    - ``spy`` is a (month,) Series of SPY adjusted closes.

    Callers must guard with ``os.path.exists(cache_path)`` before calling
    when ``fetch=False``.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached monthly panel at {cache_path}. "
                "Call fetch_monthly(fetch=True) once to populate the cache."
            )
        prices = pd.read_parquet(cache_path)
        spy = pd.read_parquet(spy_cache).iloc[:, 0]
        spy.name = "SPY"
        return prices, spy

    import yfinance as yf  # lazy import: network only on fetch=True

    tickers = LARGE_CAP_TICKERS + ["SPY"]
    raw = yf.download(
        tickers,
        start="1993-01-01",
        interval="1mo",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no monthly data")

    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=0)
    else:
        closes = raw

    closes.index = pd.to_datetime(closes.index)
    closes.index = closes.index + pd.offsets.MonthEnd(0)
    closes = closes.sort_index()

    # Separate SPY from the stock panel
    spy_col = None
    if "SPY" in closes.columns:
        spy_col = closes["SPY"].copy()
        closes = closes.drop(columns=["SPY"])

    # Drop tickers with fewer than 60 months of data
    closes = closes.loc[:, closes.count() >= 60]
    if spy_col is not None:
        spy_col = spy_col.dropna()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    closes.to_parquet(cache_path)
    if spy_col is not None:
        pd.DataFrame({"SPY": spy_col}).to_parquet(spy_cache)

    print(f"Cached monthly panel: {closes.shape[0]} months x {closes.shape[1]} tickers -> {cache_path}")
    return closes, spy_col


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a prices frame (last row)."""
    arr = df.iloc[-1].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
