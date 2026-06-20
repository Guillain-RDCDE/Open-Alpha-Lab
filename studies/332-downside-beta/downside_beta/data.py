"""Data layer for Study 332 (Downside-Beta).

Two tapes, one shape — a panel of daily stock returns plus a market return series,
from which downside-beta is estimated and the cross-section is sorted:

- ``synthetic_panel`` — a *deterministic, offline* generator. Each firm is given a
  market beta drawn cross-sectionally, an **extra downside loading** that fires only on
  market-down days, and idiosyncratic noise. A ``downside_premium`` knob then prices
  that downside loading: firms that crash *with* the market earn a higher average
  return by a *known* amount. ``downside_premium = 0`` is the null — downside-beta is
  pure risk with no compensation, so a high-minus-low β⁻ sort earns zero in
  expectation. This is the study's null in a bottle, plus its positive control.

- ``load_real`` — the real Yahoo! daily-close panel for the (current) S&P 500
  universe (via ``quantlab.universe`` / ``yfinance``), **cache-first**: it reads a
  parquet under ``_cache/`` and only touches the network on an explicit ``fetch=True``.
  The test-suite and the reproducible core never go online.

**Survivorship-bias caveat**: the real ticker list is the *current* S&P 500 membership
projected backwards, fetched through the opt-in guard
``quantlab.universe.sp500_symbols(allow_survivorship_bias=True)``. Every firm in the
panel survived to today. A downside-risk premium estimated on survivors is an upper
bound on what a real-time investor could have harvested — and survivorship can point
*against* a downside premium too (firms that took the biggest down-market hits and
*didn't* recover are simply missing). This is named on the Signal axis, not buried in
Tradability.

No look-ahead: downside-beta for the holding month ``t`` is estimated on a trailing
window ending strictly before ``t`` (see ``strategy.downside_beta``); the holding
return is month ``t``'s realised return. The one execution lag lives in the strategy.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
SHARED_CACHE = os.path.join(REPO_ROOT, "_cache")

# Real daily-return panel cache (date x ticker) + a market series alongside.
PANEL_CACHE = os.path.join(DEFAULT_CACHE, "dbeta_daily_returns.parquet")
MARKET_CACHE = os.path.join(DEFAULT_CACHE, "dbeta_market.parquet")

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 200,
    n_days: int = 2520,            # ~10 trading years
    downside_premium: float = 0.0025,  # per-day extra return per unit downside loading
    market_vol: float = 0.011,
    market_drift: float = 0.0003,
    idio_vol: float = 0.010,
    beta_spread: float = 0.5,      # cross-sectional sd of plain beta around 1.0
    dbeta_spread: float = 0.6,     # cross-sectional sd of the EXTRA down-day loading
    seed: int = 332,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """A firm × day return panel with a *known* downside-beta premium.

    The data-generating process for firm ``j`` on day ``t`` is::

        r[t,j] = alpha[j]
               + beta[j] * mkt[t]
               + gamma[j] * mkt[t] * 1{mkt[t] < 0}     # extra DOWN-day sensitivity
               + eps[t,j]

    where ``gamma[j] >= 0`` is the firm's *extra* downside loading (it amplifies losses
    on down-market days only). Left alone, that loading drags the average return of
    high-β⁻ firms down by ``gamma[j] * E[mkt·1{down}]`` (a *mechanical* deficit, since
    down days lose on average). We set::

        alpha[j] = -gamma[j] * E[mkt·1{down}] + downside_premium * gamma[j]

    so the first term exactly *neutralises* that mechanical drag and the second is the
    only thing that moves the realised high-minus-low spread. ``downside_premium`` is
    therefore a clean dial on the *priced* compensation, in per-day return units:

    - ``downside_premium = 0`` → the null: downside risk is uncompensated, so a
      high-minus-low β⁻ sort earns **zero** in expectation (the drag is offset, nothing
      is paid for bearing it).
    - ``downside_premium > 0`` → a real, plantable premium the engine must recover.

    Note the down-day loading also *creates* the downside-beta dispersion the sort keys
    on: ``beta_down[j] ≈ beta[j] + gamma[j]`` while ``beta_up[j] ≈ beta[j]``.

    Returns ``(returns_df, market, truth)`` where ``returns_df`` is a (day × firm) frame
    of simple daily returns, ``market`` is the daily market return Series, and ``truth``
    records the planted parameters (including each firm's gamma).
    """
    rng = np.random.default_rng(seed)
    # Decorative monthly-ish daily index built safely (period_range, no overflow path).
    sessions = pd.bdate_range(start="2014-01-02", periods=n_days)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Market: fat-ish tails so down days are meaningful.
    mkt = market_drift + rng.standard_t(df=6, size=n_days) * market_vol
    down = mkt < 0.0

    beta = 1.0 + rng.normal(0.0, beta_spread, n_firms)
    # Extra downside loading is non-negative (half-normal) — firms only crash *extra*,
    # they don't get *less* sensitive on down days. This is what beta_dn - beta_up keys on.
    gamma = np.abs(rng.normal(0.0, dbeta_spread, n_firms))
    # Neutralise the mechanical down-day drag, then add the priced premium on top, so
    # downside_premium is a clean dial on the realised high-minus-low spread.
    e_mkt_down = float((mkt * down).mean())     # E[mkt·1{down}] over THIS realised tape
    alpha = (-e_mkt_down + downside_premium) * gamma

    eps = rng.normal(0.0, idio_vol, (n_days, n_firms))
    mkt_col = mkt[:, None]
    down_col = down[:, None].astype(float)
    ret = (
        alpha[None, :]
        + beta[None, :] * mkt_col
        + gamma[None, :] * mkt_col * down_col
        + eps
    )

    returns_df = pd.DataFrame(ret, index=pd.DatetimeIndex(sessions, name="date"), columns=firms)
    market = pd.Series(mkt, index=returns_df.index, name="market")

    truth = {
        "downside_premium": downside_premium,
        "n_firms": n_firms,
        "n_days": n_days,
        "market_vol": market_vol,
        "idio_vol": idio_vol,
        "beta_spread": beta_spread,
        "dbeta_spread": dbeta_spread,
        "seed": seed,
        "beta": beta,
        "gamma": gamma,
        "alpha": alpha,
    }
    return returns_df, market, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily returns for the S&P 500, cache-first
# ---------------------------------------------------------------------------
def load_real(
    fetch: bool = False,
    panel_path: str = PANEL_CACHE,
    market_path: str = MARKET_CACHE,
    start: str = "2000-01-01",
    allow_survivorship_bias: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Real daily-return panel + market return; cache-first, network only on ``fetch=True``.

    Returns ``(returns_df, market)`` where ``returns_df`` is a (date × ticker) frame of
    simple daily returns for the S&P 500 universe and ``market`` is the equal-weight
    market daily return (a clean, dividend-agnostic proxy built from the same panel, so
    the cross-section and its benchmark share one adjustment convention).

    **Survivorship bias**: the universe is the current S&P 500 projected backwards via
    ``quantlab.universe.sp500_symbols(allow_survivorship_bias=True)`` — the opt-in guard.
    All real-tape results are upper bounds; the caveat travels on the Signal axis.

    Cache-first: if the parquet exists it is read offline. Callers in CI / tests must
    guard with ``os.path.exists(panel_path)`` and skip when absent.
    """
    if not fetch:
        if not (os.path.exists(panel_path) and os.path.exists(market_path)):
            raise FileNotFoundError(
                f"No cached real panel at {panel_path} / {market_path}. "
                "Call load_real(fetch=True) once to populate the cache."
            )
        returns_df = pd.read_parquet(panel_path)
        market = pd.read_parquet(market_path).iloc[:, 0]
        market.name = "market"
        return returns_df, market

    # ------------------------------------------------------------------
    # Live fetch (only on explicit fetch=True).
    # ------------------------------------------------------------------
    import yfinance as yf  # lazy import: network only on fetch=True

    if not allow_survivorship_bias:
        raise ValueError(
            "load_real(fetch=True) uses a current-membership universe; pass "
            "allow_survivorship_bias=True to opt in (the bias is named on the Signal axis)."
        )
    try:
        from quantlab.universe import sp500_symbols
        tickers = sp500_symbols(allow_survivorship_bias=True)
    except Exception:
        tickers = [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
            "UNH", "JNJ", "JPM", "V", "PG", "MA", "HD", "XOM", "CVX", "MRK",
            "ABBV", "PEP", "KO", "AVGO", "COST", "WMT", "LLY", "TMO", "MCD",
            "CSCO", "ACN", "ABT", "CRM", "DHR", "NEE", "TXN", "NFLX", "QCOM",
            "AMD", "ORCL", "INTC", "CMCSA", "ADBE", "PFE", "T", "DIS", "RTX",
            "HON", "BAC", "GE", "CAT", "GS",
        ]

    raw = yf.download(
        tickers, start=start, interval="1d", auto_adjust=True, progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no daily data")
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=0)
    else:
        closes = raw
    closes.index = pd.to_datetime(closes.index)
    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)
    closes = closes.sort_index()
    # Keep names with enough history for a stable beta estimate.
    closes = closes.loc[:, closes.count() >= 252]
    returns_df = closes.pct_change().dropna(how="all")
    returns_df.index.name = "date"
    # Equal-weight market proxy from the same panel.
    market = returns_df.mean(axis=1).rename("market")

    os.makedirs(os.path.dirname(panel_path), exist_ok=True)
    returns_df.to_parquet(panel_path)
    market.to_frame().to_parquet(market_path)
    print(f"Cached real panel: {returns_df.shape[0]} days x {returns_df.shape[1]} tickers")
    return returns_df, market


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    """A short content fingerprint (last row / value), for the as-of stamp."""
    if isinstance(df, pd.Series):
        arr = df.dropna().to_numpy(dtype=float)
    else:
        arr = df.iloc[-1].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
