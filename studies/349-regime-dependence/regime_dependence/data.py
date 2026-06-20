"""Data layer for Study 349 (Regime-Dependence).

This is a research-method demo, so the data layer's job is to manufacture the exact
situation the method is supposed to catch: a strategy whose headline (full-sample)
performance is good, but whose edge is concentrated in *one* regime. We need a tape
where we *know the ground truth* — which strategy has a durable edge and which one is
a regime artefact — so the decade-by-decade lens can be scored against the planted
answer.

Two tapes, one shape (a wide DataFrame of monthly strategy returns, plus a regime
label per period):

- ``synthetic_regimes`` — a *deterministic, offline* generator. It paints a sequence
  of ``n_regimes`` regimes (decorative monthly labels via ``period_range``, never
  ``date_range(periods=BIG, freq="M")`` which overflows ns-Timestamps), and emits two
  strategy return series on the *same* market:

    * ``durable`` — a small per-period edge present in **every** regime (the positive
      control: a genuinely stable strategy). Its decade-by-decade Sharpe is roughly
      constant.
    * ``regime_fit`` — **zero** edge in every regime except one ``hot_regime``, where a
      large edge is planted (the null for "is it skill?": the strategy that *ruled one
      era*). Its full-sample Sharpe can look excellent while every other decade is flat.

  At ``edge_durable=0`` and ``edge_hot=0`` both collapse to pure noise (a clean null:
  no strategy has any edge, so the lens should certify *neither*).

- ``load_real`` — a real canonical strategy on the broad US equity index: a 12-month
  time-series-momentum (trend) overlay and a static 60/40 stock/bond mix, via the
  shared ``quantlab.data`` loader with a cache-first parquet and graceful offline
  fallback. No survivorship issue (a single index, not a cross-section), but the
  total-return/price-only adjustment mode and one execution lag are documented in
  ``strategy.py``.

The headline question the lens answers is *not* "what is the Sharpe?" but "is the
Sharpe **stable across regimes**, and is any cross-regime gap larger than noise?".
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

MONTHS_PER_YEAR = 12

# The canonical real-tape proxy: a broad US total-return equity index and a bond proxy.
# (^SP500TR has dividends in it; we label adjustment mode explicitly in strategy.py.)
EQUITY_TICKER = "^SP500TR"
BOND_TICKER = "IEF"  # 7-10y Treasuries ETF, total return — the 40 in 60/40


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_regimes(
    n_regimes: int = 6,
    months_per_regime: int = 120,
    hot_regime: int = 2,
    edge_durable: float = 0.0020,
    edge_hot: float = 0.0120,
    market_drift: float = 0.0050,
    market_vol: float = 0.040,
    strat_vol: float = 0.030,
    seed: int = 349,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """A reproducible panel of two strategy return series across ``n_regimes`` decades.

    Each regime is ``months_per_regime`` months long (default 120 ≈ a decade). Both
    strategies share the *same* market factor draw per regime, so they differ only in
    their planted alpha:

        durable_t     = mkt_t + edge_durable + eps_d_t        (every regime)
        regime_fit_t  = mkt_t + (edge_hot if regime==hot else 0) + eps_r_t

    - ``edge_durable`` is a small constant premium present in every regime — a strategy
      whose edge is genuinely regime-independent (the positive control).
    - ``edge_hot`` is a large premium present in **only** ``hot_regime`` and zero
      elsewhere — a strategy that *ruled one decade*. Over the full sample its mean and
      Sharpe can look strong, but a decade-by-decade split exposes the single source.
    - With both edges at 0 the two series are pure noise — the clean null in which the
      lens must certify *no* durable edge for either.

    Returns ``(returns_df, regime, truth)`` where ``returns_df`` has columns
    ``["durable", "regime_fit"]`` of monthly simple returns, ``regime`` is an integer
    regime label per period, and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    n = n_regimes * months_per_regime

    # Decorative monthly labels via period_range (NEVER date_range(periods=BIG, freq="M")
    # — that overflows ns-Timestamps for long spans; see CI pitfalls). Span stays small.
    idx = pd.period_range("1960-01", periods=n, freq="M").to_timestamp()
    idx = pd.DatetimeIndex(idx, name="date")

    regime = np.repeat(np.arange(n_regimes), months_per_regime)[:n]

    # A common market factor (mild positive drift) shared by both strategies.
    mkt = rng.normal(market_drift, market_vol, n)

    eps_d = rng.normal(0.0, strat_vol, n)
    eps_r = rng.normal(0.0, strat_vol, n)

    hot_mask = (regime == hot_regime).astype(float)

    durable = mkt + edge_durable + eps_d
    regime_fit = mkt + edge_hot * hot_mask + eps_r

    df = pd.DataFrame({"durable": durable, "regime_fit": regime_fit}, index=idx)
    regime_s = pd.Series(regime, index=idx, name="regime")

    truth = {
        "n_regimes": n_regimes,
        "months_per_regime": months_per_regime,
        "hot_regime": hot_regime,
        "edge_durable": edge_durable,
        "edge_hot": edge_hot,
        "market_drift": market_drift,
        "market_vol": market_vol,
        "strat_vol": strat_vol,
        "seed": seed,
        "n": n,
    }
    return df, regime_s, truth


# ---------------------------------------------------------------------------
# Real tape — a canonical strategy on the US index, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "regime_prices.parquet")


def load_real(
    start: str = "1990-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real monthly total-return series for the equity index and the bond proxy.

    Cache-first: reads the parquet under ``_cache/`` and never touches the network
    unless ``fetch=True``. On a cache miss with ``fetch=False`` it raises
    ``FileNotFoundError`` (the offline contract) so the reproducible core and tests
    stay deterministic.

    Returns a DataFrame with columns ``["equity", "bond"]`` of monthly simple returns,
    month-end, with the in-progress final (partial) month dropped (house rule: a
    stamped run never includes a partial bar). Adjustment mode is **total return** for
    both legs (``^SP500TR`` already carries dividends; ``IEF`` is total return) — say
    so wherever a number is quoted; do not call a price-only series total return.
    """
    cpath = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(cpath):
            raise FileNotFoundError(
                f"No cached series at {cpath}. "
                f"Call load_real(fetch=True) once to populate the cache."
            )
        rets = pd.read_parquet(cpath)
    else:
        prices = _fetch_prices(start, cache_dir)
        monthly = prices.resample("ME").last()
        rets = monthly.pct_change().dropna(how="all")
        # Drop the in-progress (partial) final month.
        today = pd.Timestamp.today().normalize()
        rets = rets[rets.index < today.to_period("M").to_timestamp()]
        rets = rets.dropna()
        os.makedirs(cache_dir, exist_ok=True)
        rets.to_parquet(cpath)
    return rets.dropna()


def _fetch_prices(start: str, cache_dir: str) -> pd.DataFrame:
    """Daily total-return closes for the equity + bond legs (network on call)."""
    tickers = {"equity": EQUITY_TICKER, "bond": BOND_TICKER}
    frames = {}
    try:
        from quantlab import data as qld  # shared loader, if importable

        for name, t in tickers.items():
            try:
                bars = qld.load(t, start=start, mode="total_return")
                frames[name] = bars["close"] if "close" in bars else bars.iloc[:, 0]
            except Exception:
                continue
    except Exception:
        frames = {}

    if len(frames) < 2:
        import yfinance as yf

        for name, t in tickers.items():
            if name in frames:
                continue
            raw = yf.download(t, start=start, auto_adjust=True, progress=False)
            if raw.empty:
                continue
            col = raw["Close"]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            frames[name] = col

    if len(frames) < 2:
        raise RuntimeError("could not assemble equity+bond price series for the regime study")

    px = pd.DataFrame(frames)
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    return px.dropna(how="all")


def fingerprint(returns: pd.DataFrame) -> str:
    """A short content fingerprint of the return panel, for the as-of stamp."""
    arr = np.ascontiguousarray(returns.to_numpy().ravel())
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
