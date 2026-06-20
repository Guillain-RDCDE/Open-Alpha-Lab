"""Data layer for Study 344 (Backtest-Overfitting).

This is a *research-method* study, so the data layer's job is unusual: it must supply a
tape on which an honest backtest is **guaranteed to find nothing**, so that whatever an
optimiser "discovers" is provably overfit. That means two synthetic tapes and one real
one:

- ``synthetic_prices`` — a *deterministic, offline* geometric random walk. One knob,
  ``edge`` (a constant daily drift), turns the only real signal on or off:

    * ``edge = 0`` → a pure martingale. There is NO timing edge to find. Any strategy that
      beats buy-and-hold in-sample did so by luck; this is the **null**, and it is the
      whole point — the trap only proves itself on a tape we *know* is empty.
    * ``edge > 0`` → a tiny constant drift (a trivially real, un-timeable trend). This is
      the **positive control**: a faithful overfitting-detector must still flag the
      *grid-searched timing rules* as overfit even when a (different, buy-and-hold) edge
      exists, because the timing rules are not what's earning the drift.

- ``synthetic_panel_of_signals`` — the artefact the overfitting tools actually consume: an
  (T × N) matrix of per-period returns for **N candidate strategy configurations** run on
  one price path. This is the input to the Deflated Sharpe Ratio (N = number of trials) and
  to PBO/CSCV (the matrix it splits). Built by ``strategy.sweep`` from a price path, but
  exposed here as a one-call convenience for tests and notebooks.

- ``load_real`` — a real daily price series (default SPY) via the shared ``quantlab.data``
  loader, cache-first with a graceful offline fallback. Used for ONE real-world worked
  example: grid-search a moving-average crossover on SPY's *first half*, then watch the
  "best" rule out-of-sample on the *second half*. Survivorship is not a concern (a single
  broad index, not a cross-sectional panel), so there is no opt-in guard here.

No look-ahead is baked into the price generator. The execution-lag and cost discipline
lives in ``strategy`` (signal known at the close of *t* earns the return of *t+1*).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_prices(
    n_days: int = 2520,
    edge: float = 0.0,
    daily_vol: float = 0.011,
    seed: int = 344,
) -> tuple[pd.Series, dict]:
    """A reproducible daily price path: geometric random walk with optional constant drift.

    The log-return each day is ``edge + daily_vol * z_t`` with ``z_t`` standard normal.

    - ``edge = 0`` → a pure martingale. No day is predictable from any other; **there is no
      timing signal in this series.** Any rule that beats buy-and-hold in-sample is luck.
    - ``edge > 0`` → a tiny constant upward drift (e.g. 0.0003/day ≈ 7.8%/yr). The drift is
      real but *un-timeable* — it accrues to buy-and-hold, not to any entry/exit rule. It is
      the positive control: the detector must still call the grid-searched *timing* rules
      overfit.

    Returns ``(prices, truth)`` — ``prices`` a daily close Series, ``truth`` the planted
    parameters (so tests can assert against the ground truth).
    """
    rng = np.random.default_rng(seed)
    log_ret = edge + daily_vol * rng.standard_normal(n_days)
    log_px = np.cumsum(log_ret)
    prices = 100.0 * np.exp(log_px)

    # Decorative business-day labels via period_range -> timestamps (NEVER
    # date_range(periods=BIG, freq=...) for long spans; see CI pitfalls). n_days well under
    # any ns-Timestamp overflow bound.
    idx = pd.period_range("1990-01-01", periods=n_days, freq="D").to_timestamp()
    s = pd.Series(prices, index=pd.DatetimeIndex(idx, name="date"), name="close")

    truth = {
        "n_days": n_days,
        "edge": edge,
        "daily_vol": daily_vol,
        "seed": seed,
        # The honest answer: with edge=0 there is no timing edge; with edge>0 the only edge
        # is buy-and-hold drift, never a timing rule.
        "has_timing_edge": False,
    }
    return s, truth


def synthetic_panel_of_signals(
    n_days: int = 2520,
    n_trials: int = 200,
    edge: float = 0.0,
    daily_vol: float = 0.011,
    seed: int = 344,
) -> tuple[pd.DataFrame, dict]:
    """An (T × N) matrix of candidate-strategy per-period returns — the overfitting input.

    Builds one price path with :func:`synthetic_prices`, then runs ``n_trials`` distinct
    long/flat timing configurations over it (see :func:`strategy.sweep`). The columns are
    the trials; the rows are days. This is exactly the object the Deflated Sharpe Ratio
    (N = ``n_trials``) and PBO/CSCV consume.

    Returns ``(returns_matrix, truth)``.
    """
    from . import strategy as st

    prices, truth = synthetic_prices(n_days=n_days, edge=edge, daily_vol=daily_vol, seed=seed)
    mat = st.sweep(prices, n_trials=n_trials, cost_bps=0.0, seed=seed)
    truth = dict(truth)
    truth["n_trials"] = mat.shape[1]
    return mat, truth


# ---------------------------------------------------------------------------
# Real tape — a single broad index via the shared cache, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str, ticker: str) -> str:
    return os.path.join(cache_dir, f"overfit_{ticker}.parquet")


def load_real(
    ticker: str = "SPY",
    start: str = "1995-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.Series:
    """Real daily total-return closes for one broad index (default SPY).

    Cache-first: reads the parquet under ``_cache/`` and never touches the network unless
    ``fetch=True``. On a cache miss with ``fetch=False`` it raises ``FileNotFoundError``
    (the offline contract) so the reproducible core and the tests stay deterministic and
    network-free.

    A single broad index is used only as a *worked example* of in-sample/out-of-sample
    splitting — there is no cross-sectional selection here, so no survivorship guard is
    needed. The series is total-return (auto-adjusted) closes; the partial final month/day
    handling is left to the caller's resampling (we keep daily resolution here).

    Returns a daily close Series.
    """
    cpath = _cache_path(cache_dir, ticker)
    if not fetch:
        if not os.path.exists(cpath):
            raise FileNotFoundError(
                f"No cached series at {cpath}. "
                f"Call load_real(ticker={ticker!r}, fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(cpath)
        return df["close"]

    px = _fetch_prices(ticker, start)
    os.makedirs(cache_dir, exist_ok=True)
    px.to_frame("close").to_parquet(cpath)
    return px


def _fetch_prices(ticker: str, start: str) -> pd.Series:
    """Daily adjusted closes via the shared loader, falling back to yfinance (network)."""
    try:
        from quantlab import data as qld  # shared loader, if importable

        bars = qld.load(ticker, start=start, mode="total_return")
        s = bars["close"] if "close" in bars else bars.iloc[:, 0]
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        return s.dropna()
    except Exception:
        pass

    import yfinance as yf

    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"no price data returned for {ticker}")
    px = raw["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    if getattr(px.index, "tz", None) is not None:
        px.index = px.index.tz_localize(None)
    px.name = "close"
    return px.dropna()


def fingerprint(series: pd.Series) -> str:
    """A short content fingerprint of a price/return series, for the as-of stamp."""
    arr = np.ascontiguousarray(series.to_numpy(dtype=float).ravel())
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
