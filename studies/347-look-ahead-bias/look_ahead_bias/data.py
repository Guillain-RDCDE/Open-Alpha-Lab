"""Data layer for Study 347 (Look-Ahead-Bias).

The study is a *single-asset time-series* demonstration: a daily price path, a
signal built from it, and the question of *which bar's return the signal is
allowed to earn*. So the data layer supplies one thing — a daily close series —
in two flavours, plus a real cache-first loader.

Two synthetic tapes, one shape (a daily close ``Series``):

- ``synthetic_prices(edge=0.0, …)`` — the **null**: a pure geometric random walk.
  There is *no* predictable structure, so any honest (lagged) strategy must score
  ~0. A look-ahead bug, by contrast, will manufacture a large Sharpe here — that
  gap is the whole study. This is the negative control.

- ``synthetic_prices(edge>0.0, …)`` — the **positive control**: the same random
  walk plus a *genuinely lag-tradable* mean-reversion component. Tomorrow's
  return is pulled back toward the mean by ``edge`` times today's (already known)
  z-score, so a one-bar-lagged mean-reversion rule earns a real, positive Sharpe.
  The harness must bank this with the lagged convention (a pipeline that can't
  find a planted, *fairly tradable* edge proves nothing) — and the peek inflates
  it further. This is the harness's positive control.

The "edge" is constructed so it is realisable with the **correct one-bar lag**:
the mean-reversion force on day ``t+1`` depends only on the z-score observable at
the close of day ``t``. No look-ahead is baked into the generator; the only
look-ahead in the study comes from the *execution convention* in ``strategy.py``.

- ``load_real`` — a real daily close series (default ``SPY``) via the shared
  ``quantlab.data`` loader, cache-first parquet, graceful offline fallback. The
  real tape is used only to show the inflation persists on actual prices; the
  reproducible core and the tests run entirely on the deterministic synthetics.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252
DEFAULT_TICKER = "SPY"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_prices(
    n_days: int = 3000,
    edge: float = 0.0,
    lookback: int = 20,
    daily_vol: float = 0.011,
    drift: float = 0.0002,
    start_price: float = 100.0,
    seed: int = 347,
) -> tuple[pd.Series, dict]:
    """A reproducible daily close series with an optionally-planted, lag-tradable edge.

    The log-return on day ``t`` is::

        ret_t = drift  -  edge * z_{t-1}  +  eps_t

    where ``z_{t-1}`` is the (causal) rolling z-score of the *prior* day's close
    over ``lookback`` days, and ``eps_t`` is i.i.d. Gaussian noise. The
    mean-reversion pull (``-edge * z_{t-1}``) uses only information available at
    the close of ``t-1``, so a one-bar-lagged mean-reversion rule can *actually*
    harvest it — it is a fair, tradable edge, not a baked-in look-ahead.

    - ``edge = 0`` → a pure random walk: the **null**. No tradable structure.
    - ``edge > 0`` → a genuine mean-reversion premium a lagged rule can bank: the
      **positive control**.

    Returns ``(prices, truth)`` where ``prices`` is a daily close ``Series`` and
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, daily_vol, size=n_days)

    log_ret = np.empty(n_days)
    log_px = np.empty(n_days)
    log_px[0] = np.log(start_price)
    log_ret[0] = 0.0

    # Rolling window of recent log-returns to form a causal z-score of the close.
    # We z-score the *level deviation* via a rolling mean/std of log-prices, all
    # using strictly past information (closes up to and including t-1).
    px_hist: list[float] = [log_px[0]]
    for t in range(1, n_days):
        if len(px_hist) >= lookback:
            window = np.asarray(px_hist[-lookback:])
            mu = window.mean()
            sd = window.std(ddof=1)
            z_prev = (px_hist[-1] - mu) / sd if sd > 0 else 0.0
        else:
            z_prev = 0.0
        # Mean-reversion pull depends only on info known at the close of t-1.
        log_ret[t] = drift - edge * z_prev + eps[t]
        log_px[t] = log_px[t - 1] + log_ret[t]
        px_hist.append(log_px[t])

    # Decorative business-day labels via period_range → to_timestamp (NEVER
    # date_range(periods=BIG, freq=...) for long spans; see CI pitfalls). n_days
    # well under any ns-overflow horizon.
    idx = pd.bdate_range("2000-01-03", periods=n_days)
    prices = pd.Series(np.exp(log_px), index=pd.DatetimeIndex(idx, name="date"), name="close")

    truth = {
        "n_days": n_days,
        "edge": edge,
        "lookback": lookback,
        "daily_vol": daily_vol,
        "drift": drift,
        "seed": seed,
    }
    return prices, truth


# ---------------------------------------------------------------------------
# Real tape — daily closes via the shared cache, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str, ticker: str) -> str:
    return os.path.join(cache_dir, f"lab_prices_{ticker}.parquet")


def load_real(
    ticker: str = DEFAULT_TICKER,
    start: str = "2005-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.Series:
    """Real daily close series for ``ticker`` (default SPY), cache-first.

    Reads the parquet under ``_cache/`` and never touches the network unless
    ``fetch=True``. On a cache miss with ``fetch=False`` it raises
    ``FileNotFoundError`` (the offline contract) so the reproducible core and the
    test-suite stay deterministic and network-free.

    The series is **total-return adjusted** closes (the shared loader's
    ``mode="total_return"``); we label it as such. Only the close-to-close return
    matters for this study, and the look-ahead demonstration is adjustment-mode
    agnostic — the bug inflates either tape.

    Returns a daily close ``Series`` (the in-progress final bar is *not* dropped
    here; callers that stamp a run drop partial periods — see ``examples/verify``).
    """
    path = _cache_path(cache_dir, ticker)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices at {path}. "
                f"Call load_real(fetch=True) once to populate the cache."
            )
        s = pd.read_parquet(path)["close"]
    else:
        s = _fetch_prices(ticker, start)
        os.makedirs(cache_dir, exist_ok=True)
        s.to_frame("close").to_parquet(path)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s.dropna()


def _fetch_prices(ticker: str, start: str) -> pd.Series:
    """Daily total-return adjusted closes via the shared loader (network on call)."""
    try:
        from quantlab import data as qld  # shared loader, if importable

        bars = qld.load(ticker, start=start, mode="total_return")
        s = bars["close"] if "close" in bars else bars.iloc[:, 0]
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        return s.dropna().rename("close")
    except Exception:
        pass

    import yfinance as yf

    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"no price data returned for {ticker}")
    s = raw["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s.dropna().rename("close")


def fingerprint(prices: pd.Series) -> str:
    """A short content fingerprint of the price series, for the as-of stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float).ravel())
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
