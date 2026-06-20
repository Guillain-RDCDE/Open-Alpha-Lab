"""Data layer for Study 348 (Curve-Fitting).

The curve-fitting demo runs on a single price series: split it into an *in-sample* slice
(where we tune the moving-average crossover) and an *out-of-sample* slice (where we judge
the winner). So the data layer supplies one thing — a daily price/return series — in two
tapes:

- ``synthetic_series`` — a *deterministic, offline* generator with a single knob,
  ``trend_strength``. At ``trend_strength = 0`` the series is a pure geometric random walk:
  there is **no** crossover rule that genuinely works, so any "best" pair found by a grid
  search is fitting noise, and its edge must evaporate out of sample (the **null**). At
  ``trend_strength > 0`` the series carries a slow, persistent trend component that a
  *specific* band of crossover lengths can genuinely harvest — a real signal that should
  survive the in-sample → out-of-sample handoff (the **positive control**). One generator,
  both the trap and the proof the harness can tell them apart.

- ``load_real`` — a real daily total-return series (default SPY) via the shared
  ``quantlab.data`` loader with a **cache-first** parquet and graceful offline fallback. On
  a cache miss with ``fetch=False`` it raises ``FileNotFoundError`` (the offline contract),
  so the reproducible core and the test-suite stay deterministic and network-free.

No look-ahead is baked in here: the crossover signal known at the close of day *t* earns
the return of day *t+1* — the single execution lag lives in ``strategy.py`` (applied once,
documented there). This module only produces returns.
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
def synthetic_series(
    n_days: int = 4000,
    trend_strength: float = 0.0,
    trend_halflife: int = 60,
    daily_vol: float = 0.011,
    drift: float = 0.0002,
    seed: int = 348,
) -> tuple[pd.Series, dict]:
    """A reproducible daily *price* series with a tunable, harvestable trend component.

    The log return on day ``t`` is::

        r_t = drift + trend_strength * s_t + eps_t

    where ``eps_t`` is i.i.d. Gaussian noise (vol ``daily_vol``) and ``s_t`` is a smooth,
    mean-zero, persistent latent state — an AR(1) on the noise with the given half-life,
    standardised so ``trend_strength`` is in the same units as a daily return. A moving-
    average crossover is a trend follower, so:

    - ``trend_strength = 0`` → ``r_t`` is a pure (drifted) random walk. **No** crossover
      rule has a real edge; a grid search over windows finds the pair that best fit *this
      realisation's* noise, and that pair carries nothing out of sample. This is the null.
    - ``trend_strength > 0`` → a slow persistent trend rides under the noise. A band of
      crossover lengths can genuinely lock onto it, so the in-sample winner keeps working
      out of sample (decayed, but real). This is the positive control.

    Returns ``(prices, truth)`` where ``prices`` is a daily price ``pd.Series`` (starting
    at 100) and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)

    eps = rng.normal(0.0, daily_vol, n_days)

    # A persistent latent trend state: AR(1) driven by its own innovations, half-life set
    # by trend_halflife, then standardised to unit std so trend_strength is interpretable.
    phi = 0.5 ** (1.0 / max(trend_halflife, 1))
    innov = rng.normal(0.0, 1.0, n_days)
    s = np.empty(n_days)
    s[0] = innov[0]
    for t in range(1, n_days):
        s[t] = phi * s[t - 1] + np.sqrt(1.0 - phi * phi) * innov[t]
    s = (s - s.mean()) / (s.std() + 1e-12)

    log_ret = drift + trend_strength * s + eps
    log_price = np.cumsum(log_ret)
    price = 100.0 * np.exp(log_price)

    # Decorative business-day calendar via a SHORT span (never date_range(periods=BIG,
    # freq="M") for long spans — that overflows ns-Timestamps; here n_days of B is safe).
    idx = pd.bdate_range("2000-01-03", periods=n_days, name="date")
    prices = pd.Series(price, index=idx, name="price")

    truth = {
        "n_days": n_days,
        "trend_strength": trend_strength,
        "trend_halflife": trend_halflife,
        "daily_vol": daily_vol,
        "drift": drift,
        "seed": seed,
    }
    return prices, truth


# ---------------------------------------------------------------------------
# Real tape — a single daily series via the shared cache, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str, ticker: str) -> str:
    return os.path.join(cache_dir, f"curvefit_{ticker.upper()}.parquet")


def load_real(
    ticker: str = "SPY",
    start: str = "1993-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.Series:
    """Real daily **total-return** price series for one ticker (default SPY).

    Cache-first: reads the parquet under ``_cache/`` and never touches the network unless
    ``fetch=True``. On a cache miss with ``fetch=False`` it raises ``FileNotFoundError``
    (the offline contract) so the reproducible core and tests stay deterministic.

    Total return (``auto_adjust=True`` / shared-loader ``total_return`` mode) is the only
    fair tape for a long/flat timing rule: a price-only series mislabels reinvested
    dividends as missing return. The in-progress final day is left as-is (a daily series,
    not a resampled bar), but ``strategy`` drops the unusable head where the long MA is NaN.

    Returns a daily price ``pd.Series`` named ``price``.
    """
    cpath = _cache_path(cache_dir, ticker)
    if not fetch:
        if not os.path.exists(cpath):
            raise FileNotFoundError(
                f"No cached series at {cpath}. "
                f"Call load_real(ticker={ticker!r}, fetch=True) once to populate the cache."
            )
        s = pd.read_parquet(cpath)["price"]
        s.index.name = "date"
        return s

    px = _fetch_prices(ticker, start)
    os.makedirs(cache_dir, exist_ok=True)
    px.to_frame("price").to_parquet(cpath)
    return px


def _fetch_prices(ticker: str, start: str) -> pd.Series:
    """Daily total-return closes for one ticker via the shared loader (network on call)."""
    try:
        from quantlab import data as qld  # shared loader, if importable

        bars = qld.load(ticker, start=start, mode="total_return")
        s = bars["close"] if "close" in bars else bars.iloc[:, 0]
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        s = s.dropna()
        s.name = "price"
        return s
    except Exception:
        pass

    import yfinance as yf

    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"no price data returned for {ticker}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close = close.dropna()
    close.name = "price"
    return close


def fingerprint(prices: pd.Series) -> str:
    """A short content fingerprint of the price series, for the as-of stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float).ravel())
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
