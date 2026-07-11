"""Data layer for Study 678 — Random-Walk-Index (Poulos RWI).

Two ingredients, both offline-friendly once cached:

* **Real tape.** Daily total-return-adjusted OHLC for SPY plus a four-name liquid basket
  (QQQ, IWM, DIA, GLD), from yfinance (no key), cached as CSV under the study's own
  ``_cache/``. We fetch with ``auto_adjust=True`` so Open/High/Low/Close are all scaled
  together for splits and dividends — the Random Walk Index is a *range* statistic (a
  raw price move divided by a raw average true range), and an unadjusted split would
  otherwise print a fake giant range on the split day and poison the True Range/ATR
  window for weeks.

* **Synthetic world.** A deterministic, seeded two-state Markov regime-switch generator
  (a "trend" state and a "chop" state, same daily volatility in both, so ATR is roughly
  regime-invariant) with a TUNABLE planted edge (knob ``edge``): in the null world
  (``edge = 0``) both regimes carry the *same* expected drift, so an RWI-high flag
  (which fires on realized price displacement, not on the hidden regime label) carries
  no forward-looking information — the Welch machinery must NOT manufacture
  significance from it. With ``edge > 0`` the trend regime carries genuinely higher
  drift, so a detector that is actually picking up trend persistence must light up.

Pure numpy + pandas on the offline path. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]   # SPY is the headline; the rest is the basket
HEADLINE = "SPY"

START = "2005-01-03"       # QQQ/IWM/DIA/GLD all have clean daily history from here
AS_OF = "2026-06-30"       # last complete month at publication (2026-07-10)


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"rwi_{ticker.lower()}.csv")


def fetch(tickers: list[str] | None = None, start: str = "2004-06-01",
          end: str = "2026-07-01") -> None:
    """Download total-return-adjusted daily OHLC for each ticker; cache to CSV. Network; once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in (tickers or TICKERS):
        raw = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Open", "High", "Low", "Close"]].dropna(how="all")
        df.to_csv(_cache_path(t))


def have_real(tickers: list[str] | None = None) -> bool:
    return all(os.path.exists(_cache_path(t)) for t in (tickers or TICKERS))


def load_real(ticker: str, start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Cached (adjusted OHLC) frame for ``ticker``, sliced to [start, asof]."""
    df = pd.read_csv(_cache_path(ticker), index_col=0, parse_dates=True).sort_index()
    return df.loc[(df.index >= start) & (df.index <= asof)].copy()


def load_basket(tickers: list[str] | None = None, start: str = START,
                 asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    return {t: load_real(t, start=start, asof=asof) for t in (tickers or TICKERS)}


# --------------------------------------------------------------------------- #
# Synthetic world — two-state regime-switch generator, tunable planted edge
# --------------------------------------------------------------------------- #
def synthetic_world(edge: float = 0.0, seed: int = 678, n_days: int = 8000,
                     p_stay: float = 0.99, mu0: float = 0.0002, sigma: float = 0.010,
                     intraday_frac: float = 0.55) -> pd.DataFrame:
    """Deterministic two-regime OHLC generator with a TUNABLE planted trend-persistence edge.

    A hidden Markov state alternates between "trend" (1) and "chop" (0) with
    P(stay) = ``p_stay`` (mean regime length ~= 1/(1-p_stay) sessions, comparable to a
    multi-month market regime). Daily log-return drift is ``mu0 + edge`` in the trend
    state and ``mu0`` in the chop state; **both states share the same shock scale**
    ``sigma``, so a naive volatility read can't tell them apart — only sustained
    directional displacement can, which is exactly what the RWI is built to catch.
    ``edge = 0`` is the null world: the two regimes are statistically identical in
    expectation, so nothing about "trend detected" should predict the next day's
    return. Business-day index, span ~32 years at n_days=8000 — far below the
    ~250-year pandas ns-timestamp trap.

    High/Low are built from a small same-day noise band around Close (fraction
    ``intraday_frac`` of the day's |shock|, floored), independent of regime, so the ATR
    scale stays roughly constant across regimes and the RWI numerator (net displacement)
    is what carries the tunable edge.

    Returns a DataFrame with Open/High/Low/Close columns and a business-day index.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2001-01-02", periods=n_days)

    state = np.zeros(n_days, dtype=int)
    state[0] = rng.integers(0, 2)
    stay_draws = rng.random(n_days)
    for t in range(1, n_days):
        state[t] = state[t - 1] if stay_draws[t] < p_stay else 1 - state[t - 1]

    shocks = rng.normal(0.0, sigma, n_days)
    drift = np.where(state == 1, mu0 + edge, mu0)
    log_ret = drift + shocks

    close = 100.0 * np.exp(np.cumsum(log_ret))
    prev_close = np.roll(close, 1)
    prev_close[0] = 100.0

    band = np.abs(rng.normal(0.0, sigma, n_days)) * intraday_frac + 1e-6
    high = np.maximum(close, prev_close) * (1.0 + band)
    low = np.minimum(close, prev_close) * (1.0 - band)
    open_ = prev_close

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close}, index=idx
    )
