"""Data layer for Study 728 — "people drink more beer in recessions" (beer defensiveness).

Three sources, all offline-friendly:

* **Tradable equity names (yfinance).** The two listed beer pure-plays a public investor
  can actually buy: **Molson Coors (``TAP``)** — a large-cap mass-market brewer, the closest
  thing to a "consumer-staple beer" stock — and **Boston Beer (``SAM``)** — the craft
  brewer behind Samuel Adams / Truly. Benchmarked against **``SPY``**. Month-end Adj Close
  (total-return-ish: splits + dividends folded in by ``auto_adjust``), cached under
  ``_cache/`` so the notebooks run offline; on a cache miss *with* network we fetch via
  yfinance, otherwise we fall back to the frozen headline numbers.

* **NBER recession windows (hardcoded, cited).** The US business-cycle peaks and troughs
  are published by the **NBER Business Cycle Dating Committee** — a small, fixed, *dated*
  set, not a live feed. The three recessions inside the SAM/SPY sample are the 2001
  dot-com bust (2001-03 → 2001-11), the 2008 Global Financial Crisis (2007-12 → 2009-06)
  and the 2020 COVID contraction (2020-02 → 2020-04). These are **facts on the record**,
  cited in ``docs/references.md``. The load-bearing caveat: NBER *announces* a recession
  6–18 months **after** it begins, so knowing "we are in a recession" in real time is a
  look-ahead — which is what kills the tradability (see :mod:`beer_recession.strategy`).

* **Synthetic positive control.** A deterministic, fixed-seed generator of a stock with a
  *planted* asymmetric beta (low on the way down, high on the way up) — a genuinely
  defensive name — used to prove the bull/bear-beta engine recovers what it plants. Runs
  with no network.

Pure numpy/pandas + stdlib; the offline core never touches the network.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "_cache")

# Tradable beer names + benchmark.
BEER = {
    "TAP": "Molson Coors (NYSE) — large-cap mass-market brewer",
    "SAM": "Boston Beer (NYSE) — Samuel Adams / Truly craft brewer",
}
BENCH = "SPY"
TICKERS = list(BEER) + [BENCH]


# --------------------------------------------------------------------------- #
# NBER recession windows — hardcoded, CITED (a dated set of facts, not a feed)
# --------------------------------------------------------------------------- #
# US business-cycle contractions per the NBER Business Cycle Dating Committee
# (peak month -> trough month, inclusive). Only the three inside the SAM sample.
# Source: https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
NBER_RECESSIONS = [
    ("2001 dot-com", "2001-03-01", "2001-11-01"),
    ("2008 GFC", "2007-12-01", "2009-06-01"),
    ("2020 COVID", "2020-02-01", "2020-04-01"),
]
# The real-world announcement lag: the NBER dated the 2008 recession's start (Dec-2007)
# only in Dec-2008, and the 2020 recession's trough (Apr-2020) only in Jul-2021. You never
# know you are in a recession in real time — the tradability caveat, in one number.
NBER_ANNOUNCE_LAG_MONTHS = 12


def recession_months() -> pd.DatetimeIndex:
    """Month-start ``DatetimeIndex`` of every NBER recession month in the sample."""
    out = pd.DatetimeIndex([])
    for _, a, b in NBER_RECESSIONS:
        out = out.append(pd.date_range(a, b, freq="MS"))
    return out.sort_values()


def recession_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask over ``index`` selecting the (year, month) pairs inside a recession."""
    rec = {(d.year, d.month) for d in recession_months()}
    return np.array([(d.year, d.month) in rec for d in index])


# --------------------------------------------------------------------------- #
# Tradable equity names via yfinance (cached, offline-friendly)
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE, f"{ticker.replace('.', '_')}_monthly.csv")


def have_prices() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in TICKERS)


def fetch_prices(start: str = "1995-01-01", end: str = "2026-07-01") -> dict[str, pd.Series]:
    """Fetch month-end Adj Close levels for the beer names + SPY.

    Tries the local cache first; on a miss, fetches via yfinance and writes the cache.
    Returns ``{ticker: monthly Adj-Close Series}``. Network only on a cache miss.
    """
    os.makedirs(CACHE, exist_ok=True)
    out: dict[str, pd.Series] = {}
    need_fetch = [t for t in TICKERS if not os.path.exists(_cache_path(t))]
    fetched = {}
    if need_fetch:
        import yfinance as yf  # imported lazily so the offline core never needs it
        raw = yf.download(need_fetch, start=start, end=end, interval="1mo",
                          auto_adjust=True, progress=False)
        close = raw["Close"] if "Close" in raw else raw
        if isinstance(close, pd.Series):
            close = close.to_frame(need_fetch[0])
        for t in need_fetch:
            s = close[t].dropna()
            s.to_csv(_cache_path(t))
            fetched[t] = s
    for t in TICKERS:
        if t in fetched:
            out[t] = fetched[t]
        else:
            df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True)
            out[t] = df.iloc[:, 0].dropna()
    return out


def load_prices() -> dict[str, pd.Series]:
    """Cached month-end Adj-Close levels for the beer names + SPY (no network if cached)."""
    out = {}
    for t in TICKERS:
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True)
        out[t] = df.iloc[:, 0].dropna()
    return out


# --------------------------------------------------------------------------- #
# Synthetic positive control (deterministic, fixed seed, no network)
# --------------------------------------------------------------------------- #
def synthetic_defensive(n_months: int = 360, beta_down: float = 0.5,
                        beta_up: float = 1.0, mkt_mu: float = 0.007,
                        mkt_sigma: float = 0.045, idio_sigma: float = 0.03,
                        seed: int = 728) -> tuple[pd.Series, pd.Series]:
    """A deterministic (market, stock) return pair with a *planted asymmetric beta*.

    The stock loads ``beta_down`` on the market in down months and ``beta_up`` in up
    months — i.e. a genuinely defensive name (``beta_down`` < ``beta_up`` < 1 typically).
    Used as the positive control: :func:`strategy.bull_bear_beta` must recover a
    down-beta below the up-beta. Returns ``(mkt_returns, stock_returns)`` monthly Series.
    """
    rng = np.random.default_rng(seed)
    mkt = rng.normal(mkt_mu, mkt_sigma, n_months)
    beta = np.where(mkt < 0, beta_down, beta_up)
    stock = beta * mkt + rng.normal(0.0, idio_sigma, n_months)
    idx = pd.date_range("1996-01-31", periods=n_months, freq="ME")
    return (pd.Series(mkt, index=idx, name="synth_mkt"),
            pd.Series(stock, index=idx, name="synth_stock"))
