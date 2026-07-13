"""Data layer for Study 729 — "the ramen index" (instant-noodle sales as a downturn tell).

Three sources, all offline-friendly:

* **The "ramen index" (hardcoded, cited, APPROXIMATE).** The folklore's actual quantity is
  *instant-noodle demand* — and the authoritative figure, world **retail servings per year**,
  is published by the **World Instant Noodles Association (WINA)**, not a live API. So we
  hardcode WINA's annual global-demand series (billions of servings, ~2005–2024),
  reconstructed from WINA's public tables and clearly labelled **approximate**. The anchors a
  reader can check: demand rose through the 2008 GFC and *spiked* in 2020 (the folklore's two
  exhibits), but it also **fell in 2014–2016** (a China pullback in *no* recession) and grows
  in most expansions — i.e. it is dominated by secular Asian demand, not the US cycle. Sources
  are in ``docs/references.md``. This is a PROXY for "the ramen index," never a live feed.

* **Tradable noodle equities (yfinance).** The two listed ways to own the instant-noodle
  trade: **Nissin Foods Holdings (``2897.T``, Tokyo)** — the world's largest instant-noodle
  maker (Cup Noodle, Top Ramen, Chicken Ramen) — and **Toyo Suisan (``2875.T``, Tokyo)** —
  the Maruchan maker (the best-selling instant noodle in the US). Benchmarked against the
  **Nikkei 225 (``^N225``)**, the market they actually trade in. Month-end Adj Close
  (total-return-ish: splits + dividends folded in by ``auto_adjust``), cached under
  ``_cache/`` so the notebooks run offline; on a cache miss *with* network we fetch via
  yfinance, otherwise we fall back to the frozen headline numbers.

* **NBER recession windows (hardcoded, cited).** US business-cycle contractions per the
  **NBER Business Cycle Dating Committee** — a small, fixed, *dated* set of facts, not a
  feed. The three inside the noodle-stock sample are the 2001 dot-com bust (2001-03 →
  2001-11), the 2008 GFC (2007-12 → 2009-06) and the 2020 COVID contraction (2020-02 →
  2020-04). The load-bearing caveat, twice over: NBER *announces* a recession 6–18 months
  **after** it begins, and WINA publishes a year's demand only in mid-*next*-year — so a
  "ramen tells you a recession is coming" rule is a look-ahead on **both** legs.

* **Synthetic positive control.** A deterministic, fixed-seed generator of an index that
  *genuinely leads* market downturns by a known lag, used to prove the lead-lag engine
  recovers a planted lead. Runs with no network.

Pure numpy/pandas + stdlib; the offline core never touches the network.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "_cache")

# Tradable noodle names + benchmark.
NOODLE = {
    "2897.T": "Nissin Foods Holdings (Tokyo) — Cup Noodle / Top Ramen maker",
    "2875.T": "Toyo Suisan (Tokyo) — Maruchan instant-noodle maker",
}
BENCH = "^N225"
TICKERS = list(NOODLE) + [BENCH]


# --------------------------------------------------------------------------- #
# The "ramen index" — hardcoded, cited, APPROXIMATE (a proxy, not a live feed)
# --------------------------------------------------------------------------- #
# World instant-noodle retail demand, BILLIONS of servings per calendar year, per the
# World Instant Noodles Association (WINA) public global-demand tables (approximate; see
# docs/references.md). The shape is the load-bearing fact: broadly secular growth, a rise
# through the 2008 GFC and a jump in 2020 (the folklore's exhibits), but a 2014–2016
# *decline* in no recession at all. Year-end labelled by Dec 31.
_WINA_DEMAND_BN = {
    2005: 85.7,
    2006: 91.7,
    2007: 97.7,
    2008: 93.6,
    2009: 90.0,
    2010: 95.4,
    2011: 98.5,
    2012: 101.4,
    2013: 105.6,
    2014: 102.7,
    2015: 97.7,
    2016: 97.5,
    2017: 100.1,
    2018: 103.6,
    2019: 106.4,
    2020: 116.6,   # COVID pantry-loading spike
    2021: 118.2,
    2022: 121.2,
    2023: 120.2,
    2024: 122.0,
}
# WINA publishes a calendar year's demand only in mid-*next* year (a data-release lag on
# top of the fact that the reading itself is annual) — the tradability caveat, in one number.
WINA_RELEASE_LAG_MONTHS = 6


def load_ramen_index() -> pd.Series:
    """Annual world instant-noodle demand (billions of servings), the ``ramen index``.

    Returns a ``pd.Series`` indexed by year-end ``Timestamp``. LABELLED A PROXY:
    reconstructed from WINA's public tables (approximate), not a live data feed.
    """
    idx = pd.to_datetime([f"{y}-12-31" for y in _WINA_DEMAND_BN])
    return pd.Series(list(_WINA_DEMAND_BN.values()), index=idx, name="ramen_index")


def ramen_growth() -> pd.Series:
    """Year-over-year % growth of the (approximate, cited) ramen index."""
    s = load_ramen_index()
    return (s / s.shift(1) - 1.0).dropna() * 100.0


# --------------------------------------------------------------------------- #
# NBER recession windows — hardcoded, CITED (a dated set of facts, not a feed)
# --------------------------------------------------------------------------- #
# US business-cycle contractions per the NBER Business Cycle Dating Committee
# (peak month -> trough month, inclusive). Only the three inside the noodle-stock sample.
# Source: https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
NBER_RECESSIONS = [
    ("2001 dot-com", "2001-03-01", "2001-11-01"),
    ("2008 GFC", "2007-12-01", "2009-06-01"),
    ("2020 COVID", "2020-02-01", "2020-04-01"),
]
# The real-world announcement lag: NBER dated the 2008 recession's start (Dec-2007) only in
# Dec-2008, and the 2020 trough (Apr-2020) only in Jul-2021. You never know you are in a
# recession in real time — the other half of the tradability caveat.
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


def recession_years() -> set[int]:
    """The set of calendar years that contain any NBER recession month (for the macro test)."""
    return {d.year for d in recession_months()}


# --------------------------------------------------------------------------- #
# Tradable noodle equities via yfinance (cached, offline-friendly)
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE, f"{ticker.replace('.', '_').replace('^', '')}_monthly.csv")


def have_prices() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in TICKERS)


def fetch_prices(start: str = "2000-01-01", end: str = "2026-07-01") -> dict[str, pd.Series]:
    """Fetch month-end Adj Close levels for the noodle names + the Nikkei.

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
    """Cached month-end Adj-Close levels for the noodle names + Nikkei (no network if cached)."""
    out = {}
    for t in TICKERS:
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True)
        out[t] = df.iloc[:, 0].dropna()
    return out


# --------------------------------------------------------------------------- #
# Synthetic positive control (deterministic, fixed seed, no network)
# --------------------------------------------------------------------------- #
def synthetic_leading_index(n_years: int = 40, lead: int = 1, strength: float = 1.0,
                            noise: float = 0.4, seed: int = 729
                            ) -> tuple[pd.Series, pd.Series]:
    """A deterministic (index_growth, forward_market_return) pair with a *planted lead*.

    The ``index_growth`` at year *t* is engineered to be high right **before** a bad market
    year at *t+``lead``* — i.e. the index genuinely *leads* downturns. Used as the positive
    control: :func:`strategy.lead_lag_corr` must recover a negative correlation peaking at the
    planted ``lead``. Returns ``(index_growth, market_return)`` annual Series.
    """
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.06, 0.18, n_years)                 # annual market returns
    # index growth at t is (minus) the market return `lead` years ahead, plus noise:
    idx_growth = np.full(n_years, np.nan)
    for t in range(n_years - lead):
        idx_growth[t] = -strength * mkt[t + lead] + rng.normal(0.0, noise * 0.18)
    # fill the tail (no future known) with pure noise so lengths match
    for t in range(n_years - lead, n_years):
        idx_growth[t] = rng.normal(0.0, noise * 0.18)
    years = pd.date_range("1985-12-31", periods=n_years, freq="YE")
    return (pd.Series(idx_growth, index=years, name="synth_index_growth"),
            pd.Series(mkt, index=years, name="synth_mkt_ret"))


def synthetic_defensive(n_months: int = 300, beta_down: float = 0.5,
                        beta_up: float = 1.0, mkt_mu: float = 0.006,
                        mkt_sigma: float = 0.05, idio_sigma: float = 0.03,
                        seed: int = 7290) -> tuple[pd.Series, pd.Series]:
    """A deterministic (market, stock) return pair with a *planted asymmetric beta*.

    The stock loads ``beta_down`` in down months and ``beta_up`` in up months — a genuinely
    defensive name. Used to prove :func:`strategy.bull_bear_beta` recovers a planted defensive
    tilt. Returns ``(mkt_returns, stock_returns)`` monthly Series.
    """
    rng = np.random.default_rng(seed)
    mkt = rng.normal(mkt_mu, mkt_sigma, n_months)
    beta = np.where(mkt < 0, beta_down, beta_up)
    stock = beta * mkt + rng.normal(0.0, idio_sigma, n_months)
    idx = pd.date_range("2001-01-31", periods=n_months, freq="ME")
    return (pd.Series(mkt, index=idx, name="synth_mkt"),
            pd.Series(stock, index=idx, name="synth_stock"))
