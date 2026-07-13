"""Data layer for Study 715 — "Vinyl is back — a trend to trade?" (the vinyl revival).

Three sources, all offline-friendly:

* **The vinyl-revenue index (hardcoded, cited, approximate).** The trend everyone quotes
  is the **RIAA year-end U.S. recorded-music statistics** — the estimated *retail* value
  of vinyl LP/EP sales. The RIAA database is a public web app, **not a freely-pullable
  API**, so we hardcode a small **annual** series reconstructed from RIAA year-end
  releases and clearly labelled **approximate**. It is anchored on figures the desk could
  cite: vinyl bottomed near **$89M in 2010**, crossed **$479M in 2019**, **overtook CDs in
  revenue in 2020** ($643M, the first time since the 1980s), broke **$1.0B in 2021**,
  reached **$1.2B in 2022** (the year vinyl outsold CDs in *units*), **$1.35B in 2023**
  and **~$1.4B in 2024** — 18 straight years of growth. Sources are listed in
  ``docs/references.md``. This is a PROXY for the trend, never a tradable price.

  We also hardcode vinyl's **share of total U.S. recorded-music revenue** — the fact the
  pitch omits: even at the 2024 peak vinyl was only ~**7%** of a business ~**84%** driven
  by streaming. The revival is real; it is also a rounding error next to streaming.

* **Tradable equity proxies (yfinance).** The only listed ways to "own the vinyl trade":
  **Warner Music Group (WMG)** and **Universal Music Group (UMG.AS)** — the two majors that
  press and own the catalogue vinyl sells — and **Spotify (SPOT)** — the streaming pure-play
  that vinyl is supposedly a *reaction against*. These are *labelled proxies*: a major
  label's or a streamer's equity is not a stack of records, but they are the only things a
  public investor can actually buy that are *about* the format war. Benchmarked against
  **SPY**. Cached under ``_cache/`` so the notebooks run offline; on a cache miss with
  network we fetch via yfinance, otherwise we fall back to the frozen headline numbers.

* **Synthetic positive control.** A deterministic fixed-seed generator of a "revival" price
  path with a *known* planted CAGR and Sharpe, used to prove the inference engine recovers
  what it plants. Runs with no network.

Pure numpy/pandas + stdlib; the offline core never touches the network.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "_cache")

# Tradable equity proxies + benchmark. Labelled as PROXIES everywhere.
PROXIES = {
    "WMG": "Warner Music Group (NASDAQ) — major label / catalogue owner",
    "SPOT": "Spotify (NYSE) — streaming pure-play (the thing vinyl reacts against)",
    "UMG.AS": "Universal Music Group (Euronext AMS) — the largest major label",
}
BENCH = "SPY"
TICKERS = list(PROXIES) + [BENCH]


# --------------------------------------------------------------------------- #
# The vinyl-revenue index — hardcoded, cited, APPROXIMATE (a proxy, not a feed)
# --------------------------------------------------------------------------- #
# RIAA year-end estimated *retail* value of U.S. vinyl LP/EP sales, in $millions.
# Reconstructed from public RIAA year-end statistics (see docs/references.md). The shape
# is the load-bearing fact: an 18-year climb off a 2010 low, overtaking CDs in revenue in
# 2020 and in units in 2022, past $1B in 2021 and ~$1.4B by 2024.
_VINYL_REVENUE_M = {
    2010: 89.0,
    2011: 119.0,
    2012: 146.0,
    2013: 211.0,
    2014: 321.0,
    2015: 416.0,
    2016: 430.0,
    2017: 395.0,
    2018: 419.0,
    2019: 479.0,
    2020: 643.0,   # vinyl overtakes CDs in REVENUE (first time since the 1980s)
    2021: 1003.0,  # breaks $1.0B
    2022: 1225.0,  # vinyl outsells CDs in UNITS
    2023: 1353.0,
    2024: 1400.0,
}
# Vinyl's share of TOTAL U.S. recorded-music revenue (the fact the pitch omits).
# From RIAA year-end format mix; approximate. Streaming is ~84% of the 2024 total.
_VINYL_SHARE_PCT = {
    2010: 1.2,
    2014: 4.7,
    2019: 4.3,
    2020: 5.2,
    2021: 5.9,
    2022: 7.0,
    2023: 7.0,
    2024: 7.0,
}
_STREAMING_SHARE_2024 = 84.0  # streaming's share of the 2024 total, for context.


def load_vinyl_index(base: float = 100.0, base_year: int = 2010) -> pd.Series:
    """Year-end vinyl-revenue level, rebased to ``base`` at ``base_year``.

    Returns a ``pd.Series`` indexed by year-end ``Timestamp`` (default base 100 @ 2010).
    LABELLED A PROXY: reconstructed from public RIAA year-end statistics, not a live feed,
    and it is *revenue* — a measure of the trend, **not** a price you can buy.
    """
    ref = _VINYL_REVENUE_M[base_year]
    idx = pd.to_datetime([f"{y}-12-31" for y in _VINYL_REVENUE_M])
    vals = [v / ref * base for v in _VINYL_REVENUE_M.values()]
    return pd.Series(vals, index=idx, name="vinyl_index")


def vinyl_revenue_musd() -> pd.Series:
    """Raw RIAA vinyl revenue in $millions, year-end (the cited anchors)."""
    idx = pd.to_datetime([f"{y}-12-31" for y in _VINYL_REVENUE_M])
    return pd.Series(list(_VINYL_REVENUE_M.values()), index=idx, name="vinyl_musd")


def vinyl_annual_returns() -> pd.Series:
    """Year-over-year % change of the hardcoded vinyl-revenue series (the public anchors)."""
    s = vinyl_revenue_musd()
    return (s / s.shift(1) - 1.0).dropna() * 100.0


def vinyl_share() -> pd.Series:
    """Vinyl's share (%) of total U.S. recorded-music revenue — the omitted context."""
    idx = pd.to_datetime([f"{y}-12-31" for y in _VINYL_SHARE_PCT])
    return pd.Series(list(_VINYL_SHARE_PCT.values()), index=idx, name="vinyl_share_pct")


def streaming_share_2024() -> float:
    """Streaming's share (%) of the 2024 U.S. recorded-music total."""
    return _STREAMING_SHARE_2024


# --------------------------------------------------------------------------- #
# Tradable equity proxies via yfinance (cached, offline-friendly)
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE, f"{ticker.replace('.', '_')}_monthly.csv")


def have_proxies() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in TICKERS)


def fetch_proxies(start: str = "2018-01-01", end: str = "2026-01-01") -> dict[str, pd.Series]:
    """Fetch month-end *total-return-ish* (Adj Close) levels for the proxies + SPY.

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


def load_proxies() -> dict[str, pd.Series]:
    """Cached month-end Adj-Close levels for the proxies + SPY (no network if cached)."""
    out = {}
    for t in TICKERS:
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True)
        out[t] = df.iloc[:, 0].dropna()
    return out


# --------------------------------------------------------------------------- #
# Synthetic positive control (deterministic, fixed seed, no network)
# --------------------------------------------------------------------------- #
def synthetic_revival(n_months: int = 120, cagr: float = 0.20,
                      sigma_month: float = 0.04, start: float = 100.0,
                      seed: int = 715) -> pd.Series:
    """A deterministic 'revival' monthly level path with a planted positive drift.

    A planted log-drift of ``cagr`` plus fixed-seed Gaussian noise of ``sigma_month``.
    Used as the positive control: :func:`strategy.summarize` must recover the planted
    drift's sign and a Sharpe close to the noiseless target. Returns a month-end
    ``pd.Series``.
    """
    rng = np.random.default_rng(seed)
    mu = (1 + cagr) ** (1 / 12) - 1
    shocks = rng.normal(0.0, sigma_month, size=n_months)
    logret = np.log1p(mu) + shocks
    level = start * np.exp(np.cumsum(logret))
    idx = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    return pd.Series(level, index=idx, name="synthetic")
