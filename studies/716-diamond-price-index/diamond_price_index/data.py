"""Data layer for Study 716 — "Short the diamond" (natural-diamond price collapse).

Three sources, all offline-friendly:

* **The natural-diamond price index (hardcoded, cited, approximate).** Real polished-
  diamond price indices (the Rapaport RapNet Price Index / RAPI 1ct, the IDEX Polished
  Diamond Price Index, the Paul Zimnisky Global Polished Diamond Price Index) are **not**
  freely API-available. So we hardcode a small **annual** series of the natural polished-
  diamond price *level*, reconstructed from *public* reporting and clearly labelled as
  **approximate**. It is anchored on figures the desk could cite: prices were soft in the
  2019 midstream liquidity crisis; recovered through the 2020–21 luxury boom; **peaked in
  early 2022**; then fell as lab-grown scaled — roughly **−18%** (2023) and **−11%** (2024)
  on the 1ct RAPI, a **~-30%** peak-to-trough round-trip from the early-2022 top, while
  lab-grown *wholesale* prices collapsed ~80–90%. Sources are listed in
  ``docs/references.md``. This is a PROXY for the real tape, never the tape.

* **Tradable equity proxies (yfinance).** Two listed ways to express "short the diamond":
  **Signet Jewelers (SIG)** — the largest US diamond *jeweler* (Kay/Zales/Jared), the
  demand side — and **Lucara Diamond (LUC.TO)** — a pure-play natural-diamond *miner*
  (the Karowe mine, Botswana), the supply side. These are *labelled proxies*: a jeweler's
  equity and a single-mine miner's equity are not the price of a polished stone, but they
  are the only listed things a public investor can actually trade. Benchmarked against
  **SPY**. Cached under ``_cache/`` so the notebooks run offline; on a cache miss with
  network we fetch via yfinance, otherwise we fall back to the frozen headline numbers.

* **Synthetic positive control.** A deterministic fixed-seed generator of a
  "collapse-and-stabilise" price path with a *known* planted (negative) drift, used to
  prove the inference engine recovers what it plants — here a **down**-trend, mirroring the
  thesis. Runs with no network.

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
    "SIG": "Signet Jewelers (NYSE) — Kay/Zales/Jared, the diamond jeweler (demand side)",
    "LUC.TO": "Lucara Diamond (TSX) — Karowe pure-play diamond miner (supply side)",
}
BENCH = "SPY"
TICKERS = list(PROXIES) + [BENCH]


# --------------------------------------------------------------------------- #
# The natural-diamond price index — hardcoded, cited, APPROXIMATE (a proxy)
# --------------------------------------------------------------------------- #
# Annual natural *polished-diamond price level*, normalised to 100 at end-2018,
# reconstructed from public reporting (see docs/references.md). The shape is the
# load-bearing fact: a soft 2019 (midstream liquidity crisis), a 2020-21 luxury-boom
# recovery, an early-2022 peak, then a lab-grown-driven decline into 2024-25.
# Year-end levels (Dec each year):
#   2018 -> 100   (base)
#   2019 ->  94   (2019 midstream crisis; De Beers price cuts, oversupply)
#   2020 ->  96   (COVID dip in H1, H2 recovery -> ~flat on the year)
#   2021 -> 118   (luxury/COVID boom; strong appreciation)
#   2022 -> 122   (PEAKED early 2022 ~132, gave back H2 -> ~122 year-end)
#   2023 -> 104   (lab-grown bites; De Beers repeated rough cuts ~ -15% on the year)
#   2024 ->  92   (continued decline ~ -11%)
#   2025 ->  90   (stabilisation at a lower level)
_INDEX_YEAR_END = {
    2018: 100.0,
    2019: 94.0,
    2020: 96.0,
    2021: 118.0,
    2022: 122.0,
    2023: 104.0,
    2024: 92.0,
    2025: 90.0,
}
# The blow-off top the public reports anchor on: early 2022.
_PEAK_DATE = "2022-02-28"
_PEAK_LEVEL = 132.0  # ~ +12% above the Dec-2021 118 level; the early-2022 high.


def load_price_index() -> pd.Series:
    """Year-end levels of the (approximate, cited) natural polished-diamond price index.

    Returns a ``pd.Series`` indexed by year-end ``Timestamp``, base 100 at end-2018.
    LABELLED A PROXY: reconstructed from public reporting, not a live data feed.
    """
    idx = pd.to_datetime([f"{y}-12-31" for y in _INDEX_YEAR_END])
    return pd.Series(list(_INDEX_YEAR_END.values()), index=idx, name="diamond_price_index")


def price_peak() -> tuple[pd.Timestamp, float]:
    """The reported blow-off top (date, level) — early 2022."""
    return pd.Timestamp(_PEAK_DATE), _PEAK_LEVEL


def price_annual_returns() -> pd.Series:
    """Year-over-year % change of the hardcoded price index (the public anchors)."""
    s = load_price_index()
    return (s / s.shift(1) - 1.0).dropna() * 100.0


# --------------------------------------------------------------------------- #
# Tradable equity proxies via yfinance (cached, offline-friendly)
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE, f"{ticker.replace('.', '_')}_monthly.csv")


def have_proxies() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in TICKERS)


def fetch_proxies(start: str = "2018-12-01", end: str = "2025-12-31") -> dict[str, pd.Series]:
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
def synthetic_collapse(n_months: int = 84, top_month: int = 38,
                       boom_cagr: float = 0.20, bust_cagr: float = -0.22,
                       sigma_month: float = 0.045, start: float = 100.0,
                       seed: int = 716) -> pd.Series:
    """A deterministic 'peak-and-collapse' monthly level path (planted DOWN drift).

    A planted log-drift that ramps up modestly to ``top_month`` at ``boom_cagr`` then
    rolls over at ``bust_cagr`` (the collapse the thesis predicts), plus fixed-seed
    Gaussian noise of ``sigma_month``. Used as the positive control: the engine must
    recover the planted **negative** post-peak drift's sign — a machinery proof that a
    null on the real tape is a real null. Returns a month-end ``pd.Series``.
    """
    rng = np.random.default_rng(seed)
    mu = np.where(np.arange(n_months) < top_month,
                  (1 + boom_cagr) ** (1 / 12) - 1,
                  (1 + bust_cagr) ** (1 / 12) - 1)
    shocks = rng.normal(0.0, sigma_month, size=n_months)
    logret = np.log1p(mu) + shocks
    level = start * np.exp(np.cumsum(logret))
    idx = pd.date_range("2019-01-31", periods=n_months, freq="ME")
    return pd.Series(level, index=idx, name="synthetic_collapse")
