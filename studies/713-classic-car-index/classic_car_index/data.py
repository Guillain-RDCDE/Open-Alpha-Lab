"""Data layer for Study 713 — "classic cars are an asset class that beats equities".

Four sources, all offline-friendly:

* **The collector-car price index (hardcoded, cited, approximate).** Real collector-car
  indices — the **HAGI Top Index** (Historic Automobile Group International), the classic-
  car sleeve of the **Knight Frank Luxury Investment Index (KFLII)**, and the **Hagerty**
  Price Guide / Market Rating — are **not** freely API-available. So we hardcode a small
  **annual** series of the index *level* (base 100 @ end-2005), reconstructed from *public*
  reporting and clearly labelled **approximate**. It is anchored on figures the desk could
  cite: a powerful 2009–2015 melt-up (KFLII named classic cars the best-performing luxury
  asset of the 2010s, ~+185% over a decade), a long 2016–2020 **plateau**, a pandemic-era
  bump (KFLII cars **+25%** in 2022; HAGI Top ~+19%), and a 2023–2024 **cooling**
  (KFLII cars roughly flat-to-negative into 2024). Sources are in ``docs/references.md``.
  This is a PROXY for the real tape, never the tape.

* **Tradable equity proxies (yfinance).** The only listed ways to "own the collector-car
  trade": **Ferrari (RACE)** — the bluest-chip marque, IPO Oct-2015 — and **Aston Martin
  Lagonda (AML.L)** — IPO Oct-2018. These are *labelled proxies*: a carmaker's equity is
  not the auction price of a 1962 250 GTO, but they are the only things a public investor
  can actually buy that are *about* the collector-car trade. (They deliberately form a
  barbell — a juggernaut and a wreck — which is itself the finding: there is **no clean
  listed proxy**.)

* **Two benchmarks, labelled.** ``SPY`` (dividend-adjusted, **total return**) and ``^GSPC``
  (the S&P 500 **price-only** index). The pitch quotes a *price* index against *total-
  return* equities; the honest race puts the car price index next to ``^GSPC`` price-only,
  then notes that ``SPY`` total-return wins by even more *and* stocks pay you to hold while
  cars charge you. Cached under ``_cache/`` so the notebooks run offline.

* **Synthetic positive control.** A deterministic fixed-seed generator of a
  "boom-then-plateau" price path with a *known* planted drift, used to prove the inference
  engine recovers what it plants. Runs with no network.

Pure numpy/pandas + stdlib; the offline core never touches the network.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "_cache")

# Tradable equity proxies + benchmarks. Labelled as PROXIES / benchmarks everywhere.
PROXIES = {
    "RACE": "Ferrari NV (NYSE) — the blue-chip marque, IPO Oct-2015",
    "AML.L": "Aston Martin Lagonda (LSE) — troubled marque, IPO Oct-2018",
}
BENCH_TR = "SPY"     # dividend-adjusted -> total return
BENCH_PO = "^GSPC"   # price-only S&P 500 index
TICKERS = list(PROXIES) + [BENCH_TR, BENCH_PO]


# --------------------------------------------------------------------------- #
# The collector-car index — hardcoded, cited, APPROXIMATE (a proxy, not a feed)
# --------------------------------------------------------------------------- #
# Annual *index level*, normalised to 100 at end-2005, reconstructed from public
# reporting (see docs/references.md). The SHAPE is the load-bearing fact: a strong
# 2009-2015 melt-up, a 2016-2020 plateau, a 2021-2022 pandemic bump, then a 2023-2024
# cooling. Precise year-end values are NOT a live feed. Year-end levels (Dec each year):
#   2005 -> 100  (base)
#   ...  the 2009-2015 boom (KFLII: best luxury asset of the 2010s, ~+185%/decade) ...
#   2015 -> 372  (boom cresting)
#   2016-2020 -> ~375 plateau (HAGI Top broadly sideways)
#   2022 -> 480  (pandemic collectibles bump: KFLII cars +25%, HAGI Top ~+19%)
#   2023 -> 490  (+2%, cooling begins)
#   2024 -> 470  (-4%, market decline)
#   2025 -> 476  (+1.3%, stabilising)
_INDEX_YEAR_END = {
    2005: 100.0,
    2006: 118.0,
    2007: 138.0,
    2008: 140.0,   # GFC — collector cars notably resilient, roughly flat
    2009: 148.0,
    2010: 172.0,
    2011: 205.0,
    2012: 245.0,
    2013: 300.0,   # the mania decade accelerating (HAGI Top had ~+40% years here)
    2014: 355.0,
    2015: 372.0,   # boom cresting
    2016: 375.0,   # plateau begins
    2017: 380.0,
    2018: 378.0,   # soft
    2019: 372.0,   # -1.6%
    2020: 380.0,   # resilient through the pandemic
    2021: 410.0,   # +7.9%
    2022: 480.0,   # pandemic collectibles bump (KFLII cars +25%)
    2023: 490.0,   # +2.1%, cooling
    2024: 470.0,   # -4.1%, market decline
    2025: 476.0,   # +1.3%, stabilising
}
# The reported secondary high the public reporting anchors on (the 2023 KFLII high).
_PEAK_DATE = "2023-12-31"
_PEAK_LEVEL = 490.0


def load_car_index() -> pd.Series:
    """Year-end levels of the (approximate, cited) collector-car price index.

    Returns a ``pd.Series`` indexed by year-end ``Timestamp``, base 100 at end-2005.
    LABELLED A PROXY: reconstructed from public HAGI / KFLII / Hagerty reporting, not a
    live data feed.
    """
    idx = pd.to_datetime([f"{y}-12-31" for y in _INDEX_YEAR_END])
    return pd.Series(list(_INDEX_YEAR_END.values()), index=idx, name="car_index")


def car_peak() -> tuple[pd.Timestamp, float]:
    """The reported secondary high (date, level) — the 2023 KFLII crest."""
    return pd.Timestamp(_PEAK_DATE), _PEAK_LEVEL


def car_annual_returns() -> pd.Series:
    """Year-over-year % change of the hardcoded car index (the public anchors)."""
    s = load_car_index()
    return (s / s.shift(1) - 1.0).dropna() * 100.0


# --------------------------------------------------------------------------- #
# Tradable equity proxies + benchmarks via yfinance (cached, offline-friendly)
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE, f"{ticker.replace('.', '_').replace('^', '')}_monthly.csv")


def have_proxies() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in TICKERS)


def fetch_proxies(start: str = "2004-12-01", end: str = "2026-01-01") -> dict[str, pd.Series]:
    """Fetch month-end levels for the proxies + benchmarks.

    ``SPY`` and the equity proxies use ``auto_adjust=True`` (dividend-adjusted, total
    return); ``^GSPC`` is the price-only S&P 500 index. Tries the local cache first; on a
    miss, fetches via yfinance and writes the cache. Network only on a cache miss.
    """
    os.makedirs(CACHE, exist_ok=True)
    out: dict[str, pd.Series] = {}
    need_fetch = [t for t in TICKERS if not os.path.exists(_cache_path(t))]
    fetched = {}
    if need_fetch:
        import yfinance as yf  # imported lazily so the offline core never needs it
        for t in need_fetch:
            adjust = t != BENCH_PO  # price-only for ^GSPC, total-return for the rest
            raw = yf.download(t, start=start, end=end, interval="1mo",
                              auto_adjust=adjust, progress=False)
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            s = close.dropna()
            s.name = t
            s.index.name = "Date"
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
    """Cached month-end levels for the proxies + benchmarks (no network if cached)."""
    out = {}
    for t in TICKERS:
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True)
        out[t] = df.iloc[:, 0].dropna()
    return out


# --------------------------------------------------------------------------- #
# Synthetic positive control (deterministic, fixed seed, no network)
# --------------------------------------------------------------------------- #
def synthetic_boom(n_years: int = 20, boom_years: int = 10,
                   boom_cagr: float = 0.16, plateau_cagr: float = 0.01,
                   sigma_year: float = 0.06, start: float = 100.0,
                   seed: int = 713) -> pd.Series:
    """A deterministic 'boom-then-plateau' annual level path (the collector-car shape).

    A planted log-drift that runs at ``boom_cagr`` for ``boom_years`` then decays to
    ``plateau_cagr``, plus fixed-seed Gaussian noise of ``sigma_year``. Used as the
    positive control: :func:`strategy.summarize` must recover the planted drift's sign
    and a finite Sharpe. Returns a year-end ``pd.Series``.
    """
    rng = np.random.default_rng(seed)
    mu = np.where(np.arange(n_years) < boom_years,
                  (1 + boom_cagr) ** 1 - 1,
                  (1 + plateau_cagr) ** 1 - 1)
    shocks = rng.normal(0.0, sigma_year, size=n_years)
    logret = np.log1p(mu) + shocks
    level = start * np.exp(np.cumsum(logret))
    idx = pd.date_range("2006-12-31", periods=n_years, freq="YE")
    return pd.Series(level, index=idx, name="synthetic")
