"""Data layer for Study 358 — "Watches are an asset class" (Rolex/Patek/AP resale).

Three sources, all offline-friendly:

* **The resale index (hardcoded, cited, approximate).** Real secondary-market watch
  indices (WatchCharts Overall Market Index, the Morgan Stanley x WatchCharts review,
  the Subdial50) are **not** freely API-available. So we hardcode a small **annual**
  series of the secondary-market level, reconstructed from *public* reporting and
  clearly labelled as **approximate**. It is anchored on figures the desk could cite:
  the index peaked **March 2022**; the average secondhand price topped out near
  **$45,108**; full-year secondary prices fell **-10.7%** (2023) and **-6.1%** (2024)
  and rose **+4.9%** (2025) per the Morgan Stanley x WatchCharts quarterly reviews;
  and by Aug-2023 the average price was **-31%** off its March-2022 peak. Sources are
  listed in ``docs/references.md``. This is a PROXY for the real tape, never the tape.

* **Tradable equity proxies (yfinance).** Two listed ways to "own the watch trade":
  **Watches of Switzerland (WOSG.L)** — the UK/US luxury-watch *retailer* (the biggest
  Rolex/Patek/AP authorised dealer) — and **Richemont (CFR.SW)** — the Swiss group that
  owns Cartier/Vacheron/IWC/JLC/Piaget. These are *labelled proxies*: a retailer's
  equity and a conglomerate's equity are not the resale price of a steel Daytona, but
  they are the only thing a public investor can actually buy. Benchmarked against
  **SPY**. Cached under ``_cache/`` so the notebooks run offline; on a cache miss with
  network we fetch via yfinance, otherwise we fall back to the frozen headline numbers.

* **Synthetic positive control.** A deterministic fixed-seed generator of a
  "bubble-and-round-trip" price path with a *known* planted CAGR and Sharpe, used to
  prove the inference engine recovers what it plants. Runs with no network.

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
    "WOSG.L": "Watches of Switzerland (LSE) — luxury-watch retailer/AD",
    "CFR.SW": "Richemont (SIX) — Cartier/IWC/JLC/VC group",
}
BENCH = "SPY"
TICKERS = list(PROXIES) + [BENCH]


# --------------------------------------------------------------------------- #
# The resale index — hardcoded, cited, APPROXIMATE (a proxy, not a live feed)
# --------------------------------------------------------------------------- #
# Annual *secondary-market level*, normalised to 100 at end-2018, reconstructed from
# public reporting (see docs/references.md). The shape is the load-bearing fact: a
# 2019-21 melt-up, a March-2022 blow-off top, and a 2022-24 round-trip, then a mild
# 2025 stabilisation. Year-end levels (Dec each year):
#   2018 -> 100   (base)
#   2019 -> 112   (steady pre-pandemic appreciation)
#   2020 -> 130   (lockdown bid begins)
#   2021 -> 196   (the mania year; hype/flipping peak building)
#   2022 -> 168   (blew off in *March* 2022 ~ +5-15% above Dec, then fell H2)
#   2023 -> 150   (Morgan Stanley x WatchCharts: full-year secondary ~ -10.7%)
#   2024 -> 141   (full-year ~ -6.1%)
#   2025 -> 148   (full-year ~ +4.9%, a stabilisation)
# Plus the *intra-2022 peak* the reporting pins (March-2022), ~ +12% above Dec-2021.
_INDEX_YEAR_END = {
    2018: 100.0,
    2019: 112.0,
    2020: 130.0,
    2021: 196.0,
    2022: 168.0,
    2023: 150.0,
    2024: 141.0,
    2025: 148.0,
}
# The blow-off top the public reports anchor on: March 2022.
_PEAK_DATE = "2022-03-31"
_PEAK_LEVEL = 220.0  # ~ +12% above the Dec-2021 196 level; the mania high.


def load_resale_index() -> pd.Series:
    """Year-end levels of the (approximate, cited) secondary-market watch index.

    Returns a ``pd.Series`` indexed by year-end ``Timestamp``, base 100 at end-2018.
    LABELLED A PROXY: reconstructed from public reporting, not a live data feed.
    """
    idx = pd.to_datetime([f"{y}-12-31" for y in _INDEX_YEAR_END])
    return pd.Series(list(_INDEX_YEAR_END.values()), index=idx, name="resale_index")


def resale_peak() -> tuple[pd.Timestamp, float]:
    """The reported blow-off top (date, level) — March 2022."""
    return pd.Timestamp(_PEAK_DATE), _PEAK_LEVEL


def resale_annual_returns() -> pd.Series:
    """Year-over-year % change of the hardcoded resale index (the public anchors)."""
    s = load_resale_index()
    return (s / s.shift(1) - 1.0).dropna() * 100.0


# --------------------------------------------------------------------------- #
# Tradable equity proxies via yfinance (cached, offline-friendly)
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE, f"{ticker.replace('.', '_')}_monthly.csv")


def have_proxies() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in TICKERS)


def fetch_proxies(start: str = "2018-12-31", end: str = "2025-12-31") -> dict[str, pd.Series]:
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
def synthetic_bubble(n_months: int = 84, peak_month: int = 40,
                     boom_cagr: float = 0.55, bust_cagr: float = -0.30,
                     sigma_month: float = 0.05, start: float = 100.0,
                     seed: int = 358) -> pd.Series:
    """A deterministic 'bubble-and-round-trip' monthly level path.

    A planted log-drift that ramps up to ``peak_month`` at ``boom_cagr`` then rolls over
    at ``bust_cagr``, plus fixed-seed Gaussian noise of ``sigma_month``. Used as the
    positive control: :func:`strategy.summarize` must recover the planted drift's sign
    and a Sharpe close to the noiseless target. Returns a month-end ``pd.Series``.
    """
    rng = np.random.default_rng(seed)
    mu = np.where(np.arange(n_months) < peak_month,
                  (1 + boom_cagr) ** (1 / 12) - 1,
                  (1 + bust_cagr) ** (1 / 12) - 1)
    shocks = rng.normal(0.0, sigma_month, size=n_months)
    logret = np.log1p(mu) + shocks
    level = start * np.exp(np.cumsum(logret))
    idx = pd.date_range("2019-01-31", periods=n_months, freq="ME")
    return pd.Series(level, index=idx, name="synthetic")
