"""Data layer for Study 711 — "A Birkin beats the S&P (and even gold)" (Hermès handbags).

Three sources, all offline-friendly:

* **The Birkin resale index (hardcoded, cited, approximate).** There is no free, API-
  available secondary-market handbag index. The famous claim comes from **Baghunter's
  2016 study** (widely recycled, and echoed in Credit Suisse / Knight Frank luxury-
  investment notes): a Birkin allegedly returned **~14.2%/yr over 1980–2015**, "never
  had a down year," and **beat both the S&P (~8.7%) and gold**. So we hardcode a small
  **annual** secondary-market level, reconstructed from *public* reporting and clearly
  labelled **approximate**. It is anchored on figures the desk could cite: Hermès raises
  primary Birkin prices ~5–10%/yr; the 2020–22 resale melt-up; the **2023–24 cooling** of
  the luxury-handbag resale market (Knight Frank Luxury Investment Index had handbags as
  the *worst* luxury collectible of 2023–24, roughly flat-to-negative), and a mild 2025.
  Sources are in ``docs/references.md``. This is a PROXY for the real tape, never the tape.

* **Tradable equity proxies (yfinance).** You cannot buy "the Birkin index." The only
  *listed* ways to own the trade are the maisons: **Hermès (RMS.PA)** — the actual maker
  of the Birkin — plus **LVMH (MC.PA)** and **Kering (KER.PA)**, the other two European
  luxury majors. Benchmarked against **SPY** (the S&P the claim invokes) *and* **GLD**
  (the gold the claim also invokes). These are *labelled proxies*: a maison's equity is
  not the resale price of a Birkin, but it is the only thing a public investor can buy.
  Cached under ``_cache/`` so the notebooks run offline; on a cache miss with network we
  fetch via yfinance, otherwise we fall back to the frozen headline numbers.

* **Synthetic positive control.** A deterministic fixed-seed generator of a "steady luxury
  compounder" price path with a *known* planted CAGR and Sharpe, used to prove the
  inference engine recovers what it plants. Runs with no network.

Pure numpy/pandas + stdlib; the offline core never touches the network.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "_cache")

# Tradable equity proxies + benchmarks. Labelled as PROXIES everywhere.
PROXIES = {
    "RMS.PA": "Hermès (Euronext Paris) — the maker of the Birkin itself",
    "MC.PA": "LVMH (Euronext Paris) — Louis Vuitton / Dior luxury major",
    "KER.PA": "Kering (Euronext Paris) — Gucci / Saint Laurent / Bottega",
}
BENCH = "SPY"          # the S&P the claim says a Birkin beats
GOLD = "GLD"           # the gold the claim *also* says a Birkin beats
TICKERS = list(PROXIES) + [BENCH, GOLD]


# --------------------------------------------------------------------------- #
# The Birkin resale index — hardcoded, cited, APPROXIMATE (a proxy, not a feed)
# --------------------------------------------------------------------------- #
# Annual *secondary-market level*, normalised to 100 at end-2015, reconstructed from
# public reporting (see docs/references.md). The shape is the load-bearing fact: steady
# retail-driven appreciation, a 2020-22 resale melt-up, then the 2023-24 luxury-handbag
# resale COOLING (Knight Frank Luxury Investment Index had handbags as the worst-
# performing collectible of 2023-24), and a mild 2025. Year-end levels (Dec each year):
#   2015 -> 100   (base — the Baghunter study's end year)
#   2016 -> 106   (Hermès primary price hikes ~5-6%/yr flow to resale)
#   2017 -> 113
#   2018 -> 121
#   2019 -> 128
#   2020 -> 134   (pandemic dip then a fast recovery; scarcity bid begins)
#   2021 -> 150   (resale melt-up; the "handbags beat stocks" headlines return)
#   2022 -> 162   (peak building; Knight Frank handbags still positive)
#   2023 -> 166   (near the top; luxury resale starting to cool)
#   2024 -> 158   (handbags the WORST luxury collectible of the year — resale softens)
#   2025 -> 163   (a mild stabilisation)
_INDEX_YEAR_END = {
    2015: 100.0,
    2016: 106.0,
    2017: 113.0,
    2018: 121.0,
    2019: 128.0,
    2020: 134.0,
    2021: 150.0,
    2022: 162.0,
    2023: 166.0,
    2024: 158.0,
    2025: 163.0,
}
# The Baghunter myth the pitch anchors on: a headline "14.2%/yr, never a down year".
_MYTH_CAGR = 0.142        # Baghunter 2016: claimed 1980-2015 average annual return
_MYTH_WINDOW = "1980-2015"
_MYTH_SP500 = 0.087       # the S&P figure Baghunter quoted for the same window


def load_resale_index() -> pd.Series:
    """Year-end levels of the (approximate, cited) Birkin secondary-market index.

    Returns a ``pd.Series`` indexed by year-end ``Timestamp``, base 100 at end-2015.
    LABELLED A PROXY: reconstructed from public reporting, not a live data feed.
    """
    idx = pd.to_datetime([f"{y}-12-31" for y in _INDEX_YEAR_END])
    return pd.Series(list(_INDEX_YEAR_END.values()), index=idx, name="birkin_index")


def baghunter_myth() -> dict:
    """The headline claim under test: Baghunter's cited annual return + its S&P quote."""
    return {"cagr": _MYTH_CAGR, "window": _MYTH_WINDOW, "sp500": _MYTH_SP500}


def resale_annual_returns() -> pd.Series:
    """Year-over-year % change of the hardcoded Birkin index (the public anchors)."""
    s = load_resale_index()
    return (s / s.shift(1) - 1.0).dropna() * 100.0


# --------------------------------------------------------------------------- #
# Tradable equity proxies via yfinance (cached, offline-friendly)
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE, f"{ticker.replace('.', '_')}_monthly.csv")


def have_proxies() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in TICKERS)


def fetch_proxies(start: str = "2014-12-31", end: str = "2026-01-01") -> dict[str, pd.Series]:
    """Fetch month-end *total-return-ish* (Adj Close) levels for the proxies + benchmarks.

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
    """Cached month-end Adj-Close levels for the proxies + benchmarks (no network if cached)."""
    out = {}
    for t in TICKERS:
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True)
        out[t] = df.iloc[:, 0].dropna()
    return out


# --------------------------------------------------------------------------- #
# Synthetic positive control (deterministic, fixed seed, no network)
# --------------------------------------------------------------------------- #
def synthetic_compounder(n_months: int = 120, drift_cagr: float = 0.12,
                         sigma_month: float = 0.03, start: float = 100.0,
                         seed: int = 711) -> pd.Series:
    """A deterministic 'steady luxury compounder' monthly level path.

    A planted constant log-drift of ``drift_cagr`` plus fixed-seed Gaussian noise of
    ``sigma_month`` — the "it only ever goes up" story the pitch tells. Used as the
    positive control: :func:`strategy.summarize` must recover the planted drift's sign
    and a Sharpe close to the noiseless target. Returns a month-end ``pd.Series``.
    """
    rng = np.random.default_rng(seed)
    mu = (1 + drift_cagr) ** (1 / 12) - 1
    shocks = rng.normal(0.0, sigma_month, size=n_months)
    logret = np.log1p(mu) + shocks
    level = start * np.exp(np.cumsum(logret))
    idx = pd.date_range("2016-01-31", periods=n_months, freq="ME")
    return pd.Series(level, index=idx, name="synthetic")
