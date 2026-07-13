"""Data layer for Study 712 — "CGC-graded key comics are an asset class".

Three sources, all offline-friendly:

* **The comic price index (hardcoded, cited, approximate).** Real graded-comic price
  indices — the **GoCollect** market indices and **Heritage Auctions** realised-price
  history — are **not** freely API-available (GoCollect gates its indices behind a paid
  subscription; Heritage publishes per-lot archives, not a downloadable index). So we
  hardcode a small **annual** series of the graded-key-comic price level, reconstructed
  from *public* reporting and clearly labelled as **approximate**. It is anchored on
  figures the desk could cite: a pandemic-era 2020–2021 speculative melt-up, an **early
  2022** blow-off top, a 2022–2023 softening, and a 2024–2025 blue-chip stabilisation —
  the era of record slabs (Amazing Fantasy #15 CGC 9.6 at **$3.6M**, 2021; Action Comics
  #1 at a record **~$6.0M**, 2024). Sources are listed in ``docs/references.md``. This is
  a PROXY for the real tape, never the tape.

* **The (nearly non-existent) tradable proxy (yfinance).** There is **no** pure-play
  listed comic-book equity. CGC's parent (Certified Collectibles Group) is Blackstone-
  owned and **private**; PSA's parent (Collectors Universe, ex-`CLCT`) was **taken
  private** in Feb-2021; Heritage Auctions is **private**. The nearest *listed* proxy for
  "pop-culture collectibles as a business" is **Funko (`FNKO`)** — the Pop!-figure /
  licensed-collectibles maker that IPO'd Nov-2017. It is a *labelled proxy* and a poor
  one: a toy company's equity is not the resale price of a CGC 9.8 key. Benchmarked
  against **SPY**. Cached under ``_cache/`` so the notebooks run offline; on a cache miss
  with network we fetch via yfinance, otherwise we fall back to the frozen headline
  numbers.

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

# The single listed proxy + benchmark. Labelled as a PROXY everywhere.
PROXIES = {
    "FNKO": "Funko (Nasdaq) — pop-culture licensed-collectibles maker (nearest listed proxy)",
}
BENCH = "SPY"
TICKERS = list(PROXIES) + [BENCH]


# --------------------------------------------------------------------------- #
# The comic price index — hardcoded, cited, APPROXIMATE (a proxy, not a feed)
# --------------------------------------------------------------------------- #
# Annual *graded-key-comic price level*, normalised to 100 at end-2018, reconstructed
# from public reporting (see docs/references.md). The shape is the load-bearing fact: a
# steady 2018-19 base, a 2020-21 pandemic-speculation melt-up (movie announcements, a
# flood of new collectors, near-zero rates, slab-flipping), an early-2022 blow-off top,
# a 2022-23 giveback as rates rose and speculation cooled, and a 2024-25 stabilisation
# carried by blue-chip keys (Action Comics #1 at a record ~$6.0M in 2024). Year-end
# levels (Dec each year):
#   2018 -> 100   (base)
#   2019 -> 108   (steady pre-pandemic appreciation)
#   2020 -> 130   (the lockdown collecting boom begins mid-2020)
#   2021 -> 175   (the mania year; speculation/flipping peak building)
#   2022 -> 158   (blew off in *early* 2022, then softened H2 as rates rose)
#   2023 -> 145   (continued cooling of the speculative middle)
#   2024 -> 150   (blue-chip keys stabilise; Action #1 sets a ~$6.0M record)
#   2025 -> 152   (mild; a two-tier market — keys firm, the middle soft)
_INDEX_YEAR_END = {
    2018: 100.0,
    2019: 108.0,
    2020: 130.0,
    2021: 175.0,
    2022: 158.0,
    2023: 145.0,
    2024: 150.0,
    2025: 152.0,
}
# The blow-off top the public reports anchor on: early 2022.
_PEAK_DATE = "2022-01-31"
_PEAK_LEVEL = 188.0  # ~ +7% above the Dec-2021 175 level; the speculative high.


def load_comic_index() -> pd.Series:
    """Year-end levels of the (approximate, cited) graded-key-comic price index.

    Returns a ``pd.Series`` indexed by year-end ``Timestamp``, base 100 at end-2018.
    LABELLED A PROXY: reconstructed from public reporting, not a live data feed.
    """
    idx = pd.to_datetime([f"{y}-12-31" for y in _INDEX_YEAR_END])
    return pd.Series(list(_INDEX_YEAR_END.values()), index=idx, name="comic_index")


def comic_peak() -> tuple[pd.Timestamp, float]:
    """The reported blow-off top (date, level) — early 2022."""
    return pd.Timestamp(_PEAK_DATE), _PEAK_LEVEL


def comic_annual_returns() -> pd.Series:
    """Year-over-year % change of the hardcoded comic index (the public anchors)."""
    s = load_comic_index()
    return (s / s.shift(1) - 1.0).dropna() * 100.0


# --------------------------------------------------------------------------- #
# The single tradable equity proxy via yfinance (cached, offline-friendly)
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE, f"{ticker.replace('.', '_')}_monthly.csv")


def have_proxies() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in TICKERS)


def fetch_proxies(start: str = "2017-11-30", end: str = "2025-12-31") -> dict[str, pd.Series]:
    """Fetch month-end *total-return-ish* (Adj Close) levels for the proxy + SPY.

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
    """Cached month-end Adj-Close levels for the proxy + SPY (no network if cached)."""
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
                     seed: int = 712) -> pd.Series:
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
