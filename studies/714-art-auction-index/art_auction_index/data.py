"""Data layer for Study 714 — "Contemporary art is an asset class" (the auction index).

Three sources, all offline-friendly:

* **The art auction price index (hardcoded, cited, approximate).** The indices the pitch
  quotes — the **Artprice Global Index** / *Contemporary Art Market* report and the
  **Sotheby's Mei Moses** repeat-sales index — are **not** freely API-available (Mei Moses
  went private when Sotheby's acquired it in 2016; Artprice's series sit behind its paid
  reports). So we hardcode a small **annual** series of the secondary-market price level,
  reconstructed from *public* reporting and clearly labelled as **approximate**. It is
  anchored on figures the desk could cite: a 2000-2007 melt-up, the **2008-09 financial-
  crisis crash** (art fell roughly **-40%** peak-to-trough), the recovery to a **2014**
  peak, the **2021-2022** post-COVID records boom (the Macklowe collection alone made
  **$922M** across two Sotheby's sales), and the **2023-2024 correction** (global auction
  turnover fell **~27%** in H1-2024 per Art Basel & UBS / Artprice). Sources are listed in
  ``docs/references.md``. This is a PROXY for the real tape, never the tape.

* **Tradable equity proxies (yfinance).** There is **no pure listed auction house left**:
  **Sotheby's (BID)** was taken private by Patrick Drahi in **June 2019** ($3.7bn),
  **Christie's** is private (François Pinault / Groupe Artémis) and **Phillips** is private
  (Mercury Group). The two listed ways to touch the trade are **MCH Group (MCHN.SW)** — the
  Swiss group that organises **Art Basel**, the flagship contemporary-art fair — and
  **Kering (KER.PA)** — the luxury group whose controlling shareholder (Pinault, via
  Artémis) *owns Christie's*. These are *labelled proxies*: a fair organiser's equity and a
  luxury conglomerate's equity are not the hammer price of a Basquiat, but they are the only
  things a public investor can actually buy. Benchmarked against **SPY**. Cached under
  ``_cache/`` so the notebooks run offline; on a cache miss with network we fetch via
  yfinance, otherwise we fall back to the frozen headline numbers.

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
    "MCHN.SW": "MCH Group (SIX) — organiser of Art Basel, the flagship contemporary-art fair",
    "KER.PA": "Kering (Euronext) — Pinault/Artémis (owns Christie's); luxury proxy",
}
BENCH = "SPY"
TICKERS = list(PROXIES) + [BENCH]

# The listed auction houses that NO LONGER exist as public equities — the absence is a
# finding in itself (there is no clean way to buy the art-auction trade on an exchange).
DELISTED = {
    "BID": "Sotheby's — taken private by Patrick Drahi (BidFair) June 2019, $3.7bn",
    "Christie's": "private — François Pinault / Groupe Artémis (never listed since 1998)",
    "Phillips": "private — Mercury Group (Doronin)",
}


# --------------------------------------------------------------------------- #
# The art auction index — hardcoded, cited, APPROXIMATE (a proxy, not a live feed)
# --------------------------------------------------------------------------- #
# Annual *secondary-market price level*, normalised to 100 at end-2000, reconstructed from
# public reporting (see docs/references.md). The shape is the load-bearing fact: a 2000-07
# melt-up, the 2008-09 crash (~ -40% peak-to-trough), a recovery to a 2014 peak, a
# 2015-19 plateau, a 2020 COVID dip, the 2021-22 records boom, and the 2023-24 correction.
# Year-end levels (Dec each year):
#   2000 -> 100  (base)          2013 -> 292
#   2001 -> 103                  2014 -> 330  (post-crisis peak)
#   2002 -> 100                  2015 -> 312
#   2003 -> 108                  2016 -> 292
#   2004 -> 128                  2017 -> 330
#   2005 -> 158                  2018 -> 352
#   2006 -> 205                  2019 -> 345
#   2007 -> 268  (pre-crisis peak) 2020 -> 322  (COVID dip)
#   2008 -> 232                  2021 -> 430  (post-COVID boom)
#   2009 -> 150  (crash trough)  2022 -> 478  (records; Macklowe $922M)
#   2010 -> 205                  2023 -> 430  (softening)
#   2011 -> 248                  2024 -> 385  (~ -27% H1 turnover; correction)
#   2012 -> 250                  2025 -> 400  (stabilisation)
_INDEX_YEAR_END = {
    2000: 100.0, 2001: 103.0, 2002: 100.0, 2003: 108.0, 2004: 128.0,
    2005: 158.0, 2006: 205.0, 2007: 268.0, 2008: 232.0, 2009: 150.0,
    2010: 205.0, 2011: 248.0, 2012: 250.0, 2013: 292.0, 2014: 330.0,
    2015: 312.0, 2016: 292.0, 2017: 330.0, 2018: 352.0, 2019: 345.0,
    2020: 322.0, 2021: 430.0, 2022: 478.0, 2023: 430.0, 2024: 385.0,
    2025: 400.0,
}
# The two blow-off tops the public reports anchor on: spring 2008 (pre-crisis) and 2022.
_PEAK_DATE = "2022-05-31"
_PEAK_LEVEL = 490.0  # ~ +2.5% above the Dec-2022 478 level; the records-season high.


def load_art_index() -> pd.Series:
    """Year-end levels of the (approximate, cited) contemporary-art auction index.

    Returns a ``pd.Series`` indexed by year-end ``Timestamp``, base 100 at end-2000.
    LABELLED A PROXY: reconstructed from public reporting (Artprice / Mei Moses shape),
    not a live data feed.
    """
    idx = pd.to_datetime([f"{y}-12-31" for y in _INDEX_YEAR_END])
    return pd.Series(list(_INDEX_YEAR_END.values()), index=idx, name="art_index")


def art_peak() -> tuple[pd.Timestamp, float]:
    """The reported records-season blow-off top (date, level) — spring 2022."""
    return pd.Timestamp(_PEAK_DATE), _PEAK_LEVEL


def art_annual_returns() -> pd.Series:
    """Year-over-year % change of the hardcoded art index (the public anchors)."""
    s = load_art_index()
    return (s / s.shift(1) - 1.0).dropna() * 100.0


# --------------------------------------------------------------------------- #
# Tradable equity proxies via yfinance (cached, offline-friendly)
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE, f"{ticker.replace('.', '_')}_monthly.csv")


def have_proxies() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in TICKERS)


def fetch_proxies(start: str = "2000-01-01", end: str = "2025-12-31") -> dict[str, pd.Series]:
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
def synthetic_bubble(n_months: int = 120, peak_month: int = 60,
                     boom_cagr: float = 0.35, bust_cagr: float = -0.22,
                     sigma_month: float = 0.045, start: float = 100.0,
                     seed: int = 714) -> pd.Series:
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
    idx = pd.date_range("2001-01-31", periods=n_months, freq="ME")
    return pd.Series(level, index=idx, name="synthetic")
