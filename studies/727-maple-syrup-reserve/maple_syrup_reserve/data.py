"""Data layer for Study 727 — "the Quebec strategic maple-syrup reserve as a trade".

Three sources, all offline-friendly:

* **The PPAQ bulk maple-syrup price (hardcoded, cited, approximate).** There is no
  maple-syrup exchange, no futures, no live price API. The world price of bulk maple
  syrup is *administered*: the Quebec producers' cartel — **Producteurs et productrices
  acéricoles du Québec (PPAQ**, ex-Fédération des producteurs acéricoles du Québec) —
  which controls **~72% of world output**, negotiates a single bulk price per pound each
  season and defends it with the **Global Strategic Maple Syrup Reserve** (a barrel
  stockpile in Laurierville, QC, capacity ~100 M lb — the one raided in the 2011–12
  "Great Canadian Maple Syrup Heist", ≈ **C$18.7 M** / 3,000 t stolen). So we hardcode a
  small **annual** CAD/lb series reconstructed from PPAQ negotiated-price reporting and
  press coverage, clearly labelled **approximate**. The load-bearing fact is its *shape*:
  an administered price that barely moves (~2%/yr, near-zero volatility) — the opposite of
  a tradable commodity. Sources in ``docs/references.md``. This is a PROXY, never a feed.

* **Tradable equity proxies (yfinance).** There is essentially **no** way to buy maple.
  The single closest listed name is **Rogers Sugar (`RSI.TO`, TSX)** — the Lantic sugar
  refiner that also owns the **L.B. Maple Treat / Decacer** maple-bottling business, i.e.
  the only public company with a real maple-products segment. As a *soft-commodity
  placebo* we also carry **No. 11 sugar futures (`SB=F`)** — the nearest freely-traded
  sweetener, which has **nothing to do with maple** and is included precisely to show
  that. Both are benchmarked against the **S&P/TSX Composite (`^GSPTSE`)**, the CAD-home
  index (RSI.TO is CAD-denominated, so the benchmark is CAD too — no FX mismatch). These
  are *labelled proxies*: a diversified sugar refiner's equity is not the price of a
  barrel of syrup. Cached under ``_cache/`` so notebooks run offline; network only on a
  cache miss.

* **Synthetic positive control.** A deterministic fixed-seed generator of a monthly
  return world with a *planted* sugaring-season (Feb–Apr) premium, used to prove the
  seasonality engine recovers what it plants. Runs with no network. NEVER backs a stamp.

Pure numpy/pandas + stdlib; the offline core never touches the network.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "_cache")

# Tradable proxies + placebo + benchmark. Labelled as PROXIES everywhere.
PROXY = "RSI.TO"          # Rogers Sugar — the only listed name with a real maple segment
PLACEBO = "SB=F"          # No.11 sugar futures — nearest sweetener, NOT maple (a placebo)
BENCH = "^GSPTSE"         # S&P/TSX Composite — CAD-home benchmark (RSI.TO is CAD)
TICKERS = [PROXY, PLACEBO, BENCH]

NAMES = {
    "RSI.TO": "Rogers Sugar (TSX) — Lantic refiner + L.B. Maple Treat/Decacer maple unit",
    "SB=F": "No.11 sugar futures — nearest freely-traded sweetener (a PLACEBO, not maple)",
    "^GSPTSE": "S&P/TSX Composite — the CAD-home benchmark",
}

# The sugaring season: sap runs on the late-winter freeze/thaw cycle, Feb–Apr.
SUGARING_MONTHS = [2, 3, 4]

MONTHS = 12.0


# --------------------------------------------------------------------------- #
# The PPAQ bulk maple price — hardcoded, cited, APPROXIMATE (a proxy, not a feed)
# --------------------------------------------------------------------------- #
# Annual *administered bulk price* of Grade A maple syrup, CAD per pound, reconstructed
# from PPAQ negotiated-price reporting and press coverage (see docs/references.md). The
# shape is the load-bearing fact: a price that is NEGOTIATED, not discovered — it creeps
# up a couple of percent a year with almost no volatility, defended by the reserve. That
# near-flat, cartel-set line is exactly why maple is a curio, not a market. Year values:
_PRICE_CAD_LB = {
    2008: 2.60,
    2009: 2.65,
    2010: 2.72,
    2011: 2.85,
    2012: 2.92,
    2013: 2.94,
    2014: 2.90,
    2015: 2.86,
    2016: 2.85,
    2017: 2.88,
    2018: 2.85,
    2019: 2.84,
    2020: 2.85,
    2021: 2.86,
    2022: 3.02,
    2023: 3.35,
    2024: 3.60,
}

# Reserve / heist anchors (cited, approximate) — for the notebook's colour, not the stats.
RESERVE_FACTS = {
    "operator": "Producteurs et productrices acéricoles du Québec (PPAQ)",
    "world_share_pct": 72,          # Quebec's share of world maple output (~72%)
    "reserve_capacity_mlb": 100,    # Laurierville strategic reserve, ~100 M lb capacity
    "heist_year": "2011-2012",
    "heist_value_cad_m": 18.7,      # Great Canadian Maple Syrup Heist (~C$18.7 M)
    "heist_tonnes": 3000,           # ~3,000 tonnes of syrup stolen
}


def load_maple_price() -> pd.Series:
    """Annual PPAQ bulk price of Grade A maple syrup (CAD/lb), a cited APPROXIMATE proxy.

    Returns a ``pd.Series`` indexed by year-end ``Timestamp``. LABELLED A PROXY:
    reconstructed from PPAQ negotiated-price reporting, not a live data feed.
    """
    idx = pd.to_datetime([f"{y}-12-31" for y in _PRICE_CAD_LB])
    return pd.Series(list(_PRICE_CAD_LB.values()), index=idx, name="maple_cad_lb")


def maple_annual_returns() -> pd.Series:
    """Year-over-year % change of the hardcoded administered maple price."""
    s = load_maple_price()
    return (s / s.shift(1) - 1.0).dropna() * 100.0


# --------------------------------------------------------------------------- #
# Tradable equity proxies via yfinance (cached, offline-friendly)
# --------------------------------------------------------------------------- #
def _san(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", ticker)


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE, f"{_san(ticker)}_monthly.csv")


def have_proxies() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in TICKERS)


def fetch_proxies(start: str = "2010-01-01", end: str = "2026-06-01") -> dict[str, pd.Series]:
    """Fetch month-end Adj Close levels for the proxies + benchmark (cache-first).

    Tries the local cache first; on a miss, fetches via yfinance and writes the cache.
    Returns ``{ticker: monthly Adj-Close Series}``. Network only on a cache miss.
    """
    os.makedirs(CACHE, exist_ok=True)
    out: dict[str, pd.Series] = {}
    need_fetch = [t for t in TICKERS if not os.path.exists(_cache_path(t))]
    fetched: dict[str, pd.Series] = {}
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
    """Cached month-end Adj-Close levels for the proxies + benchmark (no network)."""
    out = {}
    for t in TICKERS:
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True)
        out[t] = df.iloc[:, 0].dropna()
    return out


def monthly_returns(level: pd.Series) -> pd.Series:
    """Simple month-over-month returns of a month-end level series."""
    return level.pct_change().dropna()


# --------------------------------------------------------------------------- #
# Synthetic positive control (deterministic, fixed seed, no network)
# --------------------------------------------------------------------------- #
def synthetic_world(n_years: int = 16, spring_premium: float = 0.06,
                    vol: float = 0.22, seed: int = 727) -> tuple[pd.DataFrame, dict]:
    """A monthly return world with a *planted* sugaring-season (Feb–Apr) premium.

    Monthly returns are i.i.d. Gaussian with annual vol ``vol`` plus ``spring_premium``
    spread across the three sugaring months (Feb–Apr) and subtracted, spread across the
    other nine, so the yearly mean is unchanged (a pure calendar re-allocation).
    ``spring_premium = 0`` is the null. Returns ``(frame, truth)`` where ``truth`` records
    the planted premium; the seasonality engine must recover its sign. No network.
    """
    rng = np.random.default_rng(seed)
    n = int(n_years * 12)
    idx = pd.date_range("2010-01-31", periods=n, freq="ME", name="date")
    months = idx.month
    base = (vol / np.sqrt(12)) * rng.standard_normal(n)
    seasonal = np.where(
        np.isin(months, SUGARING_MONTHS),
        spring_premium / len(SUGARING_MONTHS),
        -spring_premium / (12 - len(SUGARING_MONTHS)),
    )
    ret = pd.Series(base + seasonal, index=idx, name="synthetic")
    return pd.DataFrame({"ret": ret}), {"spring_premium": spring_premium, "seed": seed}
