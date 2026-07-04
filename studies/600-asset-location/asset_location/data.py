"""Data layer for Study 600 — Asset Location.

Three sources, all offline-friendly once cached:

* **Long tape (Shiller).** Robert Shiller's monthly S&P 500 series (price level, trailing
  dividend, 10-year Treasury yield "GS10"), 1871->present. We reuse the repo's shared cache
  (``_cache/_cache/shiller_sp500.parquet``) when present, else copy from the study cache, else
  fetch the `datasets/s-and-p-500` mirror on raw.githubusercontent once. From the monthly tape we
  build **annual components**: stock price return, stock dividend yield (cash paid during the
  year over the prior-December price), the 10y coupon set at the **prior December** yield (the
  one documented decision lag — the coupon a household locks in is the yield known when it
  allocates), and the annual price return of a constant-maturity 10y par bond (buy a 10y par
  bond at last December's yield, sell it one year later as a 9y bond at this December's yield).

* **Modern tape (yfinance).** SPY (stock sleeve) + IEF (7-10y Treasury fund, bond sleeve):
  unadjusted closes + cash distributions, so the income component (taxed every year in a taxable
  account) and the price component (deferrable) are split exactly as a real household statement
  splits them. One modern low-rate cohort, 2003->2025.

* **Synthetic world.** A deterministic, seeded generator with the same annual-component shape
  and a TUNABLE planted effect: the bond ``coupon`` knob. The whole asset-location benefit is
  income shielded from annual tax, so with ``coupon = 0`` **and** ``div_yield = 0`` **and** zero
  turnover every location policy is the *same* portfolio after tax — the simulator must return a
  delta of exactly 0 (the null). Raising the coupon plants a benefit that must scale with it.

Pure numpy + pandas for the offline path; network functions run once to build the cache.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
SHILLER_CACHE = os.path.join(CACHE_DIR, "al_shiller.csv")
MODERN_PRICES_CACHE = os.path.join(CACHE_DIR, "al_modern_prices.csv")
MODERN_DIVS_CACHE = os.path.join(CACHE_DIR, "al_modern_divs.csv")

# The repo-wide shared Shiller cache (already fetched by earlier studies).
SHARED_SHILLER = os.path.join(HERE, "..", "..", "..", "_cache", "_cache", "shiller_sp500.parquet")

SHILLER_URLS = [
    "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv",
    "https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/data.csv",
]

MODERN_TICKERS = ["SPY", "IEF"]  # stock sleeve, 7-10y Treasury bond sleeve


# --------------------------------------------------------------------------- #
# Long tape — Shiller monthly
# --------------------------------------------------------------------------- #
def have_shiller(path: str = SHILLER_CACHE) -> bool:
    return os.path.exists(path)


def fetch_shiller(path: str = SHILLER_CACHE) -> pd.DataFrame:
    """Build the study's Shiller cache: shared repo cache first, network last. Run once."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    shared = os.path.abspath(SHARED_SHILLER)
    if os.path.exists(shared):
        df = pd.read_parquet(shared)
    else:  # pragma: no cover — network fallback
        import io
        import urllib.request

        df = None
        for url in SHILLER_URLS:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "OpenAlphaLab research desk"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    df = pd.read_csv(io.StringIO(r.read().decode("utf-8")))
                break
            except Exception:
                continue
        if df is None:
            raise RuntimeError("could not fetch the Shiller series")
    df.to_csv(path, index=False)
    return df


def load_shiller(path: str = SHILLER_CACHE) -> pd.DataFrame:
    """Monthly Shiller frame indexed by month-start Timestamp: price / div / gs10.

    Trimmed to the rows where BOTH the dividend and the 10y yield are populated (the mirror
    zero-fills the recent tail) — so a stamped run never leans on placeholder zeros.
    """
    raw = pd.read_csv(path)
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"]),
        "price": raw["SP500"].astype(float),
        "div": raw["Dividend"].astype(float),
        "gs10": raw["Long Interest Rate"].astype(float),
    }).set_index("date").sort_index()
    out = out[(out["div"] > 0) & (out["gs10"] > 0)]
    return out


def bond_price_return(y_old: float, y_new: float, remaining: int = 9) -> float:
    """Price return of a 10y par bond bought at yield ``y_old`` and sold one year later
    as a ``remaining``-year bond at yield ``y_new`` (annual coupons, decimal yields)."""
    if y_new <= 0:
        return y_old * remaining  # linear limit as y_new -> 0
    ann = (1.0 - (1.0 + y_new) ** -remaining) / y_new
    return y_old * ann + (1.0 + y_new) ** -remaining - 1.0


def annual_components(shiller: pd.DataFrame) -> pd.DataFrame:
    """Annual return components (calendar years), indexed by year (int).

    Columns:
      stock_px — Dec->Dec S&P price return
      stock_dy — cash dividends paid during the year / prior-December price
      bond_cpn — 10y coupon locked at the PRIOR December yield (the documented decision lag)
      bond_px  — 10y constant-maturity par-bond price return over the year

    Only complete calendar years enter (a year needs its own December and the prior December).
    """
    m = shiller.copy()
    dec = m[m.index.month == 12]
    years = sorted(set(dec.index.year))
    rows = []
    for y in years[1:]:
        if (y - 1) not in years:
            continue
        p0 = float(dec.loc[dec.index.year == y - 1, "price"].iloc[0])
        p1 = float(dec.loc[dec.index.year == y, "price"].iloc[0])
        y0 = float(dec.loc[dec.index.year == y - 1, "gs10"].iloc[0]) / 100.0
        y1 = float(dec.loc[dec.index.year == y, "gs10"].iloc[0]) / 100.0
        in_year = m[m.index.year == y]
        if len(in_year) < 12:
            continue
        cash_div = float((in_year["div"] / 12.0).sum())  # Shiller div is an annualised level
        rows.append({
            "year": y,
            "stock_px": p1 / p0 - 1.0,
            "stock_dy": cash_div / p0,
            "bond_cpn": y0,
            "bond_px": bond_price_return(y0, y1),
        })
    return pd.DataFrame(rows).set_index("year")


# --------------------------------------------------------------------------- #
# Modern tape — SPY + IEF via yfinance
# --------------------------------------------------------------------------- #
def have_modern(prices_path: str = MODERN_PRICES_CACHE, divs_path: str = MODERN_DIVS_CACHE) -> bool:
    return os.path.exists(prices_path) and os.path.exists(divs_path)


def fetch_modern(prices_path: str = MODERN_PRICES_CACHE,
                 divs_path: str = MODERN_DIVS_CACHE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download SPY + IEF unadjusted closes and cash distributions. Network; run once."""
    import yfinance as yf

    px = yf.download(MODERN_TICKERS, start="2002-01-01", auto_adjust=False,
                     progress=False)["Close"].dropna(how="all")
    px.index = pd.DatetimeIndex([pd.Timestamp(t).tz_localize(None) for t in px.index])
    rows = []
    for tk in MODERN_TICKERS:
        div = yf.Ticker(tk).dividends
        for ts, amt in div.items():
            rows.append({"ticker": tk,
                         "date": pd.Timestamp(ts).tz_localize(None).normalize(),
                         "amount": float(amt)})
    divs = (pd.DataFrame(rows).drop_duplicates(["ticker", "date"])
            .sort_values(["ticker", "date"]).reset_index(drop=True))
    os.makedirs(os.path.dirname(prices_path), exist_ok=True)
    px.to_csv(prices_path)
    divs.to_csv(divs_path, index=False)
    return px, divs


def load_modern(prices_path: str = MODERN_PRICES_CACHE,
                divs_path: str = MODERN_DIVS_CACHE) -> tuple[pd.DataFrame, pd.DataFrame]:
    px = pd.read_csv(prices_path, index_col=0, parse_dates=True).sort_index()
    dv = pd.read_csv(divs_path, parse_dates=["date"])
    return px, dv


def modern_components(px: pd.DataFrame, divs: pd.DataFrame,
                      first_year: int = 2003, last_year: int = 2025) -> pd.DataFrame:
    """Annual components from SPY (stock) + IEF (bond): unadjusted price returns +
    cash-distribution yields on the prior-December close. Same shape as ``annual_components``.

    IEF's distributions are Treasury coupon pass-through (ordinary income); SPY's are
    qualified dividends — exactly the tax split the simulator charges.
    """
    dec = px.resample("YE").last()
    dec.index = dec.index.year
    rows = []
    for y in range(first_year, last_year + 1):
        if (y - 1) not in dec.index or y not in dec.index:
            continue
        s0, s1 = float(dec.loc[y - 1, "SPY"]), float(dec.loc[y, "SPY"])
        b0, b1 = float(dec.loc[y - 1, "IEF"]), float(dec.loc[y, "IEF"])
        d_spy = float(divs[(divs["ticker"] == "SPY") & (divs["date"].dt.year == y)]["amount"].sum())
        d_ief = float(divs[(divs["ticker"] == "IEF") & (divs["date"].dt.year == y)]["amount"].sum())
        rows.append({
            "year": y,
            "stock_px": s1 / s0 - 1.0,
            "stock_dy": d_spy / s0,
            "bond_cpn": d_ief / b0,
            "bond_px": b1 / b0 - 1.0,
        })
    return pd.DataFrame(rows).set_index("year")


# --------------------------------------------------------------------------- #
# Synthetic world — deterministic, tunable planted effect + a null
# --------------------------------------------------------------------------- #
def synthetic_world(n_years: int = 30, coupon: float = 0.05, div_yield: float = 0.02,
                    stock_mu: float = 0.05, stock_sig: float = 0.16, bond_sig: float = 0.05,
                    seed: int = 600) -> pd.DataFrame:
    """Seeded annual components with a TUNABLE planted asset-location effect.

    The location benefit is *income shielded from annual tax*, so the ``coupon`` knob IS the
    planted effect: ``coupon = div_yield = 0`` (with zero turnover downstream) is the exact
    null — every policy holds identical assets with identical taxation, delta must be 0.0.
    Spans <= 250 years by construction (decorative annual index of years 2000+).
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2000 + n_years)
    return pd.DataFrame({
        "year": years,
        "stock_px": rng.normal(stock_mu, stock_sig, n_years),
        "stock_dy": np.full(n_years, div_yield),
        "bond_cpn": np.full(n_years, coupon),
        "bond_px": rng.normal(0.0, bond_sig, n_years),
    }).set_index("year")
