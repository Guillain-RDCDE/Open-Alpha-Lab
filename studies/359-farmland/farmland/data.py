"""Data layer for Study 359 — Farmland.

Three sources, all offline-friendly:

* **Listed proxies (real tape).** Monthly total-return-ish price series for the two tradable
  farmland REITs **LAND** (Gladstone Land) and **FPI** (Farmland Partners) plus **SPY**, pulled
  from yfinance (auto-adjusted closes, dividends folded in) and cached under
  ``_cache/prices_monthly.csv``. We resample to month-end and work in monthly log/simple returns.
  These are the *only* farmland exposures a retail investor can actually trade.

* **Hardcoded NCREIF farmland + CPI (public, cited).** A small, transparently-cited annual
  series: the **NCREIF Farmland Index** total return (appraisal-based) and US **CPI** inflation,
  ~1992-2023. NCREIF's full index is paywalled, so we hardcode the widely-republished annual
  total returns (see ``docs/references.md`` for the cited source values) — clearly labelled as a
  cited public dataset, never fabricated to flatter a tradable result. This is the *smooth*
  series the claim leans on.

* **Synthetic appraisal-smoothing control.** A deterministic generator (fixed seed) that takes a
  *true* underlying return process with known volatility/beta and runs it through the standard
  appraisal-smoothing filter ``r_app[t] = a*r_true[t] + (1-a)*r_app[t-1]``. With a known ``a`` we
  can prove the engine recovers the closed-form variance-shrink ``(1-a)/(1+a)`` and beta-shrink,
  and that un-smoothing restores the truth — the whole "smoothness is an artifact" argument.

Pure numpy + pandas + stdlib for the offline core; yfinance only on an explicit cache miss.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "_cache", "prices_monthly.csv")
TICKERS = ["LAND", "FPI", "SPY"]


# --------------------------------------------------------------------------- #
# Listed proxies (real tape via yfinance, cached)
# --------------------------------------------------------------------------- #
def fetch_prices(tickers=TICKERS, start="2014-01-01", end=None) -> pd.DataFrame:
    """Download month-end auto-adjusted closes from yfinance. Network call; used to (re)build
    the cache. Auto-adjust folds dividends/splits in, so simple pct-change ≈ total return."""
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, interval="1d")
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    px = px[tickers]
    monthly = px.resample("ME").last()
    monthly.index.name = "date"
    return monthly


def save_cache(df: pd.DataFrame, path: str = CACHE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)


def have_real(path: str = CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = CACHE) -> pd.DataFrame:
    """Month-end price panel from the cache (LAND, FPI, SPY)."""
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df


def monthly_returns(path: str = CACHE) -> pd.DataFrame:
    """Monthly simple total returns for LAND, FPI, SPY (drops the first NaN row)."""
    px = load_prices(path)
    return px.pct_change().dropna(how="all")


def load_real(path: str = CACHE):
    """Convenience bundle used by the notebooks' boot cell. Returns a dict with the monthly
    return panel and the common-sample aligned frame, or ``None`` if the cache is absent."""
    if not have_real(path):
        return None
    rets = monthly_returns(path)
    common = rets.dropna()
    return {"returns": rets, "common": common}


# --------------------------------------------------------------------------- #
# Hardcoded NCREIF farmland total return + US CPI (public, cited annual series)
# --------------------------------------------------------------------------- #
# NCREIF Farmland Index — annual total return, %, appraisal-based (income + appreciation).
# US CPI-U — annual inflation, %. These are the widely-republished public annual figures
# (NCREIF press summaries / TIAA & Hancock farmland reviews / BLS CPI). See docs/references.md
# for the citation. Values are rounded to one decimal and used only for illustrative contrast;
# they are a CITED PUBLIC SERIES, not a fabricated tradable result.
_NCREIF_FARMLAND = {
    1992: 4.4, 1993: 6.0, 1994: 10.3, 1995: 8.9, 1996: 12.0, 1997: 10.4,
    1998: 7.6, 1999: 7.0, 2000: 8.5, 2001: 8.5, 2002: 7.4, 2003: 9.0,
    2004: 20.1, 2005: 33.9, 2006: 19.5, 2007: 21.3, 2008: 15.8, 2009: 6.4,
    2010: 9.0, 2011: 15.2, 2012: 18.5, 2013: 20.9, 2014: 10.7, 2015: 10.0,
    2016: 6.3, 2017: 6.2, 2018: 6.7, 2019: 4.8, 2020: 3.1, 2021: 7.8,
    2022: 9.6, 2023: 5.0,
}
_CPI = {
    1992: 3.0, 1993: 3.0, 1994: 2.6, 1995: 2.8, 1996: 3.0, 1997: 2.3,
    1998: 1.6, 1999: 2.2, 2000: 3.4, 2001: 2.8, 2002: 1.6, 2003: 2.3,
    2004: 2.7, 2005: 3.4, 2006: 3.2, 2007: 2.8, 2008: 3.8, 2009: -0.4,
    2010: 1.6, 2011: 3.2, 2012: 2.1, 2013: 1.5, 2014: 1.6, 2015: 0.1,
    2016: 1.3, 2017: 2.1, 2018: 2.4, 2019: 1.8, 2020: 1.2, 2021: 4.7,
    2022: 8.0, 2023: 4.1,
}
# A broad-equity annual total return proxy (S&P 500, %) for the same years, used only to
# contrast farmland's appraisal-smoothed volatility with a market series. Public annual figures.
_SPX = {
    1992: 7.6, 1993: 10.1, 1994: 1.3, 1995: 37.6, 1996: 23.0, 1997: 33.4,
    1998: 28.6, 1999: 21.0, 2000: -9.1, 2001: -11.9, 2002: -22.1, 2003: 28.7,
    2004: 10.9, 2005: 4.9, 2006: 15.8, 2007: 5.5, 2008: -37.0, 2009: 26.5,
    2010: 15.1, 2011: 2.1, 2012: 16.0, 2013: 32.4, 2014: 13.7, 2015: 1.4,
    2016: 12.0, 2017: 21.8, 2018: -4.4, 2019: 31.5, 2020: 18.4, 2021: 28.7,
    2022: -18.1, 2023: 26.3,
}


def ncreif_annual() -> pd.DataFrame:
    """Cited public annual series: NCREIF farmland TR, CPI, and S&P 500 TR (all %), 1992-2023.

    Columns ``farmland``, ``cpi``, ``spx`` indexed by year; values are fractions (not %)."""
    years = sorted(_NCREIF_FARMLAND)
    df = pd.DataFrame(
        {
            "farmland": [_NCREIF_FARMLAND[y] / 100.0 for y in years],
            "cpi": [_CPI[y] / 100.0 for y in years],
            "spx": [_SPX[y] / 100.0 for y in years],
        },
        index=pd.Index(years, name="year"),
    )
    return df


# --------------------------------------------------------------------------- #
# Synthetic appraisal-smoothing positive control
# --------------------------------------------------------------------------- #
def synthetic_smoothing(n: int = 600, a: float = 0.20, beta: float = 0.30,
                        mkt_sigma: float = 0.045, idio_sigma: float = 0.030,
                        mkt_mu: float = 0.006, seed: int = 359):
    """Deterministic true-vs-appraised return panel demonstrating appraisal smoothing.

    Generates a *true* farmland return ``r_true[t] = mkt_mu + beta*mkt[t] + idio[t]`` driven by a
    market factor, then the *appraised* (reported) series via the standard first-order smoothing
    filter ``r_app[t] = a*r_true[t] + (1-a)*r_app[t-1]`` with smoothing parameter ``a in (0,1]``.

    With this AR(1)-MA structure the reported series has, in the limit:
      * variance shrink   Var(r_app)/Var(r_true) -> a/(2-a),
      * lag-1 autocorr    corr(r_app[t], r_app[t-1]) -> (1-a),
      * beta shrink to the market by roughly the same factor.
    Returns a DataFrame with columns ``mkt``, ``true``, ``app`` (and the params via ``.attrs``).
    """
    rng = np.random.default_rng(seed)
    mkt = rng.normal(mkt_mu, mkt_sigma, size=n)
    idio = rng.normal(0.0, idio_sigma, size=n)
    r_true = mkt_mu + beta * (mkt - mkt_mu) + idio
    r_app = np.empty(n)
    r_app[0] = r_true[0]
    for t in range(1, n):
        r_app[t] = a * r_true[t] + (1.0 - a) * r_app[t - 1]
    df = pd.DataFrame({"mkt": mkt, "true": r_true, "app": r_app})
    df.attrs.update({"a": a, "beta": beta, "mkt_sigma": mkt_sigma,
                     "idio_sigma": idio_sigma, "seed": seed})
    return df
