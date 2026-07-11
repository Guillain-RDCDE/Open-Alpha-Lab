"""Data layer for Study 659 — Costless Collar.

Two ingredients, both offline-friendly once cached:

* **Real tape.** Daily SPY raw OHLC + dividend-adjusted close from yfinance (no key), cached
  as CSV under the study's own ``_cache/``. The adjusted close feeds both the **monthly
  total-return** series (own-the-index leg of the collar) and the **trailing realized
  volatility** used to price the stylized put/call legs — there are no live option chains
  here, so realized vol is our proxy for the implied vol a real market-maker would quote (see
  ``docs/references.md`` for why that's an approximation, not a chain).

* **Synthetic world.** A deterministic, seeded monthly GBM return generator with TUNABLE
  floor/cap parameters — a faithful-engine / power check for the clip-and-drag mechanics
  (``strategy.collar_returns``), never cited as market evidence. A "null" world sets the
  floor/cap so wide they essentially never bind (no detectable difference from buy & hold);
  a "planted" world uses a tight floor/cap (the difference must show up loud and clear).

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
SPY_CACHE = os.path.join(CACHE_DIR, "cc_spy.csv")

START = "1993-02-01"    # SPY inception (1993-01-29); first full trading month after
AS_OF = "2026-06-30"    # last complete calendar month at publication (2026-07-10)

# Crash windows named on the front card — peak-to-trough test beds for the drawdown claim.
GFC_WINDOW = ("2007-10-01", "2009-03-31")     # the 2008 global financial crisis drawdown
COVID_WINDOW = ("2020-01-01", "2020-04-30")   # the 2020 COVID crash + initial recovery

# --------------------------------------------------------------------------- #
# Stylized options-model assumptions — named, not hidden. This is a Black-
# Scholes approximation using REALIZED (not implied/quoted) volatility as the pricing
# input, because no live option chain is used anywhere in this study. Real listed SPY
# collars would price off the implied-vol surface, which on average sits ABOVE realized
# vol (the variance risk premium) — see docs/references.md for the honest direction of
# that bias. The risk-free rate is a single constant assumption, not a fitted curve.
# --------------------------------------------------------------------------- #
RF_ANNUAL = 0.03         # constant assumed risk-free rate (also used for the "cash" leg)
PUT_OTM = 0.05           # the put is struck 5% out of the money (the claim's own number)
VOL_WINDOW = 63          # trailing trading days (~3 months) used to estimate realized vol
OPTION_T = 1.0 / 12.0    # one-month-to-expiry approximation for both legs


def fetch(start: str = "1993-01-01", end: str = "2026-07-01") -> None:
    """Download SPY raw OHLC + dividend-adjusted close; cache as CSV. Network; runs once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    raw = yf.download("SPY", start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    out = pd.DataFrame({
        "Open": raw["Open"], "High": raw["High"], "Low": raw["Low"], "Close": raw["Close"],
        "AdjClose": raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"],
    }).dropna(how="all")
    out.to_csv(SPY_CACHE)


def have_real() -> bool:
    return os.path.exists(SPY_CACHE)


def load_real(start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Cached daily SPY frame, sliced to [start, asof]."""
    df = pd.read_csv(SPY_CACHE, index_col=0, parse_dates=True).sort_index()
    return df.loc[(df.index >= start) & (df.index <= asof)].copy()


# --------------------------------------------------------------------------- #
# Derived series: monthly total return, trailing realized vol (known ex ante).
# --------------------------------------------------------------------------- #
def monthly_total_return(daily: pd.DataFrame) -> pd.Series:
    """Month-end adjusted-close total return (dividends included via Adj Close)."""
    m = daily["AdjClose"].resample("ME").last()
    return m.pct_change().dropna().rename("spy_ret")


def trailing_realized_vol(daily: pd.DataFrame, window: int = VOL_WINDOW) -> pd.Series:
    """Annualized realized vol of daily log returns, trailing ``window`` sessions.

    Sampled at each month END, then SHIFTED one month forward before use by the caller —
    the single documented execution convention: the vol (and hence the collar's floor/cap)
    for month *t* is measured through the close of month *t-1*, known before month *t*
    begins. Zero look-ahead.
    """
    lr = np.log(daily["AdjClose"]).diff()
    rv = lr.rolling(window).std() * np.sqrt(252)
    return rv.resample("ME").last().dropna().rename("real_vol")


def month_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per month: SPY total return (realized THIS month) and the vol input that was
    KNOWN BEFORE this month began (previous month's trailing realized vol, shifted forward
    by one month — the study's single documented execution lag)."""
    ret = monthly_total_return(daily)
    vol = trailing_realized_vol(daily).shift(1)   # known before the month it prices
    df = pd.concat({"spy_ret": ret, "vol_in": vol}, axis=1).dropna()
    return df


# --------------------------------------------------------------------------- #
# Synthetic world — planted floor/cap effect (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 396, seed: int = 659,
                     mu_annual: float = 0.09, sigma_annual: float = 0.16
                     ) -> pd.DataFrame:
    """Deterministic monthly GBM total-return series with a PeriodIndex (no ns-timestamp
    risk even at long spans). ``n_months=396`` ~ 33 years, matching the real sample length.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 12.0
    mu_m = (mu_annual - 0.5 * sigma_annual ** 2) * dt
    sig_m = sigma_annual * np.sqrt(dt)
    log_ret = rng.normal(mu_m, sig_m, n_months)
    ret = np.exp(log_ret) - 1.0
    idx = pd.period_range("1993-02", periods=n_months, freq="M")
    return pd.DataFrame({"spy_ret": ret}, index=idx)
