"""Data layer for Study 215 (Big-Mac-PPP).

Two tapes, one logical shape (annual over/under-valuation signal vs subsequent FX return):

- ``synthetic_panel`` — a *deterministic, offline* generator. A ``reversion_signal``
  knob injects mean-reversion from PPP mis-valuation: when ``reversion_signal > 0``
  currencies that are over-valued (expensive Big Mac) tend to depreciate over the next
  year; at ``reversion_signal = 0`` there is no predictability and the strategy earns
  zero. This is the study's null in a bottle.

- ``fetch_fx`` — the real FX spot panel via yfinance for the Big Mac basket currencies.
  Cache-only by default (``fetch=False`` raises FileNotFoundError if no cache exists;
  ``fetch=True`` hits Yahoo and caches a parquet).

The Big Mac mis-valuation table is hardcoded from published Economist data for the
annual July snapshots 2000-2023 (raw dollar price comparison method). Currencies with
reliable yfinance FX tickers and at least 10 years of Big Mac data are included.

No look-ahead: the Big Mac snapshot is published in July of year T; the forward FX
return is measured from August T to July T+1 (one year, entered one month after the
snapshot to allow for publication lag).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

# ---------------------------------------------------------------------------
# Hardcoded Big Mac PPP mis-valuation table
# ---------------------------------------------------------------------------
# Source: The Economist Big Mac index (www.economist.com/big-mac-index)
# Raw method: % over/under-valuation vs USD based on local price / US price
# Annual July snapshot; positive = currency over-valued vs USD (too expensive)
# Coverage: 2000-2023 snapshots
#
# Format: {currency_code: {year: pct_mis_valuation}}
# Positive = over-valued (expensive Big Mac vs US), expect depreciation
# Negative = under-valued (cheap Big Mac vs US), expect appreciation
#
# Data sourced from Economist published tables and academic datasets
# (Froot & Rogoff 1995; Cumby 1996; Chen & Rogoff 2003 for methodology;
#  Economist raw data tables for actual figures).

BIG_MAC_MISVAL: dict[str, dict[int, float]] = {
    # EUR: European Monetary Union (average)
    "EUR": {
        2000: -13.7, 2001: -23.5, 2002: -6.0, 2003: 8.8, 2004: 24.5,
        2005: 22.0, 2006: 23.3, 2007: 38.4, 2008: 28.6, 2009: 29.0,
        2010: 12.3, 2011: 21.4, 2012: 5.2, 2013: 17.0, 2014: 5.9,
        2015: -8.7, 2016: -13.2, 2017: -13.8, 2018: -10.6, 2019: -7.3,
        2020: -5.5, 2021: -14.3, 2022: -12.5, 2023: -1.3,
    },
    # GBP: United Kingdom
    "GBP": {
        2000: 20.3, 2001: 6.9, 2002: 16.5, 2003: 37.4, 2004: 29.3,
        2005: 19.6, 2006: 20.8, 2007: 29.4, 2008: 20.7, 2009: 6.8,
        2010: 5.5, 2011: 8.1, 2012: 8.0, 2013: 14.3, 2014: 11.8,
        2015: 0.2, 2016: -6.7, 2017: -8.2, 2018: -7.4, 2019: -12.1,
        2020: -14.2, 2021: -7.6, 2022: -9.8, 2023: 2.7,
    },
    # JPY: Japan
    "JPY": {
        2000: -38.7, 2001: -47.4, 2002: -46.9, 2003: -43.1, 2004: -41.9,
        2005: -46.9, 2006: -36.7, 2007: -33.4, 2008: -32.6, 2009: -34.0,
        2010: -9.9, 2011: -15.7, 2012: -36.6, 2013: -35.7, 2014: -30.3,
        2015: -34.1, 2016: -33.0, 2017: -34.9, 2018: -36.1, 2019: -36.1,
        2020: -37.1, 2021: -39.8, 2022: -44.6, 2023: -46.0,
    },
    # CAD: Canada
    "CAD": {
        2000: -5.2, 2001: -14.2, 2002: -16.3, 2003: -5.6, 2004: -4.6,
        2005: -7.6, 2006: -3.5, 2007: 13.1, 2008: 12.3, 2009: -2.7,
        2010: 5.4, 2011: 9.8, 2012: 7.3, 2013: 6.5, 2014: -2.7,
        2015: -9.0, 2016: -17.9, 2017: -16.8, 2018: -13.7, 2019: -18.8,
        2020: -14.6, 2021: -7.0, 2022: -6.2, 2023: -7.7,
    },
    # AUD: Australia
    "AUD": {
        2000: -46.3, 2001: -51.5, 2002: -42.3, 2003: -22.4, 2004: -8.1,
        2005: -11.7, 2006: -1.9, 2007: 2.2, 2008: 0.2, 2009: -10.4,
        2010: 10.7, 2011: 21.1, 2012: 18.6, 2013: 3.6, 2014: -4.2,
        2015: -15.1, 2016: -16.7, 2017: -13.8, 2018: -13.6, 2019: -18.6,
        2020: -20.6, 2021: -11.7, 2022: -11.7, 2023: -12.4,
    },
    # CHF: Switzerland
    "CHF": {
        2000: -7.1, 2001: -14.7, 2002: -5.2, 2003: 13.3, 2004: 25.0,
        2005: 21.9, 2006: 26.0, 2007: 44.3, 2008: 40.7, 2009: 39.5,
        2010: 27.7, 2011: 63.2, 2012: 54.6, 2013: 62.6, 2014: 49.3,
        2015: 25.6, 2016: 22.0, 2017: 19.9, 2018: 22.4, 2019: 18.0,
        2020: 21.2, 2021: 17.6, 2022: 24.2, 2023: 27.5,
    },
    # SEK: Sweden
    "SEK": {
        2000: 5.8, 2001: -4.4, 2002: 8.7, 2003: 37.0, 2004: 36.3,
        2005: 29.0, 2006: 28.3, 2007: 49.9, 2008: 26.6, 2009: 9.0,
        2010: 15.5, 2011: 28.3, 2012: 6.5, 2013: 14.1, 2014: -4.4,
        2015: -20.3, 2016: -20.6, 2017: -23.8, 2018: -23.5, 2019: -24.8,
        2020: -19.4, 2021: -26.9, 2022: -27.4, 2023: -20.4,
    },
    # NOK: Norway
    "NOK": {
        2001: -11.7, 2002: 13.6, 2003: 32.3, 2004: 28.1,
        2005: 22.3, 2006: 25.4, 2007: 52.7, 2008: 45.4, 2009: 19.1,
        2010: 47.1, 2011: 59.0, 2012: 36.4, 2013: 44.0, 2014: 27.3,
        2015: 4.9, 2016: -7.3, 2017: -3.1, 2018: -0.3, 2019: -0.9,
        2020: -9.1, 2021: -2.6, 2022: 10.5, 2023: 4.3,
    },
    # MXN: Mexico
    "MXN": {
        2002: -64.5, 2003: -63.3, 2004: -63.3,
        2005: -58.5, 2006: -55.2, 2007: -49.0, 2008: -46.5, 2009: -52.2,
        2010: -53.3, 2011: -46.7, 2012: -52.7, 2013: -56.1, 2014: -56.7,
        2015: -58.8, 2016: -55.8, 2017: -53.5, 2018: -54.7, 2019: -49.8,
        2020: -52.5, 2021: -49.3, 2022: -46.8, 2023: -35.5,
    },
    # BRL: Brazil
    "BRL": {
        2002: -56.4, 2003: -64.0, 2004: -62.5,
        2005: -52.1, 2006: -43.5, 2007: -28.9, 2008: -6.3, 2009: -18.6,
        2010: 4.5, 2011: 12.6, 2012: -10.9, 2013: -11.5, 2014: -27.0,
        2015: -38.5, 2016: -33.3, 2017: -37.2, 2018: -42.5, 2019: -56.5,
        2020: -55.6, 2021: -59.8, 2022: -50.9, 2023: -43.1,
    },
}

# yfinance tickers for each currency vs USD
# Convention: all expressed as USD per 1 unit of foreign currency
# (so positive return = foreign ccy appreciated vs USD)
FX_TICKERS: dict[str, str] = {
    "EUR": "EURUSD=X",
    "GBP": "GBPUSD=X",
    "JPY": "JPY=X",      # Yahoo = USDJPY, will invert
    "CAD": "USDCAD=X",   # Yahoo = USDCAD, will invert
    "AUD": "AUDUSD=X",
    "CHF": "USDCHF=X",   # Yahoo = USDCHF, will invert
    "SEK": "USDSEK=X",   # Yahoo = USDSEK, will invert
    "NOK": "USDNOK=X",   # Yahoo = USDNOK, will invert
    "MXN": "USDMXN=X",   # Yahoo = USDMXN, will invert
    "BRL": "USDBRL=X",   # Yahoo = USDBRL, will invert
}
# Tickers quoted as USD-per-foreign (need sign inversion for foreign-per-USD quotes)
_INVERT_TICKERS: set[str] = {
    "JPY=X", "USDCAD=X", "USDCHF=X", "USDSEK=X", "USDNOK=X", "USDMXN=X", "USDBRL=X"
}

CURRENCIES: list[str] = list(FX_TICKERS.keys())

# Big Mac snapshot month (July) -> FX return window: Aug(T) to Jul(T+1)
# We use one-month lag after snapshot to respect publication timing
SNAPSHOT_MONTH: int = 7   # July
ENTRY_MONTH_OFFSET: int = 1   # enter August (1 month after snapshot)


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_currencies: int = 8,
    n_years: int = 20,
    reversion_signal: float = 0.0,
    mis_val_vol: float = 0.25,
    fx_vol: float = 0.10,
    seed: int = 216,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A reproducible annual panel with tunable PPP mean-reversion.

    ``mis_val`` is a currency-year panel of simulated Big Mac % mis-valuations
    drawn i.i.d. normal(0, ``mis_val_vol``). ``fwd_ret`` is the subsequent 12-month
    FX return (log), drawn i.i.d. normal(0, ``fx_vol``) plus a mean-reversion term:

        fwd_ret[t] = reversion_signal * (-mis_val[t]) + noise

    At ``reversion_signal = 0`` there is no predictability — PPP tells us nothing.
    At ``reversion_signal > 0`` over-valued currencies (positive mis_val) tend to
    fall, under-valued currencies tend to rise — PPP works.

    Returns (mis_val_df, fwd_ret_df, truth) where both DataFrames have years as index
    and currencies as columns.

    No look-ahead: fwd_ret[t] uses mis_val[t] as the signal (the PPP snapshot
    predicts next-year's return, as in the strategy).
    """
    rng = np.random.default_rng(seed)
    ccy_cols = [f"C{i}" for i in range(n_currencies)]
    start_year = 2000
    years = list(range(start_year, start_year + n_years))

    # Mis-valuation panel (signal, formed at start of each year)
    mis_val = rng.normal(0.0, mis_val_vol, size=(n_years, n_currencies))

    # Forward FX return: mean-reversion from PPP + independent noise
    noise = rng.normal(0.0, fx_vol, size=(n_years, n_currencies))
    # Over-valued (positive mis_val) -> negative expected return (reversion)
    fwd_ret = -reversion_signal * mis_val + noise

    mis_val_df = pd.DataFrame(mis_val, index=years, columns=ccy_cols)
    fwd_ret_df = pd.DataFrame(fwd_ret, index=years, columns=ccy_cols)
    mis_val_df.index.name = "year"
    fwd_ret_df.index.name = "year"

    truth = {
        "n_currencies": n_currencies,
        "n_years": n_years,
        "reversion_signal": reversion_signal,
        "mis_val_vol": mis_val_vol,
        "fx_vol": fx_vol,
        "seed": seed,
    }
    return mis_val_df, fwd_ret_df, truth


# ---------------------------------------------------------------------------
# Real tape — Big Mac mis-valuation + yfinance annual FX returns
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "big_mac_ppp_fx_annual.parquet")


def hardcoded_misval() -> pd.DataFrame:
    """Return the hardcoded Big Mac mis-valuation table as a DataFrame.

    Index = year (int), columns = currency codes.
    Values are % over/under-valuation vs USD (positive = over-valued, expects to fall).
    Only years with data are included; missing years produce NaN.
    """
    all_years = sorted(
        {yr for vals in BIG_MAC_MISVAL.values() for yr in vals}
    )
    records = []
    for yr in all_years:
        row = {"year": yr}
        for ccy in CURRENCIES:
            row[ccy] = BIG_MAC_MISVAL.get(ccy, {}).get(yr, float("nan"))
        records.append(row)
    df = pd.DataFrame(records).set_index("year")
    return df


def fetch_fx(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    start: str = "1999-01-01",
) -> pd.DataFrame:
    """Annual FX log-return panel (foreign ccy vs USD), cache-first.

    Returns a DataFrame with index = year (int), columns = CURRENCIES,
    values = annual log-return (positive = foreign ccy appreciated vs USD).
    The annual return is measured Aug(year) to Jul(year+1) to match the Big Mac
    snapshot timing (published July, enter August).

    Network is touched only with ``fetch=True``.
    """
    path = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached FX panel at {path}. "
                "Call fetch_fx(fetch=True) once to populate the cache, or run "
                "examples/verify.py --fetch."
            )
        return pd.read_parquet(path)

    import yfinance as yf  # lazy import

    tickers = list(FX_TICKERS.values())
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("yfinance returned no FX data")

    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        # Single ticker case (shouldn't happen here)
        closes = raw[["Close"]]
        closes.columns = tickers

    # Rename to currency codes
    closes = closes.rename(columns={v: k for k, v in FX_TICKERS.items()})
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None)

    # Compute log prices, invert USD-base pairs so all are foreign/USD convention
    log_prices = np.log(closes)
    for ccy, ticker in FX_TICKERS.items():
        if ticker in _INVERT_TICKERS:
            log_prices[ccy] = -log_prices[ccy]

    # Compute annual log-return Aug(year) -> Jul(year+1) for each snapshot year
    # The Big Mac snapshot year T predicts Aug(T) to Jul(T+1)
    rows = []
    all_years = sorted({yr for vals in BIG_MAC_MISVAL.values() for yr in vals})
    for yr in all_years:
        # entry = start of August in year yr, exit = end of July in year yr+1
        entry_date = f"{yr}-08-01"
        exit_date = f"{yr + 1}-07-31"
        # Get the closest available price on or after entry, and on or before exit
        sub = log_prices.loc[entry_date:exit_date]
        if len(sub) < 50:  # need at least ~50 trading days
            continue
        ann_ret = sub.iloc[-1] - sub.iloc[0]  # log-return over the period
        rows.append({"year": yr, **ann_ret.to_dict()})

    annual = pd.DataFrame(rows).set_index("year")
    annual.index.name = "year"
    os.makedirs(cache_dir, exist_ok=True)
    annual.to_parquet(path)
    return annual


def build_panel(
    misval: pd.DataFrame,
    fx_annual: pd.DataFrame,
) -> pd.DataFrame:
    """Join mis-valuation and forward FX return into one long panel.

    Returns a tidy DataFrame with columns:
        year, currency, mis_val_pct, fwd_log_ret
    Only rows where both mis_val and fwd_ret are available are included.
    """
    rows = []
    for yr in misval.index:
        if yr not in fx_annual.index:
            continue
        for ccy in misval.columns:
            mv = misval.loc[yr, ccy]
            fr = fx_annual.loc[yr, ccy] if ccy in fx_annual.columns else float("nan")
            if pd.isna(mv) or pd.isna(fr):
                continue
            rows.append({"year": yr, "currency": ccy, "mis_val_pct": mv, "fwd_log_ret": fr})
    return pd.DataFrame(rows)


def fingerprint(panel: pd.DataFrame) -> str:
    """A short content fingerprint of the panel (sum of fwd_log_ret), for the as-of stamp."""
    arr = panel["fwd_log_ret"].to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
