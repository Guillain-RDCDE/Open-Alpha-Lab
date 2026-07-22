"""Data layer for Study 797 — FX Value (PPP real-exchange-rate mean reversion).

Two ingredients, both offline-friendly once cached under the study's own ``_cache/``:

* **Real tape.**
  - **G10 FX spot** vs USD (yfinance, no key), normalised to **USD per 1 unit of
    foreign currency** so a *rise* means the foreign currency *appreciated*. Sampled
    to **month-end** (the signal and the backtest are monthly).
  - **CPI (consumer-price index)** per country, monthly where it exists:
    * IMF IFS ``M.<ISO2>.PCPI_IX`` for USD, JPY, GBP, CAD, CHF, SEK, NOK (via DBnomics),
    * Eurostat HICP ``prc_hicp_midx`` (euro-area, ``CP00.EA19``) for EUR,
    * IMF IFS **quarterly** ``Q.<ISO2>.PCPI_IX`` for AUD & NZD (Australia and New
      Zealand publish CPI only quarterly) — forward-filled to month-end inside each
      quarter. This is a real, documented coarseness on two of the ten legs.

  The **real exchange rate** of foreign currency *i* vs USD is
  ``q_i = S_i(USD/FX) * CPI_i / CPI_US`` (``log q = log S + log CPI_i - log CPI_US``).
  A *high* q means the currency is **rich** (overvalued) relative to PPP; a *low* q
  means it is **cheap** (undervalued). The value signal is the deviation of ``log q``
  from its own long trailing average — undervalued currencies are the longs.

  **Publication lag.** CPI for calendar month *m* is only released weeks into month
  *m+1*, so the signal formed at the end of month *t* uses CPI known through month
  ``t - CPI_PUB_LAG`` (default 1) — point-in-time, no look-ahead into an unreleased
  print. FX is real-time (month-end close).

* **Synthetic world.** A deterministic, seeded panel of real exchange rates that
  mean-revert to a common long-run level with a **tunable** planted reversion strength
  (knob ``value_strength``, in monthly pull-back fraction). At ``value_strength = 0``
  the real rates are driftless random walks — no PPP mean reversion — and the
  cross-sectional value sort must NOT manufacture a significant premium. Positive
  control only; never cited for the real-tape stamp.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
FX_CACHE = os.path.join(CACHE_DIR, "fx_eom.csv")
CPI_CACHE = os.path.join(CACHE_DIR, "cpi_monthly.csv")

# G10 minus USD = the nine cross rates we take a view on (USD is the numeraire).
CURRENCIES = ["EUR", "JPY", "GBP", "AUD", "NZD", "CAD", "CHF", "SEK", "NOK"]

# yfinance tickers -> (currency, invert?) so every series is USD-per-foreign-currency.
FX_TICKERS = {
    "EURUSD=X": ("EUR", False),
    "GBPUSD=X": ("GBP", False),
    "AUDUSD=X": ("AUD", False),
    "NZDUSD=X": ("NZD", False),
    "USDJPY=X": ("JPY", True),
    "USDCHF=X": ("CHF", True),
    "USDCAD=X": ("CAD", True),
    "USDNOK=X": ("NOK", True),
    "USDSEK=X": ("SEK", True),
}

# CPI (consumer-price index) sources on DBnomics. USD included (the deflator baseline).
CPI_MONTHLY = {  # currency -> IMF IFS monthly PCPI_IX
    "USD": "IMF/IFS/M.US.PCPI_IX",
    "JPY": "IMF/IFS/M.JP.PCPI_IX",
    "GBP": "IMF/IFS/M.GB.PCPI_IX",
    "CAD": "IMF/IFS/M.CA.PCPI_IX",
    "CHF": "IMF/IFS/M.CH.PCPI_IX",
    "SEK": "IMF/IFS/M.SE.PCPI_IX",
    "NOK": "IMF/IFS/M.NO.PCPI_IX",
}
CPI_EUR = "Eurostat/prc_hicp_midx/M.I15.CP00.EA19"   # euro-area HICP, monthly
CPI_QUARTERLY = {  # AUS & NZL publish CPI only quarterly -> ffill to month-end
    "AUD": "IMF/IFS/Q.AU.PCPI_IX",
    "NZD": "IMF/IFS/Q.NZ.PCPI_IX",
}

START = "1999-12-31"        # euro-era start; FX + CPI both broadly available
AS_OF = "2025-06-30"        # last complete month with CPI on every leg at publication
TRAIL_WINDOW = 60           # months in the trailing PPP-average (5 years, literature-standard)
MIN_TRAIL = 36              # minimum trailing months before a currency can be ranked
CPI_PUB_LAG = 1             # months of CPI publication lag baked into the point-in-time signal

UA = {"User-Agent": "Mozilla/5.0 (OpenAlphaLab research)"}


# --------------------------------------------------------------------------- #
# Network fetch (once) — builds the offline cache
# --------------------------------------------------------------------------- #
def _get(url: str, timeout: int = 30, tries: int = 4) -> bytes:
    last = None
    for _ in range(tries):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=timeout).read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5)
    raise last


def _dbnomics_series(sid: str) -> pd.Series:
    """One DBnomics series -> pandas Series indexed by period-end Timestamp (monthly
    or quarterly), NaNs dropped, float."""
    raw = _get(f"https://api.db.nomics.world/v22/series/{sid}?observations=1")
    d = json.loads(raw)["series"]["docs"][0]
    periods = d["period"]
    freq = "Q" if any("Q" in p for p in periods[:3]) else "M"
    idx = pd.PeriodIndex(periods, freq=freq).to_timestamp(how="end").normalize()
    s = pd.Series(d["value"], index=idx)
    return s[s.notna()].astype(float)


def _yf_eom() -> pd.DataFrame:
    """Download the FX basket via yfinance, normalise to USD-per-foreign, resample to
    month-end (last observation of each month)."""
    import yfinance as yf

    raw = yf.download(list(FX_TICKERS), period="max", interval="1d",
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw["Close"]
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    usd_per = {}
    for tk, (ccy, invert) in FX_TICKERS.items():
        if tk not in raw.columns:
            continue
        s = raw[tk].dropna()
        usd_per[ccy] = (1.0 / s) if invert else s
    df = pd.DataFrame(usd_per).sort_index()
    eom = df.resample("ME").last()
    eom.index.name = "date"
    return eom


def fetch() -> None:
    """Download FX (yfinance) + CPI (DBnomics) and cache them. Network; run once."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- CPI panel (monthly index, one column per currency incl. USD) ---
    cols: dict[str, pd.Series] = {}
    for ccy, sid in CPI_MONTHLY.items():
        try:
            cols[ccy] = _dbnomics_series(sid).resample("ME").last()
            print(f"   CPI {ccy}: {sid} ok")
        except Exception as e:  # noqa: BLE001
            print(f"   CPI {ccy}: FAIL {type(e).__name__} {str(e)[:50]}")
    try:
        cols["EUR"] = _dbnomics_series(CPI_EUR).resample("ME").last()
        print(f"   CPI EUR: {CPI_EUR} ok")
    except Exception as e:  # noqa: BLE001
        print(f"   CPI EUR: FAIL {type(e).__name__} {str(e)[:50]}")
    for ccy, sid in CPI_QUARTERLY.items():
        try:
            q = _dbnomics_series(sid)                       # quarter-end index
            m = q.resample("ME").last().ffill(limit=3)      # ffill within the quarter
            cols[ccy] = m
            print(f"   CPI {ccy} (quarterly->ffill): {sid} ok")
        except Exception as e:  # noqa: BLE001
            print(f"   CPI {ccy}: FAIL {type(e).__name__} {str(e)[:50]}")
    cpi = pd.DataFrame(cols).sort_index()
    cpi.index.name = "date"
    cpi.to_csv(CPI_CACHE)
    print(f"   cpi_monthly: {cpi.shape} {cpi.index.min().date()}..{cpi.index.max().date()} "
          f"cols={list(cpi.columns)}")

    # --- FX month-end panel ---
    fx = _yf_eom()
    fx.to_csv(FX_CACHE)
    print(f"   fx_eom: {fx.shape} {fx.index.min().date()}..{fx.index.max().date()} "
          f"cols={list(fx.columns)}")


def have_real() -> bool:
    return os.path.exists(FX_CACHE) and os.path.exists(CPI_CACHE)


# --------------------------------------------------------------------------- #
# Offline load + real-rate construction
# --------------------------------------------------------------------------- #
def load_fx(start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    df = pd.read_csv(FX_CACHE, index_col=0, parse_dates=True).sort_index()
    return df.loc[(df.index >= start) & (df.index <= asof), CURRENCIES].copy()


def load_cpi(start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    df = pd.read_csv(CPI_CACHE, index_col=0, parse_dates=True).sort_index()
    keep = ["USD"] + CURRENCIES
    mask = (df.index >= "1990-01-01") & (df.index <= asof)
    return df.loc[mask, keep].copy()


def real_rate_panel(fx: pd.DataFrame, cpi: pd.DataFrame,
                    pub_lag: int = CPI_PUB_LAG) -> pd.DataFrame:
    """log real exchange rate ``log q_i = log S_i + log CPI_i - log CPI_US`` on a
    monthly grid, with CPI lagged ``pub_lag`` months (publication lag: the print for
    month *m* is not known until month *m+1*). Returns a currency-columned frame of
    log real rates aligned to FX month-ends."""
    fx = fx.copy()
    fx.index = fx.index.to_period("M")
    cpi = cpi.copy()
    cpi.index = cpi.index.to_period("M")
    cpi_lag = cpi.shift(pub_lag)                # only CPI known pub_lag months earlier
    log_cpi = np.log(cpi_lag)
    out = {}
    for c in CURRENCIES:
        s = np.log(fx[c]) + log_cpi[c] - log_cpi["USD"]
        out[c] = s
    q = pd.DataFrame(out)
    q.index = q.index.to_timestamp(how="end").normalize()
    return q.dropna(how="all")


# --------------------------------------------------------------------------- #
# Synthetic world — planted PPP mean reversion (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(value_strength: float = 0.0, seed: int = 797,
                    n_months: int = 300, n_ccy: int = 9,
                    sigma: float = 0.025) -> pd.DataFrame:
    """Deterministic panel of log real exchange rates with a TUNABLE planted mean
    reversion. Each currency's log real rate follows

        q_{i,t+1} = q_{i,t} - value_strength * (q_{i,t} - mu_i) + eps

    where ``mu_i`` is a fixed per-currency long-run level and ``eps`` is i.i.d. normal.
    ``value_strength = 0`` -> driftless random walks (no PPP pull, the null world):
    a currency being cheap vs its own history carries NO information about its next
    move, and the cross-sectional value sort must NOT fire. ``value_strength > 0``
    plants exactly the reversion the strategy is designed to harvest.

    Returns a month-indexed frame of log real rates (columns C0..C{n-1}); feed it to
    ``strategy.value_signal`` / ``strategy.portfolio_returns`` exactly like the real
    panel. FX spot returns are recovered as the change in log real rate (holding
    relative CPI constant, a synthetic convenience — inflation is smooth vs FX).
    """
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, 0.15, n_ccy)
    q = np.zeros((n_months, n_ccy))
    q[0] = mu + rng.normal(0.0, sigma, n_ccy)
    for t in range(1, n_months):
        pull = -value_strength * (q[t - 1] - mu)
        q[t] = q[t - 1] + pull + rng.normal(0.0, sigma, n_ccy)
    idx = pd.period_range("2000-01", periods=n_months, freq="M").to_timestamp(how="end").normalize()
    return pd.DataFrame(q, index=idx, columns=[f"C{i}" for i in range(n_ccy)])
