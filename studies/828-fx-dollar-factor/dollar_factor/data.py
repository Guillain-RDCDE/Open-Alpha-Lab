"""Data layer for Study 828 — FX Dollar Factor (DOL) + dollar-timing.

The claim under test (Lustig, Roussanov & Verdelhan 2011, *"Common Risk Factors in
Currency Markets"*): the **dollar factor** ``DOL`` — the equal-weight average excess
return of a cross-section of foreign currencies against the USD — is a candidate
priced risk factor. Its *unconditional* premium is famously small, but the average
**forward discount** (the mean interest-rate gap of the basket vs the USD) is said to
**time** it: DOL is high when US rates are low relative to the world.

Two ingredients, both offline-friendly once cached under the study's own ``_cache/``:

* **Real tape — G10 FX spot vs USD.** Daily FX spot for a fixed 7-currency basket
  (EUR, GBP, JPY, AUD, CAD, CHF, NZD), pulled with yfinance (no key), sampled to
  **month-end** (the signal and the backtest are monthly). yfinance quotes some pairs
  USD-base (``JPY=X`` = USDJPY = JPY per USD) and others foreign-base (``EURUSD=X`` =
  USD per EUR); every pair is normalised to **USD-per-foreign-currency** so a *rise*
  means the foreign currency **appreciated** and a USD-funded long earns a positive
  spot return. The month-end panel is cached as a parquet.

  **The forward discount / carry.** True overnight-deposit differentials are not on
  yfinance, so the interest carry attached to each currency is a fixed, transparent
  per-currency long-run average annualised short-rate differential vs USD (a **proxy**,
  named as such everywhere). The **average forward discount** ``AFD`` is the
  cross-sectional mean of those differentials; because the proxy is static its level is
  a constant, so the *time-varying* dollar-timing conditioning variable is a **spot-only
  trailing dollar-trend proxy** (the trailing 12-month DOL) — documented as a proxy for
  the LRV average-forward-discount signal, which needs rate/forward data absent here.

* **Synthetic world — the positive control.** A deterministic, seeded panel of foreign
  currencies with a TUNABLE common **dollar drift** ``edge`` (all currencies share a
  positive appreciation-vs-USD component → a real DOL premium) and a TUNABLE **timing**
  strength that makes next-month DOL predictable from a persistent state variable. At
  ``edge = 0`` / ``timing = 0`` the currencies are driftless correlated random walks:
  DOL has no premium and no predictability, and the machinery must find nothing.

The offline path is pure numpy + pandas + stdlib. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells; ``load_panel()``
/ ``load_series()`` read the cached parquet directly (no yfinance import).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
FX_CACHE = os.path.join(CACHE_DIR, "fx_eom.parquet")

# The 7-currency basket vs USD. yfinance serves some pairs foreign-base (``EURUSD=X``
# already USD-per-foreign) and others USD-base (``JPY=X`` = USDJPY, JPY per USD). We map
# every ticker to (currency, invert) so the loader normalises ALL series to
# USD-per-foreign-currency (a rise => the foreign currency appreciated vs the dollar).
FX_TICKERS = {
    "EURUSD=X": ("EUR", False),   # USD per EUR      -> keep
    "GBPUSD=X": ("GBP", False),   # USD per GBP      -> keep
    "AUDUSD=X": ("AUD", False),   # USD per AUD      -> keep
    "NZDUSD=X": ("NZD", False),   # USD per NZD      -> keep
    "JPY=X":    ("JPY", True),    # JPY per USD      -> invert
    "CAD=X":    ("CAD", True),    # CAD per USD      -> invert
    "CHF=X":    ("CHF", True),    # CHF per USD      -> invert
}

CURRENCIES = ["EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]

# Transparent fixed carry PROXY: a long-run average annualised short-rate differential
# vs USD (in percent) — the interest a USD investor earns by holding that currency.
# Deliberately round numbers reflecting the 2004-2024 ordering (commodity AUD/NZD on
# top, funding JPY/CHF deeply negative). Used ONLY to attach an excess-return leg to DOL
# and to compute the average forward discount; the DOL premium verdict rests on spot.
CARRY_PROXY = {
    "AUD": 1.6, "NZD": 1.8, "CAD": 0.2, "GBP": 0.4,
    "EUR": -0.4, "CHF": -1.4, "JPY": -1.6,
}

START = "2003-12-01"        # yfinance FX history broadly begins here
AS_OF = "2026-06-30"        # last complete calendar month at publication

__all__ = [
    "CURRENCIES", "FX_TICKERS", "CARRY_PROXY", "START", "AS_OF", "CACHE_DIR", "FX_CACHE",
    "fetch", "have_real", "load_panel", "load_series", "synthetic_panel",
]


# --------------------------------------------------------------------------- #
# Real tape — network fetch (once), then offline load
# --------------------------------------------------------------------------- #
def fetch(start: str = START, tries: int = 4) -> pd.DataFrame:
    """Download the FX basket via yfinance and cache a month-end USD-per-foreign parquet.

    Network-only; used once to build ``_cache/fx_eom.parquet``. Never imported by the
    offline notebook cells. Retries up to ``tries`` times with a short sleep on failure.
    ``auto_adjust=True``. Each pair is normalised to USD-per-foreign-currency, then
    resampled to month-end (last observation of each month).
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    last = None
    raw = None
    for attempt in range(tries):
        try:
            raw = yf.download(list(FX_TICKERS), start=start, interval="1d",
                              auto_adjust=True, progress=False)
            if raw is not None and len(raw):
                break
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(2.0 * (attempt + 1))
    if raw is None or not len(raw):
        raise RuntimeError(f"yfinance returned no data after {tries} tries: {last}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw["Close"]
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    usd_per: dict[str, pd.Series] = {}
    for tk, (ccy, invert) in FX_TICKERS.items():
        if tk not in raw.columns:
            continue
        s = raw[tk].astype(float).dropna()
        usd_per[ccy] = (1.0 / s) if invert else s
    df = pd.DataFrame(usd_per).sort_index()
    eom = df.resample("ME").last()[CURRENCIES]
    eom.index.name = "date"
    eom.to_parquet(FX_CACHE)
    return eom


def have_real() -> bool:
    return os.path.exists(FX_CACHE)


def load_panel(start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Cached month-end USD-per-foreign panel, sliced to ``[start, asof]``.

    Index = month-end date, columns = ``CURRENCIES``. Reads the parquet directly —
    OFFLINE, no yfinance import.
    """
    df = pd.read_parquet(FX_CACHE).sort_index()
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    df = df.loc[(df.index >= lo) & (df.index <= hi)]
    return df[[c for c in CURRENCIES if c in df.columns]].copy()


def load_series(start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Alias kept for symmetry with the sibling studies (same object as load_panel)."""
    return load_panel(start, asof)


# --------------------------------------------------------------------------- #
# Synthetic world — planted dollar premium + timing (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    timing: float = 0.0,
    seed: int = 828,
    n_months: int = 300,
    n_ccy: int = 7,
    sigma: float = 0.026,
    common_load: float = 0.55,
    state_rho: float = 0.9,
) -> pd.DataFrame:
    """Deterministic seeded month-end USD-per-foreign panel with a TUNABLE planted
    dollar premium and dollar-timing relation.

    A common **dollar factor** ``d_t`` (the systematic USD move that hits every
    currency) plus a persistent latent **state** ``x_t`` (an AR(1), the stand-in for the
    average forward discount) drive each currency's monthly log spot return:

        d_t = mu_common + timing * x_{t-1} + eta_t              (common dollar move)
        r[i,t] = edge + common_load * d_t + idio_i_t           (per-currency return)

    where ``mu_common = edge`` plants a flat DOL premium and ``timing * x_{t-1}`` makes
    next-month DOL predictable from the observable trailing dollar trend (``x`` is
    persistent, so the trailing 12-month DOL proxies it). ``edge = 0`` AND
    ``timing = 0`` is the null world: correlated driftless random walks, DOL flat and
    unpredictable. Returns a month-indexed frame of **price levels** (USD-per-foreign),
    columns C0..C{n-1}; feed it to the strategy exactly like the real panel.
    """
    rng = np.random.default_rng(seed)
    innov_sd = np.sqrt(max(1.0 - state_rho ** 2, 1e-9))
    x = np.empty(n_months)
    x[0] = rng.normal(0.0, 1.0)
    eps = rng.normal(0.0, innov_sd, n_months)
    for t in range(1, n_months):
        x[t] = state_rho * x[t - 1] + eps[t]

    eta = rng.normal(0.0, sigma * 0.6, n_months)
    x_lag = np.concatenate([[0.0], x[:-1]])
    d = edge + timing * x_lag + eta                 # common dollar move per month

    idx = pd.period_range("2000-01", periods=n_months, freq="M").to_timestamp(how="end").normalize()
    cols = {}
    for i in range(n_ccy):
        idio = rng.normal(0.0, sigma, n_months)
        r = common_load * d + idio                  # per-currency simple spot return
        # cumprod(1+r) (not exp(cumsum)) so pct_change recovers r exactly -> the null
        # DOL premium centres at zero (no simple-return convexity/Jensen drift).
        cols[f"C{i}"] = 1.0 * np.cumprod(1.0 + r)
    return pd.DataFrame(cols, index=idx)
