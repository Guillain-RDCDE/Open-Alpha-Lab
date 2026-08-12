"""Data layer for Study 867 — Currency Crash Risk (Brunnermeier-Nagel-Pedersen).

The claim under test (Brunnermeier, Nagel & Pedersen 2008, *"Carry Trades and
Currency Crashes"*): **high-interest-rate (carry) currencies are exposed to CRASH
risk**. Their returns are **negatively skewed** — carry "goes up by the stairs and
down by the elevator": long stretches of gentle appreciation punctuated by sudden,
violent depreciations (unwinds). So (a) the *higher* a currency's carry, the *more
negative* its realized return skewness, and (b) a long-high / short-low carry basket
carries a deep negative skew — the premium is compensation for a sold-crash tail.

Two ingredients, both offline-friendly once cached under the study's own ``_cache/``:

* **Real tape — 8 currencies vs USD.** Daily FX spot for a fixed basket
  (EUR, GBP, JPY, AUD, CAD, CHF, NZD, **MXN** — the last a notorious high-carry
  crash currency), pulled with yfinance (no key), resampled to **weekly** (Friday
  last) — the natural Brunnermeier-et-al frequency, with enough observations to
  estimate a third moment. yfinance quotes some pairs USD-base (``JPY=X`` = USDJPY =
  JPY per USD, ``MXN=X`` = USDMXN) and others foreign-base (``EURUSD=X`` = USD per
  EUR); every pair is normalised to **USD-per-foreign-currency** so a *rise* means the
  foreign currency **appreciated** and a USD-funded long earns a positive spot return.
  The weekly panel is cached as a parquet.

  **The carry proxy.** True overnight-deposit / forward-discount differentials are not
  on yfinance, so the interest carry attached to each currency is a fixed, transparent
  per-currency long-run average annualised short-rate differential vs USD (a **proxy**,
  named as such everywhere). The high-yielders (MXN, NZD, AUD) sit on top, the funding
  currencies (JPY, CHF) deeply negative. The crash verdict rests on the realized
  **skewness** measured off spot, not on the exact carry values — only their ordering.

* **Synthetic world — the positive control.** A deterministic, seeded panel of
  currencies with carries evenly spaced across the cross-section, driven by a common
  **risk-off factor** on which each currency loads in proportion to its carry (high
  carry = high positive loading, so it falls hardest in a risk-off event). A TUNABLE
  knob ``edge`` controls the **fat negative tail** of that factor: at ``edge = 0`` the
  factor is symmetric and NO currency is skewed (the null — carry present but no crash
  asymmetry); at ``edge > 0`` the factor gets occasional violent down-jumps, so the
  high-carry currencies become **negatively skewed** and the skew-carry slope goes
  negative — the planted Brunnermeier-et-al pattern the machinery must recover.

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
FX_CACHE = os.path.join(CACHE_DIR, "fx_weekly.parquet")

# The 8-currency basket vs USD. yfinance serves some pairs foreign-base (``EURUSD=X``
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
    "MXN=X":    ("MXN", True),    # MXN per USD      -> invert
}

CURRENCIES = ["EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "MXN"]

# Transparent fixed carry PROXY: a long-run average annualised short-rate differential
# vs USD (in percent) — the interest a USD investor earns by holding that currency.
# Deliberately round numbers reflecting the 2004-2024 ordering (EM high-yielder MXN on
# top, commodity AUD/NZD next, funding JPY/CHF deeply negative). Used ONLY to rank the
# cross-section and to attach a carry leg to the basket; the crash verdict rests on the
# realized skewness measured off spot, not on the exact carry magnitudes.
CARRY_PROXY = {
    "MXN": 5.5, "NZD": 1.8, "AUD": 1.6, "GBP": 0.4,
    "CAD": 0.2, "EUR": -0.4, "CHF": -1.4, "JPY": -1.6,
}

START = "2003-12-01"        # yfinance FX history broadly begins here
AS_OF = "2026-06-30"        # last complete calendar month at publication
PERIODS_PER_YEAR = 52       # weekly sampling

__all__ = [
    "CURRENCIES", "FX_TICKERS", "CARRY_PROXY", "START", "AS_OF",
    "CACHE_DIR", "FX_CACHE", "PERIODS_PER_YEAR",
    "fetch", "have_real", "load_panel", "load_series", "load_real", "synthetic_panel",
]


# --------------------------------------------------------------------------- #
# Real tape — network fetch (once), then offline load
# --------------------------------------------------------------------------- #
def fetch(start: str = START, tries: int = 4) -> pd.DataFrame:
    """Download the FX basket via yfinance and cache a weekly USD-per-foreign parquet.

    Network-only; used once to build ``_cache/fx_weekly.parquet``. Never imported by the
    offline notebook cells. Retries up to ``tries`` times with a short sleep on failure.
    ``auto_adjust=True``. Each pair is normalised to USD-per-foreign-currency, then
    resampled to weekly (Friday, last observation of each week).
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
    weekly = df.resample("W-FRI").last()[CURRENCIES]
    weekly.index.name = "date"
    weekly.to_parquet(FX_CACHE)
    return weekly


def have_real() -> bool:
    return os.path.exists(FX_CACHE)


def load_panel(start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Cached weekly USD-per-foreign panel, sliced to ``[start, asof]``.

    Index = week-end (Friday) date, columns = ``CURRENCIES``. Reads the parquet directly
    — OFFLINE, no yfinance import.
    """
    df = pd.read_parquet(FX_CACHE).sort_index()
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    df = df.loc[(df.index >= lo) & (df.index <= hi)]
    return df[[c for c in CURRENCIES if c in df.columns]].copy()


def load_series(start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Alias kept for symmetry with the sibling studies (same object as load_panel)."""
    return load_panel(start, asof)


def load_real(start: str = START, asof: str = AS_OF,
              carry: dict[str, float] | None = None) -> dict:
    """Convenience bundle: cached weekly spot -> weekly spot/total returns + carry map.

    ``total_ret`` = spot return + carry/100/52 (a USD-funded *long* position's weekly
    total return). Mirrors the synthetic bundle so the strategy is source-agnostic.
    """
    carry = CARRY_PROXY if carry is None else carry
    spot = load_panel(start, asof)
    spot_ret = spot.pct_change().dropna(how="all")
    carry_w = pd.Series({c: carry.get(c, 0.0) / 100.0 / PERIODS_PER_YEAR
                         for c in spot.columns})
    total_ret = spot_ret.add(carry_w, axis=1)
    return {"spot": spot, "spot_ret": spot_ret, "total_ret": total_ret,
            "carry": {c: carry.get(c, 0.0) for c in spot.columns}}


# --------------------------------------------------------------------------- #
# Synthetic world — planted carry-crash (skew-carry) relation (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 867,
    n_weeks: int = 1100,
    n_ccy: int = 8,
    carry_spread: float = 0.06,
    idio_vol: float = 0.010,
    factor_vol: float = 0.010,
    crash_prob: float = 0.03,
) -> dict:
    """Deterministic seeded weekly panel with a TUNABLE planted carry-crash relation.

    ``n_ccy`` currencies carry evenly spaced annualised carries over
    ``[-carry_spread/2, +carry_spread/2]``. A common **risk-off factor** ``f_t`` drives
    every currency; the loading ``beta_i`` is proportional to the currency's carry, so a
    high-carry currency *appreciates* gently with the factor and *crashes* with it. The
    factor's downside tail is the knob:

        f_t = N(0, factor_vol)                     if edge == 0  (symmetric — the null)
        f_t = N(0, factor_vol) - crash_jump_t      if edge  > 0  (fat negative tail)

    where ``crash_jump_t`` fires with probability ``crash_prob`` and has size
    ``edge * |t_3|`` (a fat-tailed Student-t draw). With ``edge > 0`` the high-carry
    currencies inherit the factor's negative tail -> **negative skew that deepens with
    carry** (the Brunnermeier-Nagel-Pedersen pattern). ``edge = 0`` is the null: the
    factor is symmetric, no currency is skewed, and the skew-carry slope is ~0.

    Returns the same bundle shape as :func:`load_real` (``spot``, ``spot_ret``,
    ``total_ret``, ``carry``) on a plain integer RangeIndex — no timestamp horizon risk.
    """
    rng = np.random.default_rng(seed)
    if n_ccy > 1:
        carries = np.linspace(-carry_spread / 2.0, carry_spread / 2.0, n_ccy)
    else:
        carries = np.zeros(n_ccy)
    ccy = [f"C{i}" for i in range(n_ccy)]
    carry_map = {c: float(k * 100.0) for c, k in zip(ccy, carries)}
    # loadings proportional to carry (unit spread), centred so the factor is ~dollar-neutral
    beta = np.linspace(0.4, 1.6, n_ccy) if n_ccy > 1 else np.ones(n_ccy)

    base = rng.normal(0.0, factor_vol, size=n_weeks)
    if edge > 0:
        fires = rng.random(n_weeks) < crash_prob
        jumps = np.abs(rng.standard_t(3, size=n_weeks)) * edge
        factor = base - np.where(fires, jumps, 0.0)
    else:
        factor = base

    spot_ret = np.empty((n_weeks, n_ccy))
    for j in range(n_ccy):
        idio = rng.normal(0.0, idio_vol, size=n_weeks)
        spot_ret[:, j] = beta[j] * factor + idio

    idx = pd.RangeIndex(n_weeks)
    spot_ret_df = pd.DataFrame(spot_ret, index=idx, columns=ccy)
    carry_w = pd.Series({c: carry_map[c] / 100.0 / PERIODS_PER_YEAR for c in ccy})
    total_ret_df = spot_ret_df.add(carry_w, axis=1)
    spot_lvl = 100.0 * (1.0 + spot_ret_df).cumprod()
    return {"spot": spot_lvl, "spot_ret": spot_ret_df, "total_ret": total_ret_df,
            "carry": carry_map}
