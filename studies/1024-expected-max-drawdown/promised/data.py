"""Data layer for Study 1024 — The Drawdown You Were Promised.

Four real tapes, all frozen inside Python packages and SHA-256 pinned by
:mod:`quantlab.bundled` — no network, no Yahoo, no cache that drifts:

==================  ===========================================  ======================  ==========
tape                what it is                                   span used               frequency
==================  ===========================================  ======================  ==========
``sp500``           S&P 500 **price index** (skfolio)            1990-01-02 → 2022-11-30  daily
``nasdaq``          Nasdaq Composite **price index** (arch,      1999-01-04 → 2018-12-31  daily
                    ``Adj Close``; an index, so = ``Close``)
``ff_market``       Fama-French US market, **total return**      1926-07 → 2018-11        monthly
                    (``Mkt-RF + RF``, arch ``frenchdata``)
``stocks``          20 large US stocks, adjusted closes —        1990-01-02 → 2022-11-30  daily
                    a **survivor sample** (skfolio)
==================  ===========================================  ======================  ==========

Two labels matter for a drawdown study and are carried everywhere:

* **Price index vs total return.** The two daily indices exclude dividends, so their
  drawdowns are slightly *deeper* and their drifts slightly *lower* than an investor's. That
  bias is the same for the promise and for the outcome (both are computed from the same
  series), so it does not tilt the coverage test — but it does move the levels.
* **Survivors.** The 20 stocks were large and alive in 2022. Names that drew down 100% are
  not in the panel, so every realised stock drawdown here is a **lower bound** on what a
  random large-cap of 1990 delivered, and a breach rate measured on them understates the
  model's failure.

``AS_OF`` is the last *complete* month of the newest tape: the skfolio tapes stop on
2022-12-28, so December 2022 is partial and dropped. Window construction uses whole calendar
years only, so no partial year ever enters a test.

The synthetic generator (``synthetic_returns``) is a GARCH(1,1) with Student-*t* shocks whose
``signal_strength`` knob scales the two things a Gaussian drawdown formula ignores: at ``0.0``
the returns are **i.i.d. Gaussian** — the world in which the textbook formula is exactly right
and must be shown to be calibrated — and at ``1.0`` they carry equity-like volatility
clustering (GARCH persistence 0.997, ARCH loading 0.10) and fat tails (*t* with 4
degrees of freedom) at the **same
unconditional mean and variance**.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from quantlab import bundled  # noqa: E402

TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12

# Last complete month of each tape. December 2022 is partial in skfolio (ends 12-28).
AS_OF_TAPE = {
    "sp500": "2022-11-30",
    "nasdaq": "2018-12-31",
    "ff_market": "2018-11-30",
    "stocks": "2022-11-30",
}
AS_OF = "2022-11-30"

# Steps per year by tape — the simulation samples at the tape's own frequency.
PERIODS_PER_YEAR = {
    "sp500": TRADING_DAYS_PER_YEAR,
    "nasdaq": TRADING_DAYS_PER_YEAR,
    "ff_market": MONTHS_PER_YEAR,
    "stocks": TRADING_DAYS_PER_YEAR,
}

LABELS = {
    "sp500": "S&P 500 price index, daily (skfolio 1.8.1)",
    "nasdaq": "Nasdaq Composite price index, daily (arch)",
    "ff_market": "Fama-French US market total return, monthly (arch frenchdata)",
    "stocks": "20 large US stocks, adjusted closes, daily — SURVIVOR sample (skfolio 1.8.1)",
}
SOURCES = {
    "sp500": ("skfolio", "sp500_index"),
    "nasdaq": ("arch", "nasdaq"),
    "ff_market": ("arch", "frenchdata"),
    "stocks": ("skfolio", "sp500_dataset"),
}
INDEX_TAPES = ("sp500", "nasdaq", "ff_market")


# --------------------------------------------------------------------------- #
# Availability and provenance
# --------------------------------------------------------------------------- #
def have_real() -> bool:
    """True when every tape this study reads is available offline."""
    try:
        return bool(bundled.have_arch("nasdaq") and bundled.have_arch("frenchdata")
                    and bundled.have_skfolio())
    except Exception:  # pragma: no cover - defensive
        return False


def sha_pin(tape: str) -> str:
    """The SHA-256 pin (first 12 hex) of the file a tape is read from."""
    src, name = SOURCES[tape]
    if src == "arch":
        return bundled.ARCH_FILES[name][2][:12]
    return bundled.SKFOLIO_FILES[name][1][:12]


def fingerprint(obj) -> str:
    """Content fingerprint (12 hex) of a frame or series — `quantlab.bundled.fingerprint`."""
    return bundled.fingerprint(obj)


# --------------------------------------------------------------------------- #
# Real tapes — each returns a LOG-return series (or frame) cut at its as-of
# --------------------------------------------------------------------------- #
def _cut(obj, tape):
    return obj[obj.index <= pd.Timestamp(AS_OF_TAPE[tape])]


def load_sp500() -> pd.Series:
    """Daily log returns of the S&P 500 **price index**, 1990-01-03 → 2022-11-30."""
    px = _cut(bundled.load_skfolio("sp500_index")["SP500"].dropna(), "sp500")
    r = np.log(px).diff().dropna()
    r.name = "sp500"
    return r


def load_nasdaq() -> pd.Series:
    """Daily log returns of the Nasdaq Composite **price index**, 1999-01-05 → 2018-12-31."""
    px = _cut(bundled.load_arch("nasdaq")["Adj Close"].dropna(), "nasdaq")
    r = np.log(px).diff().dropna()
    r.name = "nasdaq"
    return r


def load_ff_market() -> pd.Series:
    """Monthly log **total** returns of the Fama-French US market, 1926-07 → 2018-11."""
    ff = _cut(bundled.ff_monthly_total_return(), "ff_market")
    r = np.log1p(ff["mkt"]).dropna()
    r.name = "ff_market"
    return r


def load_stocks() -> pd.DataFrame:
    """Daily log returns of the 20-stock **survivor** panel, 1990-01-03 → 2022-11-30."""
    px = _cut(bundled.load_skfolio("sp500_dataset"), "stocks")
    return np.log(px).diff().iloc[1:]


def load_tape(tape: str):
    """Dispatch by tape name (see ``AS_OF_TAPE``)."""
    return {"sp500": load_sp500, "nasdaq": load_nasdaq,
            "ff_market": load_ff_market, "stocks": load_stocks}[tape]()


def load_all() -> dict:
    """Every tape as log returns, keyed as in ``AS_OF_TAPE``."""
    return {t: load_tape(t) for t in AS_OF_TAPE}


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core
# --------------------------------------------------------------------------- #
def synthetic_returns(
    n_years: int = 60,
    signal_strength: float = 1.0,
    mu_ann: float = 0.06,           # annual LOG drift
    vol_ann: float = 0.16,          # annual unconditional volatility
    persistence: float = 0.997,     # alpha + beta at signal_strength = 1
    alpha: float = 0.10,            # ARCH loading at signal_strength = 1
    nu: float = 4.0,                # Student-t degrees of freedom at signal_strength = 1
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    start: str = "1950-01-02",
    seed: int = 1024,
) -> tuple[pd.Series, dict]:
    """Daily log returns from a GARCH(1,1)-*t* whose departure from Gaussian is dialled.

    ``signal_strength`` scales both the volatility clustering and the tail fatness:

    * ARCH loading ``alpha * s`` and persistence ``persistence * s`` (so ``beta`` scales too);
    * Student-*t* degrees of freedom ``nu / s`` (infinite — i.e. Gaussian — at ``s = 0``).

    The unconditional mean and variance are held fixed at ``mu_ann`` and ``vol_ann`` for every
    ``s``, so the moments a Gaussian risk model "knows" are identical across worlds; only the
    *shape* the model ignores moves. At ``s = 0`` the series is exactly i.i.d.
    ``N(mu, sigma^2)``. Deterministic given ``seed``; returns ``(returns, truth)``.
    """
    s = float(np.clip(signal_strength, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    n = int(n_years * periods_per_year)
    mu = mu_ann / periods_per_year
    var = vol_ann ** 2 / periods_per_year
    a = alpha * s
    p = persistence * s
    b = max(p - a, 0.0)
    omega = var * (1.0 - a - b)
    if s > 0:
        dof = nu / s
        z = rng.standard_t(dof, n) * np.sqrt((dof - 2.0) / dof)
    else:
        dof = np.inf
        z = rng.standard_normal(n)
    r = np.empty(n)
    s2 = var
    for t in range(n):
        e = np.sqrt(s2) * z[t]
        r[t] = mu + e
        s2 = omega + a * e * e + b * s2
    if periods_per_year == MONTHS_PER_YEAR:
        idx = pd.date_range(start=start, periods=n, freq="MS") + pd.offsets.MonthEnd(0)
    else:
        idx = pd.bdate_range(start=start, periods=n)
    ret = pd.Series(r, index=pd.DatetimeIndex(idx, name="date"), name="synthetic")
    truth = {"signal_strength": s, "mu_ann": mu_ann, "vol_ann": vol_ann,
             "mu": mu, "sigma": float(np.sqrt(var)), "alpha": a, "beta": b,
             "persistence": a + b, "nu": dof, "n": n, "seed": seed,
             "periods_per_year": periods_per_year}
    return ret, truth
