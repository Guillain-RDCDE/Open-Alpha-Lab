"""Data layer for Study 1019 — The Market Runs on Volume Time.

Two real tapes, one synthetic world, one shape (``open, high, low, close, volume``).

**Real tapes — frozen inside the ``arch`` package.** ``load_arch("sp500")`` and
``load_arch("nasdaq")`` from :mod:`quantlab.bundled`: daily OHLC, close and volume for the
S&P 500 and the Nasdaq Composite, 1999-01-04 → 2018-12-31, SHA-256 pinned so a package upgrade
that swaps the bytes fails loudly instead of moving a published number. Three labels travel
with every number this study prints:

* these are **price indices** — dividends are not in them (``Adj Close`` equals ``Close``);
* the ``Volume`` column is the **composite volume Yahoo! Finance reports for the index**, not a
  futures or ETF volume: it is a sum of share counts across constituents, so it drifts with
  share issuance, splits and venue fragmentation as well as with trading interest. That is why
  the study never uses the raw level — only volume relative to its own *past-only* rolling
  median;
* the Nasdaq tape carries two sessions with a volume of exactly zero; they are set to missing,
  never filled.

The early S&P 500 open prints are a known Yahoo artefact (on ~40% of 1999-2002 sessions the
open equals the previous close), so nothing here uses the open. The range estimator reads only
high and low.

**Synthetic world — a subordinated process with a dial.** ``synthetic_tape`` draws daily
returns as ``N(0, sigma^2 * m_t)`` — Gaussian conditional on a random variance multiplier
``m_t`` — realised through an intraday Brownian path so that high and low exist too. The
volume series is generated from a *clock* ``c_t``. ``signal_strength`` decides how much of
``log m_t`` is that clock:

    log m_t = sqrt(s) * clock_t + sqrt(1 - s) * other_t        (both unit-scaled, independent)

At ``s = 1`` the variance runs entirely through the clock — Clark's world. At ``s = 0`` the
variance runs on an independent clock of identical statistics, so the raw returns are exactly
as fat-tailed but volume knows nothing about them: the matched null. Because the two
components have the same distribution, the raw kurtosis is the same at every ``s``; only the
link to volume moves.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quantlab import bundled  # noqa: E402

TRADING_DAYS = 252
TAPES = ("sp500", "nasdaq")
TAPE_LABELS = {"sp500": "S&P 500 (price index)", "nasdaq": "Nasdaq Composite (price index)"}
# The tapes end on 2018-12-31, the last session of a complete year — no partial period.
AS_OF = "2018-12-31"
START = "1999-01-04"
SOURCE = "arch (bundled data, via quantlab.bundled.load_arch)"


# --------------------------------------------------------------------------- #
# Real tapes
# --------------------------------------------------------------------------- #
def have_real() -> bool:
    """True iff both arch tapes are present and importable (always so in CI)."""
    return all(bundled.have_arch(t) for t in TAPES)


def load_tape(name: str = "sp500", asof: str = AS_OF) -> pd.DataFrame:
    """One index tape as ``open, high, low, close, volume``, sliced to ``asof``.

    Zero volume is a missing print, not a quiet day: it becomes NaN. Nothing is forward-filled.
    """
    if name not in TAPES:
        raise KeyError(f"unknown tape {name!r}; one of {TAPES}")
    raw = bundled.load_arch(name)
    df = pd.DataFrame({
        "open": raw["Open"], "high": raw["High"], "low": raw["Low"],
        "close": raw["Close"], "volume": raw["Volume"].astype(float),
    })
    df.loc[df["volume"] <= 0, "volume"] = np.nan
    df = df[df.index <= pd.Timestamp(asof)]
    df.index.name = "date"
    return df.sort_index()


def load_all(asof: str = AS_OF) -> dict:
    """Both tapes, keyed by name."""
    return {t: load_tape(t, asof) for t in TAPES}


def load_daily_rf(index: pd.DatetimeIndex) -> pd.Series:
    """Daily cash rate on ``index`` from the Fama-French monthly RF (decimal).

    Each month's RF is spread evenly over that month's sessions in ``index``. The bundled
    French file ends in 2018-11, so **December 2018 carries November's rate forward** — a
    one-month, ~0.2%-a-year approximation that is the same for every overlay compared.
    """
    ff = bundled.ff_monthly_total_return()["rf"]
    month = index.to_period("M")
    rf_m = ff.copy()
    rf_m.index = rf_m.index.to_period("M")
    per_month = pd.Series(1, index=index).groupby(month).transform("count")
    vals = pd.Series(month, index=index).map(rf_m)
    vals = vals.ffill()
    return (vals / per_month).astype(float).rename("rf")


def fingerprint(obj) -> str:
    """Content fingerprint (12 hex chars) — :func:`quantlab.bundled.fingerprint`."""
    return bundled.fingerprint(obj)


def sha_pin(name: str) -> str:
    """The SHA-256 pin of one arch file, as recorded in :mod:`quantlab.bundled`."""
    return bundled.ARCH_FILES[name][2]


def arch_version() -> str:
    try:
        import arch
        return str(arch.__version__)
    except ImportError:  # pragma: no cover
        return "absent"


# --------------------------------------------------------------------------- #
# Synthetic tape — a subordinated process with a signal_strength dial
# --------------------------------------------------------------------------- #
def _clock(rng: np.random.Generator, n: int, phi: float, sd_slow: float,
           sd_fast: float) -> np.ndarray:
    """A persistent AR(1) component plus an i.i.d. daily component, mean zero."""
    eps = rng.normal(0.0, sd_slow * np.sqrt(1.0 - phi ** 2), n)
    slow = np.empty(n)
    slow[0] = rng.normal(0.0, sd_slow)
    for t in range(1, n):
        slow[t] = phi * slow[t - 1] + eps[t]
    return slow + rng.normal(0.0, sd_fast, n)


def synthetic_tape(
    n_days: int = 5000,
    signal_strength: float = 1.0,
    sigma_ann: float = 0.18,          # unconditional annualised vol of the returns
    phi: float = 0.97,                # persistence of the slow clock component
    sd_slow: float = 0.50,            # sd of the slow (clustering) component of log m
    sd_fast: float = 0.45,            # sd of the day-specific component of log m
    volume_noise: float = 0.25,       # sd of log-volume measurement noise around the clock
    volume_trend_ann: float = 0.08,   # deterministic growth of the volume level per year
    n_intraday: int = 26,             # intraday steps used to realise high and low
    start: str = "2000-01-03",
    seed: int = 1019,
) -> tuple[pd.DataFrame, dict]:
    """A daily OHLCV tape from a subordinated process with a known volume clock.

    Returns ``(tape, truth)``. ``tape`` has ``open, high, low, close, volume`` like the real
    loader. Returns are Gaussian *conditional on* ``m_t``; ``signal_strength`` (``s``) sets how
    much of ``log m_t`` the volume clock carries (see the module docstring). Volume carries a
    deterministic upward trend so the detrending step is exercised, and log-normal measurement
    noise so the clock is never observed perfectly.

    Deterministic given ``seed``; the date index stays well inside pandas' ns horizon.
    """
    s = float(np.clip(signal_strength, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    clock = _clock(rng, n_days, phi, sd_slow, sd_fast)
    other = _clock(rng, n_days, phi, sd_slow, sd_fast)
    log_m = np.sqrt(s) * clock + np.sqrt(1.0 - s) * other
    var_log_m = sd_slow ** 2 + sd_fast ** 2
    m = np.exp(log_m - 0.5 * var_log_m)                 # E[m] = 1
    sig_d = sigma_ann / np.sqrt(TRADING_DAYS)

    # Intraday Brownian path: the day's return and its high / low come from the same path.
    steps = rng.normal(0.0, 1.0, (n_days, n_intraday)) * (
        sig_d * np.sqrt(m)[:, None] / np.sqrt(n_intraday))
    path = np.cumsum(steps, axis=1)
    day_ret = path[:, -1]
    hi = np.maximum(path.max(axis=1), 0.0)
    lo = np.minimum(path.min(axis=1), 0.0)
    log_close = np.log(1000.0) + np.cumsum(day_ret)
    prev_close = np.concatenate([[np.log(1000.0)], log_close[:-1]])

    years = np.arange(n_days) / TRADING_DAYS
    log_vol = (np.log(1e9) + volume_trend_ann * years + clock
               + rng.normal(0.0, volume_noise, n_days))

    dates = pd.bdate_range(start=start, periods=n_days)
    tape = pd.DataFrame({
        "open": np.exp(prev_close),
        "high": np.exp(prev_close + hi),
        "low": np.exp(prev_close + lo),
        "close": np.exp(log_close),
        "volume": np.exp(log_vol),
    }, index=pd.DatetimeIndex(dates, name="date"))
    truth = {
        "signal_strength": s, "n_days": n_days, "seed": seed, "sigma_ann": sigma_ann,
        "phi": phi, "sd_slow": sd_slow, "sd_fast": sd_fast, "volume_noise": volume_noise,
        "var_log_m": var_log_m,
        # Excess kurtosis of N(0,1)*sqrt(m) with log m ~ N: 3*(exp(var) - 1).
        "theoretical_excess_kurtosis": float(3.0 * (np.exp(var_log_m) - 1.0)),
        "clock_share_of_log_var": s,
    }
    return tape, truth
