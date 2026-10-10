"""Data layer for Study 1020 — More Volatile Than Ever?

Three frozen tapes, each chosen for a job the others cannot do, all read through
:mod:`quantlab.bundled` (shipped inside the ``arch`` and ``skfolio`` packages, SHA-256 pinned,
no network):

``load_monthly``  — Fama-French monthly market return (``Mkt-RF + RF``), **total return**,
    1926-07 → 2018-11. The longest view on the desk: ninety-one complete calendar years, the
    1929-32 collapse included. Monthly data cannot see a single crash *day*, so it is used for
    volatility levels and trends, never for daily extremes.
``load_daily``    — skfolio's daily S&P 500 **price index** (no dividends), 1990-01-02 →
    2021-12-31. The tape for realised volatility per year, the extreme-day counts and the
    points-versus-percent mechanism. The source runs to 2022-12-28; the incomplete year 2022 is
    dropped so every year counted is a whole year.
``load_ohlc``     — arch's daily S&P 500 Open/High/Low/Close, **price index**, 1999-01-04 →
    2018-12-31. A second, independent estimator of daily volatility from the intraday range
    (Parkinson 1980), so a trend that exists only in close-to-close returns gets caught.

Price index versus total return matters less here than in most studies — dividends move the
*level* of a return, not its dispersion — but it matters for the points-versus-percent section,
where the index level is the whole point, and it is labelled wherever a number appears.

``synthetic_returns`` is the deterministic offline world the test-suite runs on: a stationary
stochastic-volatility process (log volatility is a persistent AR(1)) with an optional upward
trend in log volatility planted on top. ``signal_strength = 1`` makes volatility end the sample
``planted_multiple`` times higher than it started; ``signal_strength = 0`` is the matched null,
the same persistent process with no trend at all — which is exactly the null under which naive
trend tests on volatility misfire.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from quantlab import bundled  # noqa: E402

TRADING_DAYS = 252
MONTHS = 12

# --------------------------------------------------------------------------- #
# As-of stamps — the last COMPLETE period of each tape, never in the future.
# --------------------------------------------------------------------------- #
AS_OF_MONTHLY = "2018-11-30"      # Fama-French ends 2018-11 (a complete month)
LAST_FULL_YEAR_MONTHLY = 2017     # 2018 has only 11 months -> annual stats stop at 2017
FIRST_FULL_YEAR_MONTHLY = 1927    # 1926 starts in July -> annual stats start at 1927
AS_OF_DAILY = "2021-12-31"        # skfolio ends 2022-12-28: partial 2022 dropped
AS_OF_OHLC = "2018-12-31"         # arch sp500 ends on the last session of 2018
AS_OF = AS_OF_DAILY               # the study-wide stamp: the latest tape's last full year

# Eras for the long view (inclusive calendar years). The last one runs to the tape's end.
ERAS = (("1926-1945", 1926, 1945), ("1946-1969", 1946, 1969),
        ("1970-1989", 1970, 1989), ("1990-2018", 1990, 2018))

TAPES = {
    "monthly": {"package": "arch", "dataset": "frenchdata",
                "file": bundled.ARCH_FILES["frenchdata"][1],
                "sha256": bundled.ARCH_FILES["frenchdata"][2],
                "label": "Fama-French Mkt-RF + RF, monthly, TOTAL return",
                "as_of": AS_OF_MONTHLY},
    "daily": {"package": "skfolio 1.8.1", "dataset": "sp500_index",
              "file": bundled.SKFOLIO_FILES["sp500_index"][0],
              "sha256": bundled.SKFOLIO_FILES["sp500_index"][1],
              "label": "S&P 500 index level, daily, PRICE index (no dividends)",
              "as_of": AS_OF_DAILY},
    "ohlc": {"package": "arch", "dataset": "sp500",
             "file": bundled.ARCH_FILES["sp500"][1],
             "sha256": bundled.ARCH_FILES["sp500"][2],
             "label": "S&P 500 Open/High/Low/Close, daily, PRICE index",
             "as_of": AS_OF_OHLC},
}


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
def have_monthly() -> bool:
    return bundled.have_arch("frenchdata")


def have_daily() -> bool:
    return bundled.have_skfolio()


def have_ohlc() -> bool:
    return bundled.have_arch("sp500")


def have_real() -> bool:
    """True iff all three tapes this study reads are available offline."""
    return have_monthly() and have_daily() and have_ohlc()


# --------------------------------------------------------------------------- #
# Real tapes
# --------------------------------------------------------------------------- #
def load_monthly(asof: str = AS_OF_MONTHLY) -> pd.Series:
    """Monthly US market **total** return (decimal, simple), 1926-07 → ``asof``.

    ``Mkt-RF + RF`` from Ken French's library as frozen inside ``arch``. Month-end index.
    """
    ff = bundled.ff_monthly_total_return()
    s = ff["mkt"].rename("mkt")
    return s[s.index <= pd.Timestamp(asof)].dropna()


def load_daily(asof: str = AS_OF_DAILY) -> pd.Series:
    """Daily S&P 500 **price index** level, 1990-01-02 → ``asof`` (partial 2022 dropped)."""
    df = bundled.load_skfolio("sp500_index")
    s = df["SP500"].rename("spx").dropna()
    return s[s.index <= pd.Timestamp(asof)]


def load_ohlc(asof: str = AS_OF_OHLC) -> pd.DataFrame:
    """Daily S&P 500 OHLC (**price index**), lower-case columns open/high/low/close."""
    df = bundled.load_arch("sp500")
    out = df[["Open", "High", "Low", "Close"]].rename(columns=str.lower)
    return out[out.index <= pd.Timestamp(asof)].dropna()


def fingerprint(obj) -> str:
    """Short content fingerprint of a frame or series (the desk's ``bundled.fingerprint``)."""
    return bundled.fingerprint(obj)


def provenance() -> list[dict]:
    """One row per tape: package, file, SHA-256 pin, span, rows, content fingerprint."""
    rows = []
    loaders = {"monthly": (have_monthly, load_monthly), "daily": (have_daily, load_daily),
               "ohlc": (have_ohlc, load_ohlc)}
    for key, meta in TAPES.items():
        have, load = loaders[key]
        if not have():
            continue
        x = load()
        rows.append({"tape": key, **meta, "rows": int(len(x)),
                     "first": str(x.index[0].date()), "last": str(x.index[-1].date()),
                     "fingerprint": fingerprint(x)})
    return rows


# --------------------------------------------------------------------------- #
# Synthetic — stationary stochastic volatility, optionally with a planted trend
# --------------------------------------------------------------------------- #
def synthetic_returns(n_years: int = 90, freq: str = "M", signal_strength: float = 1.0,
                      planted_multiple: float = 3.0, base_vol: float = 0.16,
                      logvol_sd: float = 0.35, monthly_phi: float = 0.95,
                      drift: float = 0.07, seed: int = 1020) -> tuple[pd.DataFrame, dict]:
    """Returns from a stochastic-volatility world with a KNOWN trend in log volatility.

    ``log σ_t = log(base_vol) + x_t + g·τ_t`` where ``x_t`` is a stationary AR(1) with
    persistence ``monthly_phi`` per month (rescaled to the daily clock when ``freq="D"``) and
    stationary standard deviation ``logvol_sd``; ``τ_t`` is time in years and
    ``g = signal_strength · log(planted_multiple) / n_years``. At ``signal_strength = 1`` the
    volatility *trend* multiplies by ``planted_multiple`` from first to last observation; at
    ``0`` there is no trend and only the persistent wandering a real volatility series has —
    the null on which a naive OLS trend test over-rejects.

    Returns ``(frame, truth)``. The frame has ``ret`` (simple return), ``logret``, ``true_vol``
    (annualised) and ``price`` (a level starting at 100 with annual ``drift`` — used by the
    points-versus-percent section). Deterministic given ``seed``; the date index starts in
    1926 for monthly and 1990 for daily so it stays inside pandas' nanosecond horizon.
    """
    from scipy.signal import lfilter

    freq = freq.upper()
    if freq not in ("M", "D"):
        raise ValueError("freq must be 'M' or 'D'")
    per_year = MONTHS if freq == "M" else TRADING_DAYS
    n = int(n_years * per_year)
    phi = monthly_phi if freq == "M" else monthly_phi ** (MONTHS / TRADING_DAYS)
    eta = logvol_sd * np.sqrt(1.0 - phi ** 2)

    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, eta, n)
    x = lfilter([1.0], [1.0, -phi], shocks)
    x = x + rng.normal(0.0, logvol_sd) * phi ** np.arange(1, n + 1)   # stationary start
    tau = np.arange(n) / per_year
    g = float(signal_strength) * np.log(planted_multiple) / n_years
    log_sig = np.log(base_vol) + x + g * tau
    vol = np.exp(log_sig)
    z = rng.standard_normal(n)
    sig_p = vol / np.sqrt(per_year)
    logret = (drift / per_year - 0.5 * sig_p ** 2) + sig_p * z
    ret = np.expm1(logret)
    price = 100.0 * np.exp(np.cumsum(logret))

    if freq == "M":
        idx = pd.date_range("1926-07-31", periods=n, freq="ME")
    else:
        idx = pd.bdate_range("1990-01-02", periods=n)
    idx = pd.DatetimeIndex(idx, name="date")
    frame = pd.DataFrame({"ret": ret, "logret": logret, "true_vol": vol, "price": price},
                         index=idx)
    truth = {"n_years": n_years, "freq": freq, "n_obs": n, "seed": seed,
             "signal_strength": float(signal_strength), "planted_multiple": planted_multiple,
             "trend_log_per_year": g, "base_vol": base_vol, "phi": phi,
             "logvol_sd": logvol_sd, "drift": drift}
    return frame, truth
