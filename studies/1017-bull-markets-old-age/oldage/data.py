"""Data layer for Study 1017 — Do Bull Markets Die of Old Age?

Two real tapes, both frozen inside Python packages and SHA-256 pinned by
:mod:`quantlab.bundled` (no network, no Yahoo):

- ``load_ff_monthly`` — the Fama-French monthly market return, **total return** (``Mkt-RF + RF``,
  dividends included), with the one-month T-bill ``rf``, 1926-07 → 2018-11. Ninety-two years and
  the only sample on this desk with enough bull markets to estimate a hazard at all. Shipped
  inside ``arch``.
- ``load_sp500_daily`` — the daily S&P 500 **price index** (no dividends), 1990-01-02 →
  2022-11-30, shipped inside ``skfolio`` 1.8.1. The partial December 2022 is dropped so the stamped
  run never contains an in-progress month. Used as an out-of-sample, finer-grained cross-check —
  it holds only a handful of complete bulls, which is the honest constraint of this whole topic.

Plus ``synthetic_monthly`` — a deterministic, offline generator with a ``signal_strength`` knob.
At ``0.0`` it is an i.i.d. lognormal random walk (the exact null this study tests against). At
``1.0`` a genuine ageing mechanism is planted: once a (causally tracked) bull passes a planted age,
each month carries a rising probability of a crash large enough to end it — bulls really do die
of old age there, at a known rate. Every strategy test runs on it.
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

MONTHS_PER_YEAR = 12
TRADING_DAYS_PER_YEAR = 252

# Last FULL period of each tape. The FF tape ends on a complete month (2018-11); the skfolio
# S&P tape ends 2022-12-28, mid-month, so its stamped run stops at the last full month.
AS_OF_FF = "2018-11-30"
AS_OF_SP = "2022-11-30"
AS_OF = AS_OF_SP            # the latest data point any number in this study rests on

FF_SOURCE = "arch (bundled) · frenchdata.csv.gz · Fama-French Mkt-RF + RF, monthly, percent"
SP_SOURCE = "skfolio 1.8.1 (bundled wheel) · sp500_index.csv.gz · S&P 500 price index, daily"


# --------------------------------------------------------------------------- #
# Real tapes
# --------------------------------------------------------------------------- #
def have_real() -> bool:
    """True iff both tapes this study needs are available offline."""
    return bool(bundled.have_arch("frenchdata") and bundled.have_skfolio())


def have_ff() -> bool:
    """True iff the Fama-French monthly tape (inside ``arch``) is available."""
    return bool(bundled.have_arch("frenchdata"))


def load_ff_monthly(asof: str = AS_OF_FF) -> pd.DataFrame:
    """Monthly decimal returns: ``mkt`` (**total return**), ``rf`` (T-bill), ``mkt_rf``.

    Indexed by month-end, sliced to ``asof``. ``mkt`` is ``Mkt-RF + RF`` — a value-weighted
    total-return market, dividends in, which is what a bull market should be dated on if the
    question is about an investor's wealth rather than a price chart.
    """
    ff = bundled.ff_monthly_total_return()[["mkt", "rf", "mkt_rf"]]
    return ff[ff.index <= pd.Timestamp(asof)].copy()


def load_sp500_daily(asof: str = AS_OF_SP) -> pd.Series:
    """Daily S&P 500 **price index** (dividends NOT included), sliced to ``asof``."""
    s = bundled.load_skfolio("sp500_index")["SP500"].dropna()
    s = s[s.index <= pd.Timestamp(asof)]
    s.name = "SP500"
    return s


def total_return_index(r: pd.Series, base: float = 1.0) -> pd.Series:
    """Compound simple returns into a level index starting at ``base``."""
    return base * (1.0 + r.astype(float)).cumprod()


def fingerprint(obj) -> str:
    """Short content fingerprint (12 hex) — the desk-wide :func:`quantlab.bundled.fingerprint`."""
    return bundled.fingerprint(obj)


def provenance() -> list[dict]:
    """Package, file and SHA-256 pin of each tape — printed into ``docs/results.md``."""
    return [
        {"tape": "Fama-French monthly (total return)", "source": FF_SOURCE,
         "sha256": bundled.ARCH_FILES["frenchdata"][2], "as_of": AS_OF_FF,
         "label": "total return, nominal, USD"},
        {"tape": "S&P 500 daily", "source": SP_SOURCE,
         "sha256": bundled.SKFOLIO_FILES["sp500_index"][1], "as_of": AS_OF_SP,
         "label": "price index (no dividends), nominal, USD"},
    ]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core
# --------------------------------------------------------------------------- #
def synthetic_monthly(
    n_months: int = 1100,
    signal_strength: float = 1.0,
    mu: float = 0.0075,            # monthly mean LOG return (≈ the FF market's)
    sigma: float = 0.052,          # monthly log-return vol
    rf: float = 0.0025,            # flat monthly bill rate
    threshold: float = 0.20,       # the dating threshold the planted clock reads
    onset_months: int = 36,        # age at which planted ageing starts
    ramp: float = 0.04,            # extra monthly death probability per year beyond onset
    crash: float = -0.30,          # log size of the planted bull-ending crash
    start: str = "1926-07-31",
    seed: int = 1017,
) -> tuple[pd.DataFrame, dict]:
    """A monthly market with **planted, known** duration dependence in its bull markets.

    The base process is an i.i.d. normal log-return random walk with drift ``mu`` and vol
    ``sigma``. A causal bull/bear clock (the same ±``threshold`` rule the study dates with,
    run in real time) tracks the age of the current bull. Once the age passes
    ``onset_months``, every month carries a probability

        ``h(age) = signal_strength × ramp × (age − onset) / 12``   (capped at 0.6)

    of a ``crash`` (a −26% simple-return month) that ends the bull. At ``signal_strength=0`` the
    crash never fires and the series is **exactly** the random-walk null; at ``1.0`` an old bull
    is genuinely more likely to die, and its next-12-month return is genuinely lower.

    Returns ``(df, truth)`` with ``df`` columns ``mkt`` (simple return), ``rf`` and ``mkt_rf``
    on a month-end index (inside pandas' ns horizon), and the planted parameters in ``truth``.
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(mu, sigma, n_months)
    u = rng.random(n_months)
    s = float(signal_strength)
    logr = np.empty(n_months)
    level = 1.0
    state, hi, lo = 0, 1.0, 1.0          # 0 undetermined, +1 bull, -1 bear
    trough_i = 0
    n_crash = 0
    for t in range(n_months):
        x = eps[t]
        if state == 1 and s > 0:
            age = t - trough_i
            h = min(0.6, s * ramp * max(0.0, age - onset_months) / 12.0)
            if u[t] < h:
                x = x + crash
                n_crash += 1
        logr[t] = x
        level *= np.exp(x)
        # causal clock — identical logic to strategy.realtime_state
        if state == 0:
            if level >= lo * (1 + threshold):
                state, hi = 1, level
            elif level <= hi * (1 - threshold):
                state, lo = -1, level
            else:
                if level > hi:
                    hi = level
                if level < lo:
                    lo, trough_i = level, t
        elif state == 1:
            if level > hi:
                hi = level
            elif level <= hi * (1 - threshold):
                state, lo = -1, level
                trough_i = t
        else:
            if level < lo:
                lo, trough_i = level, t
            elif level >= lo * (1 + threshold):
                state, hi = 1, level
    idx = pd.date_range(start=start, periods=n_months, freq=pd.offsets.MonthEnd())
    mkt = np.expm1(logr)
    df = pd.DataFrame({"mkt": mkt, "rf": np.full(n_months, rf)}, index=idx)
    df["mkt_rf"] = df["mkt"] - df["rf"]
    df.index.name = "date"
    truth = {"n_months": n_months, "signal_strength": s, "mu": mu, "sigma": sigma,
             "rf": rf, "threshold": threshold, "onset_months": onset_months,
             "ramp": ramp, "crash": crash, "n_crash": int(n_crash), "seed": seed}
    return df, truth
