"""Data layer for Study 1016 — Stairs Up, Elevator Down.

Three real tapes, all frozen inside Python packages and pinned by SHA-256 through
:mod:`quantlab.bundled` — no network, no Yahoo, nothing that can drift between reruns:

- ``load_sp500()`` — daily **S&P 500 price index** (``skfolio`` 1.8.1, ``sp500_index``),
  1990-01-02 → 2022-11-30. The tape ends on 2022-12-28, mid-month; December 2022 is dropped so
  every weekly and monthly aggregate is a full period. **Price-only**: no dividends.
- ``load_nasdaq()`` — daily **Nasdaq Composite price index** (``arch``, ``nasdaq``, the
  ``Adj Close`` column), 1999-01-04 → 2018-12-31. **Price-only** as well.
- ``load_ff_market()`` — monthly **Fama-French US market total return** (``Mkt-RF + RF``,
  dividends included) and the one-month T-bill ``rf``, 1926-07 → 2018-11.

The primary tape for the Signal stamp is the S&P 500 daily series: it is the longest daily
record on the desk's bundled shelf and the one the proverb is usually said about. Nasdaq is the
out-of-index replication; the Fama-French monthly series is the long-horizon, total-return one.

``synthetic_returns`` is the deterministic offline world every strategy test runs on: a
GJR-GARCH(1,1) with symmetric Student-t shocks whose **leverage term is scaled by**
``signal_strength``. At ``1.0`` the asymmetry is planted at a size close to what the S&P 500
fit delivers; at ``0.0`` the leverage term is switched off *and* its contribution to
persistence is handed to the symmetric ARCH term, so the null keeps exactly the same
volatility clustering and differs only in the sign asymmetry. That matched null is the point:
a symmetric-shock process with volatility clustering is sign-symmetric, so every
"elevator" statistic must read zero there.
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

# Last FULL period of each tape. skfolio's S&P 500 index stops on 2022-12-28, three sessions
# short of the month-end, so December 2022 is cut. arch's Nasdaq runs to the last session of
# 2018. Ken French's file in arch ends with November 2018.
SP500_AS_OF = "2022-11-30"
NASDAQ_AS_OF = "2018-12-31"
FF_AS_OF = "2018-11-30"
AS_OF = SP500_AS_OF          # study-wide as-of: the latest full period used anywhere

PROVENANCE = {
    "sp500": {"package": "skfolio 1.8.1 (pinned wheel, cached)", "tape": "sp500_index",
              "column": "SP500", "kind": "daily price index (no dividends)",
              "sha256": bundled.SKFOLIO_FILES["sp500_index"][1]},
    "nasdaq": {"package": "arch", "tape": "nasdaq", "column": "Adj Close",
               "kind": "daily price index (no dividends)",
               "sha256": bundled.ARCH_FILES["nasdaq"][2]},
    "ff": {"package": "arch", "tape": "frenchdata", "column": "Mkt-RF + RF",
           "kind": "monthly total return (dividends included), decimal",
           "sha256": bundled.ARCH_FILES["frenchdata"][2]},
}


# --------------------------------------------------------------------------- #
# Real tapes
# --------------------------------------------------------------------------- #
def have_real() -> bool:
    """True iff all three tapes are available offline (arch installed, skfolio cached)."""
    try:
        return bool(bundled.have_arch("nasdaq") and bundled.have_arch("frenchdata")
                    and bundled.have_skfolio())
    except Exception:  # pragma: no cover - defensive
        return False


def load_sp500(asof: str = SP500_AS_OF) -> pd.Series:
    """Daily S&P 500 **price index** close, 1990-01-02 → ``asof`` (default 2022-11-30)."""
    s = bundled.load_skfolio("sp500_index")["SP500"].dropna()
    s = s[s.index <= pd.Timestamp(asof)]
    s.name = "sp500"
    return s


def load_nasdaq(asof: str = NASDAQ_AS_OF) -> pd.Series:
    """Daily Nasdaq Composite **price index** (arch ``Adj Close``), 1999-01-04 → ``asof``."""
    s = bundled.load_arch("nasdaq")["Adj Close"].dropna()
    s = s[s.index <= pd.Timestamp(asof)]
    s.name = "nasdaq"
    return s


def load_ff_market(asof: str = FF_AS_OF) -> pd.DataFrame:
    """Monthly Fama-French market **total return** ``mkt`` and T-bill ``rf`` (decimal)."""
    ff = bundled.ff_monthly_total_return()[["mkt", "rf"]]
    return ff[ff.index <= pd.Timestamp(asof)].copy()


def ff_index(ff: pd.DataFrame | None = None) -> pd.Series:
    """The Fama-French market as a total-return level series (1926-06 = 1)."""
    ff = load_ff_market() if ff is None else ff
    lvl = (1.0 + ff["mkt"]).cumprod()
    first = lvl.index[0] - pd.offsets.MonthEnd(1)
    lvl = pd.concat([pd.Series([1.0], index=[first]), lvl])
    lvl.name = "ff_mkt"
    return lvl


def daily_rf(index: pd.DatetimeIndex, ff: pd.DataFrame | None = None) -> pd.Series:
    """Daily cash return on a trading-day ``index`` from Ken French's monthly T-bill.

    Each month's rate is spread geometrically over that month's sessions on ``index``. Days
    after the last Fama-French month (2018-11) come back NaN — the trading tests are cut there
    rather than inventing a cash rate.
    """
    ff = load_ff_market() if ff is None else ff
    idx = pd.DatetimeIndex(index)
    month = idx.to_period("M")
    counts = pd.Series(1, index=idx).groupby(month).transform("size").to_numpy()
    rf_m = ff["rf"].copy()
    rf_m.index = rf_m.index.to_period("M")
    m = rf_m.reindex(month).to_numpy()
    out = (1.0 + m) ** (1.0 / counts) - 1.0
    return pd.Series(out, index=idx, name="rf")


def fingerprint(obj) -> str:
    """Short content fingerprint (12 hex) — delegated to ``quantlab.bundled.fingerprint``."""
    return bundled.fingerprint(obj)


# --------------------------------------------------------------------------- #
# Synthetic tape — GJR-GARCH with a leverage knob
# --------------------------------------------------------------------------- #
def _std_t(rng: np.random.Generator, nu: float, size) -> np.ndarray:
    """Student-t draws rescaled to unit variance (``nu`` > 2)."""
    return rng.standard_t(nu, size=size) * np.sqrt((nu - 2.0) / nu)


def simulate_gjr(n: int, mu: float, omega: float, alpha: float, gamma: float, beta: float,
                 nu: float, n_paths: int = 1, seed: int = 1016, burn: int = 500,
                 shocks: np.ndarray | None = None) -> np.ndarray:
    """Simulate ``n_paths`` GJR-GARCH(1,1) return paths of length ``n`` (vectorised).

    ``r_t = mu + e_t``, ``e_t = sigma_t z_t``,
    ``sigma_t^2 = omega + (alpha + gamma 1[e_{t-1}<0]) e_{t-1}^2 + beta sigma_{t-1}^2``.
    ``z_t`` is a unit-variance Student-t (symmetric) unless ``nu`` is ``inf`` (normal) or a
    ``shocks`` array of shape ``(n + burn, n_paths)`` is supplied. Units follow ``omega``.
    Returns an array of shape ``(n, n_paths)``.
    """
    rng = np.random.default_rng(seed)
    T = n + burn
    if shocks is None:
        z = rng.standard_normal((T, n_paths)) if not np.isfinite(nu) else _std_t(
            rng, nu, (T, n_paths))
    else:
        z = np.asarray(shocks, dtype=float)
    pers = alpha + 0.5 * gamma + beta
    var = np.full(n_paths, omega / max(1.0 - pers, 1e-6))
    out = np.empty((T, n_paths))
    for t in range(T):
        e = np.sqrt(var) * z[t]
        out[t] = e
        var = omega + (alpha + gamma * (e < 0.0)) * e * e + beta * var
    return mu + out[burn:]


def simulate_egarch(n: int, mu: float, omega: float, alpha: float, gamma: float,
                    beta: float, nu: float, n_paths: int = 1, seed: int = 1016,
                    burn: int = 500) -> np.ndarray:
    """Simulate EGARCH(1,1,1) paths in ``arch``'s parameterisation.

    ``ln sigma_t^2 = omega + alpha (|z_{t-1}| - sqrt(2/pi)) + gamma z_{t-1}
    + beta ln sigma_{t-1}^2`` with unit-variance Student-t ``z``. Shape ``(n, n_paths)``.
    """
    rng = np.random.default_rng(seed)
    T = n + burn
    z = rng.standard_normal((T, n_paths)) if not np.isfinite(nu) else _std_t(
        rng, nu, (T, n_paths))
    lv = np.full(n_paths, omega / max(1.0 - beta, 1e-6))
    out = np.empty((T, n_paths))
    c = np.sqrt(2.0 / np.pi)
    for t in range(T):
        out[t] = np.exp(0.5 * lv) * z[t]
        lv = omega + alpha * (np.abs(z[t]) - c) + gamma * z[t] + beta * lv
    return mu + out[burn:]


# Planted parameters (decimal daily returns), close to a GJR-t fit on the S&P 500 index.
PLANT = {"mu": 0.0003, "omega": 1.5e-6, "alpha": 0.01, "gamma": 0.15, "beta": 0.895,
         "nu": 7.0}


def synthetic_returns(n_days: int = 8000, signal_strength: float = 1.0, seed: int = 1016,
                      start: str = "1990-01-02", **over) -> tuple[pd.DataFrame, dict]:
    """A deterministic daily index with a **planted leverage effect**.

    GJR-GARCH(1,1) with symmetric unit-variance Student-t shocks. ``signal_strength`` ``s``
    scales the leverage term, ``gamma_eff = s * gamma``, and the persistence it removes is
    handed back to the symmetric term, ``alpha_eff = alpha + (1 - s) * gamma / 2``, so
    ``alpha + gamma/2 + beta`` — and therefore the amount of volatility clustering — is the
    same in every world. ``s = 0`` is the matched, sign-symmetric null.

    The shocks are drawn once per ``seed``, so the planted world and its null share the same
    random numbers and differ *only* through the knob. Returns ``(frame, truth)``: the frame
    has ``ret`` (simple daily return), ``logret`` and ``price`` (starting at 100) on a business-
    day index that stays well inside pandas' nanosecond horizon.
    """
    p = dict(PLANT)
    p.update(over)
    s = float(signal_strength)
    gamma_eff = s * p["gamma"]
    alpha_eff = p["alpha"] + (1.0 - s) * 0.5 * p["gamma"]
    lr = simulate_gjr(n_days, p["mu"], p["omega"], alpha_eff, gamma_eff, p["beta"],
                      p["nu"], n_paths=1, seed=seed)[:, 0]
    dates = pd.bdate_range(start=start, periods=n_days)
    price = 100.0 * np.exp(np.cumsum(lr))
    df = pd.DataFrame({"logret": lr, "ret": np.expm1(lr), "price": price},
                      index=pd.DatetimeIndex(dates, name="date"))
    truth = {"signal_strength": s, "gamma_eff": gamma_eff, "alpha_eff": alpha_eff,
             "beta": p["beta"], "omega": p["omega"], "mu": p["mu"], "nu": p["nu"],
             "persistence": alpha_eff + 0.5 * gamma_eff + p["beta"], "n_days": n_days,
             "seed": seed}
    return df, truth
