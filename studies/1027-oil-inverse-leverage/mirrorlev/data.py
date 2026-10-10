"""Data layer for Study 1027 — Oil Is Equity in a Mirror.

Three real tapes and one synthetic world, all offline.

Real tapes (frozen inside the ``arch`` wheel, SHA-256 pinned by :mod:`quantlab.bundled`)
----------------------------------------------------------------------------------------
- ``load_wti`` — **daily WTI spot**, Cushing, FRED ``DCOILWTICO``, 1986-01-02 → 2018-12-31.
  This is a *spot* price. Nobody can buy and hold a barrel at Cushing for a daily mark, and
  an investable oil position is a rolled futures contract whose return differs from the spot
  change by the roll yield. Every trading number on this tape is therefore a **spot proxy**:
  we treat the daily spot change as if it were the excess return of a fully collateralised
  front-month position. That caps the Tradability stamp, and we say so wherever it appears.
- ``load_sp500`` — **daily S&P 500 price index** (dividends *not* included), 1999-01-04 →
  2018-12-31: the equity mirror.
- ``load_crude_monthly`` — **monthly Brent and WTI**, 1987-05 → 2019-12. These are **monthly
  averages of daily prices**, not month-end marks (we checked: the WTI column matches the
  calendar-month mean of the daily tape to a cent). Averaging smooths returns and induces
  autocorrelation (Working 1960), so this tape is a *cross-check on direction only*.
- ``load_rf_daily`` — the Fama-French one-month T-bill rate (``frenchdata``, monthly,
  decimal), spread evenly across the trading days of each month. Used only to put the S&P
  overlay on an excess-of-cash footing; it ends 2018-11, so the S&P excess series ends there.

``AS_OF`` pins the daily window to the last full month of both daily tapes (the WTI tape's
three January-2019 prints are a partial month and are dropped). ``AS_OF_MONTHLY`` pins the
monthly tape to its last full month (the January-2020 row is dropped too, conservatively).

Synthetic world
---------------
``synthetic_gjr`` simulates a GJR-GARCH(1,1,1) return series whose asymmetry is set by
``signal_strength``: ``+1`` plants an **equity-type** leverage effect (a down shock raises
next-day variance six times as much as an equal up shock), ``-1`` plants the **inverse**
(oil-type) effect at the same size, and ``0`` is the **symmetric null**. The unconditional
volatility, persistence and fat tails are held fixed across the knob, so the only thing that
changes is the sign and size of the asymmetry — exactly what the estimators must recover.
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

AS_OF = "2018-12-31"            # last full month common to the daily WTI and S&P tapes
AS_OF_MONTHLY = "2019-12-31"    # last full month of the monthly Brent/WTI tape
WTI_START = "1986-01-02"
SP_START = "1999-01-04"

TAPES = ("wti", "sp500", "crude", "frenchdata")


# --------------------------------------------------------------------------- #
# Real tapes
# --------------------------------------------------------------------------- #
def have_real() -> bool:
    """True when every arch tape this study needs is present (always, if arch is installed)."""
    return all(bundled.have_arch(t) for t in TAPES)


def load_wti(asof: str = AS_OF) -> pd.Series:
    """Daily WTI spot (USD/bbl, FRED ``DCOILWTICO``), missing prints dropped, never filled."""
    s = bundled.load_arch("wti")["wti"].astype(float)
    s = s[s.index <= pd.Timestamp(asof)]
    s.name = "wti"
    return s


def load_sp500(asof: str = AS_OF) -> pd.Series:
    """Daily S&P 500 **price index** close (no dividends)."""
    df = bundled.load_arch("sp500")
    s = df["Adj Close"].astype(float)
    s = s[s.index <= pd.Timestamp(asof)]
    s.name = "sp500"
    return s


def load_crude_monthly(asof: str = AS_OF_MONTHLY) -> pd.DataFrame:
    """Monthly **average** Brent and WTI prices (USD/bbl), month-end indexed."""
    df = bundled.load_arch("crude")[["Brent", "WTI"]].astype(float)
    df = df[df.index <= pd.Timestamp(asof)]
    return df.rename(columns={"Brent": "brent", "WTI": "wti"})


def load_rf_daily(index: pd.DatetimeIndex) -> pd.Series:
    """Daily risk-free accrual on ``index`` from the Fama-French monthly T-bill rate.

    Each month's (decimal) rate is spread geometrically across that month's trading days in
    ``index``. Months the FF tape does not cover (after 2018-11) are left **NaN**, never
    filled — downstream excess-return calculations drop them.
    """
    rf_m = bundled.ff_monthly_total_return()["rf"]
    idx = pd.DatetimeIndex(index)
    month = idx.to_period("M")
    counts = pd.Series(1, index=idx).groupby(month).transform("count").to_numpy()
    key = rf_m.copy()
    key.index = key.index.to_period("M")
    m = key.reindex(month).to_numpy()
    out = (1.0 + m) ** (1.0 / counts) - 1.0
    return pd.Series(out, index=idx, name="rf")


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily (or monthly) log returns, first observation dropped."""
    r = np.log(prices.astype(float)).diff().dropna()
    r.name = prices.name
    return r


def simple_returns(prices: pd.Series) -> pd.Series:
    r = prices.astype(float).pct_change().dropna()
    r.name = prices.name
    return r


def fingerprint(obj) -> str:
    """12-hex content fingerprint (delegates to :func:`quantlab.bundled.fingerprint`)."""
    return bundled.fingerprint(obj)


def provenance() -> list[dict]:
    """Package, file, SHA-256 pin and content fingerprint of every tape used."""
    import arch

    rows = []
    loaders = {"wti": lambda: load_wti(), "sp500": lambda: load_sp500(),
               "crude": lambda: load_crude_monthly(),
               "frenchdata": lambda: bundled.ff_monthly_total_return()["rf"]}
    labels = {"wti": "daily WTI **spot** (FRED DCOILWTICO), USD/bbl",
              "sp500": "daily S&P 500 **price index** (no dividends)",
              "crude": "monthly **average** Brent and WTI spot",
              "frenchdata": "Fama-French monthly RF (T-bill), decimal"}
    for name in TAPES:
        sub, fname, sha = bundled.ARCH_FILES[name]
        obj = loaders[name]()
        rows.append({"tape": name, "package": f"arch {arch.__version__}",
                     "file": f"arch/data/{sub}/{fname}", "sha256": sha,
                     "label": labels[name], "rows": int(len(obj)),
                     "first": str(obj.index[0].date()), "last": str(obj.index[-1].date()),
                     "fingerprint": fingerprint(obj)})
    return rows


# --------------------------------------------------------------------------- #
# Synthetic world — a GJR process with a dialled asymmetry
# --------------------------------------------------------------------------- #
def synthetic_gjr(
    n_years: int = 20,
    signal_strength: float = 1.0,
    seed: int = 1027,
    ann_vol: float = 0.30,
    base_impact: float = 0.07,
    asym_half: float = 0.05,
    beta: float = 0.90,
    nu: float = 8.0,
    drift_ann: float = 0.03,
    start: str = "2000-01-03",
) -> tuple[pd.Series, dict]:
    """Daily prices from a GJR-GARCH(1,1,1) with a known asymmetry.

    Variance recursion (``e`` the return shock, ``I`` the down-shock indicator)::

        s2[t] = omega + (alpha + gamma * I[e[t-1] < 0]) * e[t-1]**2 + beta * s2[t-1]
        alpha = base_impact - asym_half * k,   gamma = 2 * asym_half * k,   k = signal_strength

    So a **down** shock carries ``base_impact + asym_half*k`` and an **up** shock
    ``base_impact - asym_half*k``. ``k=+1`` is equity-type (down 0.12, up 0.02), ``k=-1`` the
    inverse oil-type mirror (down 0.02, up 0.12), ``k=0`` symmetric (0.07 each). Persistence
    ``beta + base_impact`` and the unconditional variance (set from ``ann_vol``) do not move
    with ``k``. Innovations are Student-t(``nu``) scaled to unit variance. ``|k|`` must be at
    most ``base_impact / asym_half`` so both impacts stay non-negative.

    Returns ``(prices, truth)`` where ``truth['gamma']`` is the planted GJR gamma in the
    arch package's convention (positive = equity-type).
    """
    k = float(signal_strength)
    if abs(k) * asym_half > base_impact + 1e-12:
        raise ValueError("signal_strength too large: an impact would turn negative")
    rng = np.random.default_rng(seed)
    n = int(n_years * TRADING_DAYS_PER_YEAR)
    alpha = base_impact - asym_half * k
    gamma = 2.0 * asym_half * k
    persistence = beta + base_impact
    var_d = (ann_vol ** 2) / TRADING_DAYS_PER_YEAR
    omega = var_d * (1.0 - persistence)
    z = rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2.0))
    mu = drift_ann / TRADING_DAYS_PER_YEAR
    e = np.empty(n)
    s2 = np.empty(n)
    s2_prev, e_prev = var_d, 0.0
    for t in range(n):
        s2_t = omega + (alpha + gamma * (e_prev < 0.0)) * e_prev ** 2 + beta * s2_prev
        e_t = np.sqrt(s2_t) * z[t]
        e[t], s2[t] = e_t, s2_t
        s2_prev, e_prev = s2_t, e_t
    r = mu + e
    dates = pd.bdate_range(start=start, periods=n)
    px = pd.Series(50.0 * np.exp(np.cumsum(r)), index=pd.DatetimeIndex(dates, name="date"),
                   name="synthetic")
    truth = {"signal_strength": k, "alpha": alpha, "gamma": gamma, "beta": beta,
             "omega": omega, "persistence": persistence, "ann_vol": ann_vol, "nu": nu,
             "n_days": n, "seed": seed,
             "down_impact": alpha + gamma, "up_impact": alpha}
    return px, truth
