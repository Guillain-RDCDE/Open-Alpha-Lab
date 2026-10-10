"""Data layer for Study 1025 — Don't Look.

Three legs, one evaluation clock. Benartzi & Thaler (1995) compared a loss-averse investor's
feelings about **stocks**, **bonds** and **bills** at different evaluation horizons, so the study
needs all three on a long monthly tape, plus a daily tape for the "check it every day" end of
the curve.

Every real number comes from :mod:`quantlab.bundled` — tapes frozen inside the ``arch`` and
``skfolio`` wheels, SHA-256 pinned, no network:

- **Stocks** — Fama-French ``Mkt-RF + RF`` (``ff_monthly_total_return()["mkt"]``): the CRSP
  value-weighted market, **total return** (dividends in), monthly, 1926-07 → 2018-11.
- **Bills** — Fama-French ``RF``: one-month T-bill return, monthly, same span.
- **Bonds** — *constructed*, and the construction is a decision, not a detail. ``arch``'s
  ``default`` tape carries Moody's seasoned **AAA corporate yield** (percent, monthly average).
  A monthly total return is approximated from yield changes with the **duration-convexity
  approximation** for a constant-maturity par bond (default 20 years, the order of Moody's
  seasoned-issue maturities)::

      r_t  ≈  y_{t-1}/12  −  D_{t-1} · Δy_t  +  ½ · C_{t-1} · Δy_t²

  where ``D`` and ``C`` are the modified duration and convexity of a par bond at last month's
  yield. It is an approximation in three ways that the results carry as caveats: the yield is a
  **monthly average**, not a month-end print (which smooths the series and adds a little
  positive autocorrelation); the bond is **corporate AAA**, not Treasury (a small credit
  spread, almost no defaults); and maturity is held constant. ``bond_total_return`` also offers
  an exact par-bond repricing, which the tests show agrees closely.
- **Inflation** — ``core_cpi`` (``CPILFESL``, **core** CPI: ex food and energy — headline CPI
  is not in the bundle), monthly from 1957-01, for the **real-terms** check.
- **Daily stocks** — ``skfolio`` ``sp500_index``: the S&P 500 **price index** (no dividends),
  1990-01-02 → 2022-12-28, used for the daily and weekly ends of the curve and for the daily /
  weekly myopic switcher. Dividends are about 2% a year — under one basis point a day — so the
  omission is negligible at a one-day horizon and is labelled everywhere it is used.

The study is pinned at ``AS_OF = 2018-11-30``: the last full month of the Fama-French and core
CPI tapes. The AAA tape (to 2018-12) and the daily tape (to 2022-12) are cut there too, so every
leg shares one window and one bill rate. Nothing after the as-of enters a number.

``synthetic_monthly`` / ``synthetic_daily`` are the deterministic offline worlds the test-suite
runs on: i.i.d. lognormal returns with a **planted** equity premium scaled by
``signal_strength``. At ``1.0`` stocks earn a premium of known size and a loss-averse investor's
break-even horizon exists; at ``0.0`` stocks earn no more than bills on average and the
break-even horizon against bills must not exist.
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

AS_OF = "2018-11-30"            # last full month of the FF and core-CPI tapes
START = "1926-07-31"            # first Fama-French month
REAL_START = "1957-02-28"       # first month with a core-CPI inflation rate
SPLIT = "1970-01-01"            # pre / post sub-period boundary (fixed in the brief)
BOND_MATURITY = 20.0            # years, constant-maturity par bond on the AAA yield
TRADING_DAYS_PER_YEAR = 252

TAPES = {
    "stocks + bills": ("arch", "frenchdata", bundled.ARCH_FILES["frenchdata"][2]),
    "bond yield (AAA)": ("arch", "default", bundled.ARCH_FILES["default"][2]),
    "core CPI": ("arch", "core_cpi", bundled.ARCH_FILES["core_cpi"][2]),
    "daily S&P 500 (price)": ("skfolio 1.8.1", "sp500_index",
                              bundled.SKFOLIO_FILES["sp500_index"][1]),
}


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
def have_monthly() -> bool:
    """True iff the three arch tapes this study needs are installed."""
    return all(bundled.have_arch(n) for n in ("frenchdata", "default", "core_cpi"))


def have_daily() -> bool:
    """True iff the skfolio daily S&P 500 tape is cached (never fetches)."""
    return bundled.have_skfolio()


def have_real() -> bool:
    """True iff every real tape is available offline."""
    return have_monthly() and have_daily()


# --------------------------------------------------------------------------- #
# The bond leg
# --------------------------------------------------------------------------- #
def par_bond_duration_convexity(y: np.ndarray, maturity: float = BOND_MATURITY,
                                freq: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Modified duration and convexity of a par bond at yield ``y`` (annual decimal).

    Closed forms for a bond paying ``y/freq`` every ``1/freq`` years, priced at par. Both are
    expressed per unit of *annual* yield, so ``ΔP/P ≈ −D·Δy + ½·C·Δy²``.
    """
    y = np.asarray(y, dtype=float)
    n = maturity * freq
    k = y / freq
    t = np.arange(1, int(round(n)) + 1)[None, :] / freq          # years
    cf = np.full((y.size, t.shape[1]), 1.0)
    cf = cf * k.reshape(-1, 1)
    cf[:, -1] += 1.0
    disc = (1.0 + k.reshape(-1, 1)) ** (-t * freq)
    pv = cf * disc
    price = pv.sum(axis=1)
    mac = (pv * t).sum(axis=1) / price
    mod = mac / (1.0 + k)
    conv = (pv * t * (t + 1.0 / freq)).sum(axis=1) / (price * (1.0 + k) ** 2)
    return mod.reshape(y.shape), conv.reshape(y.shape)


def _par_bond_price(coupon: np.ndarray, y: np.ndarray, maturity: float, freq: int = 2):
    n = int(round(maturity * freq))
    k = y / freq
    c = coupon / freq
    t = np.arange(1, n + 1)[None, :]
    disc = (1.0 + k.reshape(-1, 1)) ** (-t)
    return (c.reshape(-1, 1) * disc).sum(axis=1) + disc[:, -1]


def bond_total_return(yield_pct: pd.Series, maturity: float = BOND_MATURITY,
                      method: str = "duration") -> pd.Series:
    """Monthly total return of a constant-maturity par bond from a yield series (percent).

    ``method="duration"`` (the default, as documented in the module docstring) uses
    ``y_{t-1}/12 − D·Δy + ½·C·Δy²`` with ``D``, ``C`` of a par bond at ``y_{t-1}``.
    ``method="reprice"`` reprices last month's par bond (coupon ``y_{t-1}``) at today's yield
    with ``maturity − 1/12`` years left — the exact object the approximation targets.
    The first month has no prior yield and is dropped.
    """
    y = yield_pct.astype(float) / 100.0
    y0 = y.shift(1)
    dy = y - y0
    ok = y0.notna()
    y0v, yv, dyv = y0[ok].to_numpy(), y[ok].to_numpy(), dy[ok].to_numpy()
    if method == "duration":
        D, C = par_bond_duration_convexity(y0v, maturity)
        r = y0v / 12.0 - D * dyv + 0.5 * C * dyv ** 2
    elif method == "reprice":
        p1 = _par_bond_price(y0v, yv, maturity - 1.0 / 12.0)
        r = p1 - 1.0 + y0v / 12.0
    else:
        raise ValueError("method must be 'duration' or 'reprice'")
    return pd.Series(r, index=y.index[ok], name="bond")


# --------------------------------------------------------------------------- #
# Real tapes
# --------------------------------------------------------------------------- #
def load_monthly(asof: str = AS_OF, maturity: float = BOND_MATURITY,
                 bond_method: str = "duration") -> pd.DataFrame:
    """Monthly decimal returns: ``stock`` (total return), ``bond`` (constructed), ``bill``.

    Plus ``infl`` (core-CPI monthly inflation, NaN before 1957-02). Month-end index,
    1926-07 → ``asof``; no partial month is ever included.
    """
    ff = bundled.ff_monthly_total_return()
    aaa = bundled.load_arch("default")["AAA"]
    cpi = bundled.load_arch("core_cpi")["CPILFESL"]
    bond = bond_total_return(aaa, maturity=maturity, method=bond_method)
    df = pd.DataFrame({"stock": ff["mkt"], "bill": ff["rf"]})
    df["bond"] = bond.reindex(df.index)
    df["infl"] = cpi.pct_change().reindex(df.index)
    df = df[["stock", "bond", "bill", "infl"]]
    df = df[(df.index >= pd.Timestamp(START)) & (df.index <= pd.Timestamp(asof))]
    df.index.name = "date"
    return df


def to_real(df: pd.DataFrame, cols=("stock", "bond", "bill")) -> pd.DataFrame:
    """Deflate nominal monthly returns by core-CPI inflation: ``(1+r)/(1+π) − 1``.

    Rows without an inflation print (before 1957-02) are dropped, never filled.
    """
    d = df.dropna(subset=["infl"])
    out = pd.DataFrame({c: (1.0 + d[c]) / (1.0 + d["infl"]) - 1.0 for c in cols},
                       index=d.index)
    return out


def load_daily(asof: str = AS_OF) -> pd.DataFrame:
    """Daily ``stock`` (S&P 500 **price** return, no dividends) and ``bill`` returns.

    The bill leg spreads each Fama-French monthly ``RF`` evenly (geometrically) over that
    month's trading days. 1990-01-03 → ``asof``.
    """
    px = bundled.load_skfolio("sp500_index")["SP500"]
    px = px[px.index <= pd.Timestamp(asof)]
    r = px.pct_change().dropna()
    rf_m = bundled.ff_monthly_total_return()["rf"]
    month = r.index.to_period("M")
    n_in_month = pd.Series(1, index=r.index).groupby(month).transform("size")
    rf_month = rf_m.copy()
    rf_month.index = rf_month.index.to_period("M")
    rf = rf_month.reindex(month).to_numpy()
    bill = (1.0 + rf) ** (1.0 / n_in_month.to_numpy()) - 1.0
    out = pd.DataFrame({"stock": r.to_numpy(), "bill": bill}, index=r.index)
    out.index.name = "date"
    return out.dropna()


def fingerprint(obj) -> str:
    """Short content fingerprint (12 hex) — ``quantlab.bundled.fingerprint``."""
    return bundled.fingerprint(obj)


def provenance() -> list[dict]:
    """One row per tape: package, dataset, pinned SHA-256 (first 16 hex)."""
    return [{"leg": leg, "package": pkg, "dataset": name, "sha256": sha[:16]}
            for leg, (pkg, name, sha) in TAPES.items()]


# --------------------------------------------------------------------------- #
# Synthetic worlds — the deterministic offline core
# --------------------------------------------------------------------------- #
def synthetic_monthly(n_months: int = 1104, signal_strength: float = 1.0,
                      seed: int = 1025, bill_rate: float = 0.0030,
                      equity_premium: float = 0.0055, stock_vol: float = 0.054,
                      bond_premium: float = 0.0010, bond_vol: float = 0.020,
                      start: str = "1926-07-31") -> tuple[pd.DataFrame, dict]:
    """i.i.d. lognormal monthly ``stock`` / ``bond`` / ``bill`` returns with a planted premium.

    ``1 + r_stock = exp(m + σ·z)`` with ``m`` chosen so the **arithmetic** monthly premium over
    bills is exactly ``signal_strength × equity_premium`` (6.6% a year at 1.0). At
    ``signal_strength = 0`` stocks have the same expected return as bills — and a lower median,
    since volatility drags the median below the mean — so no loss-averse investor should ever
    prefer them to bills at any horizon. Bonds carry a small fixed premium and lower volatility;
    bills are a constant rate. Defaults (≈19% stock vol, ≈7% bond vol, 3.6% bills) are of the
    order of the historical tape. Deterministic in ``seed``; dates stay well inside pandas' ns
    horizon.
    """
    rng = np.random.default_rng(seed)
    prem = float(signal_strength) * equity_premium
    ms = np.log(1.0 + bill_rate + prem) - 0.5 * stock_vol ** 2
    mb = np.log(1.0 + bill_rate + bond_premium) - 0.5 * bond_vol ** 2
    z = rng.standard_normal((n_months, 2))
    stock = np.exp(ms + stock_vol * z[:, 0]) - 1.0
    bond = np.exp(mb + bond_vol * z[:, 1]) - 1.0
    idx = pd.date_range(start=start, periods=n_months, freq="ME")
    df = pd.DataFrame({"stock": stock, "bond": bond,
                       "bill": np.full(n_months, bill_rate)}, index=idx)
    df.index.name = "date"
    truth = {"signal_strength": float(signal_strength), "seed": seed,
             "equity_premium_monthly": prem, "bill_rate": bill_rate,
             "stock_vol": stock_vol, "bond_premium": bond_premium, "bond_vol": bond_vol,
             "log_mean_stock": float(ms), "log_mean_bond": float(mb),
             "n_months": n_months}
    return df, truth


def synthetic_daily(n_years: int = 30, signal_strength: float = 1.0, seed: int = 1025,
                    bill_rate_ann: float = 0.03, equity_premium_ann: float = 0.066,
                    stock_vol_ann: float = 0.17, start: str = "1990-01-02"
                    ) -> tuple[pd.DataFrame, dict]:
    """i.i.d. lognormal daily ``stock`` and constant ``bill`` returns, planted premium.

    Same knob as :func:`synthetic_monthly`: the arithmetic daily premium is
    ``signal_strength × equity_premium_ann / 252``.
    """
    rng = np.random.default_rng(seed)
    n = int(n_years * TRADING_DAYS_PER_YEAR)
    rf = (1.0 + bill_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    prem = float(signal_strength) * equity_premium_ann / TRADING_DAYS_PER_YEAR
    sd = stock_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    m = np.log(1.0 + rf + prem) - 0.5 * sd ** 2
    stock = np.exp(m + sd * rng.standard_normal(n)) - 1.0
    idx = pd.bdate_range(start=start, periods=n)
    df = pd.DataFrame({"stock": stock, "bill": np.full(n, rf)}, index=idx)
    df.index.name = "date"
    return df, {"signal_strength": float(signal_strength), "seed": seed,
                "equity_premium_daily": prem, "bill_rate_daily": rf, "n_days": n}
