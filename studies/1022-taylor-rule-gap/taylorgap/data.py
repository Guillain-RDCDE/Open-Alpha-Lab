"""Data layer for Study 1022 — Behind the Curve.

Three real tapes, all shipped inside Python packages and SHA-256 pinned (no network, ever):

- **US quarterly macro**, statsmodels ``macrodata`` (1959Q1 → 2009Q3): real GDP, the CPI,
  the 3-month T-bill rate and the unemployment rate. **Final vintage** — these are the
  numbers as revised years later, not the numbers on a desk at the time. The T-bill rate is
  the **policy-rate proxy** (the effective fed funds rate is not on the tape; the two track
  each other closely but not exactly, especially before 1980). It is the *quarterly average*
  of the monthly secondary-market rate, so it is fully known at the end of its quarter.
- **Fama-French monthly**, via ``arch`` (``frenchdata``): ``mkt`` is a **total return**
  (dividends in), ``rf`` the one-month T-bill. Compounded here to calendar quarters.
- **Moody's AAA corporate yield**, monthly, via ``arch`` (``default``): sampled at each
  quarter-end month. It stands in for "the long bond". There is no long-bond *return* on any
  bundled tape, so one is **approximated** from yield changes (see :func:`long_bond_return`)
  — an approximation, labelled as one everywhere it is used.

Release lag: macro values dated quarter *t* are treated as known only at the end of quarter
*t+1* (GDP's advance estimate lands about a month after the quarter; the CPI mid-month).
That lag is applied **once**, in :func:`taylorgap.strategy.build_signals`, never here.

The sample ends where the macro tape ends: the last signal is 2009Q3, which forecasts the
return of 2009Q4. ``AS_OF`` is therefore 2009-12-31 — the last full quarter any number in
the study uses — even though the return tapes run to 2018.

The synthetic world (:func:`synthetic_quarterly`) is the offline core: an AR(1) Taylor gap,
equity excess returns and yield changes with a planted predictive slope scaled by
``signal_strength`` (1.0 = planted, 0.0 = matched null), and return innovations correlated
with the gap's innovations exactly the way that creates Stambaugh bias in the real data.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

import sys  # noqa: E402

if REPO not in sys.path:
    sys.path.insert(0, REPO)

from quantlab import bundled  # noqa: E402

QUARTERS_PER_YEAR = 4

# statsmodels ships macrodata.csv as a plain file; bundled.load_macrodata does not pin it,
# so this study does. A statsmodels release that swapped the tape would fail loudly here.
MACRODATA_SHA256 = "d93c0d3a7a77ef83c3af14e46032bb1d02ae3a512b22ab94159a8ca226fcf708"

START = "1959-03-31"
MACRO_LAST = "2009-09-30"     # last quarter on the macro tape (2009Q3)
AS_OF = "2009-12-31"          # last full quarter any number uses (the 2009Q4 return)

# Taylor (1993): i* = r* + pi + 0.5 (pi - pi*) + 0.5 gap, with r* = pi* = 2.
R_STAR = 2.0
PI_STAR = 2.0

# The long-bond approximation: a par bond of this maturity, semi-annual coupons.
BOND_MATURITY_YEARS = 20


# --------------------------------------------------------------------------- #
# Real tapes
# --------------------------------------------------------------------------- #
def macrodata_path() -> str:
    from statsmodels.datasets import macrodata
    return os.path.join(os.path.dirname(macrodata.__file__), "macrodata.csv")


def macrodata_sha256() -> str:
    with open(macrodata_path(), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def have_real() -> bool:
    """True when every tape this study needs is installed (arch + statsmodels)."""
    try:
        return (bundled.have_arch("frenchdata") and bundled.have_arch("default")
                and os.path.exists(macrodata_path()))
    except ImportError:
        return False


def load_macro() -> pd.DataFrame:
    """Quarterly US macro (final vintage), quarter-end indexed, the columns this study uses.

    ``realgdp`` (bn chained 2005 $), ``cpi`` (end-of-quarter CPI-U), ``tbilrate`` (quarterly
    average of the 3-month T-bill, percent — the **policy-rate proxy**), ``unemp`` (percent).
    Raises if the shipped file's SHA-256 differs from the pin.
    """
    got = macrodata_sha256()
    if got != MACRODATA_SHA256:
        raise bundled.TapeMismatch(
            f"statsmodels macrodata SHA-256 {got[:12]} != pinned {MACRODATA_SHA256[:12]}")
    m = bundled.load_macrodata()
    return m[["realgdp", "cpi", "tbilrate", "unemp"]].astype(float)


def _quarter_end(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.PeriodIndex(idx, freq="Q").to_timestamp(how="end").normalize()


def load_equity_quarterly() -> pd.DataFrame:
    """Fama-French market **total return** and T-bill, compounded to calendar quarters.

    Columns ``mkt``, ``rf`` and ``eq_xs = mkt - rf`` (decimal, per quarter). Only quarters with
    all three months present are kept — no partial quarter is ever stamped.
    """
    ff = bundled.ff_monthly_total_return()[["mkt", "rf"]]
    q = _quarter_end(ff.index)
    g = (1.0 + ff).groupby(q)
    out = g.prod() - 1.0
    out = out[g.size() == 3]
    out["eq_xs"] = out["mkt"] - out["rf"]
    out.index.name = "date"
    return out


def load_aaa_quarterly() -> pd.Series:
    """Moody's AAA yield (percent) at each quarter-end month."""
    d = bundled.load_arch("default")["AAA"].dropna()
    q = _quarter_end(d.index)
    last = d.groupby(q).last()
    n = d.groupby(q).size()
    last = last[n == 3]
    last.index.name = "date"
    return last.rename("aaa")


def par_bond_duration_convexity(y: np.ndarray, maturity: float = BOND_MATURITY_YEARS,
                                freq: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Modified duration and convexity of a par bond at yield ``y`` (decimal, annual).

    Closed form for a par bond: ``D_mod = (1 - (1 + y/f)^(-f N)) / y``. Convexity from the
    exact price function by central finite difference (so it is the bond's own, not a
    textbook shortcut).
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = int(round(maturity * freq))
    t = np.arange(1, n + 1)

    def price(yy, cpn):
        disc = (1.0 + yy[:, None] / freq) ** (-t[None, :])
        cf = np.repeat((cpn / freq * 100.0)[:, None], n, axis=1)
        cf[:, -1] += 100.0
        return (cf * disc).sum(axis=1)

    cpn = y.copy()
    h = 1e-4
    p0 = price(y, cpn)
    pu = price(y + h, cpn)
    pd_ = price(y - h, cpn)
    dmod = (1.0 - (1.0 + y / freq) ** (-n)) / y
    conv = (pu + pd_ - 2.0 * p0) / (p0 * h * h)
    return dmod, conv


def long_bond_return(yield_pct: pd.Series, maturity: float = BOND_MATURITY_YEARS) -> pd.Series:
    """**Approximate** quarterly total return of a long par bond from yield changes.

    With ``y`` the start-of-quarter yield and ``dy`` the change over the quarter (decimals)::

        r  ≈  y / 4  -  D_mod(y) * dy  +  0.5 * C(y) * dy**2

    — carry (a quarter of the coupon on a par bond) plus the second-order Taylor expansion
    of the price in the yield, with ``D_mod`` and ``C`` from :func:`par_bond_duration_convexity`
    at a ``maturity``-year par bond. It is an **approximation**: it ignores roll-down, the call
    features and credit migration of the bonds behind Moody's AAA index, and the fact that the
    index is not a single constant-maturity bond. Good to a few basis points a quarter for the
    rate moves seen here; not a substitute for a real total-return index.
    """
    y = yield_pct.astype(float) / 100.0
    y0 = y.shift(1)
    dy = y - y0
    dmod, conv = par_bond_duration_convexity(y0.fillna(0.05).to_numpy(), maturity)
    r = y0 / QUARTERS_PER_YEAR - dmod * dy + 0.5 * conv * dy ** 2
    return r.rename("bond_ret")


def load_quarterly() -> pd.DataFrame:
    """One aligned quarterly frame, 1959Q1 → ``AS_OF``.

    Convention: row *t* holds values **dated** quarter *t* — the macro prints for quarter *t*
    (not yet released at the end of *t*; the release lag is applied in ``build_signals``) and
    the returns **realised during** quarter *t*.

    Columns: ``realgdp cpi tbilrate unemp`` (macro, final vintage), ``mkt rf eq_xs`` (Fama-
    French total return, decimal per quarter), ``aaa`` (percent, quarter-end), ``d_aaa``
    (change in percentage points over the quarter), ``bond_ret`` (approximate long-bond total
    return) and ``bond_xs = bond_ret - rf``.
    """
    macro = load_macro()
    eq = load_equity_quarterly()
    aaa = load_aaa_quarterly()
    df = pd.concat([macro, eq, aaa], axis=1, sort=True)
    df["d_aaa"] = df["aaa"].diff()
    df["bond_ret"] = long_bond_return(df["aaa"])
    df["bond_xs"] = df["bond_ret"] - df["rf"]
    df = df.loc[(df.index >= pd.Timestamp(START)) & (df.index <= pd.Timestamp(AS_OF))]
    df.index.name = "date"
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """12-hex content fingerprint (``quantlab.bundled.fingerprint``)."""
    return bundled.fingerprint(df)


def provenance() -> dict:
    """Package versions and SHA-256 pins of every tape this study reads."""
    import arch
    import statsmodels
    return {
        "statsmodels": statsmodels.__version__,
        "arch": arch.__version__,
        "macrodata_sha256": macrodata_sha256(),
        "frenchdata_sha256": bundled.ARCH_FILES["frenchdata"][2],
        "default_sha256": bundled.ARCH_FILES["default"][2],
    }


# --------------------------------------------------------------------------- #
# Synthetic world — the offline core
# --------------------------------------------------------------------------- #
def synthetic_quarterly(n_quarters: int = 200, signal_strength: float = 1.0,
                        seed: int = 1022, rho: float = 0.92, gap_sd: float = 2.5,
                        beta_eq: float = -0.012, beta_bond: float = -0.08,
                        eq_mu: float = 0.015, eq_vol: float = 0.08,
                        bond_vol: float = 0.45, corr_uv: float = -0.4,
                        rf_q: float = 0.01, start: str = "1960-03-31"
                        ) -> tuple[pd.DataFrame, dict]:
    """A quarterly world with a **planted** Taylor-gap predictability, shaped like the real one.

    - ``tgap`` follows an AR(1) with persistence ``rho`` and stationary sd ``gap_sd`` (pp);
      row *t*'s value is known at the end of quarter *t*.
    - ``eq_xs[t+1] = eq_mu + signal_strength * beta_eq * tgap[t] + u[t+1]`` — with the
      claim's sign: a *negative* gap (Fed behind the curve) predicts *higher* equity returns.
    - ``d_aaa[t+1] = signal_strength * beta_bond * tgap[t] - 0.03 * (aaa[t] - 6) + w[t+1]``
      — behind the curve, long yields rise (a bond headwind).
    - ``corr(u[t+1], v[t+1]) = corr_uv`` with ``v`` the gap's own innovation, the source of
      Stambaugh bias; yield shocks correlate +0.3 with ``v``.

    ``signal_strength = 0`` keeps every other feature (persistence, the innovation
    correlation, the volatilities) and removes only the slope — the matched null.
    Returns ``(panel, truth)``; the panel has the columns the strategy functions read:
    ``tgap eq_xs mkt rf aaa d_aaa bond_ret bond_xs``. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n = int(n_quarters)
    s = float(signal_strength)
    v_sd = gap_sd * np.sqrt(1.0 - rho ** 2)
    z = rng.standard_normal((n, 3))
    # Cholesky for (v, u, w): corr(u, v) = corr_uv, corr(w, v) = 0.3, corr(u, w) = -0.3
    C = np.array([[1.0, corr_uv, 0.3], [corr_uv, 1.0, -0.3], [0.3, -0.3, 1.0]])
    L = np.linalg.cholesky(C)
    e = z @ L.T
    v = e[:, 0] * v_sd
    u = e[:, 1] * eq_vol
    w = e[:, 2] * bond_vol

    g = np.zeros(n)
    eq = np.zeros(n)
    dy = np.zeros(n)
    aaa = np.zeros(n)
    g[0] = rng.normal(0.0, gap_sd)
    aaa[0] = 6.0
    eq[0] = eq_mu + u[0]
    for t in range(1, n):
        g[t] = rho * g[t - 1] + v[t]
        eq[t] = eq_mu + s * beta_eq * g[t - 1] + u[t]
        dy[t] = s * beta_bond * g[t - 1] - 0.03 * (aaa[t - 1] - 6.0) + w[t]
        aaa[t] = aaa[t - 1] + dy[t]
    idx = pd.date_range(start=start, periods=n, freq="QE-DEC")
    panel = pd.DataFrame({"tgap": g, "eq_xs": eq, "rf": rf_q, "aaa": aaa}, index=idx)
    panel["mkt"] = panel["eq_xs"] + panel["rf"]
    panel["d_aaa"] = panel["aaa"].diff()
    panel["bond_ret"] = long_bond_return(panel["aaa"])
    panel["bond_xs"] = panel["bond_ret"] - panel["rf"]
    panel.index.name = "date"
    truth = {"n_quarters": n, "seed": seed, "signal_strength": s, "rho": rho,
             "gap_sd": gap_sd, "beta_eq": s * beta_eq, "beta_bond": s * beta_bond,
             "corr_uv": corr_uv, "eq_vol": eq_vol, "bond_vol": bond_vol}
    return panel, truth
