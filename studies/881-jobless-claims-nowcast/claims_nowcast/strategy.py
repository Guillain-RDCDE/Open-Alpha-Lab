"""Strategy + inference for Study 881 — Jobless-Claims Sector Rotation.

The claim (labour nowcast): the **4-week change in initial jobless claims** should
drive a **cyclical-vs-defensive rotation**. When claims *rise* (labour market cooling),
the market should reward **defensives** (XLP, XLU) over **cyclicals** (XLY, XLI), so the
**cyclical-minus-defensive** return spread should be *pushed down*. This is distinct
from Study 385 (aggregate claims momentum → the whole market): here the outcome is a
long-short **sector rotation**, not a market-timing overlay.

Operationalised on the monthly tape:

    Let ``c`` = the 4-week-MA initial-claims level (thousands) and
    ``dcl_t = c_t / c_{t-k} - 1`` its k-month change (default k = 1 ≈ the "4-week
    change"). Let ``spread_t`` = the equal-weight **cyclical (XLY,XLI) minus defensive
    (XLP,XLU)** monthly return. The claim says a *rising* ``dcl`` predicts a *negative*
    forward spread — i.e. the predictive-regression slope of ``spread_{t+lag}`` on
    ``dcl_t`` should be **negative**.

Method:

  * **Predictive Newey-West regression.** OLS of the forward cyclical-minus-defensive
    spread on the claims change, with a HAC (Newey-West, Bartlett) *t* on the slope — a
    monthly overlapping-signal *t* would overstate significance. Report slope, HAC *t*,
    R².
  * **One documented execution lag.** The month-``t`` claims 4-week MA is fully printed
    by month-end ``t``; to stay strictly non-look-ahead the position is held over month
    ``t+lag`` (default lag = 1) — signal end of ``t`` → forward spread of ``t+1``.
  * **Permutation placebo.** Shuffle the claims-change column against the forward spread
    (breaking the signal → outcome link) and ask how often a shuffled slope is at least
    as extreme as observed — the honest small-sample test of whether the fitted slope is
    a lucky alignment (here, of the single COVID episode).
  * **Era cut + COVID sensitivity.** Split the sample and drop the 2020 claims spike —
    the decisive robustness check for a labour-nowcast whose entire signal history is
    dominated by one outlier.
  * **Costed rotation timer.** Flip a long-short cyclical/defensive book with the claim's
    sign; charge one-way cost × NAV per leg per rebalance plus borrow on the short leg.
  * **Inference primitives + a seeded synthetic control** (the machinery proof).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


# --------------------------------------------------------------------------- #
# Signal + cyclical-minus-defensive spread
# --------------------------------------------------------------------------- #
def claims_change(frame: pd.DataFrame, k: int = 1) -> pd.Series:
    """k-month change of the 4-week-MA claims level: ``c_t / c_{t-k} - 1``.

    ``k = 1`` is the "4-week change in initial claims" at the monthly cadence. A positive
    value means claims are RISING (labour market softening).
    """
    c = frame["claims"]
    return c / c.shift(k) - 1.0


def sector_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Monthly simple returns of the four sector ETFs (index=month, columns=ticker)."""
    from . import data
    px = frame[data.SECTORS]
    return px.pct_change()


def cyc_def_spread(frame: pd.DataFrame) -> pd.Series:
    """Equal-weight **cyclical (XLY,XLI) minus defensive (XLP,XLU)** monthly return.

    Positive = cyclicals out-earned defensives that month (risk-on); the claim says a
    rising claims change should *predict a negative* value next month.
    """
    from . import data
    r = sector_returns(frame)
    cyc = r[data.CYCLICALS].mean(axis=1)
    dfn = r[data.DEFENSIVES].mean(axis=1)
    return (cyc - dfn).rename("spread")


def build_xy(frame: pd.DataFrame, k: int = 1, lag: int = 1):
    """(x, y, index): the claims change ``dcl_t`` and the forward spread ``spread_{t+lag}``.

    Vectorised, NaNs dropped. The forward alignment (``.shift(-lag)``) makes the signal
    strictly precede the outcome — one documented execution lag, zero look-ahead.
    """
    x = claims_change(frame, k=k).rename("x")
    fwd = cyc_def_spread(frame).shift(-lag).rename("y")
    d = pd.concat([x, fwd], axis=1).dropna()
    return d["x"].to_numpy(dtype=float), d["y"].to_numpy(dtype=float), d.index


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 6) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# The predictive Newey-West regression (the headline)
# --------------------------------------------------------------------------- #
def _hac_ols(x: np.ndarray, y: np.ndarray, lags: int = 6) -> dict:
    """OLS ``y = a + b x`` with Newey-West (HAC) standard errors on the slope.

    Fully vectorised (the HAC sandwich sums a small number of outer-product lags via
    matrix products, no per-observation loop). Returns slope, intercept, HAC-*t* of the
    slope, R², and n.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 5:
        return {"slope": float("nan"), "intercept": float("nan"),
                "t_nw": float("nan"), "r2": float("nan"), "n": n}
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    g = X * u[:, None]                       # n x 2 score contributions
    S = g.T @ g
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        G = g[l:].T @ g[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    r2 = 1.0 - u.var() / y.var() if y.var() > 0 else float("nan")
    return {"slope": float(beta[1]), "intercept": float(beta[0]),
            "t_nw": float(beta[1] / se[1]) if se[1] > 0 else float("nan"),
            "r2": float(r2), "n": n}


def predictive_regression(frame: pd.DataFrame, k: int = 1, lag: int = 1,
                          lags: int = 6) -> dict:
    """Predictive HAC regression of the forward cyclical-minus-defensive spread on the
    claims change. The claim holds only if the slope is **negative** (rising claims →
    cyclicals under-earn defensives)."""
    x, y, _ = build_xy(frame, k=k, lag=lag)
    out = _hac_ols(x, y, lags=lags)
    out["corr"] = float(np.corrcoef(x, y)[0, 1]) if len(x) > 3 else float("nan")
    out["spread_mean_bps"] = float(np.nanmean(y) * 1e4)
    return out


def era_regressions(frame: pd.DataFrame, split: str = "2012-01-01",
                    k: int = 1, lag: int = 1, lags: int = 6) -> dict:
    """The predictive slope/t in two eras split at ``split`` (a robustness cut)."""
    x, y, idx = build_xy(frame, k=k, lag=lag)
    early = idx < pd.Timestamp(split)
    late = ~early
    return {
        "split": split,
        "early": _hac_ols(x[early], y[early], lags=lags),
        "late": _hac_ols(x[late], y[late], lags=lags),
    }


def covid_sensitivity(frame: pd.DataFrame, k: int = 1, lag: int = 1, lags: int = 6,
                      lo: str = "2020-02-01", hi: str = "2020-12-31") -> dict:
    """The full slope vs the slope with the 2020 claims spike removed, and a
    winsorised (1/99 pct on the claims change) slope — is the fit one outlier?"""
    x, y, idx = build_xy(frame, k=k, lag=lag)
    keep = ~((idx >= pd.Timestamp(lo)) & (idx <= pd.Timestamp(hi)))
    q = np.percentile(x, [1, 99])
    xw = np.clip(x, q[0], q[1])
    return {
        "full": _hac_ols(x, y, lags=lags),
        "ex_covid": _hac_ols(x[keep], y[keep], lags=lags),
        "winsor": _hac_ols(xw, y, lags=lags),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the fitted slope a lucky alignment of the sort?
# --------------------------------------------------------------------------- #
def placebo_pvalue(frame: pd.DataFrame, k: int = 1, lag: int = 1,
                   n_draws: int = 2000, seed: int = 881) -> dict:
    """Shuffle the claims-change column against the forward spread (breaking the
    signal → outcome link) ``n_draws`` times; p = share of shuffled |slope| ≥ |observed|
    (two-sided). A real predictor => small p."""
    x, y, _ = build_xy(frame, k=k, lag=lag)
    n = len(x)
    if n < 5:
        return {"obs_slope": float("nan"), "placebo_mean": float("nan"),
                "placebo_sd": float("nan"), "p_value": float("nan"), "n_draws": 0}
    X0 = np.column_stack([np.ones(n), x])
    obs = float((np.linalg.inv(X0.T @ X0) @ X0.T @ y)[1])
    rng = np.random.default_rng(seed)
    xc = x - x.mean()
    denom = float(xc @ xc)
    yc = y - y.mean()
    draws = np.empty(n_draws)
    for i in range(n_draws):
        xp = rng.permutation(x)
        xpc = xp - xp.mean()
        draws[i] = float(xpc @ yc) / float(xpc @ xpc)
    p = float((np.abs(draws) >= abs(obs)).mean())
    return {"obs_slope": obs, "placebo_mean": float(draws.mean()),
            "placebo_sd": float(draws.std(ddof=1)), "p_value": p,
            "n_draws": n_draws, "draws": draws}


# --------------------------------------------------------------------------- #
# The costed rotation timer
# --------------------------------------------------------------------------- #
def rotation_timer(frame: pd.DataFrame, k: int = 1, lag: int = 1, thresh: float = 0.0,
                   cost_bps: float = 10.0, borrow_bps_yr: float = 50.0) -> dict:
    """Trade the claim as a long-short cyclical/defensive rotation.

    Position each month: **rising** claims (``dcl > thresh``) → tilt **defensive**
    (short the cyclical-minus-defensive spread, ``pos = -1``); **falling** claims → tilt
    **cyclical** (``pos = +1``). Realised return over month ``t+lag`` is
    ``pos_t · spread_{t+lag}``. A flip trades the whole 2×-NAV long-short book, so we
    charge ``|Δpos| · cost`` per leg (two legs) plus borrow on the short side. Gross and
    net both reported.
    """
    dcl = claims_change(frame, k=k)
    spread = cyc_def_spread(frame)
    pos = -np.sign(dcl - thresh)     # rising -> defensive (-1), falling -> cyclical (+1)
    d = pd.concat([pos.rename("pos"), spread.shift(-lag).rename("fwd")], axis=1).dropna()
    p = d["pos"].to_numpy(dtype=float)
    fwd = d["fwd"].to_numpy(dtype=float)
    n = len(p)
    gross = p * fwd
    # turnover: |Δpos| units of NAV traded per leg (first month opens the book)
    dpos = np.abs(np.diff(p, prepend=0.0))
    c = cost_bps / 1e4
    borrow = (borrow_bps_yr / 1e4) / 12.0
    cost = dpos * 2.0 * c + borrow           # two legs per unit turn, monthly borrow
    net = gross - cost
    n_switch = int((np.abs(np.diff(p)) > 0).sum())

    def _ann(v):
        mu = v.mean() * MONTHS
        vol = v.std(ddof=1) * np.sqrt(MONTHS) if n > 1 else float("nan")
        return mu, (mu / vol if vol and vol > 0 else float("nan"))

    g_mu, g_sh = _ann(gross)
    n_mu, n_sh = _ann(net)
    return {
        "n_months": n, "n_switches": n_switch,
        "gross_ann_pct": g_mu * 100, "gross_sharpe": g_sh, "gross_t": one_sample_t(gross),
        "net_ann_pct": n_mu * 100, "net_sharpe": n_sh, "net_t": one_sample_t(net),
        "cost_bps": cost_bps,
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(frame: pd.DataFrame, k: int = 1, lag: int = 1) -> dict:
    """Run the predictive regression on a synthetic frame (slope, HAC t, n)."""
    r = predictive_regression(frame, k=k, lag=lag)
    return {"slope": r["slope"], "t_nw": r["t_nw"], "r2": r["r2"], "n": r["n"]}
