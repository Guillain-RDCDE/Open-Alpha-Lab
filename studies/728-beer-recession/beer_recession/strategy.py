"""Strategy + inference for Study 728 — "beer stocks are defensive/counter-cyclical".

The claim: because people supposedly *drink more beer in a recession*, beer stocks are a
defensive, counter-cyclical hold — they fall less than the market, and they beat it when a
downturn hits. We test the strongest tradable version of that against SPY:

1. **Downside beta** — split the tape into down-market (SPY < 0) and up-market months and
   measure beta on each side. A defensive name has *bear-beta < bull-beta* (and both < 1):
   it participates less on the way down. This is the systematic-defensiveness test.
2. **Recession-window returns** — over the three NBER recessions in the sample, is each
   name's monthly return *above* SPY's? Paired-*t* of the recession-window excess. This is
   the direct counter-cyclical test the folklore actually asserts.
3. **CAPM alpha** — Newey-West (HAC) *t* of the full-sample alpha vs SPY: after adjusting
   for the (low) market beta, is there any excess return to the "beer defensiveness" at all?
4. **The tradability kill** — NBER dates a recession only ex-post (a ~12-month announce
   lag), so a "rotate into beer at the downturn" rule is a look-ahead; and holding beer
   "for defense" over the full sample cost a large opportunity-cost vs SPY.

Inference: bull/bear conditional betas, a Newey-West (HAC) CAPM alpha *t*, and a
recession-window paired excess *t*. Pure numpy/pandas; scipy only for the *t*-distribution
p-value (optional, guarded).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MONTHS = 12.0


# --------------------------------------------------------------------------- #
# Return / risk primitives
# --------------------------------------------------------------------------- #
def cagr(level: pd.Series) -> float:
    """Compound annual growth rate of a level series (index carries the dates)."""
    yrs = (level.index[-1] - level.index[0]).days / 365.25
    if yrs <= 0 or level.iloc[0] <= 0:
        return float("nan")
    return (level.iloc[-1] / level.iloc[0]) ** (1.0 / yrs) - 1.0


def max_drawdown(level: pd.Series) -> float:
    """Worst peak-to-trough drawdown (a negative fraction)."""
    roll_max = level.cummax()
    dd = level / roll_max - 1.0
    return float(dd.min())


def ann_vol(returns: pd.Series, periods_per_year: float = MONTHS) -> float:
    """Annualised volatility of periodic simple returns."""
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe(returns: pd.Series, rf_annual: float = 0.0,
           periods_per_year: float = MONTHS) -> float:
    """Annualised Sharpe of periodic simple returns (excess of a flat rf)."""
    rf_p = (1 + rf_annual) ** (1 / periods_per_year) - 1
    ex = returns - rf_p
    sd = ex.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(ex.mean() / sd * math.sqrt(periods_per_year))


def summarize(level: pd.Series, periods_per_year: float = MONTHS) -> dict:
    """CAGR / annualised vol / Sharpe / max-drawdown of a level series."""
    rets = level.pct_change().dropna()
    return {
        "cagr": cagr(level),
        "vol": ann_vol(rets, periods_per_year),
        "sharpe": sharpe(rets, 0.0, periods_per_year),
        "mdd": max_drawdown(level),
        "n": int(len(rets)),
    }


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def _t_p_value(t: float, df: int) -> float:
    """Two-sided p-value for a t-stat (scipy if available, else a normal approx)."""
    if not np.isfinite(t):
        return float("nan")
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), df))
    except Exception:  # pragma: no cover - scipy is in requirements but stay safe
        return float(2 * 0.5 * math.erfc(abs(t) / math.sqrt(2)))


def newey_west_alpha_t(stock_ret: pd.Series, bench_ret: pd.Series,
                       lags: int = 6) -> dict:
    """Newey-West (HAC) *t* of the monthly CAPM alpha from r_stock = a + b*r_bench + e.

    Returns the OLS alpha (monthly + annualised), beta, the HAC standard error of alpha
    and its *t*. ``lags`` is the Bartlett-kernel truncation. The Signal-axis statistic:
    after paying for market beta, is there any excess to "beer defensiveness"?
    """
    j = pd.concat([stock_ret, bench_ret], axis=1, keys=["y", "x"]).dropna()
    y = j["y"].to_numpy()
    x = j["x"].to_numpy()
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        Xe = X * resid[:, None]
        Gamma = Xe[L:].T @ Xe[:-L]
        S += w * (Gamma + Gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    se_alpha = math.sqrt(cov[0, 0])
    a_m = float(beta[0])
    t_a = a_m / se_alpha if se_alpha > 0 else float("nan")
    return {
        "alpha_m": a_m,
        "alpha_ann": (1 + a_m) ** 12 - 1,
        "beta": float(beta[1]),
        "se_alpha": float(se_alpha),
        "t_alpha": float(t_a),
        "p_alpha": _t_p_value(t_a, n - 2),
        "n": n,
    }


# --------------------------------------------------------------------------- #
# The systematic-defensiveness test — bull vs bear beta
# --------------------------------------------------------------------------- #
def _slope(y: np.ndarray, x: np.ndarray) -> float:
    """OLS slope of y on x (single regressor, with intercept)."""
    if len(x) < 2 or np.var(x, ddof=1) == 0:
        return float("nan")
    return float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))


def bull_bear_beta(stock_ret: pd.Series, bench_ret: pd.Series,
                   split: float = 0.0) -> dict:
    """Conditional beta in down (bench < ``split``) vs up (bench >= ``split``) months.

    A genuinely *defensive* name has ``down_beta`` < ``up_beta`` (and both < 1): it
    participates less on the way down than on the way up. ``split=0.0`` is the standard
    bear/bull split; pass the bench mean for the Ang–Chen–Xing convention. Also returns
    the full-sample OLS beta for reference. No look-ahead (contemporaneous risk metric).
    """
    j = pd.concat([stock_ret, bench_ret], axis=1, keys=["y", "x"]).dropna()
    down = j[j["x"] < split]
    up = j[j["x"] >= split]
    db = _slope(down["y"].to_numpy(), down["x"].to_numpy())
    ub = _slope(up["y"].to_numpy(), up["x"].to_numpy())
    fb = _slope(j["y"].to_numpy(), j["x"].to_numpy())
    return {
        "down_beta": db,
        "up_beta": ub,
        "full_beta": fb,
        "asymmetry": db - ub,        # < 0 = defensive (loads LESS on the downside)
        "defensive": int(db < ub and db < 1.0),
        "n_down": int(len(down)),
        "n_up": int(len(up)),
    }


# --------------------------------------------------------------------------- #
# The direct counter-cyclical test — recession-window returns vs SPY
# --------------------------------------------------------------------------- #
def recession_excess_t(stock_ret: pd.Series, bench_ret: pd.Series,
                       rec_mask_fn) -> dict:
    """Paired *t* that the stock's monthly return **beats** SPY during recession months.

    ``rec_mask_fn(index)`` returns a boolean mask of recession months. The counter-cyclical
    claim predicts a *positive* mean excess with *t* > 2 — the stock outperforms when the
    economy contracts. Also reports the compounded recession-window return of each leg.
    """
    j = pd.concat([stock_ret, bench_ret], axis=1, keys=["y", "x"]).dropna()
    mask = rec_mask_fn(j.index)
    jr = j[mask]
    ex = jr["y"] - jr["x"]
    n = len(ex)
    if n < 2:
        return {"mean_excess": float("nan"), "t": float("nan"), "p": float("nan"), "n": n}
    se = ex.std(ddof=1) / math.sqrt(n)
    t = ex.mean() / se if se > 0 else float("nan")
    return {
        "mean_excess": float(ex.mean()),
        "t": float(t),
        "p": _t_p_value(t, n - 1),
        "n": n,
        "mean_stock": float(jr["y"].mean()),
        "mean_bench": float(jr["x"].mean()),
        "cum_stock": float((1 + jr["y"]).prod() - 1),
        "cum_bench": float((1 + jr["x"]).prod() - 1),
    }


def recession_breakdown(stock_ret: pd.Series, bench_ret: pd.Series,
                        recessions: list) -> dict:
    """Per-recession compounded return of the stock vs SPY (the aggregate's provenance).

    ``recessions`` is a list of ``(name, start, end)`` tuples. Returns
    ``{name: {stock, bench, n}}`` with compounded % over each window — exposing whether an
    aggregate "beat SPY in recessions" is broad or driven by one idiosyncratic window.
    """
    j = pd.concat([stock_ret, bench_ret], axis=1, keys=["y", "x"]).dropna()
    out = {}
    for name, a, b in recessions:
        m = (j.index >= pd.Timestamp(a)) & (j.index <= pd.Timestamp(b))
        w = j[m]
        out[name] = {
            "stock": float((1 + w["y"]).prod() - 1),
            "bench": float((1 + w["x"]).prod() - 1),
            "n": int(len(w)),
        }
    return out


# --------------------------------------------------------------------------- #
# Tradability — the opportunity cost of holding "beer for defense"
# --------------------------------------------------------------------------- #
def terminal_wealth(level: pd.Series, start: float = 1.0) -> float:
    """Growth of ``start`` invested at the first date to the last (buy-and-hold)."""
    return float(start * level.iloc[-1] / level.iloc[0])


def control_recovers(stock_ret: pd.Series, mkt_ret: pd.Series,
                     planted_down: float, planted_up: float) -> dict:
    """Positive control: bull/bear-beta recovers a *planted* asymmetric (defensive) beta."""
    bb = bull_bear_beta(stock_ret, mkt_ret)
    return {
        "down_beta": bb["down_beta"],
        "up_beta": bb["up_beta"],
        "planted_down": planted_down,
        "planted_up": planted_up,
        "recovered_defensive": int(bb["down_beta"] < bb["up_beta"]),
    }
