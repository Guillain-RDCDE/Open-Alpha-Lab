"""Strategy + inference for Study 729 — "the ramen index" as a downturn tell.

The claim: instant-noodle demand is a leading tell of hard times — when a downturn is
coming, people trade down to cheap noodles, so the "ramen index" *rises ahead of* a
recession / an equity drawdown, and the noodle makers are a defensive place to hide. We test
the strongest tradable version of that:

1. **Does the ramen index LEAD?** — the headline test. Cross-correlate the (cited,
   approximate) WINA world-demand growth against the market's return at various leads/lags.
   A genuine downturn tell shows a *negative* correlation that peaks when demand-growth
   *precedes* a bad market year (index up now → market down later). We also compare
   demand-growth inside vs outside NBER recession years.
2. **Is the noodle maker defensive?** — split the tape into up- and down-market months and
   measure Nissin/Toyo Suisan's beta on each side (bear-beta < bull-beta < 1 = defensive),
   and their recession-window excess vs the Nikkei.
3. **CAPM alpha** — Newey-West (HAC) *t* of the full-sample alpha vs the Nikkei: any excess
   after paying for market beta?
4. **The tradability kill** — WINA publishes a year's demand only mid-*next*-year, and NBER
   dates a recession ~12 months ex-post, so a "ramen says a recession is coming" rule is a
   double look-ahead; and holding the noodle maker "for defense" carried an opportunity cost.

Inference: a lead-lag cross-correlation with a small-sample *t*, bull/bear conditional betas,
a Newey-West (HAC) CAPM alpha *t*, and a recession-window paired excess *t*. Pure
numpy/pandas; scipy only for the *t*-distribution p-value (optional, guarded).
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


def _corr_t(r: float, n: int) -> float:
    """Small-sample *t* of a Pearson correlation: t = r*sqrt((n-2)/(1-r^2))."""
    if not np.isfinite(r) or n < 3 or abs(r) >= 1.0:
        return float("nan")
    return float(r * math.sqrt((n - 2) / (1.0 - r * r)))


# --------------------------------------------------------------------------- #
# THE HEADLINE TEST — does the ramen index LEAD the market?
# --------------------------------------------------------------------------- #
def lead_lag_corr(index_growth: pd.Series, market_ret: pd.Series,
                  leads: range = range(-2, 3)) -> dict:
    """Cross-correlation of the ramen-index growth against the market return at each lead.

    ``lead = k`` correlates ``index_growth`` at year *t* with ``market_ret`` at year
    *t + k*: a **positive** ``lead`` means the index *precedes* the market. A genuine
    downturn tell predicts a **negative** correlation at a positive lead (index up now →
    market down later). Both series are aligned on the calendar year. Returns
    ``{lead: {r, t, p, n}}`` plus the ``best_lead`` (largest |t| at lead ≥ 1) — the honest
    "does it lead?" statistic on a small annual sample.
    """
    a = index_growth.copy()
    b = market_ret.copy()
    a.index = a.index.year
    b.index = b.index.year
    per_lead: dict[int, dict] = {}
    for k in leads:
        pairs = pd.concat([a, b.shift(-k)], axis=1, keys=["idx", "mkt"]).dropna()
        n = len(pairs)
        if n < 3:
            per_lead[k] = {"r": float("nan"), "t": float("nan"), "p": float("nan"), "n": n}
            continue
        r = float(np.corrcoef(pairs["idx"], pairs["mkt"])[0, 1])
        t = _corr_t(r, n)
        per_lead[k] = {"r": r, "t": t, "p": _t_p_value(t, n - 2), "n": n}
    lead_ge1 = {k: v for k, v in per_lead.items() if k >= 1 and np.isfinite(v["t"])}
    best = max(lead_ge1, key=lambda k: abs(lead_ge1[k]["t"])) if lead_ge1 else None
    return {"per_lead": per_lead, "best_lead": best,
            "best": per_lead[best] if best is not None else None}


def demand_in_vs_out_recession(index_growth: pd.Series, rec_years: set) -> dict:
    """Mean ramen-index growth in NBER recession years vs all other years (a Welch *t*).

    The folklore says demand *jumps* in a downturn; this is the direct contrast. ``rec_years``
    is the set of calendar years containing an NBER recession month. Returns the two means, the
    difference, and a two-sample (Welch) *t* — on a tiny sample, so a weak test by design.
    """
    g = index_growth.copy()
    g.index = g.index.year
    inr = g[[y in rec_years for y in g.index]]
    out = g[[y not in rec_years for y in g.index]]
    ni, no = len(inr), len(out)
    if ni < 2 or no < 2:
        return {"in_mean": float("nan"), "out_mean": float("nan"),
                "diff": float("nan"), "t": float("nan"), "p": float("nan"),
                "n_in": ni, "n_out": no}
    vi, vo = inr.var(ddof=1), out.var(ddof=1)
    se = math.sqrt(vi / ni + vo / no)
    diff = float(inr.mean() - out.mean())
    t = diff / se if se > 0 else float("nan")
    # Welch–Satterthwaite dof
    dof = (vi / ni + vo / no) ** 2 / ((vi / ni) ** 2 / (ni - 1) + (vo / no) ** 2 / (no - 1))
    return {"in_mean": float(inr.mean()), "out_mean": float(out.mean()), "diff": diff,
            "t": float(t), "p": _t_p_value(t, dof), "n_in": ni, "n_out": no}


# --------------------------------------------------------------------------- #
# The systematic-defensiveness test — bull vs bear beta + CAPM alpha
# --------------------------------------------------------------------------- #
def _slope(y: np.ndarray, x: np.ndarray) -> float:
    """OLS slope of y on x (single regressor, with intercept)."""
    if len(x) < 2 or np.var(x, ddof=1) == 0:
        return float("nan")
    return float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))


def bull_bear_beta(stock_ret: pd.Series, bench_ret: pd.Series,
                   split: float = 0.0) -> dict:
    """Conditional beta in down (bench < ``split``) vs up (bench >= ``split``) months.

    A genuinely *defensive* name has ``down_beta`` < ``up_beta`` (and both < 1). ``split=0.0``
    is the standard bear/bull split. No look-ahead (a contemporaneous risk metric).
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


def newey_west_alpha_t(stock_ret: pd.Series, bench_ret: pd.Series,
                       lags: int = 6) -> dict:
    """Newey-West (HAC) *t* of the monthly CAPM alpha from r_stock = a + b*r_bench + e.

    Returns the OLS alpha (monthly + annualised), beta, the HAC standard error of alpha and
    its *t*. ``lags`` is the Bartlett-kernel truncation. The Signal-axis statistic: after
    paying for market beta, is there any excess to the "noodle-defensiveness" trade?
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
# The recession-window test — noodle-maker returns vs the Nikkei
# --------------------------------------------------------------------------- #
def recession_excess_t(stock_ret: pd.Series, bench_ret: pd.Series, rec_mask_fn) -> dict:
    """Paired *t* that the stock's monthly return **beats** the Nikkei during recession months.

    ``rec_mask_fn(index)`` returns a boolean mask of recession months. The defensive claim
    predicts a *positive* mean excess with *t* > 2. Also reports the compounded recession-
    window return of each leg.
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
    """Per-recession compounded return of the stock vs the Nikkei (the aggregate's provenance).

    ``recessions`` is a list of ``(name, start, end)`` tuples. Returns
    ``{name: {stock, bench, n}}`` — exposing whether an aggregate "beat the market in
    recessions" is broad or driven by one idiosyncratic window.
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
# Tradability — the opportunity cost of holding "noodles for defense"
# --------------------------------------------------------------------------- #
def terminal_wealth(level: pd.Series, start: float = 1.0) -> float:
    """Growth of ``start`` invested at the first date to the last (buy-and-hold)."""
    return float(start * level.iloc[-1] / level.iloc[0])


# --------------------------------------------------------------------------- #
# Positive controls
# --------------------------------------------------------------------------- #
def control_recovers_lead(index_growth: pd.Series, market_ret: pd.Series,
                          planted_lead: int) -> dict:
    """Positive control: the lead-lag engine recovers a *planted* lead (a real tell)."""
    ll = lead_lag_corr(index_growth, market_ret, leads=range(-2, 4))
    return {
        "best_lead": ll["best_lead"],
        "best_r": ll["best"]["r"] if ll["best"] else float("nan"),
        "best_t": ll["best"]["t"] if ll["best"] else float("nan"),
        "planted_lead": planted_lead,
        "recovered_lead_ok": int(ll["best_lead"] == planted_lead),
    }


def control_recovers_defensive(stock_ret: pd.Series, mkt_ret: pd.Series,
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
