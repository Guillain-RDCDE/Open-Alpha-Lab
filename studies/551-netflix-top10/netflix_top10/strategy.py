"""The engine and its honest controls — Study 551 (Netflix-Top10).

The claim, at full strength: **streaming-engagement momentum** — the acceleration in how many
hours the world watches Netflix's Top-10 — *predicts* forward returns for NFLX and, by spillover,
consumer-discretionary (XLY). The tradable expression is a timing rule: hold the stock when
engagement momentum is high, stand aside (or short) when it is low.

This module measures that honestly on the synthetic weekly world (no real tape exists for a
no-key stack — see ``data.py``):

1. **The signal.** ``eng_mom`` — the standardised gap between the engagement index and its
   trailing mean (higher = engagement accelerating). Formed from data up to week ``t`` only.

2. **The predictive regression.** OLS of the *forward* return on ``eng_mom``. The slope's sign
   and *t*-stat are the headline test (the claim = a positive, significant slope for NFLX).

3. **The placebo null.** Shuffle the engagement-momentum labels against forward returns; the
   p-value is the fraction of shuffles whose |slope| >= the observed. A real link sits in the
   tail; noise does not.

4. **The timing backtest with costs.** A long/flat rule (in when ``eng_mom`` > 0) and its
   gross vs net return; when the rule also *shorts* the low-engagement weeks, the short leg pays
   a borrow. Costs are one-way x NAV on every position change.

5. **A robustness sweep.** Re-run the regression across forward horizons (2/4/8/13 weeks) — a
   real signal is not knife-edge to one horizon.

6. **A seed-robust synthetic positive control.** Average the slope-*t* over >= 20 seeds at a
   planted ``beta`` and at the null, proving the engine catches a genuine link and stays flat
   when there is none.

Execution convention: the signal at week ``t`` uses engagement up to ``t`` (no look-ahead); the
position is entered at that week's close and held over the forward window. A one-week-later entry
is a documented convention that moves the headline slope negligibly and changes no stamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52


# ---------------------------------------------------------------------------
# The predictive regression — the sign & t of the slope ARE the claim
# ---------------------------------------------------------------------------
def predictive_regression(weekly: pd.DataFrame, target: str = "nflx_fwd") -> dict:
    """OLS of the forward return (``target``) on engagement momentum.

    Slope > 0 with *t* >= 2 is the claim (engagement momentum predicts forward returns).
    Returns the slope, its *t*-stat (HAC-lite: plain OLS SE — the forward windows overlap, so
    we also expose ``t_nw`` with a Newey-West correction for the overlap), the correlation and n.
    """
    if weekly.empty or len(weekly) < 8:
        return {"slope": float("nan"), "slope_t": float("nan"), "t_nw": float("nan"),
                "corr": float("nan"), "n": 0}
    x = weekly["eng_mom"].to_numpy(dtype=float)
    y = weekly[target].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, kk = X.shape
    dof = max(n - kk, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_ols = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    slope_t = float(coef[1] / se_ols) if se_ols > 0 else float("nan")

    # Newey-West on the slope for the overlapping forward windows (lag = horizon guess ~ 4)
    t_nw = _newey_west_slope_t(x, y, coef, lag=4)

    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return {
        "slope": float(coef[1]),
        "slope_t": slope_t,
        "t_nw": t_nw,
        "corr": corr,
        "n": int(n),
    }


def _newey_west_slope_t(x, y, coef, lag: int = 4) -> float:
    """Newey-West HAC t-stat for the slope of a simple OLS (overlap-robust)."""
    n = len(x)
    if n < lag + 3:
        return float("nan")
    xc = x - x.mean()
    resid = y - (coef[0] + coef[1] * x)
    sxx = float(xc @ xc)
    if sxx <= 0:
        return float("nan")
    u = xc * resid                       # score for the slope
    s0 = float(u @ u)
    for L in range(1, lag + 1):
        w = 1.0 - L / (lag + 1.0)
        cov = float(u[L:] @ u[:-L])
        s0 += 2.0 * w * cov
    var_slope = s0 / (sxx ** 2)
    se = np.sqrt(var_slope) if var_slope > 0 else float("nan")
    return float(coef[1] / se) if se and se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(weekly: pd.DataFrame, target: str = "nflx_fwd",
                   n_perm: int = 2000, seed: int = 551) -> float:
    """Label-shuffle placebo p-value for the predictive slope.

    Shuffle ``eng_mom`` against the forward returns ``n_perm`` times; the p-value is the fraction
    of shuffles whose |slope| >= the observed |slope|.
    """
    if weekly.empty or len(weekly) < 8:
        return float("nan")
    obs = abs(predictive_regression(weekly, target)["slope"])
    rng = np.random.default_rng(seed)
    y = weekly[target].to_numpy(dtype=float)
    x = weekly["eng_mom"].to_numpy(dtype=float)
    xc = x - x.mean()
    sxx = float(xc @ xc)
    if sxx <= 0:
        return float("nan")
    count = 0
    for _ in range(n_perm):
        ys = rng.permutation(y)
        slope = float(xc @ (ys - ys.mean())) / sxx
        if abs(slope) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# The timing backtest with costs + borrow
# ---------------------------------------------------------------------------
def timing_backtest(weekly: pd.DataFrame, target_ret: str = "nflx_ret",
                    allow_short: bool = False, cost_bps: float = 5.0,
                    borrow_ann_bps: float = 100.0) -> dict:
    """Long/flat (or long/short) timing on engagement momentum, gross and net.

    Rule: position at week ``t`` = +1 if ``eng_mom`` > 0 (engagement accelerating); else 0
    (long/flat) or -1 (long/short, when ``allow_short``). The position earns the *following*
    week's return ``target_ret`` — one execution lag. Costs are one-way x NAV on every position
    change; the short leg (if any) pays an annual borrow pro-rated to weekly.

    Returns gross/net total & annualised return, the buy-and-hold benchmark, and turnover.
    """
    if weekly.empty or len(weekly) < 8:
        return {}
    w = weekly.sort_index()
    sig = np.where(w["eng_mom"].to_numpy() > 0, 1.0, (-1.0 if allow_short else 0.0))
    r = w[target_ret].to_numpy(dtype=float)
    # position formed at t, earns r[t+1]; align by shifting the realised return back one step
    pos = sig[:-1]
    fwd = r[1:]
    strat = pos * fwd

    # costs: one-way per unit turnover on each position change (entry & exit both charged)
    turns = np.abs(np.diff(np.concatenate([[0.0], pos])))
    cost = turns * cost_bps * 1e-4
    # borrow on the short leg (position < 0), weekly pro-rate
    borrow = np.where(pos < 0, borrow_ann_bps * 1e-4 / WEEKS_PER_YEAR, 0.0)
    net = strat - cost - borrow

    def _ann(total_growth, n_weeks):
        yrs = n_weeks / WEEKS_PER_YEAR
        return float(total_growth ** (1.0 / yrs) - 1.0) if yrs > 0 and total_growth > 0 else float("nan")

    n = len(fwd)
    gross_growth = float(np.prod(1.0 + strat))
    net_growth = float(np.prod(1.0 + net))
    bh_growth = float(np.prod(1.0 + fwd))
    return {
        "gross_total": gross_growth - 1.0,
        "net_total": net_growth - 1.0,
        "buyhold_total": bh_growth - 1.0,
        "gross_ann": _ann(gross_growth, n),
        "net_ann": _ann(net_growth, n),
        "buyhold_ann": _ann(bh_growth, n),
        "turnover": float(turns.sum()),
        "weeks": int(n),
        "allow_short": allow_short,
    }


# ---------------------------------------------------------------------------
# Robustness — across forward horizons
# ---------------------------------------------------------------------------
def horizon_sweep(data_mod, beta: float, horizons=(2, 4, 8, 13), seed: int = 551) -> list:
    """Re-fit the predictive regression across forward horizons for one synthetic world.

    Returns a list of ``(horizon, slope, slope_t, t_nw)`` — a real signal is not knife-edge to
    a single horizon.
    """
    out = []
    for h in horizons:
        weekly, _ = data_mod.synthetic_series(beta=beta, horizon=h, seed=seed)
        reg = predictive_regression(weekly, "nflx_fwd")
        out.append((h, reg["slope"], reg["slope_t"], reg["t_nw"]))
    return out


# ---------------------------------------------------------------------------
# Seed-robust synthetic positive control (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, beta: float, n_seeds: int = 25, base_seed: int = 551) -> float:
    """Average the predictive slope-*t* over ``n_seeds`` synthetic worlds at a planted ``beta``.

    House rule: any synthetic-dependent claim averages a Welch/OLS *t* over >= 20 seeds so no
    single lucky RNG seed can manufacture significance. Returns the mean slope-*t* (NW, overlap-
    robust) across seeds.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        weekly, _ = data_mod.synthetic_series(beta=beta, seed=s)
        ts.append(predictive_regression(weekly, "nflx_fwd")["t_nw"])
    return float(np.nanmean(ts))
