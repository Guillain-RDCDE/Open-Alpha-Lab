"""The engine and its honest controls — Study 577 (MBS-OAS-Signal).

The claim, at full strength: the agency-MBS **option-adjusted spread** leads risk assets. A
*widening* in OAS this week signals mortgage investors demanding more risk compensation, and that
risk-off move front-runs weakness in equities and credit — so a rising OAS should predict *low*
forward returns, and a rule that de-risks when OAS widens should dodge drawdowns.

This module measures that lead, honestly:

1. **The signal.** The standardised weekly OAS change ``d_oas_z`` (a widening is positive), known at
   the close of week *t*. (We also expose a *level*-based z as an alternative view.) The lead
   hypothesis is that this predicts *next* week's return with a **negative** sign.

2. **The predictive test.** A one-variable OLS of forward return on the signal — the *slope* and its
   *t*-stat. A negative slope at ``|t| >= 2`` is the folklore. We report the slope, its t, and the
   in-sample R^2.

3. **A timing overlay.** A tradable expression: hold the risk asset, but **step to cash** the week
   after OAS widens beyond a threshold (a risk-off switch). We report its gross and net return vs
   buy-and-hold, with realistic costs — the strategy trades, so it pays.

4. **A placebo / label-shuffle null.** Shuffle the signal against the forward returns many times; the
   p-value is the tail probability of the observed |slope-t|. A real lead sits in the tail.

5. **A robustness sweep.** Re-estimate the slope-t across several signal definitions / horizons; a
   real lead keeps its sign.

6. **A synthetic positive control.** A deterministic world where the lead is planted
   (``lead_beta < 0``); the engine must recover a negative slope-t and stay flat at the null,
   averaged over >= 20 seeds so no single lucky RNG seed manufactures significance.

Execution convention: the signal is measured at the close of week *t*; the position it implies is
held over week *t+1* (the ``fwd_ret`` column). That is the single, explicit execution lag — the
timing overlay never acts on a return it could not have known was coming.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52
SEED = 577  # placebo / control default seed


# ---------------------------------------------------------------------------
# The predictive relation — the sign IS the claim
# ---------------------------------------------------------------------------
def predictive_regression(frame: pd.DataFrame, signal_col: str = "d_oas_z") -> dict:
    """OLS of forward return on the OAS signal across all weeks.

    Slope < 0 (at |t| >= 2) = the risk-off lead (OAS widening -> lower forward return). Returns the
    slope, its t-stat, the Pearson correlation, the in-sample R^2, and n.
    """
    df = frame.dropna(subset=[signal_col, "fwd_ret"])
    if len(df) < 8:
        return {"slope": float("nan"), "slope_t": float("nan"), "corr": float("nan"),
                "r2": float("nan"), "n": 0}
    x = df[signal_col].to_numpy(dtype=float)
    y = df["fwd_ret"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, kk = X.shape
    dof = max(n - kk, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_slope = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return {
        "slope": float(coef[1]),
        "slope_t": float(coef[1] / se_slope) if se_slope > 0 else float("nan"),
        "corr": corr,
        "r2": float(r2),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# The timing overlay — a tradable risk-off switch
# ---------------------------------------------------------------------------
def timing_overlay(
    frame: pd.DataFrame,
    signal_col: str = "d_oas_z",
    widen_z: float = 1.0,
    cost_bps: float = 5.0,
) -> dict:
    """De-risk to cash the week after OAS widens past ``widen_z`` sd; else hold the risk asset.

    The signal at week *t* decides the week-*t+1* position (``fwd_ret``): if OAS widened by more than
    ``widen_z`` standardised units, step to cash (position 0) for that forward week; otherwise stay
    fully invested (position 1). Costs are charged one-way on every change of position (a switch in
    *or* out), on NAV.

    Returns gross/net timed CAGR, buy-and-hold CAGR, the Sharpe of each, weeks in cash, and switches.
    """
    df = frame.dropna(subset=[signal_col, "fwd_ret"]).copy()
    if len(df) < 8:
        return {}
    sig = df[signal_col].to_numpy(dtype=float)
    fwd = df["fwd_ret"].to_numpy(dtype=float) / 100.0  # % -> fraction
    pos = np.where(sig > widen_z, 0.0, 1.0)            # de-risk the week AFTER a widening

    # costs on position changes (entering the sample = enter from flat)
    prev = np.concatenate([[0.0], pos[:-1]])
    turnover = np.abs(pos - prev)
    cost = turnover * (cost_bps * 1e-4)

    timed_gross = pos * fwd
    timed_net = timed_gross - cost
    bh = fwd

    def _cagr(r):
        g = float(np.prod(1.0 + r))
        yrs = len(r) / WEEKS_PER_YEAR
        return g ** (1.0 / yrs) - 1.0 if yrs > 0 and g > 0 else float("nan")

    def _sharpe(r):
        sd = r.std(ddof=1)
        return float(r.mean() / sd * np.sqrt(WEEKS_PER_YEAR)) if sd > 0 else float("nan")

    return {
        "bh_cagr": _cagr(bh),
        "timed_gross_cagr": _cagr(timed_gross),
        "timed_net_cagr": _cagr(timed_net),
        "bh_sharpe": _sharpe(bh),
        "timed_gross_sharpe": _sharpe(timed_gross),
        "timed_net_sharpe": _sharpe(timed_net),
        "weeks_in_cash": int((pos == 0).sum()),
        "n_weeks": int(len(pos)),
        "switches": int((turnover > 0).sum()),
        "widen_z": float(widen_z),
        "cost_bps": float(cost_bps),
    }


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(
    frame: pd.DataFrame, signal_col: str = "d_oas_z", n_perm: int = 2000, seed: int = SEED
) -> float:
    """Label-shuffle placebo p-value for the predictive slope-t.

    Shuffle the signal against forward returns ``n_perm`` times; the p-value is the fraction of
    shuffles whose |slope-t| >= the observed |slope-t|. A real lead sits in the tail; noise does not.
    """
    df = frame.dropna(subset=[signal_col, "fwd_ret"])
    if len(df) < 8:
        return float("nan")
    obs = abs(predictive_regression(df, signal_col)["slope_t"])
    x = df[signal_col].to_numpy(dtype=float)
    y = df["fwd_ret"].to_numpy(dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        xs = x[rng.permutation(n)]
        X = np.column_stack([np.ones_like(xs), xs])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        dof = max(n - 2, 1)
        sigma2 = float(resid @ resid) / dof
        se = float(np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1]))
        t = coef[1] / se if se > 0 else 0.0
        if abs(t) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Robustness — slope-t across signal definitions
# ---------------------------------------------------------------------------
def robustness_sweep(frame: pd.DataFrame) -> list[tuple[str, float, float]]:
    """Slope-t of forward return on several signal definitions.

    Returns a list of ``(label, slope, slope_t)`` for: the standardised weekly change (headline),
    the OAS *level* z-score, a 4-week change, and a 4-week momentum of the level. A real lead keeps
    its (negative) sign across reasonable definitions.
    """
    df = frame.copy()
    # level z
    lvl = df["oas"]
    df["level_z"] = (lvl - lvl.mean()) / lvl.std(ddof=1)
    # 4-week change, standardised
    d4 = df["oas"].diff(4)
    df["d4_z"] = (d4 - d4.mean()) / d4.std(ddof=1)
    out = []
    for lab, col in [
        ("weekly change (headline)", "d_oas_z"),
        ("level z-score", "level_z"),
        ("4-week change", "d4_z"),
    ]:
        r = predictive_regression(df, col)
        out.append((lab, r["slope"], r["slope_t"]))
    return out


# ---------------------------------------------------------------------------
# Synthetic positive control — seed-robust (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, lead_beta: float, n_seeds: int = 25, base_seed: int = SEED) -> float:
    """Average the predictive slope-t over ``n_seeds`` synthetic worlds for a planted ``lead_beta``.

    The house rule: any synthetic-dependent claim averages the stat over >= 20 seeds so no single
    lucky RNG seed can manufacture significance. Returns the mean slope-t across seeds.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        frame, _ = data_mod.synthetic_series(lead_beta=lead_beta, seed=s)
        ts.append(predictive_regression(frame)["slope_t"])
    return float(np.nanmean(ts))
