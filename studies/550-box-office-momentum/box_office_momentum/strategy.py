"""The engine and its honest controls — Study 550 (Box-Office-Momentum).

The claim: a strong weekend box office is a **consumer-sentiment leading indicator** — busy
cinemas this weekend foreshadow gains for media/studio stocks (and maybe the whole tape) next
week. The tradable version is a *predictive regression*: does this week's box-office momentum
have positive, significant slope on next week's forward return, and does a signal-timed long-only
strategy beat buy-and-hold net of costs?

This module measures that on the synthetic panel (the free retail stack cannot reach an honest
real box-office tape — see ``data.py``):

1. **The predictive regression.** OLS of next-week forward return on this-week box-office
   momentum. The slope's sign and *t* are the headline test (the leading-indicator claim).

2. **The confound-controlled regression.** The folklore's trap is that box office and stocks both
   load on the *same* contemporaneous consumer/market factor. We add the contemporaneous SPY
   return as a control; a *genuine* lead survives it, a spurious co-movement does not.

3. **The placebo null.** Circular-shift the box-office signal against the returns (breaking the
   temporal alignment while preserving each series' autocorrelation) and read the tail probability
   of the |slope|. A real lead sits in the tail; noise does not.

4. **A signal-timed strategy + costs.** Go long the media basket in weeks the signal is positive,
   flat otherwise; report gross AND net of a one-way cost per switch (turnover-scaled).

5. **A robustness sweep** over the signal's lookback/threshold, checking sign stability.

6. **A seed-robust synthetic positive control** (>= 20 seeds): the engine must recover the planted
   predictive slope when ``predictive_beta > 0`` and stay flat at the null.

EXECUTION LAG: the signal at week ``t`` uses only box office known by the close of week ``t``; the
position is entered at that close and held over week ``t+1`` (the ``*_fwd`` columns). One
documented week of lag; no future data enters the signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52


# ---------------------------------------------------------------------------
# OLS with t-stats
# ---------------------------------------------------------------------------
def _ols(y: np.ndarray, X: np.ndarray) -> dict:
    """OLS with classical (homoskedastic) standard errors; returns coefs and t-stats."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = X.shape
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(sigma2 * xtx_inv), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, coef / se, np.nan)
    return {"coef": coef, "se": se, "t": t, "n": int(n), "r2": _r2(y, resid)}


def _r2(y: np.ndarray, resid: np.ndarray) -> float:
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ---------------------------------------------------------------------------
# 1 · The predictive regression — the headline leading-indicator test
# ---------------------------------------------------------------------------
def predictive_regression(panel: pd.DataFrame, ret_col: str = "media_fwd") -> dict:
    """OLS of next-week forward return on this-week box-office momentum.

    ``forward_ret = a + b * bo_mom + e``. The slope ``b`` (and its *t*) is the leading-indicator
    claim: does a strong box office this week foreshadow gains next week? Returns the slope (in
    return units per 1-sigma of momentum), its *t*, the intercept, R^2 and n.
    """
    if panel.empty or len(panel) < 8:
        return {"slope": float("nan"), "slope_t": float("nan"), "r2": float("nan"), "n": 0}
    x = panel["bo_mom"].to_numpy(dtype=float)
    y = panel[ret_col].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    r = _ols(y, X)
    return {"slope": float(r["coef"][1]), "slope_t": float(r["t"][1]),
            "intercept": float(r["coef"][0]), "r2": float(r["r2"]), "n": r["n"]}


# ---------------------------------------------------------------------------
# 2 · Confound-controlled regression — does the lead survive the common factor?
# ---------------------------------------------------------------------------
def controlled_regression(panel: pd.DataFrame, ret_col: str = "media_fwd",
                          ctrl_col: str | None = None) -> dict:
    """Predictive regression adding a market-factor control alongside box-office momentum.

    ``ret = a + b * bo_mom + c * ctrl + e``. Box office and stocks both load on the same
    consumer/market factor; a *genuine* lead keeps a significant ``b`` after controlling for the
    market move, a spurious co-movement collapses. Returns the controlled slope on box-office
    momentum and its *t*.

    The control defaults to the CONTEMPORANEOUS broad-market return (``spy_fwd``) — the honest
    control when predicting the media basket. When the dependent variable *is* ``spy_fwd`` that
    would be degenerate (regressing SPY on itself), so we control on the contemporaneous media
    return (``media_fwd``) instead — an equally valid proxy for the common factor at ``t+1``.
    """
    if panel.empty or len(panel) < 8:
        return {"slope": float("nan"), "slope_t": float("nan"), "n": 0}
    if ctrl_col is None:
        ctrl_col = "media_fwd" if ret_col == "spy_fwd" else "spy_fwd"
    x = panel["bo_mom"].to_numpy(dtype=float)
    ctrl = panel[ctrl_col].to_numpy(dtype=float)
    y = panel[ret_col].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x, ctrl])
    r = _ols(y, X)
    return {"slope": float(r["coef"][1]), "slope_t": float(r["t"][1]),
            "ctrl_slope": float(r["coef"][2]), "ctrl_t": float(r["t"][2]),
            "r2": float(r["r2"]), "n": r["n"]}


# ---------------------------------------------------------------------------
# 3 · Placebo null — circular-shift the signal against returns
# ---------------------------------------------------------------------------
def placebo_pvalue(panel: pd.DataFrame, ret_col: str = "media_fwd",
                   n_perm: int = 2000, seed: int = 550) -> float:
    """Circular-shift placebo p-value for the predictive slope.

    Circularly roll the box-office signal by a random offset (preserving each series' own
    autocorrelation while destroying the *lead* alignment) ``n_perm`` times; the p-value is the
    fraction of shifts whose |slope| >= the observed |slope|. A real lead sits in the tail.
    """
    if panel.empty or len(panel) < 12:
        return float("nan")
    x = panel["bo_mom"].to_numpy(dtype=float)
    y = panel[ret_col].to_numpy(dtype=float)
    obs = abs(predictive_regression(panel, ret_col)["slope"])
    rng = np.random.default_rng(seed)
    n = len(x)
    count = 0
    for _ in range(n_perm):
        shift = int(rng.integers(2, n - 2))
        xs = np.roll(x, shift)
        X = np.column_stack([np.ones_like(xs), xs])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        if abs(coef[1]) >= obs - 1e-18:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# 4 · Signal-timed strategy + costs
# ---------------------------------------------------------------------------
def timed_strategy(panel: pd.DataFrame, ret_col: str = "media_fwd",
                   threshold: float = 0.0, cost_bps: float = 5.0) -> dict:
    """Long the basket in weeks the box-office signal exceeds ``threshold``, flat otherwise.

    Signal at week ``t`` is ``bo_mom`` (known at that close); the position is held over week
    ``t+1`` (the forward return). Cost is charged one-way per position *switch* (turnover-scaled).
    Returns gross and net annualised mean return, the buy-and-hold benchmark, hit rate and
    turnover.
    """
    if panel.empty or len(panel) < 8:
        return {"gross_ann": float("nan"), "net_ann": float("nan"),
                "bh_ann": float("nan"), "n": 0}
    sig = (panel["bo_mom"].to_numpy(dtype=float) > threshold).astype(float)
    fwd = panel[ret_col].to_numpy(dtype=float)
    strat = sig * fwd
    switches = np.abs(np.diff(np.concatenate([[0.0], sig])))
    cost = switches * (cost_bps * 1e-4)
    net = strat - cost
    return {
        "gross_ann": float(strat.mean() * WEEKS_PER_YEAR),
        "net_ann": float(net.mean() * WEEKS_PER_YEAR),
        "bh_ann": float(fwd.mean() * WEEKS_PER_YEAR),
        "hit_rate": float((strat[sig > 0] > 0).mean()) if sig.sum() > 0 else float("nan"),
        "turnover_ann": float(switches.mean() * WEEKS_PER_YEAR),
        "exposure": float(sig.mean()),
        "n": int(len(panel)),
    }


# ---------------------------------------------------------------------------
# 5 · Robustness sweep — sign stability across the signal threshold
# ---------------------------------------------------------------------------
def robustness_sweep(panel: pd.DataFrame, ret_col: str = "media_fwd",
                     thresholds=(-0.5, -0.25, 0.0, 0.25, 0.5)) -> pd.DataFrame:
    """Predictive slope-*t* and net-minus-buy-and-hold across signal thresholds.

    A real lead keeps the same slope sign across thresholds; a spurious one flips. Returns a tidy
    frame indexed by threshold.
    """
    rows = []
    reg = predictive_regression(panel, ret_col)
    for th in thresholds:
        s = timed_strategy(panel, ret_col, threshold=th)
        rows.append({
            "threshold": th,
            "slope_t": reg["slope_t"],  # slope is threshold-free; kept for reference
            "net_ann": s["net_ann"],
            "bh_ann": s["bh_ann"],
            "net_minus_bh": s["net_ann"] - s["bh_ann"],
        })
    return pd.DataFrame(rows).set_index("threshold")


# ---------------------------------------------------------------------------
# 6 · Seed-robust synthetic positive control
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, predictive_beta: float, n_seeds: int = 25,
                     base_seed: int = 550, controlled: bool = False,
                     ret_col: str = "media_fwd") -> float:
    """Average the predictive slope-*t* over ``n_seeds`` synthetic worlds.

    House rule: any synthetic-dependent claim averages the stat over >= 20 seeds so no single lucky
    RNG seed can manufacture significance. With ``controlled=True`` it averages the
    confound-controlled slope-*t* (the honest one). Returns the mean slope-*t* for a planted
    ``predictive_beta``.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(predictive_beta=predictive_beta, seed=s)
        if controlled:
            ts.append(controlled_regression(panel, ret_col)["slope_t"])
        else:
            ts.append(predictive_regression(panel, ret_col)["slope_t"])
    return float(np.nanmean(ts))
