"""The engine and its honest controls — Study 567 (Uncertainty-Word-Count).

The claim, at full strength (Loughran & McDonald 2011, *When Is a Liability Not a Liability?*, JF;
Campbell, Chen, Dhaliwal, Lu & Steele 2014; Kravet & Muslu 2013; Bochkay, Chychyla & Nanda 2020):
the *density of uncertainty / hedging words* in a firm's disclosure — its share of tokens in the LM
**Uncertainty** word list — forecasts **higher future realised volatility** and (the softer half)
**lower future returns**. Hedgy language flags ambiguity, and ambiguity resolves as vol.

The honesty pivot of this study is **confounding by recent volatility.** Firms that are already
jumpy write hedgier text, and realised vol is strongly autocorrelated, so a naive
uncertainty→forward-vol regression double-counts vol persistence. The engine therefore reports both:

1. **The naive uncertainty slope** — OLS of forward vol on ``uncert`` alone. Inflated by vol
   persistence.
2. **The trail-vol-controlled slope** — OLS of forward vol on ``[uncert, trail_vol]``; the *uncert*
   coefficient (and its *t*) is the isolated **textual** edge, the honest headline statistic.

Plus the desk's standard controls: a **label-shuffle placebo** null (shuffle uncertainty against
the outcome and read the controlled-slope tail), a **long-short** on the uncertainty tails (long
hedgy / short plain, on the return axis) with costs + short borrow, a **multi-window robustness**
sweep (does the sign hold across sub-panels?), and a **seed-robust synthetic positive control**
(≥ 20 seeds) proving the engine banks a planted uncertainty→vol link and stays flat at the null.

Execution convention: uncertainty density is known *at* the filing (the document is public that
day); the forward vol / return are measured strictly *after* the filing period, and the tradable
position is entered the next session. No future data enters the score — one documented execution
lag, applied identically in the sweep and the synthetic control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core OLS helper
# ---------------------------------------------------------------------------
def _ols_t(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Plain OLS: return (coef, t-stats) with classical standard errors."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = X.shape
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.clip(sigma2 * np.diag(xtx_inv), 0, np.inf))
    t = np.divide(coef, se, out=np.full_like(coef, np.nan), where=se > 0)
    return coef, t


# ---------------------------------------------------------------------------
# The headline regression: forward vol on uncertainty, controlling for trailing vol
# ---------------------------------------------------------------------------
def uncertainty_vol_regression(panel: pd.DataFrame, control_trail_vol: bool = True) -> dict:
    """Regress forward realised vol on the uncertainty density (optionally control trailing vol).

    - ``control_trail_vol=False`` → the **naive** slope of forward vol on uncertainty (inflated by
      vol persistence).
    - ``control_trail_vol=True``  → forward vol on ``[uncert, trail_vol]``; the *uncert* coefficient
      is the isolated **textual** edge — the honest headline statistic.

    Returns the uncertainty slope, its *t*-stat, the trail-vol slope (if controlled), and n.
    """
    if panel.empty or len(panel) < 5:
        return {"uncert_slope": float("nan"), "uncert_t": float("nan"),
                "trail_slope": float("nan"), "n": 0}
    u = panel["uncert"].to_numpy(dtype=float)
    fv = panel["fwd_vol"].to_numpy(dtype=float)
    if control_trail_vol and "trail_vol" in panel.columns:
        tv = panel["trail_vol"].to_numpy(dtype=float)
        X = np.column_stack([np.ones_like(u), u, tv])
        coef, t = _ols_t(fv, X)
        return {"uncert_slope": float(coef[1]), "uncert_t": float(t[1]),
                "trail_slope": float(coef[2]), "n": int(len(fv))}
    X = np.column_stack([np.ones_like(u), u])
    coef, t = _ols_t(fv, X)
    return {"uncert_slope": float(coef[1]), "uncert_t": float(t[1]),
            "trail_slope": float("nan"), "n": int(len(fv))}


# ---------------------------------------------------------------------------
# The secondary axis: forward return on uncertainty (the "lower returns" half)
# ---------------------------------------------------------------------------
def uncertainty_return_regression(panel: pd.DataFrame, control_trail_vol: bool = True) -> dict:
    """Regress forward return on the uncertainty density (the softer 'lower returns' claim).

    A *negative* uncert slope is the return drag. Controlling for trailing vol keeps the read
    consistent with the vol regression. Returns the uncert slope, its *t*, and n.
    """
    if panel.empty or len(panel) < 5:
        return {"uncert_slope": float("nan"), "uncert_t": float("nan"), "n": 0}
    u = panel["uncert"].to_numpy(dtype=float)
    fr = panel["fwd_ret"].to_numpy(dtype=float)
    if control_trail_vol and "trail_vol" in panel.columns:
        tv = panel["trail_vol"].to_numpy(dtype=float)
        X = np.column_stack([np.ones_like(u), u, tv])
    else:
        X = np.column_stack([np.ones_like(u), u])
    coef, t = _ols_t(fr, X)
    return {"uncert_slope": float(coef[1]), "uncert_t": float(t[1]), "n": int(len(fr))}


# ---------------------------------------------------------------------------
# The long-short on the uncertainty tails (return axis — the tradable expression)
# ---------------------------------------------------------------------------
def uncertainty_long_short(panel: pd.DataFrame, frac: float = 0.3) -> dict:
    """Sort by uncertainty density; short the hedgy tail / long the plain tail; compare forward ret.

    The claim says hedgy firms earn *lower* returns, so the tradable book is **long plain / short
    hedgy**. Returns the plain-bucket mean return, the hedgy-bucket mean return, their spread
    (plain − hedgy), a two-sample (Welch) *t*, and bucket size ``k``. This does not control for
    trailing vol, so it is the naive tradable read (the regression is the honest one).
    """
    if panel.empty or len(panel) < 6:
        return {"plain_mean": float("nan"), "hedgy_mean": float("nan"),
                "spread": float("nan"), "t": float("nan"), "k": 0}
    df = panel.sort_values("uncert")
    k = max(1, int(round(len(df) * frac)))
    plain = df.head(k)["fwd_ret"].to_numpy(dtype=float)   # lowest uncertainty
    hedgy = df.tail(k)["fwd_ret"].to_numpy(dtype=float)   # highest uncertainty
    va, vb = plain.var(ddof=1), hedgy.var(ddof=1)
    se = np.sqrt(va / len(plain) + vb / len(hedgy))
    t = (plain.mean() - hedgy.mean()) / se if se > 0 else float("nan")
    return {"plain_mean": float(plain.mean()), "hedgy_mean": float(hedgy.mean()),
            "spread": float(plain.mean() - hedgy.mean()), "t": float(t), "k": int(k)}


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null (on the trail-vol-controlled forward-vol slope)
# ---------------------------------------------------------------------------
def placebo_pvalue(panel: pd.DataFrame, n_perm: int = 2000, seed: int = 567) -> float:
    """Label-shuffle placebo p-value for the trail-vol-controlled uncertainty→forward-vol slope.

    Shuffle the ``uncert`` labels against (forward vol, trailing vol) ``n_perm`` times and refit the
    controlled regression; the p-value is the fraction of shuffles whose |uncert slope| ≥ the
    observed |uncert slope|. A real textual edge sits in the tail; a confound-only signal does not.
    """
    if panel.empty or len(panel) < 8:
        return float("nan")
    obs = abs(uncertainty_vol_regression(panel, control_trail_vol=True)["uncert_slope"])
    if not np.isfinite(obs):
        return float("nan")
    fv = panel["fwd_vol"].to_numpy(dtype=float)
    tv = panel["trail_vol"].to_numpy(dtype=float)
    u = panel["uncert"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    n = len(fv)
    count = 0
    for _ in range(n_perm):
        us = u[rng.permutation(n)]
        X = np.column_stack([np.ones(n), us, tv])
        coef, _t = _ols_t(fv, X)
        if abs(coef[1]) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs + borrow on the tradable long-short (return axis)
# ---------------------------------------------------------------------------
def net_spread(
    ls: dict,
    turns_per_year: float = 4.0,
    cost_bps: float = 5.0,
    borrow_ann_bps: float = 50.0,
) -> float:
    """Uncertainty long-short spread net of costs and short borrow, annualised.

    The book rebalances every filing (four filings/yr → ``turns_per_year`` ≈ 4), each turn crossing
    the spread on both legs (long plain, short hedgy). The hedgy (short) leg pays an annual borrow.
    The ``spread`` is a per-event forward return, realised ~4×/yr, so we annualise by
    ``turns_per_year`` then net the frictions.
    """
    if not ls or not np.isfinite(ls.get("spread", float("nan"))):
        return float("nan")
    gross_ann = ls["spread"] * turns_per_year
    # entry + exit, two legs -> 4 one-way crossings of cost_bps per turn, `turns_per_year` turns
    cost = turns_per_year * 4.0 * cost_bps * 1e-4
    borrow = borrow_ann_bps * 1e-4
    return float(gross_ann - cost - borrow)


# ---------------------------------------------------------------------------
# Robustness — sign stability across sub-panels
# ---------------------------------------------------------------------------
def robustness_windows(panel: pd.DataFrame, n_windows: int = 4) -> list[tuple]:
    """Split the periods into ``n_windows`` contiguous sub-panels; report the controlled slope-t.

    Returns a list of ``(label, controlled uncert slope, controlled uncert-t)`` — the sign should
    hold if the textual edge is real, and wander if it is noise.
    """
    if panel.empty or "period" not in panel.columns:
        return []
    ps = sorted(panel["period"].unique())
    if len(ps) < n_windows:
        return []
    chunks = np.array_split(np.array(ps), n_windows)
    out = []
    for ch in chunks:
        sub = panel[panel["period"].isin(list(ch))]
        reg = uncertainty_vol_regression(sub, control_trail_vol=True)
        lab = f"{ch[0]}-{ch[-1]}"
        out.append((lab, reg["uncert_slope"], reg["uncert_t"]))
    return out


# ---------------------------------------------------------------------------
# Seed-robust synthetic positive control (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(
    data_mod, uncert_beta: float, n_seeds: int = 25, base_seed: int = 567,
    control_trail_vol: bool = True,
) -> float:
    """Average the (trail-vol-controlled) uncertainty→forward-vol slope-t over ``n_seeds`` worlds.

    The house rule: any synthetic-dependent claim averages the stat over ≥ 20 seeds so no single
    lucky RNG seed manufactures significance. Returns the mean uncertainty slope-t across seeds for a
    planted ``uncert_beta`` (0 = null, >0 = the vol anomaly).
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(uncert_beta=uncert_beta, seed=s)
        ts.append(
            uncertainty_vol_regression(panel, control_trail_vol=control_trail_vol)["uncert_t"]
        )
    return float(np.nanmean(ts))
