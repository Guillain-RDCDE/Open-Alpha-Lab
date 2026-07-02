"""The engine and its honest controls — Study 566 (Earnings-Call-Tone).

The claim, at full strength (Loughran & McDonald 2011; Price, Doran, Peterson & Bliss 2012;
Mayew & Venkatachalam 2012): the *net emotional tone* of an earnings-call transcript — its
(positive − negative) sentiment-word share under a finance lexicon — forecasts the stock's
**post-call drift** (the cumulative abnormal return over the days after the call). It is a
linguistic cousin of post-earnings-announcement drift (PEAD, Study 363): the *number* drifts, and
the claim is that the *words* drift on top of it.

The honesty pivot of this study is **confounding by the number.** Transcripts sound upbeat because
the quarter beat, so a naive tone→CAR regression double-counts the numeric surprise. The engine
therefore reports both:

1. **The naive tone slope** — OLS of post-call CAR on ``net_tone`` alone. Inflated by PEAD.
2. **The surprise-controlled tone slope** — OLS of CAR on ``[net_tone, surprise]``; the *net_tone*
   coefficient (and its *t*) is the isolated **linguistic** edge, the honest headline statistic.

Plus the desk's standard controls: a **label-shuffle placebo** null (shuffle tone against CAR and
read the controlled-slope tail), a **long-short** on the tone tails with costs + short borrow, a
**multi-window robustness** sweep (does the sign hold across sub-panels?), and a **seed-robust
synthetic positive control** (≥ 20 seeds) proving the engine banks a planted linguistic drift and
stays flat at the null.

Execution convention: tone is known *at* the call (the transcript is public that evening); the CAR
is measured strictly *after* the call, and the tradable position is entered the next session. No
future data enters the tone score — one documented execution lag, applied identically in the sweep
and the synthetic control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core regression: CAR on tone, optionally controlling for the numeric surprise
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


def tone_regression(panel: pd.DataFrame, control_surprise: bool = True) -> dict:
    """Regress post-call CAR on net tone (optionally controlling for the numeric surprise).

    - ``control_surprise=False`` → the **naive** slope of CAR on tone (inflated by PEAD).
    - ``control_surprise=True``  → CAR on ``[tone, surprise]``; the *tone* coefficient is the
      isolated **linguistic** edge — the honest headline statistic.

    Returns the tone slope, its *t*-stat, the surprise slope (if controlled), and n.
    """
    if panel.empty or len(panel) < 5:
        return {"tone_slope": float("nan"), "tone_t": float("nan"),
                "surprise_slope": float("nan"), "n": 0}
    tone = panel["net_tone"].to_numpy(dtype=float)
    car = panel["car"].to_numpy(dtype=float)
    if control_surprise and "surprise" in panel.columns:
        sup = panel["surprise"].to_numpy(dtype=float)
        X = np.column_stack([np.ones_like(tone), tone, sup])
        coef, t = _ols_t(car, X)
        return {"tone_slope": float(coef[1]), "tone_t": float(t[1]),
                "surprise_slope": float(coef[2]), "n": int(len(car))}
    X = np.column_stack([np.ones_like(tone), tone])
    coef, t = _ols_t(car, X)
    return {"tone_slope": float(coef[1]), "tone_t": float(t[1]),
            "surprise_slope": float("nan"), "n": int(len(car))}


# ---------------------------------------------------------------------------
# The long-short on the tone tails
# ---------------------------------------------------------------------------
def tone_long_short(panel: pd.DataFrame, frac: float = 0.3) -> dict:
    """Sort by net tone, go long the upbeat tail / short the guarded tail; compare post-call CAR.

    Returns the upbeat-bucket mean CAR, the guarded-bucket mean CAR, their spread (upbeat −
    guarded), a two-sample (Welch) *t* on the difference, and bucket size ``k``. This is the
    *tradable* expression of the claim — but note it does **not** control for surprise, so it is the
    naive, PEAD-inflated read (the regression is the honest one).
    """
    if panel.empty or len(panel) < 6:
        return {"upbeat_mean": float("nan"), "guarded_mean": float("nan"),
                "spread": float("nan"), "t": float("nan"), "k": 0}
    df = panel.sort_values("net_tone")
    k = max(1, int(round(len(df) * frac)))
    guarded = df.head(k)["car"].to_numpy(dtype=float)   # lowest tone
    upbeat = df.tail(k)["car"].to_numpy(dtype=float)    # highest tone
    va, vb = upbeat.var(ddof=1), guarded.var(ddof=1)
    se = np.sqrt(va / len(upbeat) + vb / len(guarded))
    t = (upbeat.mean() - guarded.mean()) / se if se > 0 else float("nan")
    return {"upbeat_mean": float(upbeat.mean()), "guarded_mean": float(guarded.mean()),
            "spread": float(upbeat.mean() - guarded.mean()), "t": float(t), "k": int(k)}


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null (on the surprise-controlled tone slope)
# ---------------------------------------------------------------------------
def placebo_pvalue(panel: pd.DataFrame, n_perm: int = 2000, seed: int = 566) -> float:
    """Label-shuffle placebo p-value for the surprise-controlled tone slope.

    Shuffle the ``net_tone`` labels against (CAR, surprise) ``n_perm`` times and refit the
    controlled regression; the p-value is the fraction of shuffles whose |tone slope| ≥ the observed
    |tone slope|. A real linguistic edge sits in the tail; a confound-only signal does not.
    """
    if panel.empty or len(panel) < 8:
        return float("nan")
    obs = abs(tone_regression(panel, control_surprise=True)["tone_slope"])
    if not np.isfinite(obs):
        return float("nan")
    car = panel["car"].to_numpy(dtype=float)
    sup = panel["surprise"].to_numpy(dtype=float)
    tone = panel["net_tone"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    n = len(car)
    count = 0
    for _ in range(n_perm):
        ts = tone[rng.permutation(n)]
        X = np.column_stack([np.ones(n), ts, sup])
        coef, _t = _ols_t(car, X)
        if abs(coef[1]) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs + borrow on the tradable long-short
# ---------------------------------------------------------------------------
def net_spread(
    ls: dict,
    turns_per_year: float = 4.0,
    cost_bps: float = 5.0,
    borrow_ann_bps: float = 50.0,
) -> float:
    """Tone long-short spread net of costs and short borrow, annualised.

    The book rebalances every quarter (four calls/yr → ``turns_per_year`` ≈ 4), each turn crossing
    the spread on both legs (long upbeat, short guarded). The guarded (short) leg pays an annual
    borrow. The ``spread`` is a per-event CAR, realised ~4×/yr, so we annualise by ``turns_per_year``
    then net the frictions.
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
    """Split the quarters into ``n_windows`` contiguous sub-panels; report the controlled slope-t.

    Returns a list of ``(label, controlled tone slope, controlled tone-t)`` — the sign should hold
    if the linguistic edge is real, and wander if it is noise.
    """
    if panel.empty or "quarter" not in panel.columns:
        return []
    qs = sorted(panel["quarter"].unique())
    if len(qs) < n_windows:
        return []
    chunks = np.array_split(np.array(qs), n_windows)
    out = []
    for i, ch in enumerate(chunks):
        sub = panel[panel["quarter"].isin(list(ch))]
        reg = tone_regression(sub, control_surprise=True)
        lab = f"{ch[0]}-{ch[-1]}"
        out.append((lab, reg["tone_slope"], reg["tone_t"]))
    return out


# ---------------------------------------------------------------------------
# Seed-robust synthetic positive control (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(
    data_mod, tone_beta: float, n_seeds: int = 25, base_seed: int = 566,
    control_surprise: bool = True,
) -> float:
    """Average the (surprise-controlled) tone slope-t over ``n_seeds`` synthetic worlds.

    The house rule: any synthetic-dependent claim averages the stat over ≥ 20 seeds so no single
    lucky RNG seed manufactures significance. Returns the mean tone slope-t across seeds for a
    planted ``tone_beta`` (0 = null, >0 = the linguistic drift).
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(tone_beta=tone_beta, seed=s)
        ts.append(tone_regression(panel, control_surprise=control_surprise)["tone_t"])
    return float(np.nanmean(ts))
