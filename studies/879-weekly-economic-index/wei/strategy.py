"""Strategy + inference for Study 879 — Weekly Economic Index.

The claim, operationalised on a weekly frame of ``wei`` (the nowcast level), ``dwei`` (its
weekly change), and forward market returns (``spy_h1``/``spy_h4`` broad tape; ``rot_h1``/
``rot_h4`` = forward XLY minus forward XLP cyclical-vs-defensive rotation):

    A *higher-frequency* growth nowcast should time the market. When the WEI is **high**
    (or **rising**), be long stocks / tilt **cyclical** (XLY) over **defensive** (XLP); the
    level and the weekly change are supposed to predict a higher forward return and a
    cyclical outperformance.

We test it four ways:

  * **Predictive regression, Newey-West t.** Regress the forward return on a constant and
    the standardized nowcast level & weekly change; the overlapping weekly forward window
    is serially correlated, so the *t* is a HAC (Newey-West, Bartlett kernel) *t* — a plain
    OLS *t* would overstate significance. R^2 reports how much forward variance the nowcast
    explains at all.
  * **Conditional vs unconditional forward returns.** Mean forward return when the nowcast
    is above its own median (or rising) vs the unconditional base rate, with a Welch *t*.
  * **A placebo / randomization null.** Resample the same number of random entry weeks many
    times and ask how often chance matches the conditioned set — the honest small-sample
    test.
  * **A costed rotation overlay.** A long-cyclical / short-defensive (XLY-XLP) book driven
    by the nowcast sign, rebalanced weekly with a one-way cost per leg and borrow on the
    short, raced against always-hold on a Sharpe basis (the Tradability axis).

The decisive question is whether the *weekly* nowcast beats what the *monthly* macro tape
already tells the market — or whether it is just a smooth, slow proxy for the recession /
recovery cycle whose apparent edge is the market's own trend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52


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


def newey_west_t(x: np.ndarray, lags: int = 8) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        var += 2.0 * w * cov
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
# Newey-West predictive regression  y = a + X b + e
# --------------------------------------------------------------------------- #
def nw_regression(y: np.ndarray, X: np.ndarray, lags: int = 8) -> dict:
    """OLS of ``y`` on a constant plus the columns of ``X`` with Newey-West (HAC) *t*.

    ``X`` is an ``(n, k)`` matrix of regressors (a constant is added automatically). Returns
    the coefficient vector (const first), the HAC *t* on each coefficient (Bartlett kernel,
    ``lags``), the R^2, and ``n``. The HAC sandwich uses the same Bartlett weights as
    :func:`newey_west_t`; on a single regressor with a constant the slope *t* matches a
    univariate HAC test. Rows with any NaN are dropped.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    Z = np.column_stack([np.ones(len(y)), X])
    ok = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    y, Z = y[ok], Z[ok]
    n, k = Z.shape
    if n <= k + 2:
        nan = np.full(k, np.nan)
        return {"beta": nan, "t": nan, "r2": float("nan"), "n": int(n)}
    XtX_inv = np.linalg.inv(Z.T @ Z)
    beta = XtX_inv @ (Z.T @ y)
    resid = y - Z @ beta
    # Newey-West HAC meat: S = sum_l w_l (Gamma_l + Gamma_l')
    S = np.zeros((k, k))
    Xu = Z * resid[:, None]
    S += Xu.T @ Xu
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        G = Xu[l:].T @ Xu[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"beta": beta, "t": t, "r2": r2, "n": int(n)}


def _z(x: np.ndarray) -> np.ndarray:
    """Full-sample z-score (scale only — HAC *t* on a slope is scale-invariant)."""
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    return (x - mu) / sd if sd > 0 else x - mu


def predict(frame: pd.DataFrame, target: str, lags: int = 8) -> dict:
    """Predictive regression of ``frame[target]`` on the standardized ``wei`` & ``dwei``.

    Returns the HAC *t* on the level (``t_level``) and the weekly change (``t_dwei``) from
    the **joint** regression, plus each **univariate** HAC *t* (``t_level_uni`` /
    ``t_dwei_uni``) and the joint R^2. A positive *t* means a higher / rising nowcast
    predicts a higher forward return (the claim's sign).
    """
    y = frame[target].to_numpy(dtype=float)
    lvl = _z(frame["wei"].to_numpy())
    chg = _z(frame["dwei"].to_numpy())
    joint = nw_regression(y, np.column_stack([lvl, chg]), lags=lags)
    uni_l = nw_regression(y, lvl, lags=lags)
    uni_c = nw_regression(y, chg, lags=lags)
    return {
        "target": target,
        "n": joint["n"],
        "r2": joint["r2"],
        "beta_level": float(joint["beta"][1]), "t_level": float(joint["t"][1]),
        "beta_dwei": float(joint["beta"][2]), "t_dwei": float(joint["t"][2]),
        "t_level_uni": float(uni_l["t"][1]),
        "t_dwei_uni": float(uni_c["t"][1]),
    }


# --------------------------------------------------------------------------- #
# Conditional vs unconditional forward returns
# --------------------------------------------------------------------------- #
def conditional(frame: pd.DataFrame, target: str, signal: str = "wei",
                rule: str = "above_median") -> dict:
    """Mean forward ``target`` return when the nowcast is 'on' vs the base rate.

    ``rule='above_median'`` conditions on ``frame[signal]`` above its full-sample median;
    ``rule='positive'`` conditions on ``frame[signal] > 0`` (natural for ``dwei``). Returns
    the conditional mean/win-rate, the unconditional base rate, and a Welch *t*.
    """
    s = frame[signal].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
    if rule == "positive":
        on = s > 0
    else:
        on = s > np.nanmedian(s)
    ok = np.isfinite(y)
    cond = y[on & ok]
    base = y[ok]
    return {
        "target": target, "signal": signal, "rule": rule,
        "n_on": int(len(cond)),
        "cond_mean": float(cond.mean()) if len(cond) else float("nan"),
        "cond_win": float((cond > 0).mean()) if len(cond) else float("nan"),
        "base_mean": float(base.mean()) if len(base) else float("nan"),
        "base_win": float((base > 0).mean()) if len(base) else float("nan"),
        "welch_t": welch_t(cond, base),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the regression slope a lucky alignment?
# --------------------------------------------------------------------------- #
def placebo_pvalue(frame: pd.DataFrame, target: str, signal: str = "wei",
                   lags: int = 8, n_draws: int = 2000, seed: int = 879) -> dict:
    """Block-shuffle placebo: keep the forward-return series, permute the nowcast signal
    (breaking the signal->outcome link) and re-run the univariate HAC slope *t* many times.
    p = share of permuted worlds whose |t| >= observed |t| (a two-sided realness check)."""
    y = frame[target].to_numpy(dtype=float)
    x = _z(frame[signal].to_numpy())
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    obs = abs(nw_regression(y, x, lags=lags)["t"][1])
    rng = np.random.default_rng(seed)
    n = len(x)
    hit = 0
    ts = np.empty(n_draws)
    for i in range(n_draws):
        xp = x[rng.permutation(n)]
        ti = nw_regression(y, xp, lags=lags)["t"][1]
        ts[i] = ti
        if abs(ti) >= obs:
            hit += 1
    return {"target": target, "signal": signal, "obs_t": obs,
            "placebo_mean_abs_t": float(np.nanmean(np.abs(ts))),
            "p_value": hit / n_draws, "n_draws": n_draws}


# --------------------------------------------------------------------------- #
# Era cut
# --------------------------------------------------------------------------- #
def era_split(frame: pd.DataFrame, target: str, split: str = "2017-01-01",
              lags: int = 8) -> dict:
    """Univariate HAC slope *t* of ``target`` on ``wei`` and on ``dwei`` in each era."""
    early = frame[frame.index < pd.Timestamp(split)]
    late = frame[frame.index >= pd.Timestamp(split)]
    out = {"split": split}
    for tag, sub in (("early", early), ("late", late)):
        pl = predict(sub, target, lags=lags) if len(sub) > 20 else None
        out[tag] = {
            "n": int(len(sub)),
            "t_level": pl["t_level_uni"] if pl else float("nan"),
            "t_dwei": pl["t_dwei_uni"] if pl else float("nan"),
        }
    return out


# --------------------------------------------------------------------------- #
# Costed rotation overlay (the Tradability axis)
# --------------------------------------------------------------------------- #
def rotation_overlay(frame: pd.DataFrame, signal: str = "wei",
                     rule: str = "above_median", cost_bps: float = 5.0,
                     borrow_bps_yr: float = 50.0, allow_short: bool = True) -> dict:
    """Long-cyclical / short-defensive (XLY-XLP) overlay driven by the nowcast sign.

    Position for week ``w`` is decided by the nowcast known that week (the frame's forward
    ``rot_h1`` already carries the one-week execution lag): +1 when the signal is 'on'
    (``above_median`` on ``wei``, or ``positive`` on ``dwei``); −1 (long-short) or 0
    (long-cyclical/flat) otherwise. A one-way ``cost_bps`` is charged per leg on each change
    of position (turnover), and the short leg pays ``borrow_bps_yr`` borrow. The realized
    weekly P&L is the position times the forward rotation return ``rot_h1``. Raced against
    always-holding the rotation (buy-cyclical/sell-defensive) on a Sharpe basis."""
    r = frame["rot_h1"].to_numpy(dtype=float)
    s = frame[signal].to_numpy(dtype=float)
    ok = np.isfinite(r) & np.isfinite(s)
    r, s = r[ok], s[ok]
    if rule == "positive":
        on = s > 0
    else:
        on = s > np.median(s)
    pos = np.where(on, 1.0, (-1.0 if allow_short else 0.0))
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    c = cost_bps / 1e4
    borrow_wk = (borrow_bps_yr / 1e4) / WEEKS_PER_YEAR
    short_leg = (pos < 0).astype(float)
    gross = pos * r
    net = gross - turn * c - short_leg * borrow_wk

    def _stats(x: np.ndarray) -> dict:
        mu, sd = x.mean() * WEEKS_PER_YEAR, x.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)
        return {"ann_ret": float(mu), "ann_vol": float(sd),
                "sharpe": float(mu / sd) if sd > 0 else float("nan")}

    return {
        "signal": signal, "rule": rule, "n_weeks": int(len(r)),
        "n_turns": float(turn.sum()), "exposure": float((pos != 0).mean()),
        "gross": _stats(gross), "net": _stats(net),
        "hold": _stats(r), "t_net": one_sample_t(net),
        "cost_bps": cost_bps, "allow_short": allow_short,
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(frame: pd.DataFrame, target: str = "spy_h1", lags: int = 8) -> dict:
    """Run the headline predictive regression on a synthetic frame."""
    p = predict(frame, target, lags=lags)
    return {"t_level": p["t_level"], "t_dwei": p["t_dwei"],
            "t_level_uni": p["t_level_uni"], "r2": p["r2"], "n": p["n"]}
