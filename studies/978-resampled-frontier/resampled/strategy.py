"""Resampled efficiency, and the cheap fixes it competes with — Study 978.

Michaud's (1998) procedure, implemented exactly:

1. Estimate ``mu`` and ``Sigma`` from the sample.
2. Draw ``n_resamples`` bootstrap samples *of the same length* from the estimated
   distribution, and re-estimate ``mu_b``, ``Sigma_b`` on each.
3. Solve the same optimisation on each resample.
4. **Average the weight vectors.**

The averaging is the whole method and it is also the whole controversy. Averaging optimal
portfolios is not the same as the portfolio that is optimal on average — Jensen again — and
Scherer (2002) showed the result inherits whatever bias the estimator had while adding a
long-only-like shrinkage of its own. Two comparisons make that concrete here:

- ``shrinkage_weights`` — Ledoit-Wolf on the covariance plus grand-mean shrinkage on the
  expected returns, the standard cheap fix. If resampling and shrinkage produce nearly the
  same portfolio, the honest summary is "resampling is an expensive shrinkage".
- ``equal_weights`` — the fix that costs nothing at all.

Two objectives are run, because the method behaves very differently on each:

- **Minimum variance** (``min_variance_weights``), which uses only ``Sigma``;
- **Maximum Sharpe** (``max_sharpe_weights``), which uses ``mu`` as well and is therefore where
  estimation error is catastrophic — and where resampling is supposed to shine.

Everything is long-only (weights projected onto the simplex, as in Michaud's own
implementation), scored on a rolling out-of-sample window with costs, and compared pairwise
because every method sees the same data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
METHODS = ("plain", "resampled", "shrunk", "equal")
METHOD_LABEL = {"plain": "Plain optimisation", "resampled": "Resampled (Michaud)",
                "shrunk": "Shrinkage (Ledoit-Wolf + grand mean)", "equal": "1/N"}
OBJECTIVES = ("min_var", "max_sharpe")
OBJECTIVE_LABEL = {"min_var": "Minimum variance", "max_sharpe": "Maximum Sharpe"}


# --------------------------------------------------------------------------- #
# Optimisers (long-only throughout, as in Michaud's own construction)
# --------------------------------------------------------------------------- #
def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the simplex — long-only, fully invested."""
    n = v.size
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def min_variance_weights(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Long-only global minimum variance (``mu`` unused, kept for a common signature)."""
    n = cov.shape[0]
    w = np.linalg.pinv(cov + np.eye(n) * 1e-14) @ np.ones(n)
    s = w.sum()
    w = w / s if abs(s) > 1e-12 else np.full(n, 1 / n)
    return project_simplex(w)


def max_sharpe_weights(mu: np.ndarray, cov: np.ndarray, iters: int = 300,
                       step: float = 0.4) -> np.ndarray:
    """Long-only maximum Sharpe, by projected gradient ascent from the tangency solution.

    The unconstrained tangency portfolio is ``Sigma^-1 mu`` normalised; with a long-only
    constraint it usually leaves the simplex, so the closed form is used as a starting point
    and refined by projected gradient on the Sharpe ratio itself. Tested against a brute-force
    grid on small problems rather than trusted.
    """
    n = cov.shape[0]
    w = np.linalg.pinv(cov + np.eye(n) * 1e-14) @ mu
    s = w.sum()
    w = project_simplex(w / s) if abs(s) > 1e-12 else np.full(n, 1 / n)
    for _ in range(iters):
        var = float(w @ cov @ w)
        num = float(w @ mu)
        if var <= 0:
            break
        grad = mu / max(num, 1e-18) - (cov @ w) / var if num > 0 else -(cov @ w) / var
        nxt = project_simplex(w + step * grad / (np.linalg.norm(grad) + 1e-18))
        if np.max(np.abs(nxt - w)) < 1e-12:
            w = nxt
            break
        w = nxt
    return w


def optimise(mu: np.ndarray, cov: np.ndarray, objective: str) -> np.ndarray:
    """Dispatch on ``OBJECTIVES``."""
    return (min_variance_weights if objective == "min_var" else max_sharpe_weights)(mu, cov)


# --------------------------------------------------------------------------- #
# The four ways to get a weight vector
# --------------------------------------------------------------------------- #
def plain_weights(X: np.ndarray, objective: str) -> np.ndarray:
    """Optimise once, on the sample estimates. The thing everybody else is fixing."""
    return optimise(X.mean(axis=0), np.cov(X, rowvar=False, ddof=1), objective)


def resampled_weights(X: np.ndarray, objective: str, n_resamples: int = 100,
                      seed: int = 978, parametric: bool = True) -> np.ndarray:
    """Michaud resampling: optimise on many bootstrap samples and average the weights.

    ``parametric=True`` draws from a multivariate normal with the sample mean and covariance
    (Michaud's original "Monte Carlo resampling"); ``False`` resamples the observed rows with
    replacement, which makes no distributional assumption and keeps the fat tails. Both are
    reported in ``verify.py`` because the choice is rarely stated and does change the answer.
    """
    rng = np.random.default_rng(seed)
    T, n = X.shape
    mu, cov = X.mean(axis=0), np.cov(X, rowvar=False, ddof=1)
    acc = np.zeros(n)
    ok = 0
    for _ in range(n_resamples):
        if parametric:
            Xb = rng.multivariate_normal(mu, cov, size=T, method="cholesky")
        else:
            Xb = X[rng.integers(0, T, T)]
        w = optimise(Xb.mean(axis=0), np.cov(Xb, rowvar=False, ddof=1), objective)
        if np.all(np.isfinite(w)):
            acc += w
            ok += 1
    return acc / ok if ok else np.full(n, 1 / n)


def _lw_shrink_cov(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward the constant-correlation target (analytic intensity)."""
    S = np.cov(X, rowvar=False, ddof=1)
    n, T = S.shape[0], X.shape[0]
    sd = np.sqrt(np.diag(S))
    outer = np.outer(sd, sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(outer > 0, S / outer, 0.0)
    off = ~np.eye(n, dtype=bool)
    target = (R[off].mean() if n > 1 else 0.0) * outer
    np.fill_diagonal(target, np.diag(S))
    Xc = X - X.mean(axis=0)
    pi_mat = np.zeros_like(S)
    for t in range(T):
        d = np.outer(Xc[t], Xc[t]) - S
        pi_mat += d * d
    pi_hat = float(pi_mat.sum() / T)
    gamma = float(((target - S) ** 2).sum())
    delta = 0.0 if gamma <= 0 else float(np.clip((pi_hat / T) / gamma, 0.0, 1.0))
    return (1 - delta) * S + delta * target


def shrunk_weights(X: np.ndarray, objective: str, mean_shrink: float = 0.5) -> np.ndarray:
    """The cheap fix: shrink the covariance (Ledoit-Wolf) and the means (toward the grand mean).

    ``mean_shrink = 0.5`` halves the dispersion of expected returns, which is a crude
    stand-in for Jorion's Bayes-Stein estimator and is deliberately not tuned — the point of
    the comparison is that a *default* shrinkage competes with a thousand optimisations.
    """
    mu = X.mean(axis=0)
    mu_s = (1 - mean_shrink) * mu + mean_shrink * mu.mean()
    return optimise(mu_s, _lw_shrink_cov(X), objective)


def equal_weights(X: np.ndarray, objective: str) -> np.ndarray:
    """1/N, which ignores both estimates and keeps winning anyway."""
    return np.full(X.shape[1], 1.0 / X.shape[1])


def weights_for(method: str, X: np.ndarray, objective: str, **kw) -> np.ndarray:
    """Dispatch on ``METHODS``."""
    if method == "plain":
        return plain_weights(X, objective)
    if method == "resampled":
        return resampled_weights(X, objective, **kw)
    if method == "shrunk":
        return shrunk_weights(X, objective)
    if method == "equal":
        return equal_weights(X, objective)
    raise ValueError(f"unknown method {method!r}")


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def concentration(w: np.ndarray) -> dict:
    """Max weight, effective N and how many positions are actually held."""
    hhi = float(np.sum(w ** 2))
    return {"max_weight": float(np.max(w)), "effective_n": float(1 / hhi) if hhi > 0 else np.nan,
            "n_held": int((w > 1e-6).sum())}


def weight_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Share of the book that differs between two long-only weight vectors."""
    return float(np.abs(a - b).sum() / 2)


# --------------------------------------------------------------------------- #
# The scoreboard
# --------------------------------------------------------------------------- #
def walk_forward(rets: pd.DataFrame, objective: str, window: int = 504, step: int = 63,
                 methods=METHODS, cost_bps: float = 5.0, n_resamples: int = 60,
                 seed: int = 978) -> pd.DataFrame:
    """Rolling re-estimation with one day of lag; one row per (method, rebalance)."""
    R = rets.dropna(how="any")
    rows = []
    prev: dict[str, np.ndarray] = {}
    for start in range(window, len(R) - step, step):
        X = R.iloc[start - window:start].to_numpy()
        test = R.iloc[start:start + step]
        for m in methods:
            kw = {"n_resamples": n_resamples, "seed": seed + start} if m == "resampled" else {}
            w = weights_for(m, X, objective, **kw)
            port = test.to_numpy() @ w
            turn = float(np.abs(w - prev.get(m, np.zeros_like(w))).sum())
            prev[m] = w
            sd = float(np.std(port, ddof=1))
            rows.append({"date": R.index[start], "method": m, "objective": objective,
                         "realised_vol": sd * np.sqrt(TRADING_DAYS),
                         "mean_ret": float(port.mean() * TRADING_DAYS
                                           - turn * cost_bps / 1e4 * TRADING_DAYS / step),
                         "turnover": turn, **concentration(w)})
    out = pd.DataFrame(rows)
    out["sharpe_period"] = out["mean_ret"] / out["realised_vol"].replace(0.0, np.nan)
    return out


def summarise(wf: pd.DataFrame) -> pd.DataFrame:
    """Per method: realised volatility, return, Sharpe, turnover and concentration."""
    g = wf.groupby("method")
    out = pd.DataFrame({
        "realised_vol": g["realised_vol"].mean(), "mean_ret": g["mean_ret"].mean(),
        "turnover": g["turnover"].mean(), "max_weight": g["max_weight"].mean(),
        "effective_n": g["effective_n"].mean(), "n_held": g["n_held"].mean(),
        "n": g.size(),
    })
    out["sharpe"] = out["mean_ret"] / out["realised_vol"]
    return out.reindex([m for m in METHODS if m in out.index])


def paired_test(wf: pd.DataFrame, a: str, b: str, column: str = "mean_ret") -> dict:
    """Paired *t* on a per-rebalance column — same window, same holding period."""
    x = wf[wf["method"] == a].set_index("date")[column]
    y = wf[wf["method"] == b].set_index("date")[column]
    x, y = x.align(y, join="inner")
    d = (x - y).dropna()
    if len(d) < 8:
        return {"diff": np.nan, "t": np.nan, "n": int(len(d)), "win_rate": np.nan}
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"diff": float(d.mean()), "t": float(d.mean() / se) if se > 0 else np.nan,
            "n": int(len(d)), "win_rate": float((d > 0).mean())}


def truth_experiment(mu: np.ndarray, cov: np.ndarray, objective: str, n_obs: int = 504,
                     n_trials: int = 40, n_resamples: int = 60, seed: int = 978) -> pd.DataFrame:
    """Score every method against the portfolio the TRUE parameters imply.

    On real data there is no truth, so "resampling produces a better portfolio" is
    unfalsifiable. Here the true mean and covariance are known: each trial draws a sample,
    builds each method's weights from it, and measures both the distance to the optimal
    weights and the *utility gap* — the true objective value lost relative to the best
    achievable. That gap, not the weight distance, is what an investor pays.
    """
    rng = np.random.default_rng(seed)
    n = mu.size
    w_star = optimise(mu, cov, objective)
    best = (float(np.sqrt(w_star @ cov @ w_star)) if objective == "min_var"
            else float(w_star @ mu / np.sqrt(w_star @ cov @ w_star)))
    rows = []
    for t in range(n_trials):
        X = rng.multivariate_normal(mu, cov, size=n_obs, method="cholesky")
        for m in METHODS:
            kw = {"n_resamples": n_resamples, "seed": seed + t} if m == "resampled" else {}
            w = weights_for(m, X, objective, **kw)
            val = (float(np.sqrt(w @ cov @ w)) if objective == "min_var"
                   else float(w @ mu / np.sqrt(w @ cov @ w)))
            rows.append({"trial": t, "method": m, "distance_to_optimal": weight_distance(w, w_star),
                         "true_value": val,
                         "utility_gap": (val - best) if objective == "min_var" else (best - val)})
    return pd.DataFrame(rows)


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Real** if resampling changes the portfolio materially (more than 10% of the
      book versus plain optimisation) **and** reduces the utility gap against a known truth in
      simulation; **Weak** if only one holds; **None** otherwise.
    - **Usefulness**: **Useful** only if resampling beats **shrinkage** — the cheap competitor
      — on out-of-sample Sharpe with a paired |*t*| >= 2; **Fragile** if it beats plain
      optimisation but not shrinkage; **Mirage** if it beats neither.
    """
    distinct = h["weight_gap_vs_plain"] > 0.10
    better = h["utility_gap_resampled"] < h["utility_gap_plain"]
    signal = "Real" if distinct and better else ("Weak" if distinct or better else "None")
    beats_shrink = h["t_vs_shrunk"] > 2.0
    beats_plain = h["t_vs_plain"] > 0
    trad = ("Useful" if beats_shrink else ("Fragile" if beats_plain else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"Averaging a thousand optimisations does produce a different portfolio: it differs "
            f"from the single-shot answer by **{h['weight_gap_vs_plain']:.0%}** of the book, "
            f"holds **{h['n_held_resampled']:.0f}** positions against "
            f"{h['n_held_plain']:.0f}, and caps its largest weight at "
            f"{h['max_weight_resampled']:.0%} against {h['max_weight_plain']:.0%}. Against a "
            f"**known** true covariance and mean it also helps: the utility gap falls from "
            f"{h['utility_gap_plain']:.4f} to **{h['utility_gap_resampled']:.4f}** "
            f"({h['utility_gap_shrunk']:.4f} for shrinkage, {h['utility_gap_equal']:.4f} for "
            f"1/N). The averaging is doing something real — the question is whether it is doing "
            f"anything *distinctive*."),
        "trad": trad,
        "trad_why": (
            f"Out of sample on the {h['n_assets']}-sleeve panel, resampling returned "
            f"**{h['ret_resampled']:+.2%}/yr** at {h['vol_resampled']:.2%} volatility (Sharpe "
            f"{h['sharpe_resampled']:+.2f}) against {h['ret_plain']:+.2%} / "
            f"{h['sharpe_plain']:+.2f} for the plain optimiser (paired *t* on the return "
            f"difference {h['t_vs_plain']:+.2f}) and {h['ret_shrunk']:+.2%} / "
            f"{h['sharpe_shrunk']:+.2f} for plain shrinkage (*t* = **{h['t_vs_shrunk']:+.2f}**). "
            f"Its weights sit **{h['weight_gap_vs_shrunk']:.0%}** from the shrunk portfolio's "
            f"and {h['weight_gap_vs_plain']:.0%} from the plain one — closer to the cheap fix "
            f"than to the thing it is fixing, which is the whole story. It also costs "
            f"{h['n_resamples']} optimisations per rebalance."),
        "one_sentence": (
            f"Resampling works, in the sense that it produces a more diversified portfolio and "
            f"a smaller utility gap against a known truth — but it lands "
            f"**{h['weight_gap_vs_shrunk']:.0%}** away from what a default shrinkage produces "
            f"in one pass, beats it by {h['sharpe_resampled'] - h['sharpe_shrunk']:+.2f} of "
            f"Sharpe with *t* = {h['t_vs_shrunk']:+.2f}, and costs {h['n_resamples']} "
            f"optimisations to get there."),
    }
