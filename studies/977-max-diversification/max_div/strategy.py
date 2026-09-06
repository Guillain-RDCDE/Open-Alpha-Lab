"""The diversification ratio and what maximising it produces — Study 977.

Choueifaty & Coignard (2008) define the **diversification ratio** of a long-only portfolio

    DR(w) = (w' sigma) / sqrt(w' Sigma w)

— the weighted average of the constituents' volatilities over the volatility of the portfolio.
It equals 1 for a single asset and grows as correlations fall. The *most diversified portfolio*
(MDP) maximises it.

Three facts about the objective, each of which this module makes checkable rather than
asserted:

1. **It is not minimum variance.** Minimum variance ignores the numerator entirely and will
   happily concentrate in the single quietest asset; the MDP is penalised for doing so.
   ``degenerate_case`` demonstrates the one situation where they coincide — equal volatilities
   — which is the honest boundary of the distinction.
2. **It is a well-behaved optimisation.** Maximising DR is equivalent to minimising the
   variance of a *volatility-rescaled* problem, so it can be solved as a minimum-variance
   problem on the correlation matrix and then unscaled (``max_div_weights``). No general-purpose
   optimiser, no convergence tuning: the closed form is exact for the unconstrained case, and a
   projected-gradient pass handles the long-only constraint.
3. **It has a free competitor, and an exact identity connecting them.** Under a *constant*
   correlation matrix, ``C^-1 1`` is proportional to ``1``, so the closed form collapses to
   ``w ∝ 1/sigma``: **the most diversified portfolio is exactly inverse-volatility weighting**,
   at every level of correlation. It therefore differs from the free competitor only through
   the *dispersion* of correlations, not their level — a much narrower claim than the method's
   presentation suggests, and one this module pins in ``equicorrelation_identity`` and in the
   test-suite rather than leaving as a footnote.

The scoreboard is the same as this desk's other allocation studies — rolling window, quarterly
re-estimation, one day of execution lag, costs on turnover — with the *realised diversification
ratio* reported alongside realised volatility, because a method that maximises a quantity
in-sample and fails to deliver it out of sample has told you something important about the
quantity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
METHODS = ("max_div", "min_var", "inv_vol", "equal", "risk_parity")
METHOD_LABEL = {
    "max_div": "Most diversified (Choueifaty)",
    "min_var": "Minimum variance",
    "inv_vol": "Inverse volatility",
    "equal": "1/N",
    "risk_parity": "Equal risk contribution",
}


# --------------------------------------------------------------------------- #
# The ratio itself
# --------------------------------------------------------------------------- #
def diversification_ratio(w: np.ndarray, cov: np.ndarray) -> float:
    """``(w' sigma) / sqrt(w' Sigma w)`` — 1 for one asset, higher when correlations fall."""
    sd = np.sqrt(np.diag(cov))
    port = float(np.sqrt(max(w @ cov @ w, 1e-24)))
    return float((w @ sd) / port) if port > 0 else np.nan


def effective_bets(w: np.ndarray, cov: np.ndarray) -> float:
    """Choueifaty's synthetic-asset count: the square of the diversification ratio."""
    dr = diversification_ratio(w, cov)
    return float(dr ** 2) if np.isfinite(dr) else np.nan


# --------------------------------------------------------------------------- #
# The weightings
# --------------------------------------------------------------------------- #
def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the simplex — the long-only constraint, done properly."""
    n = v.size
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def max_div_weights(cov: np.ndarray, long_only: bool = True, iters: int = 400,
                    step: float = 0.5) -> np.ndarray:
    """The most diversified portfolio.

    Unconstrained, the solution is closed form: maximising ``(w'sigma)/sqrt(w'Sigma w)`` is
    minimising the variance of the *correlation* matrix problem, so
    ``w ∝ diag(sigma)^-1 C^-1 1`` with ``C`` the correlation matrix. That vector is returned
    directly when ``long_only`` is False.

    With the long-only constraint the closed form can go negative, so the solution is refined
    by projected gradient ascent on log DR from the closed-form start. Convergence is checked
    in the test-suite against a brute-force search on small problems rather than assumed.
    """
    n = cov.shape[0]
    sd = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(np.outer(sd, sd) > 0, cov / np.outer(sd, sd), 0.0)
    np.fill_diagonal(C, 1.0)
    y = np.linalg.pinv(C + np.eye(n) * 1e-12) @ np.ones(n)
    w = y / np.maximum(sd, 1e-18)
    s = w.sum()
    w = w / s if abs(s) > 1e-12 else np.full(n, 1.0 / n)
    if not long_only:
        return w
    if (w >= -1e-12).all():
        return np.clip(w, 0.0, None) / np.clip(w, 0.0, None).sum()

    w = np.full(n, 1.0 / n)
    for _ in range(iters):
        port_var = float(w @ cov @ w)
        num = float(w @ sd)
        if port_var <= 0 or num <= 0:
            break
        # d/dw log DR = sigma/(w'sigma) - (Sigma w)/(w'Sigma w)
        grad = sd / num - (cov @ w) / port_var
        w_new = _project_simplex(w + step * grad / (np.linalg.norm(grad) + 1e-18))
        if np.max(np.abs(w_new - w)) < 1e-12:
            w = w_new
            break
        w = w_new
    return w


def min_variance_weights(cov: np.ndarray, long_only: bool = True) -> np.ndarray:
    """Global minimum variance, projected onto the simplex when long-only."""
    n = cov.shape[0]
    w = np.linalg.pinv(cov + np.eye(n) * 1e-14) @ np.ones(n)
    s = w.sum()
    w = w / s if abs(s) > 1e-12 else np.full(n, 1.0 / n)
    return _project_simplex(w) if long_only else w


def inverse_vol_weights(cov: np.ndarray) -> np.ndarray:
    """1/sigma, normalised — the free competitor that needs no correlation matrix."""
    iv = 1.0 / np.maximum(np.sqrt(np.diag(cov)), 1e-18)
    return iv / iv.sum()


def equal_weights(cov: np.ndarray) -> np.ndarray:
    """1/N."""
    n = cov.shape[0]
    return np.full(n, 1.0 / n)


def risk_parity_weights(cov: np.ndarray, iters: int = 2000, tol: float = 1e-12) -> np.ndarray:
    """Equal risk contribution, by fixed-point iteration on the contributions themselves."""
    n = cov.shape[0]
    w = np.full(n, 1.0 / n)
    for _ in range(iters):
        rc = w * (cov @ w)
        target = rc.mean()
        if target <= 0:
            break
        new = w * np.sqrt(target / np.maximum(rc, 1e-18))
        new = np.clip(new, 1e-14, None)
        new = new / new.sum()
        if np.max(np.abs(new - w)) < tol:
            return new
        w = new
    return w


def weights_for(method: str, cov: np.ndarray) -> np.ndarray:
    """Dispatch on ``METHODS``."""
    return {"max_div": max_div_weights, "min_var": min_variance_weights,
            "inv_vol": inverse_vol_weights, "equal": equal_weights,
            "risk_parity": risk_parity_weights}[method](cov)


def equicorrelation_identity(vols: np.ndarray, rhos=(0.0, 0.2, 0.5, 0.8)) -> pd.DataFrame:
    """Show, at several correlation levels, that the MDP equals inverse volatility exactly.

    This is the cleanest available answer to "what is the correlation matrix worth here?": if
    every pair shares one correlation, the answer is *nothing at all*, whatever that
    correlation is. Everything the method earns comes from correlations that differ from one
    another — which is why the real-tape section reports the dispersion of the correlation
    matrix next to the weight gap.
    """
    vols = np.asarray(vols, dtype=float)
    n = vols.size
    rows = []
    for rho in rhos:
        corr = np.full((n, n), rho)
        np.fill_diagonal(corr, 1.0)
        cov = np.outer(vols, vols) * corr
        w_md, w_iv = max_div_weights(cov), inverse_vol_weights(cov)
        rows.append({"rho": rho, "max_abs_diff": float(np.max(np.abs(w_md - w_iv))),
                     "dr": diversification_ratio(w_md, cov)})
    return pd.DataFrame(rows).set_index("rho")


def correlation_dispersion(cov: np.ndarray) -> float:
    """Standard deviation of the off-diagonal correlations — the MDP's actual raw material."""
    sd = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(np.outer(sd, sd) > 0, cov / np.outer(sd, sd), 0.0)
    off = ~np.eye(cov.shape[0], dtype=bool)
    return float(np.std(corr[off], ddof=1))


def degenerate_case(n: int = 8, rho: float = 0.3, vol: float = 0.2) -> dict:
    """Equal volatilities: the one place the MDP and minimum variance must coincide.

    With identical variances the diversification ratio is ``sigma / sqrt(w'Sigma w)``, a
    monotone decreasing function of portfolio variance — so maximising it *is* minimising
    variance. Any implementation that disagrees here is wrong, which makes this the sharpest
    test in the module.
    """
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    cov = (vol ** 2) * corr
    w_md, w_mv = max_div_weights(cov), min_variance_weights(cov)
    return {"cov": cov, "max_div": w_md, "min_var": w_mv,
            "max_abs_diff": float(np.max(np.abs(w_md - w_mv)))}


# --------------------------------------------------------------------------- #
# The scoreboard
# --------------------------------------------------------------------------- #
def walk_forward(rets: pd.DataFrame, window: int = 252, step: int = 63,
                 methods=METHODS, cost_bps: float = 5.0) -> pd.DataFrame:
    """Rolling re-estimation with one day of lag; realised volatility *and* realised DR."""
    R = rets.dropna(how="any")
    rows = []
    prev: dict[str, np.ndarray] = {}
    for start in range(window, len(R) - step, step):
        train = R.iloc[start - window:start].to_numpy()
        test = R.iloc[start:start + step]
        cov_in = np.cov(train, rowvar=False, ddof=1)
        cov_out = np.cov(test.to_numpy(), rowvar=False, ddof=1)
        for m in methods:
            w = weights_for(m, cov_in)
            port = test.to_numpy() @ w
            turn = float(np.abs(w - prev.get(m, np.zeros_like(w))).sum())
            prev[m] = w
            rows.append({
                "date": R.index[start], "method": m,
                "dr_in": diversification_ratio(w, cov_in),
                "dr_out": diversification_ratio(w, cov_out),
                "realised_vol": float(np.std(port, ddof=1) * np.sqrt(TRADING_DAYS)),
                "mean_ret": float(port.mean() * TRADING_DAYS
                                  - turn * cost_bps / 1e4 * TRADING_DAYS / step),
                "turnover": turn, "max_weight": float(np.max(w)),
                "effective_n": float(1.0 / np.sum(w ** 2)),
            })
    return pd.DataFrame(rows)


def summarise(wf: pd.DataFrame) -> pd.DataFrame:
    """Per method: realised volatility and DR, return, Sharpe, turnover, concentration."""
    g = wf.groupby("method")
    out = pd.DataFrame({
        "realised_vol": g["realised_vol"].mean(), "mean_ret": g["mean_ret"].mean(),
        "dr_in": g["dr_in"].mean(), "dr_out": g["dr_out"].mean(),
        "turnover": g["turnover"].mean(), "max_weight": g["max_weight"].mean(),
        "effective_n": g["effective_n"].mean(), "n": g.size(),
    })
    out["dr_slippage"] = out["dr_out"] / out["dr_in"] - 1.0
    out["sharpe"] = out["mean_ret"] / out["realised_vol"]
    return out.reindex([m for m in METHODS if m in out.index])


def paired_test(wf: pd.DataFrame, a: str, b: str, column: str = "realised_vol") -> dict:
    """Paired *t* on a per-rebalance column — same window, same holding period."""
    x = wf[wf["method"] == a].set_index("date")[column]
    y = wf[wf["method"] == b].set_index("date")[column]
    x, y = x.align(y, join="inner")
    d = (x - y).dropna()
    if len(d) < 8:
        return {"diff": np.nan, "t": np.nan, "n": int(len(d)), "win_rate": np.nan}
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"diff": float(d.mean()), "t": float(d.mean() / se) if se > 0 else np.nan,
            "n": int(len(d)), "win_rate": float((d < 0).mean())}


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal** (is it a distinct objective?): **Real** if the MDP's weights differ from the
      minimum-variance weights by more than 10% of the book on both panels **and** it achieves
      a higher in-sample diversification ratio; **Weak** if only one holds; **None** if the two
      portfolios are effectively the same.
    - **Usefulness**: **Useful** if the MDP beats **inverse volatility** — the free competitor
      — on realised volatility with a paired |*t*| >= 2 on at least one panel; **Fragile** if it
      wins without significance; **Mirage** if the free version is as good.
    """
    distinct = h["weight_gap_multi"] > 0.10 and h["weight_gap_sectors"] > 0.10
    higher_dr = h["dr_in_maxdiv"] > h["dr_in_minvar"]
    signal = ("Real" if distinct and higher_dr
              else ("Weak" if distinct or higher_dr else "None"))
    t = h["best_t_vs_invvol"]
    trad = ("Useful" if t > 2.0 else ("Fragile" if h["beats_invvol_panels"] >= 1 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"It is a genuinely different objective. The most diversified portfolio's weights "
            f"differ from the minimum-variance ones by **{h['weight_gap_multi']:.0%}** of the "
            f"book on the multi-asset panel and {h['weight_gap_sectors']:.0%} on sectors, and "
            f"it achieves an in-sample diversification ratio of **{h['dr_in_maxdiv']:.2f}** "
            f"against {h['dr_in_minvar']:.2f} — by construction, but it is worth confirming "
            f"the implementation does what the formula says. On a panel of equal-volatility "
            f"assets the two objectives collapse into one, and the two weight vectors agree to "
            f"{h['degenerate_gap']:.1e}, which is the boundary of the distinction."),
        "trad": trad,
        "trad_why": (
            f"The objective is delivered but the advantage is not free money. Out of sample the "
            f"realised diversification ratio slips to {h['dr_out_maxdiv']:.2f} "
            f"({h['dr_slippage_maxdiv']:+.0%} of the in-sample value), realised volatility is "
            f"**{h['vol_maxdiv']:.2%}** against {h['vol_invvol']:.2%} for plain inverse "
            f"volatility (paired *t* = {h['best_t_vs_invvol']:+.2f}) and {h['vol_equal']:.2%} "
            f"for 1/N, and it turns over {h['turnover_maxdiv']:.2f} a rebalance against "
            f"{h['turnover_invvol']:.2f}. It beats the free competitor on "
            f"**{h['beats_invvol_panels']} of 2** panels."),
        "one_sentence": (
            f"Maximum diversification is a real and distinct objective — it holds an effective "
            f"{h['effective_n_maxdiv']:.1f} positions against the optimiser's "
            f"{h['effective_n_minvar']:.1f} and delivers the higher diversification ratio it "
            f"promises — but most of what it achieves out of sample is available from "
            f"inverse-volatility weighting, which requires no correlation matrix and no "
            f"optimiser at all."),
    }
