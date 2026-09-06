"""Covariance estimation and the portfolios it builds — Study 975.

The sample covariance matrix is unbiased and, when the number of assets ``N`` approaches the
number of observations ``T``, useless. Its eigenvalues are spread far wider than the truth's —
the largest too large, the smallest far too small — and a minimum-variance optimiser is
attracted to exactly those smallest eigenvalues, because they look like free risk reduction.
That is Michaud's error maximisation stated in linear algebra.

Four estimators, each a different bet about what to believe when the data is thin:

- ``sample_cov`` — believe everything.
- ``shrink_identity`` — Ledoit-Wolf (2004) toward a scaled identity: all variances equal, all
  correlations zero. The intensity is computed, not chosen.
- ``shrink_constant_correlation`` — Ledoit-Wolf (2003) toward the constant-correlation matrix:
  every pair gets the *average* pairwise correlation. Usually the better target on equities,
  because the average correlation is the one thing a market really does have.
- ``diagonal_cov`` — believe nothing but the variances. The crude limit case, included because
  it is often close to as good as the clever ones and needs no algebra at all.

Scored three ways, in increasing order of what a user cares about:

1. ``frobenius_error`` — distance to a known true matrix (simulation only; there is no truth on
   the real tape).
2. ``condition_number`` — how invertible the matrix is, which is what the optimiser actually
   touches.
3. ``walk_forward`` — the honest test: re-estimate every quarter on a rolling window, build the
   minimum-variance portfolio, hold it, and measure the volatility it *delivered* against the
   volatility it *promised*. Everything else in this module is a diagnostic; this is the
   scoreboard.

The Ledoit-Wolf intensity here is the standard analytic one: ``delta = pi_hat / gamma_hat``,
clipped to [0, 1], where ``pi_hat`` estimates the sum of asymptotic variances of the sample
covariance entries and ``gamma_hat`` the squared distance between sample and target. No
cross-validation, no tuning knob — which is exactly why the estimator is usable in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
ESTIMATORS = ("sample", "identity", "constant_corr", "diagonal")
ESTIMATOR_LABEL = {
    "sample": "Sample covariance",
    "identity": "Ledoit-Wolf -> identity",
    "constant_corr": "Ledoit-Wolf -> constant correlation",
    "diagonal": "Diagonal only",
}


# --------------------------------------------------------------------------- #
# The estimators
# --------------------------------------------------------------------------- #
def sample_cov(X: np.ndarray) -> np.ndarray:
    """Plain sample covariance (rows are observations)."""
    return np.cov(X, rowvar=False, ddof=1)


def diagonal_cov(X: np.ndarray) -> np.ndarray:
    """Variances only — the estimator that refuses to believe in correlation."""
    return np.diag(np.var(X, axis=0, ddof=1))


def _lw_pi_gamma(X: np.ndarray, S: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """The two Ledoit-Wolf quantities: estimator noise (pi) and target bias (gamma)."""
    T = X.shape[0]
    Xc = X - X.mean(axis=0)
    # pi: sum over entries of the asymptotic variance of the sample covariance
    pi_mat = np.zeros_like(S)
    for t in range(T):
        d = np.outer(Xc[t], Xc[t]) - S
        pi_mat += d * d
    pi_hat = float(pi_mat.sum() / T)
    gamma_hat = float(((target - S) ** 2).sum())
    return pi_hat, gamma_hat


def shrink_identity(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) toward ``mu * I``, with the analytic intensity."""
    S = sample_cov(X)
    n = S.shape[0]
    mu = float(np.trace(S) / n)
    target = mu * np.eye(n)
    pi_hat, gamma_hat = _lw_pi_gamma(X, S, target)
    T = X.shape[0]
    delta = 0.0 if gamma_hat <= 0 else float(np.clip((pi_hat / T) / gamma_hat, 0.0, 1.0))
    return (1 - delta) * S + delta * target, delta


def shrink_constant_correlation(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2003) toward the constant-correlation matrix.

    Target: every asset keeps its own variance, and every pair is assigned the *average*
    sample correlation. On equities this is a strong target — average correlation is a real,
    stable feature of a market — which is why the intensity it earns is usually lower than the
    identity target's and the result usually better.
    """
    S = sample_cov(X)
    n = S.shape[0]
    sd = np.sqrt(np.diag(S))
    outer_sd = np.outer(sd, sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(outer_sd > 0, S / outer_sd, 0.0)
    off = ~np.eye(n, dtype=bool)
    r_bar = float(R[off].mean()) if n > 1 else 0.0
    target = r_bar * outer_sd
    np.fill_diagonal(target, np.diag(S))
    pi_hat, gamma_hat = _lw_pi_gamma(X, S, target)
    T = X.shape[0]
    delta = 0.0 if gamma_hat <= 0 else float(np.clip((pi_hat / T) / gamma_hat, 0.0, 1.0))
    return (1 - delta) * S + delta * target, delta


def estimate(X: np.ndarray, method: str) -> tuple[np.ndarray, float]:
    """Dispatch: ``(covariance, shrinkage intensity)`` for one of ``ESTIMATORS``."""
    if method == "sample":
        return sample_cov(X), 0.0
    if method == "diagonal":
        return diagonal_cov(X), 1.0
    if method == "identity":
        return shrink_identity(X)
    if method == "constant_corr":
        return shrink_constant_correlation(X)
    raise ValueError(f"unknown estimator {method!r}")


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def condition_number(C: np.ndarray) -> float:
    """Ratio of largest to smallest eigenvalue — how much the optimiser can misbehave."""
    w = np.linalg.eigvalsh(C)
    lo = max(float(w.min()), 1e-18)
    return float(w.max() / lo)


def frobenius_error(C: np.ndarray, truth: np.ndarray) -> float:
    """Relative Frobenius distance to a known true matrix (simulation only)."""
    return float(np.linalg.norm(C - truth, "fro") / np.linalg.norm(truth, "fro"))


def eigen_spread(C: np.ndarray) -> dict:
    """The shape of the eigenvalue spectrum — where the estimation error lives."""
    w = np.sort(np.linalg.eigvalsh(C))[::-1]
    return {"max": float(w[0]), "min": float(w[-1]),
            "top_share": float(w[0] / w.sum()) if w.sum() > 0 else np.nan,
            "n_below_1pct": int((w < 0.01 * w.max()).sum())}


def min_variance_weights(C: np.ndarray, long_only: bool = False) -> np.ndarray:
    """Global minimum-variance weights; ``long_only`` clips and renormalises."""
    n = C.shape[0]
    inv = np.linalg.pinv(C + np.eye(n) * 1e-14)
    w = inv @ np.ones(n)
    s = w.sum()
    w = w / s if abs(s) > 1e-12 else np.full(n, 1 / n)
    if long_only:
        w = np.clip(w, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.full(n, 1 / n)
    return w


# --------------------------------------------------------------------------- #
# The scoreboard
# --------------------------------------------------------------------------- #
def walk_forward(rets: pd.DataFrame, window: int = 252, step: int = 63,
                 methods=ESTIMATORS, long_only: bool = False,
                 cost_bps: float = 5.0) -> pd.DataFrame:
    """Re-estimate every ``step`` days on a rolling ``window``; hold; measure what happened.

    The weights formed at the close of day ``t`` apply from day ``t+1`` — one execution lag,
    and the only one — and each rebalance is charged ``cost_bps`` on the notional it turns
    over. Returns one row per (method, rebalance) with the promised in-sample volatility, the
    realised out-of-sample volatility over the holding period, the turnover and the
    concentration.
    """
    R = rets.dropna(how="any")
    rows = []
    prev: dict[str, np.ndarray] = {}
    for start in range(window, len(R) - step, step):
        train = R.iloc[start - window:start].to_numpy()
        test = R.iloc[start:start + step]
        for m in methods:
            C, delta = estimate(train, m)
            w = min_variance_weights(C, long_only)
            port = test.to_numpy() @ w
            turn = float(np.abs(w - prev.get(m, np.zeros_like(w))).sum())
            prev[m] = w
            cost = turn * cost_bps / 1e4
            rows.append({
                "date": R.index[start], "method": m, "delta": delta,
                "promised_vol": float(np.sqrt(w @ C @ w * TRADING_DAYS)),
                "realised_vol": float(np.std(port, ddof=1) * np.sqrt(TRADING_DAYS)),
                "mean_ret": float(port.mean() * TRADING_DAYS - cost * TRADING_DAYS / step),
                "turnover": turn, "max_weight": float(np.max(np.abs(w))),
                "short_weight": float(np.sum(np.clip(-w, 0, None))),
                "condition": condition_number(C),
            })
    return pd.DataFrame(rows)


def summarise(wf: pd.DataFrame) -> pd.DataFrame:
    """Per estimator: realised volatility, optimism, turnover, concentration, intensity."""
    g = wf.groupby("method")
    out = pd.DataFrame({
        "realised_vol": g["realised_vol"].mean(),
        "promised_vol": g["promised_vol"].mean(),
        "mean_ret": g["mean_ret"].mean(),
        "turnover": g["turnover"].mean(),
        "max_weight": g["max_weight"].mean(),
        "short_weight": g["short_weight"].mean(),
        "condition": g["condition"].median(),
        "delta": g["delta"].mean(),
        "n": g.size(),
    })
    out["optimism"] = out["promised_vol"] / out["realised_vol"] - 1.0
    out["sharpe"] = out["mean_ret"] / out["realised_vol"]
    return out.reindex([m for m in ESTIMATORS if m in out.index])


def paired_vol_test(wf: pd.DataFrame, a: str, b: str) -> dict:
    """Paired *t* on the per-rebalance realised volatility of two estimators.

    Paired, because both estimators saw the same window and the same holding period; a
    two-sample test on these would ignore the strongest structure in the data.
    """
    x = wf[wf["method"] == a].set_index("date")["realised_vol"]
    y = wf[wf["method"] == b].set_index("date")["realised_vol"]
    x, y = x.align(y, join="inner")
    d = (x - y).dropna()
    if len(d) < 8:
        return {"diff": np.nan, "t": np.nan, "n": int(len(d))}
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"diff": float(d.mean()), "t": float(d.mean() / se) if se > 0 else np.nan,
            "n": int(len(d)), "win_rate": float((d < 0).mean())}


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Real** if the sample covariance's in-sample optimism exceeds 20% on the
      wide cross-section (it promises a fifth less risk than it delivers); **Weak** above 5%;
      **None** below.
    - **Usefulness**: **Useful** if the best shrinkage estimator cuts realised volatility by
      at least 5% relative to the sample matrix with a paired |*t*| >= 2; **Fragile** if it
      wins without significance; **Mirage** if it does not win.
    """
    opt = h["wide_optimism_sample"]
    signal = "Real" if opt >= 0.20 else ("Weak" if opt >= 0.05 else "None")
    gain, t = h["wide_vol_saving"], h["wide_paired_t"]
    trad = ("Useful" if gain >= 0.05 and abs(t) >= 2.0
            else ("Fragile" if gain > 0 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"On the wide cross-section ({h['n_names']} names, {h['n_params_wide']} covariance "
            f"parameters, {h['window']} rows of history) the sample matrix promises "
            f"{h['wide_promised_sample']:.2%} annualised volatility and delivers "
            f"**{h['wide_realised_sample']:.2%}** — it is **{opt:.0%}** optimistic, and its "
            f"median condition number is **{h['wide_condition_sample']:,.0f}** against "
            f"{h['wide_condition_best']:,.0f} for the shrunk version. On the eleven-sector "
            f"sleeve, where rows outnumber parameters {h['sector_rows_per_param']:.0f} to one, "
            f"the same optimism is only {h['sector_optimism_sample']:.0%} — the problem is "
            f"arithmetic, not equities."),
        "trad": trad,
        "trad_why": (
            f"**{h['best_method']}** cut realised volatility from {h['wide_realised_sample']:.2%} "
            f"to **{h['wide_realised_best']:.2%}** — a **{gain:.0%}** saving, paired *t* = "
            f"**{t:+.2f}** across {h['n_rebalances']} rebalances, winning "
            f"{h['wide_win_rate']:.0%} of them. It also turns over "
            f"{h['turnover_best']:.2f} against {h['turnover_sample']:.2f} per rebalance and "
            f"holds a maximum weight of {h['max_weight_best']:.0%} against "
            f"{h['max_weight_sample']:.0%}. Three lines of algebra, no tuning parameter, no "
            f"look-ahead."),
        "one_sentence": (
            f"The sample covariance matrix is not wrong so much as overconfident: on forty "
            f"names estimated from a year of data it under-promises risk by **{opt:.0%}** and "
            f"builds portfolios with {h['max_weight_sample']:.0%} single-name weights, while a "
            f"shrinkage estimator with no free parameters cuts realised volatility by "
            f"**{gain:.0%}** — and on eleven sectors, where the arithmetic is comfortable, it "
            f"changes almost nothing."),
    }
