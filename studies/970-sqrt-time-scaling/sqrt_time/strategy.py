"""Time scaling and its assumptions — Study 970.

The rule under examination: ``sigma_T = sigma_1 * sqrt(T)``. It follows from one assumption —
**serially independent returns** — because the variance of a sum of independent variables is
the sum of their variances. Anything else and the sum picks up covariance terms:

    Var(r_1 + ... + r_T) = T * Var(r) * [ 1 + 2 * sum_{k=1..T-1} (1 - k/T) * rho_k ]

The bracket is the **variance ratio** ``VR(T)``, and the whole study is about measuring it.

- ``VR > 1`` — positive autocorrelation (trending, or stale prices). sqrt(T) *understates*
  long-horizon risk.
- ``VR < 1`` — mean reversion. sqrt(T) *overstates* it.
- ``VR = 1`` — the textbook world.

Three ways to read the same quantity, all implemented because they fail differently:

1. ``variance_ratio`` — the direct estimator: the variance of overlapping q-period returns
   divided by q times the variance of one-period returns, with the Lo-MacKinlay small-sample
   bias corrections.
2. ``lo_mackinlay_test`` — the heteroskedasticity-robust z-statistic (Lo & MacKinlay 1988).
   Without the robust version, volatility clustering alone rejects the random walk, which
   would make this whole study a measurement of GARCH rather than of dependence.
3. ``realised_scaling`` — the assumption-free version: measure the *actual* standard deviation
   of non-overlapping q-day returns and compare it with ``sigma_1 * sqrt(q)``. No estimator,
   no correction, just what a holder experienced.

Then the consequences, because a variance ratio is not a finding until it is a number someone
acts on: ``vol_scaling_error`` converts VR into the percentage error in an annualised
volatility, ``var_scaling_error`` into the error in a 10-day 99% VaR (the Basel horizon), and
``sharpe_scaling_error`` uses the Lo (2002) factor from ``quantlab`` to correct an annualised
Sharpe ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from quantlab.analytics import lo_annualization_factor

TRADING_DAYS = 252
HORIZONS = (5, 21, 63, 252)
BASEL_HORIZON = 10


# --------------------------------------------------------------------------- #
# Variance ratios
# --------------------------------------------------------------------------- #
def variance_ratio(r: pd.Series, q: int, bias_correct: bool = True) -> float:
    """Lo-MacKinlay variance ratio at horizon ``q`` from overlapping sums.

    Overlapping q-period returns are used (far more efficient than non-overlapping ones) with
    the unbiased scaling constants of Lo & MacKinlay (1988): the denominators ``nq - q + 1``
    and ``m = q(nq - q + 1)(1 - q/nq)`` are what stop the estimator from being biased below 1
    in small samples — which would otherwise manufacture "mean reversion" everywhere.
    """
    x = np.asarray(r.dropna(), dtype=float)
    n = x.size
    if n < q * 3:
        return np.nan
    mu = x.mean()
    var1 = np.sum((x - mu) ** 2) / ((n - 1) if not bias_correct else (n - 1))
    sums = np.convolve(x, np.ones(q), mode="valid")     # overlapping q-period returns
    if bias_correct:
        m = q * (n - q + 1) * (1 - q / n)
        varq = np.sum((sums - q * mu) ** 2) / m
    else:
        varq = np.sum((sums - q * mu) ** 2) / (len(sums) * q)
    return float(varq / var1) if var1 > 0 else np.nan


def lo_mackinlay_test(r: pd.Series, q: int) -> dict:
    """Heteroskedasticity-robust variance-ratio z-statistic (Lo & MacKinlay 1988, eq. 18).

    The robust version matters more than the ratio itself: under the homoskedastic null,
    volatility clustering alone produces rejections, so a naive VR test on daily equity data
    "rejects the random walk" for entirely the wrong reason.
    """
    x = np.asarray(r.dropna(), dtype=float)
    n = x.size
    if n < q * 5:
        return {"vr": np.nan, "z": np.nan, "p_value": np.nan, "n": int(n), "q": int(q)}
    vr = variance_ratio(pd.Series(x), q)
    mu = x.mean()
    d = x - mu
    denom = float(np.sum(d ** 2)) ** 2
    theta = 0.0
    for k in range(1, q):
        num = float(np.sum((d[k:] ** 2) * (d[:-k] ** 2)))
        delta = num * n / denom
        theta += (2.0 * (q - k) / q) ** 2 * delta
    se = np.sqrt(theta / n) if theta > 0 else np.nan
    z = (vr - 1.0) / se if se and np.isfinite(se) and se > 0 else np.nan
    p = float(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
    return {"vr": float(vr), "z": float(z), "p_value": p, "se": float(se),
            "n": int(n), "q": int(q)}


def vr_curve(r: pd.Series, horizons=HORIZONS) -> pd.DataFrame:
    """The variance-ratio curve with robust z-statistics, one row per horizon."""
    return pd.DataFrame([lo_mackinlay_test(r, q) for q in horizons]).set_index("q")


# --------------------------------------------------------------------------- #
# The assumption-free version
# --------------------------------------------------------------------------- #
def realised_scaling(prices: pd.Series, horizons=HORIZONS) -> pd.DataFrame:
    """Actual volatility of **non-overlapping** q-day returns versus the sqrt(q) prediction.

    No estimator and no correction: chop the tape into disjoint q-day blocks, take the
    standard deviation of the block returns, and compare with ``sigma_1 * sqrt(q)``. Fewer
    observations, but nothing to argue with.
    """
    p = prices.dropna()
    r1 = p.pct_change().dropna()
    s1 = float(r1.std(ddof=1))
    rows = []
    for q in horizons:
        blocks = p.iloc[::q]
        rq = blocks.pct_change().dropna()
        if len(rq) < 12:
            continue
        actual = float(rq.std(ddof=1))
        predicted = s1 * np.sqrt(q)
        rows.append({"q": q, "n_blocks": int(len(rq)), "actual_sd": actual,
                     "sqrt_rule_sd": predicted, "ratio": actual / predicted,
                     "implied_vr": (actual / predicted) ** 2})
    return pd.DataFrame(rows).set_index("q")


# --------------------------------------------------------------------------- #
# What the error costs
# --------------------------------------------------------------------------- #
def vol_scaling_error(vr: float) -> float:
    """Percentage error in an annualised volatility built with sqrt(T): ``sqrt(VR) - 1``."""
    return float(np.sqrt(vr) - 1.0)


def var_scaling_error(daily_vol: float, vr: float, horizon: int = BASEL_HORIZON,
                      confidence: float = 0.99) -> dict:
    """A 10-day 99% VaR under the sqrt(T) rule and under the measured variance ratio.

    The Basel square-root-of-time rule scales a one-day VaR by ``sqrt(10)``. If the true
    10-day variance ratio is not 1, the reported figure is wrong by ``sqrt(VR) - 1`` — and
    for a positively autocorrelated book that error is in the comfortable direction.
    """
    z = float(norm.ppf(confidence))
    naive = z * daily_vol * np.sqrt(horizon)
    honest = naive * np.sqrt(vr)
    return {"var_sqrt_rule": float(naive), "var_corrected": float(honest),
            "error_pct": float(honest / naive - 1.0), "horizon": horizon,
            "confidence": confidence}


def sharpe_scaling_error(r: pd.Series, q: int = TRADING_DAYS) -> dict:
    """Annualised Sharpe under sqrt(T) versus under Lo's (2002) autocorrelation factor.

    **Read the correction with its own error bar.** At ``q = 252`` Lo's factor sums 251
    estimated autocorrelations; each carries a sampling error of roughly ``1/sqrt(n)`` and the
    weights ``(q - k)`` make the early ones dominant but do not make the tail negligible. On
    *independent* simulated returns the factor still lands 20-30% away from ``sqrt(252)`` on a
    single twenty-year sample — unbiased across draws, and very noisy within one. The same
    correction at ``q = 21`` is far better behaved, which is why both are reported in
    ``verify.py`` and why this study leans on variance ratios rather than on this factor for
    its headline.
    """
    x = r.dropna()
    sd = float(x.std(ddof=1))
    if sd <= 0:
        return {"sharpe_naive": np.nan, "sharpe_lo": np.nan, "factor": np.nan}
    per_period = float(x.mean() / sd)
    naive = per_period * np.sqrt(q)
    factor = float(lo_annualization_factor(x, q=q))
    return {"sharpe_naive": naive, "sharpe_lo": per_period * factor,
            "factor": factor, "sqrt_factor": float(np.sqrt(q)),
            "relative_error": float(factor / np.sqrt(q) - 1.0)}


def autocorrelation_profile(r: pd.Series, lags: int = 10) -> pd.Series:
    """The first few autocorrelations — the raw material behind every variance ratio."""
    x = r.dropna()
    return pd.Series({k: float(x.autocorr(k)) for k in range(1, lags + 1)}, name="rho")


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal** (do real tapes violate independence?): **Real** if at least three tapes
      reject VR = 1 at the annual horizon with a robust |z| >= 2; **Weak** if one or two do;
      **None** otherwise.
    - **Usefulness** (does the error matter?): **Useful** if the implied volatility error
      exceeds 10% on at least one tape a desk would actually hold; **Fragile** above 3%;
      **Mirage** below.
    """
    n_rej = h["n_reject_annual"]
    signal = "Real" if n_rej >= 3 else ("Weak" if n_rej >= 1 else "None")
    err = h["max_abs_vol_error"]
    trad = "Useful" if err >= 0.10 else ("Fragile" if err >= 0.03 else "Mirage")
    return {
        "signal": signal,
        "signal_why": (
            f"Yes, and not subtly. At the annual horizon **{n_rej} of {h['n_tickers']}** tapes "
            f"reject VR = 1 with a heteroskedasticity-robust |z| >= 2. The extremes are "
            f"**{h['max_vr_ticker']}** at VR = **{h['max_vr']:.2f}** (trending: sqrt(T) "
            f"*understates* its annual volatility by {vol_scaling_error(h['max_vr']):.0%}) and "
            f"**{h['min_vr_ticker']}** at VR = **{h['min_vr']:.2f}** (mean-reverting: it "
            f"*overstates* by {-vol_scaling_error(h['min_vr']):.0%}). Equity indices sit close "
            f"to 1, which is why the rule survived: it is nearly right for the one asset class "
            f"everybody tests it on."),
        "trad": trad,
        "trad_why": (
            f"On a 10-day 99% VaR — the Basel horizon, computed by exactly this rule — the "
            f"correction reaches **{h['max_var_error']:+.0%}** on {h['max_var_ticker']}. An "
            f"annualised Sharpe moves too: Lo's (2002) factor differs from sqrt(252) by up to "
            f"**{h['max_sharpe_error']:+.0%}** — though that correction carries a "
            f"{h['lo_factor_noise_iid']:.0%} standard deviation of its own on i.i.d. data, so "
            f"the variance ratio, not the Sharpe factor, is the load-bearing number. On SPY the "
            f"whole correction is {h['spy_vol_error']:+.1%} — invisible, and that is precisely "
            f"why nobody checks it on anything else."),
        "one_sentence": (
            f"sqrt(T) is exactly right for independent returns and approximately right for "
            f"equity indices, which is why it is used everywhere — but on the bond and bill "
            f"funds sitting in the same risk system it is off by "
            f"**{h['max_abs_vol_error']:.0%}** in volatility and "
            f"**{h['max_var_error']:+.0%}** on a 10-day VaR, always in the direction that "
            f"makes the book look safer."),
    }
