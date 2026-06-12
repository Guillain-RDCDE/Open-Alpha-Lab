"""Fitting the arrival kernel — and asking whether it is exponential at all.

The Avellaneda-Stoikov optimum needs a single number, ``k``, from the assumed kernel
``lambda(delta) = A e^{-k delta}``. This module (1) recovers ``(A, k)`` from observed fills,
(2) tests whether the exponential form is even the right shape versus a heavy-tailed power
law, and (3) prices the consequence of a *static* ``k`` when the true one drifts — the
"phantom parameter" cost that flows straight into the spread.

Fitting is a weighted log-linear regression: with fills bucketed by quote distance,
``log lambda(delta) = log A - k * delta`` is linear in ``delta`` (exponential kernel) and
``log lambda(delta) = log A' - alpha * log(delta)`` is linear in ``log delta`` (power-law
kernel). Counts are weighted by themselves (a Poisson-variance weighting) so dense, reliable
buckets dominate the noisy tail. Those binned fits are *descriptive* (they recover ``k`` and
draw the curve); the *verdict* between the two shapes comes from
:func:`reach_likelihood_test` — a per-observation likelihood comparison on the raw execution
depths, where each order counts exactly once, scored by AIC/BIC and Vuong's test. Pure numpy —
no scipy — to keep the study's dependency surface as light as the rest of the desk.
"""

from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------- #
# Rigorous tail discrimination: power-law vs exponential (Clauset/Vuong)
# --------------------------------------------------------------------------- #
def _normal_sf(z: float) -> float:
    """Upper-tail of the standard normal (no scipy)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def tail_test(x: np.ndarray, xmin_quantiles=None) -> dict:
    """Is the upper tail of ``x`` power-law or exponential? (Clauset-Shalizi-Newman + Vuong.)

    Binned log-count regression is grid-sensitive and not a reliable distribution test; this is
    the standard alternative. It (1) picks the tail cutoff ``xmin`` by minimising the KS
    distance to a fitted continuous power law, (2) fits both a power law and a (shifted)
    exponential to the tail by MLE, and (3) runs Vuong's normalised likelihood-ratio test.

    Returns ``alpha`` (power-law exponent), ``lam`` (exponential rate), ``xmin``, ``n_tail``,
    the Vuong statistic ``V`` (``> 0`` favours the power law), its two-sided ``p``, and a
    ``winner`` that is only declared when ``p < 0.05`` (else ``"inconclusive"``).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    if xmin_quantiles is None:
        xmin_quantiles = np.linspace(0.50, 0.98, 40)

    best = None
    for q in xmin_quantiles:
        xmin = float(np.quantile(x, q))
        tail = np.sort(x[x >= xmin])
        n = tail.size
        if n < 200 or xmin <= 0:
            continue
        alpha = 1.0 + n / float(np.sum(np.log(tail / xmin)))
        # KS distance to the fitted continuous power law S(x) = (x/xmin)^-(alpha-1).
        cdf_emp = np.arange(1, n + 1) / n
        cdf_fit = 1.0 - (tail / xmin) ** (-(alpha - 1.0))
        ks = float(np.max(np.abs(cdf_emp - cdf_fit)))
        if best is None or ks < best["ks"]:
            best = {"xmin": xmin, "n": n, "alpha": alpha, "ks": ks, "tail": tail}

    if best is None:
        return {"winner": "inconclusive", "n_tail": 0}

    tail, xmin, n, alpha = best["tail"], best["xmin"], best["n"], best["alpha"]
    lam = 1.0 / (tail.mean() - xmin) if tail.mean() > xmin else float("inf")

    # Per-point log-likelihoods on the same tail, then Vuong's normalised LR test.
    ll_pl = math.log(alpha - 1.0) - math.log(xmin) - alpha * np.log(tail / xmin)
    ll_exp = math.log(lam) - lam * (tail - xmin)
    d = ll_pl - ll_exp
    lr = float(d.sum())
    sd = float(d.std(ddof=1))
    V = lr / (math.sqrt(n) * sd) if sd > 0 else 0.0
    p = 2.0 * _normal_sf(abs(V))
    winner = ("power-law" if V > 0 else "exponential") if p < 0.05 else "inconclusive"
    return {
        "alpha": round(alpha, 3),
        "lam": lam,
        "xmin": xmin,
        "n_tail": n,
        "ks": round(best["ks"], 4),
        "LR": round(lr, 1),
        "V": round(V, 2),
        "p": p,
        "winner": winner,
    }


# --------------------------------------------------------------------------- #
# Weighted linear regression (the shared primitive)
# --------------------------------------------------------------------------- #
def _wls(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Weighted least squares ``y = a + b x``; returns ``(a, b, weighted_R2)``."""
    sw = w.sum()
    swx = (w * x).sum()
    swy = (w * y).sum()
    swxx = (w * x * x).sum()
    swxy = (w * x * y).sum()
    denom = sw * swxx - swx * swx
    if denom == 0:
        return float("nan"), float("nan"), float("nan")
    b = (sw * swxy - swx * swy) / denom
    a = (swy - b * swx) / sw
    yhat = a + b * x
    ybar = swy / sw
    ss_res = float((w * (y - yhat) ** 2).sum())
    ss_tot = float((w * (y - ybar) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), float(r2)


def reach_likelihood_test(reach: np.ndarray) -> dict:
    """Exponential vs power-law on the **individual execution depths** — the valid H1 test.

    Each order contributes exactly one observation (its reach), so under either hypothesis the
    depths are iid draws from the reach law and the two log-likelihoods are legitimate:

    * **exponential** ``f(x) = lam e^{-lam x}`` (MLE ``lam = 1/mean``; 1 parameter) — the law
      whose survival function *is* the AS kernel;
    * **power law** (Pareto I) ``f(x) = (alpha/xm)(x/xm)^{-(alpha+1)}`` on ``x >= xm``
      (MLE ``xm = min(x)``, ``alpha = n / sum log(x/xm)``; 2 parameters) — the heavy tail.

    The two models are non-nested, so they are scored by AIC/BIC *and* by Vuong's normalised
    likelihood-ratio test (Vuong 1989): ``V > 0`` favours the power law, two-sided ``p``, and a
    ``winner`` declared only at ``p < 0.05``. ``ll_per_obs`` is the mean per-order log-likelihood
    edge of the power law — the magnitude that scales honestly.

    Why this replaces the study's original test: the first version compared Poisson AICs across
    80 **cumulative survival counts** of the *same* orders (the count at depth ``delta`` and at
    ``delta'`` share orders), so the buckets were neither independent nor Poisson and the AIC
    gap grew without bound in ``n_orders`` — its magnitude was meaningless (its direction, it
    turns out, was right). Here every order enters the likelihood exactly once.
    """
    x = np.asarray(reach, float)
    x = x[np.isfinite(x) & (x > 0)]
    n = x.size
    if n < 100:
        return {"winner": "inconclusive", "n": int(n)}

    # Exponential MLE (1 parameter).
    lam = 1.0 / float(x.mean())
    ll_exp = math.log(lam) - lam * x

    # Pareto I MLE (2 parameters: the scale is the smallest depth, then alpha in closed form).
    xm = float(x.min())
    logs = np.log(x / xm)
    alpha = n / float(logs.sum())
    ll_pl = math.log(alpha) - math.log(xm) - (alpha + 1.0) * logs

    d = ll_pl - ll_exp
    lr = float(d.sum())
    sd = float(d.std(ddof=1))
    V = lr / (math.sqrt(n) * sd) if sd > 0 else 0.0
    p = 2.0 * _normal_sf(abs(V))
    sum_exp, sum_pl = float(ll_exp.sum()), float(ll_pl.sum())
    winner = ("power-law" if V > 0 else "exponential") if p < 0.05 else "inconclusive"
    return {
        "n": int(n),
        "lam": lam,
        "alpha_mle": alpha,
        "xm": xm,
        "aic_exp": 2 * 1 - 2 * sum_exp,
        "aic_pow": 2 * 2 - 2 * sum_pl,
        "aic_gap": (2 * 1 - 2 * sum_exp) - (2 * 2 - 2 * sum_pl),   # > 0: power law preferred
        "bic_gap": (math.log(n) * 1 - 2 * sum_exp) - (math.log(n) * 2 - 2 * sum_pl),
        "ll_per_obs": lr / n,
        "V": V,
        "p": p,
        "winner": winner,
    }


# --------------------------------------------------------------------------- #
# The two competing kernel fits
# --------------------------------------------------------------------------- #
def fit_exponential(deltas: np.ndarray, counts: np.ndarray) -> dict:
    """Fit ``lambda(delta) = A e^{-k delta}`` (the AS kernel) to bucketed fill counts.

    Returns the recovered ``k`` and ``A`` (relative) and the weighted R^2 of the log-linear
    fit. A clean exponential world gives R^2 ~ 1; the worse the fit, the more ``k`` is a
    fiction. Descriptive only — the exponential-vs-power-law *verdict* is
    :func:`reach_likelihood_test` on the raw depths.
    """
    deltas = np.asarray(deltas, float)
    counts = np.asarray(counts, float)
    m = counts > 0
    x, c = deltas[m], counts[m]
    a, b, r2 = _wls(x, np.log(c), c)
    return {
        "k": -b,
        "A": math.exp(a) if np.isfinite(a) else float("nan"),
        "r2": r2,
        "n_buckets": int(m.sum()),
    }


def fit_powerlaw(deltas: np.ndarray, counts: np.ndarray) -> dict:
    """Fit ``lambda(delta) = A' delta^{-alpha}`` (a heavy-tailed kernel) to the same counts.

    The alternative hypothesis: order reach is power-law, so the kernel decays polynomially,
    not exponentially. Descriptive, like :func:`fit_exponential`.
    """
    deltas = np.asarray(deltas, float)
    counts = np.asarray(counts, float)
    m = (counts > 0) & (deltas > 0)
    x, c = deltas[m], counts[m]
    a, b, r2 = _wls(np.log(x), np.log(c), c)
    return {
        "alpha": -b,
        "r2": r2,
        "n_buckets": int(m.sum()),
    }


def goodness_of_fit(reach: np.ndarray, deltas: np.ndarray, counts: np.ndarray) -> dict:
    """Head-to-head: is the fill kernel exponential (AS) or a power law (heavy-tailed)?

    Two layers. The **descriptive** layer is the binned weighted-R^2 fits (they recover the
    ``k`` a practitioner would plug in, and the power-law slope). The **verdict** layer is
    :func:`reach_likelihood_test` on the raw execution depths — each order counted exactly
    once — which supplies the legitimate AIC/BIC gap and Vuong statistic.

    *Restated:* an earlier version of this function scored the binned fits by Poisson AIC.
    Those 80 buckets are nested cumulative survival counts of the same orders — not
    independent Poisson draws — so that AIC gap scaled with ``n_orders`` and its magnitude was
    statistically meaningless. The per-observation test replaces it; the *direction* of the
    old result (power law wins in the friction world) survives.
    """
    exp = fit_exponential(deltas, counts)
    pw = fit_powerlaw(deltas, counts)
    lt = reach_likelihood_test(reach)
    return {
        "r2_exp": exp["r2"],
        "r2_pow": pw["r2"],
        "k_exp": exp["k"],
        "alpha_pow": pw["alpha"],
        "n_orders": lt["n"],
        "alpha_mle": lt.get("alpha_mle", float("nan")),
        "aic_gap": lt.get("aic_gap", float("nan")),    # > 0 : power law preferred
        "bic_gap": lt.get("bic_gap", float("nan")),
        "ll_per_obs": lt.get("ll_per_obs", float("nan")),
        "V": lt.get("V", float("nan")),
        "p": lt.get("p", float("nan")),
        "winner": lt["winner"],
    }


# --------------------------------------------------------------------------- #
# The phantom-parameter cost: a static k when the true k drifts
# --------------------------------------------------------------------------- #
def static_k_spread_error(
    k_values: np.ndarray,
    deltas: np.ndarray,
    n_orders: int = 200_000,
    gamma: float = 0.1,
    sigma: float = 0.4,
    horizon: float = 1.0,
    eval_t: float = 0.0,
    seed: int = 0,
) -> dict:
    """Quantify the spread error from using one *static* k while the true k moves by regime.

    The paper itself concedes ``k`` swings 3-5x intraday. Here each regime really is
    exponential with its own ``k`` (so an *adaptive* estimator could track it), but base AS
    fits a single pooled ``k`` over the whole session. We recover each regime's ``k``, the
    pooled ``k``, and translate the gap into the quantity that matters: the percentage error
    in the AS *optimal half-spread* the static fit would quote in each regime.

    ``horizon`` and ``eval_t`` set where the spread is evaluated, and they matter: the k-term
    ``(2/gamma) ln(1 + gamma/k)`` is constant in time while the inventory term
    ``gamma sigma^2 (T - t)`` scales with the remaining horizon, so a short horizon makes the
    k-error the whole spread (the worst case) and a long one dilutes it. Quote the number at
    the horizon the rest of the study actually trades (the tournament's ``T``), and keep the
    short-horizon figure as a labelled upper bound.

    Imports the spread formula from :mod:`phantom_kernel.strategies` lazily to avoid a cycle.
    """
    from .strategies import optimal_spread

    from .sim import World, fill_counts

    deltas = np.asarray(deltas, float)
    k_values = np.asarray(k_values, float)

    per_regime_k, pooled_counts = [], np.zeros_like(deltas)
    for i, k in enumerate(k_values):
        w = World(name=f"regime{i}", reach_kind="exponential", reach_k=float(k))
        c = fill_counts(w, deltas, n_orders=n_orders, seed=seed + i)
        per_regime_k.append(fit_exponential(deltas, c)["k"])
        pooled_counts = pooled_counts + c
    k_pooled = fit_exponential(deltas, pooled_counts)["k"]

    # Spread the static (pooled) k would quote vs the spread each regime's true k warrants.
    spread_true = np.array([optimal_spread(eval_t, gamma, sigma, k, horizon) for k in k_values])
    spread_static = np.array([optimal_spread(eval_t, gamma, sigma, k_pooled, horizon)] * len(k_values))
    pct_err = 100.0 * (spread_static - spread_true) / spread_true

    return {
        "k_true": k_values.tolist(),
        "k_recovered_per_regime": [round(x, 4) for x in per_regime_k],
        "k_pooled_static": round(float(k_pooled), 4),
        "k_true_ratio": float(k_values.max() / k_values.min()),
        "horizon": horizon,
        "eval_t": eval_t,
        "spread_pct_error_per_regime": [round(float(x), 2) for x in pct_err],
        "max_abs_spread_pct_error": round(float(np.max(np.abs(pct_err))), 2),
    }
