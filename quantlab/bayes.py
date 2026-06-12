"""Two pieces of heavy artillery: a transparent Bayesian posterior on the
manipulation hypothesis, and a White (2000) Reality Check for data-snooping.

**Bayesian posterior.** Turn the §8 likelihood argument into a number — honestly.
We do *not* pretend the likelihood ratios are objective; they are stated, with
reasons, and shipped with a sensitivity analysis so a reader can plug in their
own. The maths is just Bayes in odds form:

    posterior_odds = prior_odds * Π LR_i,   LR_i = P(E_i | manip) / P(E_i | micro)

**Reality Check.** "The 25 most problematic markets" is selection. White's
Reality Check asks: is the *best* overnight Sharpe in a universe of N series
better than you'd expect from searching N series of pure noise? We bootstrap the
distribution of the max Sharpe under the mean-zero null and get a multiplicity-
aware p-value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .decompose import TRADING_DAYS_PER_YEAR

# Discriminating evidence and its likelihood ratio LR = P(E|manip)/P(E|micro).
# LR < 1 favours "risk premium + microstructure"; LR = 1 is non-discriminating.
# These are explicit, defensible priors — vary them with `posterior_sensitivity`.
# LRs are deliberately *moderate* and defensible (not extreme): the aim is a
# robust direction, not a falsely precise number. They are combined assuming
# rough independence, which overstates the evidence when items are correlated —
# so read the posterior together with `posterior_sensitivity`, never alone.
DEFAULT_EVIDENCE = [
    ("Basic night > day pattern", 1.00, "predicted by both hypotheses; non-discriminating"),
    ("Foreign-ETF sign inversion (listing clock)", 0.40, "predicted by microstructure; awkward for one global actor"),
    ("Chinese T+1 inversion", 0.50, "a settlement-rule natural experiment, not manipulation"),
    ("Per-hour edge ~1.4x (clock illusion)", 0.65, "much of the magnitude is calendar time, not a scheme"),
    ("Negative net P&L at world-moving scale", 0.35, "the alleged mechanism is economically self-defeating"),
    ("Post-publication Sharpe decay (~2 to ~0.5)", 0.55, "an ongoing scheme would not fade as it is documented"),
]


def manipulation_posterior(prior_manip: float = 0.5, evidence=None) -> dict:
    """Posterior P(manipulation | evidence) by combining likelihood ratios.

    ``prior_manip`` is the prior probability of the manipulation hypothesis;
    ``evidence`` is a list of ``(name, LR, rationale)``. Returns the posterior
    and the running odds so the contribution of each item is visible.
    """
    ev = DEFAULT_EVIDENCE if evidence is None else evidence
    odds = prior_manip / (1.0 - prior_manip)
    steps = []
    for name, lr, *_ in ev:
        odds *= lr
        steps.append({"evidence": name, "LR": lr, "running_posterior": odds / (1.0 + odds)})
    return {
        "prior_manip": prior_manip,
        "posterior_manip": odds / (1.0 + odds),
        "combined_LR": float(np.prod([lr for _, lr, *_ in ev])),
        "steps": pd.DataFrame(steps),
    }


def posterior_sensitivity(
    priors=(0.1, 0.3, 0.5, 0.7, 0.9),
    lr_scales=(0.5, 1.0, 2.0),
    evidence=None,
) -> pd.DataFrame:
    """Posterior across a grid of priors and LR-strength multipliers.

    ``lr_scale`` raises every LR to a power: <1 weakens the evidence toward 1,
    >1 strengthens it. Even hostile settings (strong prior, halved evidence)
    leave the posterior modest — that robustness is the point.
    """
    ev = DEFAULT_EVIDENCE if evidence is None else evidence
    out = {}
    for scale in lr_scales:
        scaled = [(n, lr**scale, *rest) for (n, lr, *rest) in ev]
        out[f"LR^{scale}"] = {
            p: manipulation_posterior(p, scaled)["posterior_manip"] for p in priors
        }
    df = pd.DataFrame(out)
    df.index.name = "prior_manip"
    return df


def _stationary_bootstrap_indices(rng: np.random.Generator, n: int, mean_block: float) -> np.ndarray:
    """Row indices for one Politis-Romano (1994) stationary-bootstrap resample.

    Blocks have geometric length with mean ``mean_block`` (restart probability
    ``p = 1/mean_block``) and wrap around the end of the sample. Returns ``n``
    indices into the original rows.
    """
    p = 1.0 / float(mean_block)
    new_block = rng.random(n) < p
    new_block[0] = True
    start_pos = np.flatnonzero(new_block)
    starts = rng.integers(0, n, start_pos.size)
    block_id = np.cumsum(new_block) - 1
    offsets = np.arange(n) - start_pos[block_id]
    return (starts[block_id] + offsets) % n


def reality_check(
    panel: pd.DataFrame,
    n_boot: int = 2000,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    seed: int = 0,
    mean_block: float | None = None,
    method: str = "stationary",
) -> dict:
    """White (2000) Reality Check for the best overnight Sharpe in a universe.

    ``panel`` has one column per series (e.g. each market's overnight returns).
    H0: no series has predictive overnight return. We recentre every column to
    mean zero (impose the null), resample whole *rows* of the panel, and build
    the null distribution of the **maximum** annualised Sharpe across columns.
    The Reality-Check p-value is P(bootstrap max Sharpe >= observed max Sharpe).

    Resampling follows White's prescription: the **stationary bootstrap** of
    Politis & Romano (1994) — blocks of geometrically distributed length
    (mean ``mean_block``) that wrap circularly — so the serial dependence in
    returns survives into the null distribution. An i.i.d. row bootstrap
    understates the variance of the max-Sharpe null and yields p-values that
    are too small on autocorrelated data. The *same* block index sequence is
    applied to every column, preserving cross-sectional dependence. Recentring
    each column on its sample mean is the "centered" variant of the test,
    close in spirit to Hansen's (2005) SPA recentring.

    Parameters
    ----------
    mean_block : float, optional
        Mean block length. ``None`` (default) uses the simple n^(1/3) rate
        rule, ``max(1, n ** (1/3))``; the value used is reported in the output
        as ``mean_block_length``.
    method : {'stationary', 'iid'}
        ``'iid'`` reproduces the legacy independent row resampling (kept for
        comparison; it is anti-conservative on dependent data).
    """
    X = panel.dropna(how="all").to_numpy(dtype=float)
    n, m = X.shape
    ann = np.sqrt(periods_per_year)

    if method not in ("stationary", "iid"):
        raise ValueError("method must be 'stationary' or 'iid'")
    if mean_block is None:
        mean_block = max(1.0, n ** (1.0 / 3.0))
    mean_block = float(min(max(mean_block, 1.0), n))

    def col_sharpe(a):
        mu = np.nanmean(a, axis=0)
        sd = np.nanstd(a, axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(sd > 0, mu / sd * ann, 0.0)

    observed = col_sharpe(X)
    best_i = int(np.nanargmax(observed))
    observed_max = float(np.nanmax(observed))

    Xc = X - np.nanmean(X, axis=0)  # impose mean-zero null ("centered" variant)
    rng = np.random.default_rng(seed)
    boot_max = np.empty(n_boot)
    for b in range(n_boot):
        if method == "iid":
            idx = rng.integers(0, n, n)
        else:
            idx = _stationary_bootstrap_indices(rng, n, mean_block)
        boot_max[b] = np.nanmax(col_sharpe(Xc[idx]))
    p = float((boot_max >= observed_max).mean())
    return {
        "n_series": m,
        "best_series_index": best_i,
        "observed_max_sharpe": observed_max,
        "reality_check_pvalue": p,
        "bootstrap_method": method,
        "mean_block_length": float(mean_block) if method == "stationary" else 1.0,
        "n_boot": int(n_boot),
        "naive_pvalue_note": "single-series 5% crit ~1.96 t; RC accounts for searching m series",
    }
