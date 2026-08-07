"""The engine and its honest controls — Study 839 (The t > 3 Threshold).

The claim, at full strength (Harvey, Liu & Zhu 2016): the published cross-sectional
"factor zoo" is the survivor set of an enormous, largely-unreported multiple-testing
exercise. With hundreds of candidate factors data-mined, the conventional single-test
**t > 2** hurdle is far too lax:

* Under the pure null, a two-sided ``|t| > 2`` fires with probability ``2 * Phi(-2) ~=
  4.55%`` — so a zoo of a few hundred noise factors *manufactures* a paper's worth of
  "significant" discoveries. ``|t| > 3`` fires with probability only ``~0.27%`` (~17x
  rarer).
* Controlling the **family-wise error rate** (Bonferroni / Holm) or the **false-discovery
  rate** (Benjamini-Hochberg / Benjamini-Yekutieli) pushes the required ``|t|`` up with
  the number of tests — into the ``3.4 - 3.8`` range for a few-hundred-factor zoo. HLZ
  round this to a practical recommendation: **a new factor should clear ``t ~ 3.0``**,
  not 2.0.

This module supplies:

* **Inference primitives** (house style): one-sample / Welch / Newey-West (HAC) / Wilson.
* **The factor zoo mechanics** — vectorised per-factor *t*-stats and two-sided p-values.
* **The threshold arithmetic** — counts clearing t>2 vs t>3, and the *expected* number of
  false discoveries under the null.
* **The corrections, expressed as an implied ``|t|`` cutoff** — Bonferroni, Holm, BH, BHY.
* **The realized-FDR proof** against known planted truth (the machinery check), and the
  **publication haircut** a claimed *t* suffers once the search size is disclosed.
* **Seed-robust controls** (>= 20 seeds, house rule): the null fraction clearing each
  bar, and the FDR collapse from t>2 to t>3 on a planted mixture.

Distinct from [346 multiple-testing](../../346-multiple-testing/) (the generic
family-wise-error problem), this study is framed specifically around the **3.0 hurdle**
and the **publication haircut** peculiar to the return-predictor factor zoo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- #
# Inference primitives (house style — shared across the desk)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    """One-sample t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch (unequal-variance) t of mean(a) - mean(b)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 6) -> float:
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
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# The factor zoo mechanics — per-factor t-stats and p-values
# --------------------------------------------------------------------------- #
def factor_tstats(returns: np.ndarray) -> np.ndarray:
    """Vectorised single-test *t*-stat of each factor's mean return vs 0.

    ``returns`` is ``(T, N)`` (rows = periods, columns = candidate factors). Returns a
    length-``N`` array of ``mean / (sd / sqrt(T))`` per column (``ddof=1``). Columns with
    zero variance return 0.
    """
    R = np.asarray(returns, dtype=float)
    T = R.shape[0]
    mu = R.mean(axis=0)
    sd = R.std(axis=0, ddof=1)
    se = sd / np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, mu / se, 0.0)
    return t


def tstat_pvalues(tstats: np.ndarray, df: int | None = None) -> np.ndarray:
    """Two-sided p-values for an array of *t*-stats.

    Uses the (large-``T``) normal approximation by default; pass ``df`` to use the exact
    Student-t survival function. For ``T = 240`` the two agree to ~4 decimals.
    """
    t = np.abs(np.asarray(tstats, dtype=float))
    if df is None:
        return 2.0 * stats.norm.sf(t)
    return 2.0 * stats.t.sf(t, df)


def prob_exceed(threshold: float) -> float:
    """Two-sided probability a single pure-noise *t* exceeds ``threshold`` (normal)."""
    return float(2.0 * stats.norm.sf(abs(threshold)))


def expected_false_positives(n_factors: int, threshold: float) -> float:
    """Expected number of noise factors clearing ``|t| > threshold`` in a zoo of size N."""
    return float(n_factors) * prob_exceed(threshold)


# --------------------------------------------------------------------------- #
# The naive threshold counts — t>2 vs t>3
# --------------------------------------------------------------------------- #
def threshold_summary(
    tstats: np.ndarray, thresholds=(2.0, 3.0), n_factors: int | None = None
) -> pd.DataFrame:
    """For each ``|t|`` threshold: observed count, observed fraction, and the *expected*
    number/fraction of pure-noise factors that would clear it.

    The gap between the observed fraction and the null-expected fraction is the tell: on a
    pure-noise zoo they coincide (nothing real), and roughly 4.55% of factors clear t>2
    versus 0.27% clearing t>3.
    """
    t = np.abs(np.asarray(tstats, dtype=float))
    N = int(n_factors if n_factors is not None else t.size)
    rows = []
    for thr in thresholds:
        obs = int(np.sum(t > thr))
        rows.append({
            "threshold": float(thr),
            "n_cleared": obs,
            "frac_cleared": obs / N if N else float("nan"),
            "exp_false_frac": prob_exceed(thr),
            "exp_false_count": expected_false_positives(N, thr),
        })
    return pd.DataFrame(rows).set_index("threshold")


# --------------------------------------------------------------------------- #
# The corrections — Bonferroni / Holm (FWER) and BH / BHY (FDR)
# --------------------------------------------------------------------------- #
def harmonic(n: int) -> float:
    """Harmonic number c(N) = sum_{i=1..N} 1/i — the Benjamini-Yekutieli dependency factor."""
    return float(np.sum(1.0 / np.arange(1, int(n) + 1)))


def bonferroni_t(n_factors: int, alpha: float = 0.05) -> float:
    """The Bonferroni family-wise ``|t|`` cutoff: two-sided per-test level alpha/N."""
    return float(stats.norm.ppf(1.0 - alpha / (2.0 * n_factors)))


def _cutoff_from_nrej(tstats: np.ndarray, n_rej: int) -> float:
    """Implied ``|t|`` cutoff of a step procedure that rejects ``n_rej`` factors: the
    smallest ``|t|`` among the rejected (i.e. the ``n_rej``-th largest ``|t|``)."""
    if n_rej <= 0:
        return float("nan")
    at = np.sort(np.abs(np.asarray(tstats, dtype=float)))[::-1]
    return float(at[min(n_rej, at.size) - 1])


def holm_reject(tstats: np.ndarray, alpha: float = 0.05, df: int | None = None) -> dict:
    """Holm (1979) step-down family-wise procedure.

    Sort p-values ascending; reject ``p_(1) <= alpha/N``, ``p_(2) <= alpha/(N-1)``, ...
    stopping at the first failure. Returns the number rejected and the implied ``|t|``
    cutoff (the smallest ``|t|`` retained as significant).
    """
    p = np.sort(tstat_pvalues(tstats, df=df))
    N = p.size
    thresh = alpha / (N - np.arange(N))            # alpha/N, alpha/(N-1), ...
    ok = p <= thresh
    if bool((~ok).any()):
        n_rej = int(np.argmax(~ok))                # first failure index = leading Trues
    else:
        n_rej = N
    return {"method": "Holm", "n_rej": n_rej,
            "t_cutoff": _cutoff_from_nrej(tstats, n_rej)}


def benjamini_hochberg(
    tstats: np.ndarray, alpha: float = 0.05, dependency: bool = False,
    df: int | None = None,
) -> dict:
    """Benjamini-Hochberg (FDR) step-up, with the Benjamini-Yekutieli dependency variant.

    Sort p ascending; find the largest ``k`` with ``p_(k) <= (k / (N * c)) * alpha`` and
    reject the smallest ``k`` p-values. ``c = 1`` is plain BH (independence / PRDS);
    ``c = sum_{i<=N} 1/i`` is BHY, valid under arbitrary dependence (the one HLZ lean on).
    Returns the number rejected and the implied ``|t|`` cutoff.
    """
    p = np.sort(tstat_pvalues(tstats, df=df))
    N = p.size
    c = harmonic(N) if dependency else 1.0
    ranks = np.arange(1, N + 1)
    thresh = (ranks / (N * c)) * alpha
    below = np.where(p <= thresh)[0]
    n_rej = int(below.max() + 1) if below.size else 0
    return {"method": "BHY" if dependency else "BH", "n_rej": n_rej,
            "t_cutoff": _cutoff_from_nrej(tstats, n_rej)}


def multiple_testing_table(
    tstats: np.ndarray, alpha: float = 0.05, df: int | None = None
) -> pd.DataFrame:
    """The centrepiece: the implied ``|t|`` cutoff and discovery count for every rule.

    Rows: naive ``t>2``, naive ``t>3``, Bonferroni, Holm, BH, BHY. Columns: the implied
    ``|t|`` hurdle and the number of the zoo's factors that clear it. On a pure-noise zoo
    the corrections collapse the discovery count toward zero while the naive t>2 bar keeps
    a paper's worth of false positives.
    """
    t = np.abs(np.asarray(tstats, dtype=float))
    N = t.size
    rows = []

    def add(name, cutoff, n_disc):
        rows.append({"method": name, "t_cutoff": float(cutoff), "n_discoveries": int(n_disc)})

    add("naive t>2", 2.0, np.sum(t > 2.0))
    add("naive t>3", 3.0, np.sum(t > 3.0))
    bcut = bonferroni_t(N, alpha)
    add("Bonferroni", bcut, np.sum(t > bcut))
    hm = holm_reject(tstats, alpha, df=df)
    add("Holm", hm["t_cutoff"], hm["n_rej"])
    bh = benjamini_hochberg(tstats, alpha, dependency=False, df=df)
    add("BH", bh["t_cutoff"], bh["n_rej"])
    bhy = benjamini_hochberg(tstats, alpha, dependency=True, df=df)
    add("BHY", bhy["t_cutoff"], bhy["n_rej"])
    return pd.DataFrame(rows).set_index("method")


# --------------------------------------------------------------------------- #
# The realized-FDR proof — against known planted truth (the machinery check)
# --------------------------------------------------------------------------- #
def detection(tstats: np.ndarray, is_true: np.ndarray, threshold: float) -> dict:
    """Realized discovery accounting at a ``|t|`` threshold, given the planted truth.

    Returns discoveries, true/false positives, the realized **false-discovery rate**
    (FP / discoveries) and **power** (TP / true factors). On a pure-noise zoo every
    discovery is false (FDR = 1 whenever anything fires).
    """
    t = np.abs(np.asarray(tstats, dtype=float))
    truth = np.asarray(is_true, dtype=bool)
    disc = t > threshold
    n_disc = int(disc.sum())
    tp = int((disc & truth).sum())
    fp = int((disc & ~truth).sum())
    n_true = int(truth.sum())
    return {
        "threshold": float(threshold),
        "n_disc": n_disc,
        "tp": tp,
        "fp": fp,
        "fdr": fp / n_disc if n_disc > 0 else 0.0,
        "power": tp / n_true if n_true > 0 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# The publication haircut — what a claimed t survives once the search is disclosed
# --------------------------------------------------------------------------- #
def publication_haircut(t: float, n_tests: int, method: str = "bonferroni") -> dict:
    """The multiple-testing haircut a claimed single-test ``|t|`` suffers.

    A factor reported at ``|t|`` has naive two-sided p ``p = 2*Phi(-|t|)``. Adjusting for
    ``n_tests`` searched hypotheses (Bonferroni: ``p_adj = min(1, p * n_tests)``) implies
    an *effective* ``|t|_adj = Phi^{-1}(1 - p_adj/2)``. The **haircut** is
    ``(|t| - |t|_adj) / |t|`` — the fraction of the reported significance that was an
    artefact of the search. A t=2.0 factor from a 300-test search is haircut to nothing.
    """
    t = abs(float(t))
    p = float(2.0 * stats.norm.sf(t))
    if method.lower().startswith("bonf"):
        p_adj = min(1.0, p * n_tests)
    else:
        raise ValueError(f"unknown method {method!r}")
    t_adj = float(stats.norm.ppf(1.0 - p_adj / 2.0)) if p_adj < 1.0 else 0.0
    t_adj = max(0.0, t_adj)
    return {
        "t_reported": t,
        "p_naive": p,
        "n_tests": int(n_tests),
        "p_adjusted": p_adj,
        "t_adjusted": t_adj,
        "haircut": (t - t_adj) / t if t > 0 else float("nan"),
        "survives_005": p_adj < 0.05,
    }


# --------------------------------------------------------------------------- #
# Headline packagers
# --------------------------------------------------------------------------- #
def zoo_stats(returns: np.ndarray, is_true: np.ndarray | None = None) -> dict:
    """Headline numbers for one zoo: threshold counts + the two detection accountings."""
    t = factor_tstats(returns)
    N = t.size
    out = {
        "n_factors": N,
        "n_gt2": int(np.sum(np.abs(t) > 2.0)),
        "n_gt3": int(np.sum(np.abs(t) > 3.0)),
        "frac_gt2": float(np.mean(np.abs(t) > 2.0)),
        "frac_gt3": float(np.mean(np.abs(t) > 3.0)),
        "max_t": float(np.max(np.abs(t))),
    }
    if is_true is not None:
        out["det2"] = detection(t, is_true, 2.0)
        out["det3"] = detection(t, is_true, 3.0)
    return out


# --------------------------------------------------------------------------- #
# Seed-robust controls — the >= 20-seed house rule
# --------------------------------------------------------------------------- #
def seed_robust_null(
    data_mod, n_factors: int = 1000, n_periods: int = 240,
    n_seeds: int = 20, base_seed: int = 839,
) -> dict:
    """Average the pure-null fractions clearing t>2 and t>3 over ``n_seeds`` zoos.

    The point HLZ make quantitatively: a pure-noise zoo clears t>2 ~4.55% of the time and
    t>3 ~0.27% of the time — a ~17x gap that is exactly the false-discovery inflation of
    the lax bar. Averaged over seeds so no lucky RNG can manufacture the result.
    """
    f2, f3, n2, n3, maxt = [], [], [], [], []
    for s in range(base_seed, base_seed + n_seeds):
        R, _, _ = data_mod.synthetic_zoo(
            n_factors=n_factors, n_periods=n_periods, n_true=0, seed=s
        )
        t = np.abs(factor_tstats(R))
        f2.append(np.mean(t > 2.0)); f3.append(np.mean(t > 3.0))
        n2.append(int(np.sum(t > 2.0))); n3.append(int(np.sum(t > 3.0)))
        maxt.append(float(t.max()))
    return {
        "n_seeds": n_seeds, "n_factors": n_factors,
        "mean_frac_gt2": float(np.mean(f2)), "mean_frac_gt3": float(np.mean(f3)),
        "mean_n_gt2": float(np.mean(n2)), "mean_n_gt3": float(np.mean(n3)),
        "ratio_gt2_over_gt3": float(np.mean(f2) / np.mean(f3)) if np.mean(f3) > 0 else float("nan"),
        "mean_max_t": float(np.mean(maxt)),
        "theory_frac_gt2": prob_exceed(2.0), "theory_frac_gt3": prob_exceed(3.0),
    }


def seed_robust_mixture(
    data_mod, n_factors: int = 1000, n_true: int = 50, expected_t: float = 4.0,
    n_periods: int = 240, n_seeds: int = 20, base_seed: int = 839,
) -> dict:
    """Average the realized FDR / power at t>2 and t>3 over ``n_seeds`` planted mixtures.

    With a small true subset buried in a big noisy zoo, the naive t>2 bar lets nearly as
    many *false* factors through as true ones (FDR near 50%); raising the bar to t>3
    collapses the FDR ~8x for a modest loss of power — the quantitative case for the ~3.0
    hurdle. Also averages the FDR-controlled BHY discovery count and its realized FDR.
    """
    fdr2, fdr3, pw2, pw3 = [], [], [], []
    bhy_n, bhy_fdr, bhy_cut = [], [], []
    for s in range(base_seed, base_seed + n_seeds):
        R, is_true, _ = data_mod.synthetic_zoo(
            n_factors=n_factors, n_periods=n_periods, n_true=n_true,
            expected_t=expected_t, seed=s,
        )
        t = factor_tstats(R)
        d2 = detection(t, is_true, 2.0); d3 = detection(t, is_true, 3.0)
        fdr2.append(d2["fdr"]); fdr3.append(d3["fdr"])
        pw2.append(d2["power"]); pw3.append(d3["power"])
        bhy = benjamini_hochberg(t, alpha=0.05, dependency=True)
        cut = bhy["t_cutoff"]
        bhy_n.append(bhy["n_rej"])
        bhy_cut.append(cut if np.isfinite(cut) else np.nan)
        if bhy["n_rej"] > 0:
            db = detection(t, is_true, cut - 1e-9)
            bhy_fdr.append(db["fdr"])
        else:
            bhy_fdr.append(0.0)
    return {
        "n_seeds": n_seeds, "n_factors": n_factors, "n_true": n_true,
        "expected_t": expected_t,
        "mean_fdr_t2": float(np.mean(fdr2)), "mean_fdr_t3": float(np.mean(fdr3)),
        "mean_power_t2": float(np.mean(pw2)), "mean_power_t3": float(np.mean(pw3)),
        "fdr_collapse": float(np.mean(fdr2) / np.mean(fdr3)) if np.mean(fdr3) > 0 else float("nan"),
        "mean_bhy_n": float(np.mean(bhy_n)),
        "mean_bhy_fdr": float(np.nanmean(bhy_fdr)),
        "mean_bhy_cutoff": float(np.nanmean(bhy_cut)),
    }
