"""The multiple-testing engine and its honest controls — Study 346 (Multiple-Testing).

The pipeline has three parts:

1. **Per-effect honest inference.** For every effect in the battery, compute an
   autocorrelation-robust Newey-West (HAC) *t*-stat on its daily excess return, its
   annualised mean, its Sharpe, and a circular block-bootstrap CI on the mean. These are
   the *single-test* statistics — the numbers a per-study write-up would report.

2. **The corrections (the heart of the study).** Turn the M per-effect *p*-values into
   *decisions* four ways:
     * **naive** — reject when |HAC *t*| >= ``t_thresh`` (the p-hacker's count);
     * **Bonferroni** — reject when *p* <= alpha / M (controls the family-wise error rate,
       FWER: the chance of *any* false positive);
     * **Holm** — the step-down FWER procedure, uniformly more powerful than Bonferroni at
       the same FWER guarantee;
     * **Benjamini-Hochberg (BH)** — the step-up procedure that controls the *false
       discovery rate* (FDR: the expected *fraction* of rejections that are false), the
       right target when you can tolerate a few false leads in exchange for power.

3. **Scoring against the truth (synthetic only).** With the planted truth mask we can
   measure each procedure's realised power (true positives found), false positives, and
   realised FDR — the trade-off the corrections are *named* for. On the real tape there is
   no truth mask, so the verdict is the methodology one: how the discovery count collapses
   from naive to corrected, and which procedure to report.

Costs: each effect's return is already excess-of-cash; a cost sweep charges one-way
turnover x NAV at ``cost_bps`` per entry/exit of the effect window. One execution lag:
calendar membership is known in advance, so the lag is the forward-return alignment in the
data layer, applied once — no extra shift here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Per-effect honest inference
# ---------------------------------------------------------------------------
def hac_tstat(daily: np.ndarray) -> float:
    """Newey-West (HAC) *t*-stat that the mean of a daily return series is non-zero."""
    x = np.asarray(daily, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        wk = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * wk * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def two_sided_p(t: float) -> float:
    """Two-sided p-value from a (large-sample normal) t-statistic."""
    if not np.isfinite(t):
        return float("nan")
    try:
        from scipy.stats import norm

        return float(2.0 * (1.0 - norm.cdf(abs(t))))
    except Exception:
        z = abs(t)
        return float(min(1.0, 2.0 * 0.5 * np.exp(-0.717 * z - 0.416 * z * z)))


def alpha_to_t(alpha: float) -> float:
    """Two-sided critical |t| for a target alpha (normal approx)."""
    try:
        from scipy.stats import norm

        return float(norm.ppf(1.0 - alpha / 2.0))
    except Exception:
        # Acklam rational approximation of the inverse normal CDF.
        p = 1.0 - alpha / 2.0
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        plow, phigh = 0.02425, 1 - 0.02425
        if p < plow:
            q = np.sqrt(-2 * np.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if p <= phigh:
            q = p - 0.5
            r = q * q
            return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def block_bootstrap_ci(
    daily: np.ndarray, block: int = 10, n_boot: int = 2000, seed: int = 346, alpha: float = 0.05
) -> tuple[float, float]:
    """Circular block-bootstrap CI on the *mean* daily return (preserves autocorrelation)."""
    x = np.asarray(daily, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < block + 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = x[idx][:n].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def screen_battery(returns: pd.DataFrame, cost_bps: float = 0.0) -> pd.DataFrame:
    """Per-effect honest stats for every column. Returns a table sorted by |HAC t|.

    Columns: ``effect, sharpe, hac_t, p, mean_excess_ann``. ``p`` is the two-sided p-value
    from the effect's own HAC *t* (the single-test number). A cost of ``cost_bps`` charges
    one-way turnover x NAV at each entry/exit of the effect window before the stats.
    """
    rows = []
    for col in returns.columns:
        net = _net_after_costs(returns[col], cost_bps)
        x = net.to_numpy(dtype=float)
        t = hac_tstat(x)
        sd = np.nanstd(x, ddof=1)
        sharpe = float(np.nanmean(x) / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")
        rows.append({
            "effect": col,
            "sharpe": sharpe,
            "hac_t": t,
            "p": two_sided_p(t),
            "mean_excess_ann": float(np.nanmean(x) * TRADING_DAYS),
        })
    df = pd.DataFrame(rows).set_index("effect")
    return df.reindex(df["hac_t"].abs().sort_values(ascending=False).index)


def _net_after_costs(series: pd.Series, cost_bps: float) -> pd.Series:
    """Charge one-way turnover x NAV at entry/exit of the effect window (cost_bps/leg)."""
    if cost_bps <= 0:
        return series
    on = (series != 0).astype(float)
    flips = on.diff().abs().fillna(on)            # first day establishes the book
    cost = flips * (cost_bps * 1e-4)
    return series - cost


# ---------------------------------------------------------------------------
# The corrections — the heart of the study
# ---------------------------------------------------------------------------
def naive_reject(pvals: np.ndarray, t_thresh: float = 2.0) -> np.ndarray:
    """Reject (True) when the single-test |t| clears ``t_thresh`` — the p-hacker's count."""
    alpha = two_sided_p(t_thresh)
    return np.asarray(pvals, dtype=float) <= alpha


def bonferroni_reject(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Bonferroni: reject when p <= alpha / M. Controls FWER = P(any false positive)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    return p <= (alpha / max(m, 1))


def holm_reject(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Holm step-down: sort ascending, reject p_(k) while p_(k) <= alpha/(M-k), then stop.

    Uniformly more powerful than Bonferroni at the same FWER guarantee.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    rej = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        thresh = alpha / (m - rank)
        if p[idx] <= thresh:
            rej[idx] = True
        else:
            break                                  # step-down stops at the first failure
    return rej


def bh_reject(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up: controls the FDR = E[false rejections / rejections].

    Sort ascending; find the largest k with p_(k) <= (k/M) * alpha; reject all p_(1..k).
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    sorted_p = p[order]
    thresh = (np.arange(1, m + 1) / m) * alpha
    below = sorted_p <= thresh
    rej = np.zeros(m, dtype=bool)
    if below.any():
        kmax = np.max(np.where(below)[0])          # largest passing rank (0-based)
        rej[order[: kmax + 1]] = True
    return rej


def apply_corrections(
    table: pd.DataFrame, alpha: float = 0.05, t_thresh: float = 2.0
) -> pd.DataFrame:
    """Add naive / Bonferroni / Holm / BH reject columns to a ``screen_battery`` table."""
    p = table["p"].to_numpy(dtype=float)
    out = table.copy()
    out["reject_naive"] = naive_reject(p, t_thresh=t_thresh)
    out["reject_bonferroni"] = bonferroni_reject(p, alpha=alpha)
    out["reject_holm"] = holm_reject(p, alpha=alpha)
    out["reject_bh"] = bh_reject(p, alpha=alpha)
    return out


# ---------------------------------------------------------------------------
# Scoring against the truth (synthetic positive control)
# ---------------------------------------------------------------------------
def score_against_truth(table: pd.DataFrame, truth: pd.Series) -> dict:
    """Power, false positives and realised FDR for each procedure, given the truth mask.

    Returns a dict keyed by procedure, each a sub-dict with ``n_reject``, ``true_pos``,
    ``false_pos``, ``power`` (fraction of real effects found), ``fdr`` (false rejections /
    total rejections), and a binary ``any_false_positive`` (the FWER event).
    """
    real = truth.reindex(table.index).fillna(False).to_numpy(dtype=bool)
    n_real = int(real.sum())
    out = {}
    for proc in ("naive", "bonferroni", "holm", "bh"):
        rej = table[f"reject_{proc}"].to_numpy(dtype=bool)
        tp = int(np.sum(rej & real))
        fp = int(np.sum(rej & ~real))
        n_rej = int(rej.sum())
        out[proc] = {
            "n_reject": n_rej,
            "true_pos": tp,
            "false_pos": fp,
            "power": float(tp / n_real) if n_real > 0 else float("nan"),
            "fdr": float(fp / n_rej) if n_rej > 0 else 0.0,
            "any_false_positive": bool(fp > 0),
        }
    out["_n_real"] = n_real
    out["_m_effects"] = int(len(table))
    return out


def discovery_counts(table: pd.DataFrame) -> dict:
    """How many effects each procedure declares 'real' (the headline collapse)."""
    return {
        "m_effects": int(len(table)),
        "naive": int(table["reject_naive"].sum()),
        "bonferroni": int(table["reject_bonferroni"].sum()),
        "holm": int(table["reject_holm"].sum()),
        "bh": int(table["reject_bh"].sum()),
    }


def run_experiment(
    returns: pd.DataFrame,
    truth: pd.Series | None = None,
    alpha: float = 0.05,
    t_thresh: float = 2.0,
    cost_bps: float = 0.0,
) -> dict:
    """The full battery screen + corrections (+ scoring if a truth mask is supplied).

    Returns a bundle: ``table`` (per-effect, with reject columns), ``counts``
    (discovery_counts), and — when ``truth`` is given — ``score`` (score_against_truth).
    """
    raw = screen_battery(returns, cost_bps=cost_bps)
    table = apply_corrections(raw, alpha=alpha, t_thresh=t_thresh)
    bundle = {"table": table, "counts": discovery_counts(table)}
    if truth is not None and truth.any():
        bundle["score"] = score_against_truth(table, truth)
    return bundle


# ---------------------------------------------------------------------------
# Monte-Carlo: the FWER under the global null (a machinery check)
# ---------------------------------------------------------------------------
def fwer_monte_carlo(
    m_effects: int = 100,
    n_days: int = 2520,
    n_reps: int = 200,
    alpha: float = 0.05,
    t_thresh: float = 2.0,
    seed: int = 346,
) -> dict:
    """Estimate each procedure's family-wise error rate under the *global null*.

    Repeatedly draws an all-null battery (n_true=0) and records, per procedure, whether it
    makes *any* false rejection (the FWER event) and the average number of false rejections.
    Under the null a correct FWER procedure (Bonferroni, Holm) should fire <= alpha of the
    time; the naive count should fire almost always (its whole failing).
    """
    from . import data as _data

    procs = ("naive", "bonferroni", "holm", "bh")
    any_fp = {p: 0 for p in procs}
    n_fp = {p: 0 for p in procs}
    for r in range(n_reps):
        rets, truth, _ = _data.synthetic_battery(
            n_days=n_days, m_effects=m_effects, n_true=0, seed=seed + r
        )
        res = run_experiment(rets, alpha=alpha, t_thresh=t_thresh)
        tab = res["table"]
        for p in procs:
            fp = int(tab[f"reject_{p}"].sum())     # all rejections are false under the null
            n_fp[p] += fp
            any_fp[p] += int(fp > 0)
    return {
        "m_effects": m_effects, "n_reps": n_reps, "alpha": alpha,
        "fwer": {p: any_fp[p] / n_reps for p in procs},
        "avg_false_positives": {p: n_fp[p] / n_reps for p in procs},
    }
