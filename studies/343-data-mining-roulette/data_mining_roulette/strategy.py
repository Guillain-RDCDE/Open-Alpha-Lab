"""The roulette engine and its honest controls — Study 343 (Data-Mining-Roulette).

The p-hacking machine has three parts:

1. **A rule generator.** A "rule" is a tiny long/flat (optionally long/short) timing
   strategy: pick a feature, a comparison (``>`` or ``<``), and a threshold (a quantile of
   that feature); go long the next day when the condition holds, else hold cash (or go
   short). ``random_rules`` draws ``n_rules`` of these from a seed — the roulette wheel.

2. **A backtest with one honest execution lag and costs.** The signal is known at the
   close of t; it earns ``fwd_ret[t]`` (the t→t+1 return), so the lag is encoded once in
   the data layer and never shifted again here. Costs are one-way turnover × NAV at
   ``cost_bps`` (each flip in/out of the market is a trade). The return is *excess-of-cash*
   on the days the rule sits out, so a rule that beats the market is compared like-for-like
   (excess-vs-excess): we race each rule's excess return against the buy-and-hold excess
   return, not raw Sharpe against a part-time-cash leg.

3. **The multiple-testing reckoning.** Running ``n_rules`` backtests and quoting the best
   one's *own* *t*-stat is the original sin. The honest statistics are computed on the
   **maximum** over the family:
     * a **Bonferroni** threshold on the single-rule *p*-value;
     * **White's (2000) Reality Check** *p*-value via the **stationary bootstrap**
       (Politis-Romano) — the fraction of bootstrap worlds whose *best* rule beats the
       observed best, i.e. how often pure resampling luck reproduces our champion.
   Plus per-rule honesty: Newey-West (HAC) *t*-stats and circular block-bootstrap CIs.

The whole study's verdict is a *methodology* one: on a null tape the best of N random
rules looks "real" by its own t-stat and "mirage" by the Reality Check — confirming that
luck mimics skill, and that the only defence is correcting for the search.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Rule representation & generation
# ---------------------------------------------------------------------------
def random_rules(
    feature_names: list[str],
    n_rules: int = 1000,
    long_short: bool = False,
    seed: int = 343,
) -> list[dict]:
    """Draw ``n_rules`` random timing rules — the roulette wheel.

    Each rule is a dict ``{feature, op, q, long_short}``: when ``feature`` is on the chosen
    side (``op`` ``">"`` or ``"<"``) of its ``q``-quantile threshold, the rule is "on"
    (long the next day); otherwise it is flat (cash) — or short, if ``long_short``.

    Thresholds are drawn as quantiles in [0.2, 0.8] so a rule is on/off a meaningful
    fraction of the time (avoids degenerate always-on / always-off rules). Deterministic
    in ``seed``.
    """
    rng = np.random.default_rng(seed)
    rules = []
    for _ in range(n_rules):
        feat = feature_names[rng.integers(0, len(feature_names))]
        op = ">" if rng.random() < 0.5 else "<"
        q = float(rng.uniform(0.2, 0.8))
        rules.append({"feature": feat, "op": op, "q": q, "long_short": long_short})
    return rules


def rule_position(features: pd.DataFrame, rule: dict) -> pd.Series:
    """The +1/0 (or +1/-1) position series for one rule, known at the close of t.

    No shift here: ``features`` are already close-of-t values, and the realised return they
    earn (``fwd_ret``) is already the t→t+1 return — one lag, applied once, in the data
    layer. The position formed at the close of t multiplies ``fwd_ret[t]``.
    """
    x = features[rule["feature"]]
    thr = x.quantile(rule["q"])
    on = (x > thr) if rule["op"] == ">" else (x < thr)
    if rule["long_short"]:
        pos = pd.Series(np.where(on, 1.0, -1.0), index=features.index)
    else:
        pos = pd.Series(np.where(on, 1.0, 0.0), index=features.index)
    return pos


# ---------------------------------------------------------------------------
# Backtest with one execution lag (in the data) and one-way turnover costs
# ---------------------------------------------------------------------------
def backtest_rule(
    features: pd.DataFrame,
    fwd_ret: pd.Series,
    rule: dict,
    rf_daily: float = 0.0,
    cost_bps: float = 0.0,
) -> pd.Series:
    """Net **excess-of-cash** daily return series for one rule.

    ``pos_t * (fwd_ret_t - rf)`` is the rule's excess return (it earns the market excess
    return when long, 0 when in cash, the negated excess when short). Costs: one-way
    turnover × NAV at ``cost_bps`` charged whenever the position changes (each flip is a
    trade). Excess-vs-excess by construction, so a cash-heavy rule is judged fairly against
    a fully-invested benchmark.
    """
    pos = rule_position(features, rule)
    excess = fwd_ret - rf_daily
    gross = pos * excess
    turnover = pos.diff().abs().fillna(pos.abs())          # first day establishes the book
    cost = turnover * (cost_bps * 1e-4)
    net = (gross - cost).rename("ret")
    return net


def buy_and_hold_excess(fwd_ret: pd.Series, rf_daily: float = 0.0) -> pd.Series:
    """The always-long excess-of-cash return — the benchmark every rule races."""
    return (fwd_ret - rf_daily).rename("bh")


# ---------------------------------------------------------------------------
# Metrics & per-rule honest inference
# ---------------------------------------------------------------------------
def sharpe(daily: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sharpe of an (already-excess) daily return series."""
    r = daily.dropna()
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


def hac_tstat(daily: pd.Series) -> float:
    """Newey-West (HAC) *t*-stat that the mean (excess) daily return is non-zero."""
    x = daily.dropna().to_numpy(dtype=float)
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


def block_bootstrap_ci(
    daily: pd.Series, block: int = 10, n_boot: int = 2000, seed: int = 343, alpha: float = 0.05
) -> tuple[float, float]:
    """Circular block-bootstrap CI on the *mean* daily return (preserves autocorrelation)."""
    x = daily.dropna().to_numpy(dtype=float)
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


# ---------------------------------------------------------------------------
# The roulette: backtest the whole family and rank it
# ---------------------------------------------------------------------------
def spin_roulette(
    features: pd.DataFrame,
    fwd_ret: pd.Series,
    rules: list[dict],
    rf_daily: float = 0.0,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Backtest every rule; return a per-rule table sorted by Sharpe (best first).

    Columns: ``feature, op, q, sharpe, hac_t, mean_excess_ann, naive_p`` where ``naive_p``
    is the (two-sided, normal-approx) *p*-value from each rule's *own* HAC *t* — the number
    a p-hacker would quote without correcting for the search.
    """
    rows = []
    rets = []
    for r in rules:
        net = backtest_rule(features, fwd_ret, r, rf_daily=rf_daily, cost_bps=cost_bps)
        s = sharpe(net)
        t = hac_tstat(net)
        rows.append(
            {
                "feature": r["feature"],
                "op": r["op"],
                "q": round(r["q"], 3),
                "sharpe": s,
                "hac_t": t,
                "mean_excess_ann": float(net.mean() * TRADING_DAYS),
                "naive_p": _two_sided_p(t),
            }
        )
        rets.append(net.to_numpy(dtype=float))
    df = pd.DataFrame(rows)
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    # Stash the aligned return matrix for the Reality Check (columns = rules, rows = days).
    df.attrs["ret_matrix"] = np.column_stack(rets) if rets else np.empty((0, 0))
    df.attrs["index"] = features.index
    return df


def _two_sided_p(t: float) -> float:
    """Two-sided p-value from a (large-sample normal) t-statistic."""
    if not np.isfinite(t):
        return float("nan")
    try:
        from scipy.stats import norm

        return float(2.0 * (1.0 - norm.cdf(abs(t))))
    except Exception:
        # Normal-tail approximation without scipy.
        z = abs(t)
        return float(2.0 * 0.5 * np.exp(-0.717 * z - 0.416 * z * z))


# ---------------------------------------------------------------------------
# The multiple-testing reckoning — the heart of the study
# ---------------------------------------------------------------------------
def how_many_pass(table: pd.DataFrame, t_thresh: float = 2.0) -> dict:
    """How many of the N rules clear ``|HAC t| >= t_thresh`` — the p-hacker's harvest.

    Returns counts and the expected false-positive count under independence (a back-of-
    envelope ``N * alpha`` reference), plus the Bonferroni-corrected threshold the *family*
    should have used.
    """
    n = len(table)
    t = table["hac_t"].to_numpy(dtype=float)
    n_pass = int(np.sum(np.abs(t) >= t_thresh))
    # alpha for the two-sided |t|>=thresh single test.
    alpha = float(_two_sided_p(t_thresh))
    expected_fp = n * alpha
    # Bonferroni: to hold family-wise alpha=0.05, each rule needs p <= 0.05/N.
    bonf_alpha = 0.05 / max(n, 1)
    bonf_t = _alpha_to_t(bonf_alpha)
    n_pass_bonf = int(np.sum(np.abs(t) >= bonf_t))
    return {
        "n_rules": n,
        "t_thresh": t_thresh,
        "n_pass_naive": n_pass,
        "single_test_alpha": alpha,
        "expected_false_positives": float(expected_fp),
        "bonferroni_t": float(bonf_t),
        "n_pass_bonferroni": n_pass_bonf,
    }


def _alpha_to_t(alpha: float) -> float:
    """Two-sided critical |t| for a target alpha (normal approx)."""
    try:
        from scipy.stats import norm

        return float(norm.ppf(1.0 - alpha / 2.0))
    except Exception:
        # Rational approximation (Beasley-Springer/Moro-ish) good enough for a threshold.
        p = 1.0 - alpha / 2.0
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        plow = 0.02425
        if p < plow:
            q = np.sqrt(-2 * np.log(p))
            return -(((((a[0]*q+a[1])*q+a[2])*q+a[3])*q+a[4])*q+a[5])
        if p <= 1 - plow:
            q = p - 0.5
            r = q * q
            return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
        q = np.sqrt(-2 * np.log(1 - p))
        return (((((a[0]*q+a[1])*q+a[2])*q+a[3])*q+a[4])*q+a[5])


def reality_check(
    table: pd.DataFrame,
    n_boot: int = 1000,
    avg_block: int = 10,
    seed: int = 343,
) -> dict:
    """White's (2000) Reality Check p-value on the **best** rule, stationary bootstrap.

    The null is "no rule has positive expected excess return." The test statistic is the
    best rule's mean excess return. We resample the *aligned* return matrix with the
    stationary bootstrap (Politis-Romano, geometric block length ``avg_block``), recompute
    the best demeaned rule mean in each bootstrap world, and report the fraction that
    exceed the observed best — the data-snooping-adjusted p-value.

    A small p means even after accounting for the search, the champion is unlikely under
    luck. A large p means the champion is exactly what a search of this size produces by
    chance — the verdict the study is built to surface.
    """
    M = table.attrs.get("ret_matrix")
    if M is None or M.size == 0:
        return {"reality_check_p": float("nan"), "n_boot": n_boot}
    T, N = M.shape
    col_means = M.mean(axis=0)                      # per-rule mean excess return
    observed_best = float(col_means.max())
    demeaned = M - col_means[None, :]              # impose the null (zero mean) per rule

    rng = np.random.default_rng(seed)
    p = 1.0 / avg_block                             # stationary-bootstrap restart prob
    count_exceed = 0
    for _ in range(n_boot):
        idx = _stationary_indices(T, p, rng)
        boot = demeaned[idx, :]
        boot_best = float(boot.mean(axis=0).max())
        if boot_best >= observed_best:
            count_exceed += 1
    rc_p = (count_exceed + 1) / (n_boot + 1)        # +1 smoothing (Davison-Hinkley)
    return {
        "reality_check_p": float(rc_p),
        "observed_best_mean_ann": float(observed_best * TRADING_DAYS),
        "n_boot": n_boot,
        "avg_block": avg_block,
        "n_rules": N,
    }


def _stationary_indices(T: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary-bootstrap index path of length T (geometric blocks)."""
    idx = np.empty(T, dtype=np.int64)
    cur = rng.integers(0, T)
    for i in range(T):
        idx[i] = cur
        if rng.random() < p:
            cur = rng.integers(0, T)               # restart at a random point
        else:
            cur = (cur + 1) % T                    # continue the block
    return idx


def run_experiment(
    features: pd.DataFrame,
    fwd_ret: pd.Series,
    n_rules: int = 1000,
    cost_bps: float = 0.0,
    rf_daily: float = 0.0,
    long_short: bool = False,
    t_thresh: float = 2.0,
    rc_boot: int = 1000,
    seed: int = 343,
) -> dict:
    """The full roulette spin: generate N rules, backtest, count winners, run the Reality Check.

    Returns a bundle: ``table`` (per-rule, best first), ``best`` (the champion row),
    ``pass_counts`` (how_many_pass), ``rc`` (reality_check), and the buy-and-hold benchmark
    Sharpe — everything the notebooks and results.md quote.
    """
    rules = random_rules(list(features.columns), n_rules=n_rules,
                         long_short=long_short, seed=seed)
    table = spin_roulette(features, fwd_ret, rules, rf_daily=rf_daily, cost_bps=cost_bps)
    counts = how_many_pass(table, t_thresh=t_thresh)
    rc = reality_check(table, n_boot=rc_boot, seed=seed)
    bh = buy_and_hold_excess(fwd_ret, rf_daily=rf_daily)
    best = table.iloc[0].to_dict() if len(table) else {}
    return {
        "table": table,
        "best": best,
        "pass_counts": counts,
        "rc": rc,
        "bh_sharpe": sharpe(bh),
        "n_rules": n_rules,
        "cost_bps": cost_bps,
    }
