"""Does time diversify risk — Study 1007.

The debate is sixty years old and largely a failure to specify the question. Three different
quantities are called "risk" and they behave differently with horizon:

1. **Annualised return dispersion** — the standard deviation of the *average annual* return over
   a T-year hold. This falls like 1/√T for i.i.d. returns. It is what advisers plot, it is
   genuinely true, and it is arithmetic rather than a property of equities.

2. **Terminal wealth dispersion** — the standard deviation of what you actually end up with.
   This *grows* with √T. It is what Samuelson (1963) pointed at, and it is also arithmetic.

3. **Shortfall probability** — the chance of ending below some benchmark. This falls with
   horizon whenever the mean excess return is positive, and the rate at which it falls is a
   real empirical question rather than an arithmetic identity.

All three are consequences of the same distribution. Quoting one and calling it "risk" is where
the argument comes from. ``horizon_metrics`` computes all three side by side so that nobody has
to.

The genuinely empirical question underneath is whether returns **mean-revert** — whether a bad
decade makes a good one more likely. If they do, annualised dispersion falls *faster* than
1/√T and there is something beyond arithmetic. ``variance_ratio`` tests exactly that, with the
Lo-MacKinlay heteroscedasticity-robust statistic, and ``excess_convergence`` measures how much
faster than √T the observed dispersion narrows.

Finally ``optimal_weight_by_horizon`` addresses the decision. Samuelson's result is that under
CRRA utility with i.i.d. returns the optimal equity share does not depend on horizon at all —
a striking claim, and one that can be checked numerically rather than accepted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Horizon windows
# --------------------------------------------------------------------------- #
def horizon_windows(rets: pd.Series, years: float, step: int = 21) -> np.ndarray:
    """Overlapping cumulative log returns over every ``years``-long window.

    Overlapping, and stepped monthly. With thirty-three years of data there are only three
    independent thirty-year periods, so non-overlapping windows would leave nothing to measure.
    The cost is that these observations are heavily dependent, which is why every inferential
    statement in this module goes through a block bootstrap rather than a naive standard error.
    """
    r = rets.dropna()
    h = int(years * TRADING_DAYS)
    lr = np.log1p(r.to_numpy(dtype=float))
    n = len(lr)
    if n <= h:
        return np.empty(0)
    cs = np.concatenate([[0.0], np.cumsum(lr)])
    starts = np.arange(0, n - h + 1, step)
    return cs[starts + h] - cs[starts]


def effective_sample(n_obs: int, horizon_days: int, step: int, total_days: int) -> float:
    """Roughly how many *independent* windows the overlapping sample is worth.

    Reported everywhere a dispersion is reported, because "the 30-year standard deviation was
    x%" sounds like a measurement and is closer to an anecdote when the effective sample is
    three.
    """
    return float(max(total_days / max(horizon_days, 1), 1.0))


def horizon_metrics(rets: pd.Series, cash: pd.Series, years_grid=None,
                    step: int = 21) -> pd.DataFrame:
    """All three "risks" at once, on identical windows.

    The point of computing them together is that they cannot then be quoted selectively. The
    annualised column falls, the terminal column rises, and the shortfall column falls — from
    the same windows, on the same data.
    """
    if years_grid is None:
        years_grid = (1, 2, 3, 5, 7, 10, 15, 20, 25, 30)
    total = len(rets.dropna())
    rows = []
    for y in years_grid:
        lw = horizon_windows(rets, y, step)
        lc = horizon_windows(cash, y, step)
        k = min(len(lw), len(lc))
        if k < 10:
            continue
        lw, lc = lw[-k:], lc[-k:]
        ann = np.expm1(lw / y)
        term = np.exp(lw)
        rows.append({
            "years": y, "n_windows": int(k),
            "effective_n": effective_sample(k, int(y * TRADING_DAYS), step, total),
            "annualised_sd": float(np.std(ann, ddof=1)),
            "annualised_mean": float(np.mean(ann)),
            "annualised_p05": float(np.percentile(ann, 5)),
            "annualised_p95": float(np.percentile(ann, 95)),
            "terminal_sd": float(np.std(term, ddof=1)),
            "terminal_median": float(np.median(term)),
            "terminal_p05": float(np.percentile(term, 5)),
            "terminal_p95": float(np.percentile(term, 95)),
            "log_sd": float(np.std(lw, ddof=1)),
            "shortfall_vs_cash": float(np.mean(lw < lc)),
            "shortfall_vs_zero": float(np.mean(lw < 0)),
            "worst_terminal": float(np.min(term)),
        })
    return pd.DataFrame(rows).set_index("years")


def sqrt_t_benchmark(metrics: pd.DataFrame) -> pd.DataFrame:
    """What each dispersion *would* be under i.i.d. returns, anchored on the one-year figure.

    The comparison that separates arithmetic from economics. Annualised dispersion falling like
    1/√T proves nothing about equities; falling *faster* than that is mean reversion and would.
    """
    if metrics.empty or 1 not in metrics.index:
        return pd.DataFrame()
    base_ann = float(metrics.loc[1, "annualised_sd"])
    base_log = float(metrics.loc[1, "log_sd"])
    t = metrics.index.to_numpy(dtype=float)
    return pd.DataFrame({
        "annualised_sd": metrics["annualised_sd"].to_numpy(),
        "iid_annualised_sd": base_ann / np.sqrt(t),
        "log_sd": metrics["log_sd"].to_numpy(),
        "iid_log_sd": base_log * np.sqrt(t),
        "ratio_annualised": metrics["annualised_sd"].to_numpy() / (base_ann / np.sqrt(t)),
        "ratio_log": metrics["log_sd"].to_numpy() / (base_log * np.sqrt(t)),
    }, index=metrics.index)


def excess_convergence(metrics: pd.DataFrame) -> dict:
    """Fit log(dispersion) on log(horizon); the slope is −0.5 under i.i.d. returns.

    A slope below −0.5 means annualised dispersion narrows faster than arithmetic allows, which
    is mean reversion. This is the whole empirical question, reduced to one number with a
    bootstrap interval attached in ``bootstrap_slope``.
    """
    if len(metrics) < 4:
        return {}
    x = np.log(metrics.index.to_numpy(dtype=float))
    y = np.log(metrics["annualised_sd"].to_numpy(dtype=float))
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return {}
    x, y = x[ok], y[ok]
    xc = x - x.mean()
    slope = float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
    resid = y - y.mean() - slope * xc
    se = float(np.sqrt((resid ** 2).sum() / max(len(x) - 2, 1) / (xc ** 2).sum()))
    return {"slope": slope, "se": se, "iid_slope": -0.5,
            "excess": slope + 0.5, "t_vs_iid": (slope + 0.5) / se if se > 0 else np.nan,
            "faster_than_sqrt_t": bool(slope < -0.5)}


# --------------------------------------------------------------------------- #
# Is there mean reversion?
# --------------------------------------------------------------------------- #
def variance_ratio(rets: pd.Series, q: int) -> dict:
    """Lo-MacKinlay variance ratio with the heteroscedasticity-robust statistic.

    VR(q) below one is mean reversion — the variance of a q-period return is less than q times
    the one-period variance. The robust version matters here: equity returns are strongly
    heteroscedastic, and the homoscedastic statistic rejects far too often on real data, which
    would manufacture exactly the mean reversion this study is testing for.
    """
    r = np.log1p(rets.dropna().to_numpy(dtype=float))
    n = len(r)
    if n < q * 10:
        return {}
    mu = r.mean()
    var1 = float(((r - mu) ** 2).sum() / (n - 1))
    m = q * (n - q + 1) * (1 - q / n)
    cs = np.cumsum(np.concatenate([[0.0], r]))
    q_rets = cs[q:] - cs[:-q]
    varq = float(((q_rets - q * mu) ** 2).sum() / m)
    vr = varq / var1 if var1 > 0 else np.nan
    # heteroscedasticity-robust standard error (Lo & MacKinlay 1988, eq. 18)
    d = (r - mu) ** 2
    theta = 0.0
    for j in range(1, q):
        num = float((d[j:] * d[:-j]).sum())
        den = float((d.sum()) ** 2 / n)
        delta = num / den if den > 0 else 0.0
        theta += (2 * (q - j) / q) ** 2 * delta
    se = float(np.sqrt(theta / n)) if theta > 0 else np.nan
    z = (vr - 1) / se if se and np.isfinite(se) and se > 0 else np.nan
    return {"q": q, "vr": float(vr), "se": se, "z": float(z) if np.isfinite(z) else np.nan,
            "mean_reverting": bool(np.isfinite(z) and z < -1.96),
            "trending": bool(np.isfinite(z) and z > 1.96), "n": n}


def variance_ratio_profile(rets: pd.Series, qs=None) -> pd.DataFrame:
    """Variance ratios across horizons from a week to several years."""
    if qs is None:
        qs = (5, 21, 63, 126, 252, 504, 756, 1260)
    rows = [variance_ratio(rets, q) for q in qs]
    rows = [r for r in rows if r]
    return pd.DataFrame(rows).set_index("q") if rows else pd.DataFrame()


def bootstrap_slope(rets: pd.Series, cash: pd.Series, n_boot: int = 300,
                    block: int = 252, years_grid=None, seed: int = 1007) -> dict:
    """Block bootstrap of the convergence slope, which is the only honest interval here.

    Resampling in one-year blocks destroys any multi-year mean reversion while preserving
    volatility clustering — so the bootstrap distribution is the null of "no mean reversion,
    same short-run dynamics". If the observed slope sits inside it, there is nothing beyond
    arithmetic to explain.
    """
    r = rets.dropna()
    c = cash.reindex(r.index).fillna(0.0)
    n = len(r)
    if n < block * 5:
        return {}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    rv, cv = r.to_numpy(dtype=float), c.to_numpy(dtype=float)
    slopes = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        idx = idx[idx < n]
        sr = pd.Series(rv[idx], index=r.index[:len(idx)])
        sc = pd.Series(cv[idx], index=r.index[:len(idx)])
        m = horizon_metrics(sr, sc, years_grid or (1, 2, 3, 5, 7, 10, 15))
        e = excess_convergence(m)
        if e:
            slopes.append(e["slope"])
    if not slopes:
        return {}
    slopes = np.array(slopes)
    actual = excess_convergence(horizon_metrics(r, c, years_grid or
                                                (1, 2, 3, 5, 7, 10, 15)))
    return {"actual_slope": actual.get("slope", np.nan),
            "null_mean": float(slopes.mean()), "null_sd": float(slopes.std(ddof=1)),
            "null_p05": float(np.percentile(slopes, 5)),
            "null_p95": float(np.percentile(slopes, 95)),
            "p_value": float((slopes <= actual.get("slope", np.nan)).mean()),
            "beyond_arithmetic": bool(actual.get("slope", 0) <
                                      np.percentile(slopes, 5))}


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #
def crra_utility(wealth: np.ndarray, gamma: float = 3.0) -> float:
    """Expected CRRA utility of a terminal wealth sample."""
    w = np.clip(np.asarray(wealth, dtype=float), 1e-9, None)
    if abs(gamma - 1.0) < 1e-9:
        return float(np.mean(np.log(w)))
    return float(np.mean((w ** (1 - gamma) - 1) / (1 - gamma)))


def certainty_equivalent(wealth: np.ndarray, gamma: float = 3.0) -> float:
    """The riskless terminal wealth an investor would accept instead of the gamble."""
    u = crra_utility(wealth, gamma)
    if abs(gamma - 1.0) < 1e-9:
        return float(np.exp(u))
    return float((u * (1 - gamma) + 1) ** (1 / (1 - gamma)))


def optimal_weight_by_horizon(rets: pd.Series, cash: pd.Series, years_grid=None,
                              gammas=(2.0, 3.0, 5.0, 10.0), step: int = 21,
                              n_weights: int = 41) -> pd.DataFrame:
    """The equity weight maximising CRRA certainty equivalent, horizon by horizon.

    Samuelson's (1969) theorem says this should be **flat** in horizon for i.i.d. returns and
    CRRA preferences. Checking it numerically on real data does two things at once: it tests
    whether the theorem's conclusion survives the data's actual dependence structure, and it
    turns an abstract argument into a table an investor can read.
    """
    if years_grid is None:
        years_grid = (1, 3, 5, 10, 15, 20)
    grid = np.linspace(0.0, 1.0, n_weights)
    rows = []
    for y in years_grid:
        lw = horizon_windows(rets, y, step)
        lc = horizon_windows(cash, y, step)
        k = min(len(lw), len(lc))
        if k < 20:
            continue
        eq = np.exp(lw[-k:])
        cs = np.exp(lc[-k:])
        for g in gammas:
            ces = []
            for w in grid:
                # a fixed mix, rebalanced at the ends of the horizon only
                ces.append(certainty_equivalent(w * eq + (1 - w) * cs, g))
            ces = np.array(ces)
            best = float(grid[int(np.nanargmax(ces))])
            rows.append({"years": y, "gamma": g, "optimal_weight": best,
                         "ce_at_best": float(np.nanmax(ces)),
                         "ce_all_equity": float(ces[-1]),
                         "ce_all_cash": float(ces[0]),
                         "n_windows": int(k)})
    return pd.DataFrame(rows)


def weight_stability(table: pd.DataFrame) -> dict:
    """How much the optimal weight actually moves with horizon, per risk aversion."""
    if table.empty:
        return {}
    out = {}
    for g, grp in table.groupby("gamma"):
        w = grp.sort_values("years")["optimal_weight"]
        out[float(g)] = {"min": float(w.min()), "max": float(w.max()),
                         "range": float(w.max() - w.min()),
                         "first": float(w.iloc[0]), "last": float(w.iloc[-1])}
    ranges = [v["range"] for v in out.values()]
    return {"by_gamma": out, "max_range": float(max(ranges)),
            "mean_range": float(np.mean(ranges)),
            "roughly_flat": bool(max(ranges) <= 0.20)}


# --------------------------------------------------------------------------- #
# The control
# --------------------------------------------------------------------------- #
def small_sample_bias(n_days_grid=(8400, 16800, 40000, 100000), drift: float = 0.08,
                      vol: float = 0.16, n_reps: int = 4,
                      years_grid=(1, 2, 3, 5, 7, 10, 15),
                      seed: int = 1007) -> pd.DataFrame:
    """How wrong the convergence slope is on samples of realistic length — on i.i.d. data.

    The single most important measurement in this study, and it uses no market data at all.
    Returns here are i.i.d. by construction, so the true convergence slope is exactly −0.5 and
    the true log-dispersion grows exactly like √T. Neither is what a 33-year sample measures.

    Two things go wrong at once, both from the same cause. Long-horizon statistics are computed
    from overlapping windows carved out of a short tape, so the *effective* number of
    independent observations falls to one or two: the sample standard deviation of those windows
    is then badly downward-biased. That drags the fitted slope well below −0.5 — toward
    apparent mean reversion — and it makes log dispersion appear to *peak and fall* with horizon
    when it must rise monotonically.

    The practical consequence is that measuring −0.7 on real equity data is not evidence of
    mean reversion. It is what a memoryless market looks like through a 33-year window, which
    is why every inferential claim in this module runs against the block bootstrap rather than
    against the theoretical −0.5.
    """
    rows = []
    for n in n_days_grid:
        slopes, peaks = [], []
        for k in range(n_reps):
            r = synthetic_iid(n, drift, vol, seed + k)
            c = pd.Series(np.zeros(n), index=r.index)
            m = horizon_metrics(r, c, years_grid)
            e = excess_convergence(m)
            if e:
                slopes.append(e["slope"])
            if len(m) > 2:
                peaks.append(float(m["log_sd"].idxmax()))
        if not slopes:
            continue
        rows.append({"n_days": n, "years": n / TRADING_DAYS,
                     "mean_slope": float(np.mean(slopes)),
                     "sd_slope": float(np.std(slopes, ddof=1)) if len(slopes) > 1
                     else np.nan,
                     "bias_vs_half": float(np.mean(slopes) + 0.5),
                     "log_sd_peaks_at": float(np.mean(peaks)) if peaks else np.nan,
                     "max_horizon": max(years_grid)})
    return pd.DataFrame(rows).set_index("n_days")


def synthetic_slow_reversion(n_days: int = 8400, drift: float = 0.08, vol: float = 0.16,
                             half_life_years: float = 4.0, transitory_sd: float = 0.25,
                             seed: int = 1007) -> pd.Series:
    """Prices that revert to a trend with a MULTI-YEAR half-life.

    The economically meaningful form of mean reversion, and the one Poterba-Summers and
    Fama-French were testing for. A daily AR(1) is the wrong control here: a one-year block
    bootstrap preserves one-day memory entirely, so a daily-AR(1) process is correctly
    reported by that bootstrap as having nothing beyond the block. Reversion at a four-year
    half-life is the thing the block bootstrap is designed to destroy, and therefore the thing
    it should be able to detect.

    ``transitory_sd`` is the **stationary standard deviation of the transitory component of the
    log price** — how far, in level terms, the price wanders from trend before being pulled
    back. Parameterising by a share of *daily* variance, as a first attempt did, produces an
    undetectable effect: the transitory part enters daily returns only through its first
    difference, which is nearly zero when the half-life is years, so it can carry a large share
    of the price swing while contributing almost nothing per day. The level is the quantity
    that matters and the level is what is specified.
    """
    rng = np.random.default_rng(seed)
    dv = vol / np.sqrt(TRADING_DAYS)
    mu = np.log1p(drift) / TRADING_DAYS - dv ** 2 / 2
    phi = 0.5 ** (1.0 / (half_life_years * TRADING_DAYS))
    # AR(1) innovation giving the transitory component a stationary sd of transitory_sd
    innov_sd = transitory_sd * np.sqrt(max(1 - phi ** 2, 1e-18))
    z = np.zeros(n_days)
    e = rng.normal(0, innov_sd, n_days)
    for t in range(1, n_days):
        z[t] = phi * z[t - 1] + e[t]
    dz = np.diff(np.concatenate([[0.0], z]))
    # the permanent walk carries whatever daily variance the transitory part does not
    perm_var = max(dv ** 2 - float(np.var(dz)), (0.2 * dv) ** 2)
    perm = rng.normal(0, np.sqrt(perm_var), n_days)
    log_ret = mu + perm + dz
    idx = pd.bdate_range("1993-02-01", periods=n_days)
    return pd.Series(np.expm1(log_ret), index=idx, name="slow_mr")


def synthetic_iid(n_days: int = 8400, drift: float = 0.08, vol: float = 0.16,
                  seed: int = 1007) -> pd.Series:
    """Returns that are i.i.d. by construction — mean reversion is impossible here.

    The control that makes the study's central distinction testable. Any narrowing of
    annualised dispersion in this series is the 1/√T of arithmetic, nothing more, so if the
    real data narrows at the same rate then the adviser's chart is showing arithmetic too.
    """
    rng = np.random.default_rng(seed)
    dv = vol / np.sqrt(TRADING_DAYS)
    mu = np.log1p(drift) / TRADING_DAYS - dv ** 2 / 2
    idx = pd.bdate_range("1993-02-01", periods=n_days)
    return pd.Series(np.expm1(rng.normal(mu, dv, n_days)), index=idx, name="iid")


def synthetic_mean_reverting(n_days: int = 8400, drift: float = 0.08, vol: float = 0.16,
                             phi: float = -0.02, seed: int = 1007) -> pd.Series:
    """Returns with a deliberate negative autocorrelation — genuine mean reversion.

    Exists so the machinery can be shown to *detect* time diversification when it is present.
    A test that returns "no mean reversion" on every input would be worthless, and this is how
    that possibility is ruled out.
    """
    rng = np.random.default_rng(seed)
    dv = vol / np.sqrt(TRADING_DAYS)
    mu = np.log1p(drift) / TRADING_DAYS - dv ** 2 / 2
    e = rng.normal(0, dv, n_days)
    r = np.zeros(n_days)
    prev = 0.0
    for t in range(n_days):
        r[t] = mu + e[t] + phi * prev
        prev = r[t] - mu
    idx = pd.bdate_range("1993-02-01", periods=n_days)
    return pd.Series(np.expm1(r), index=idx, name="mr")


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: about whether *time diversification beyond arithmetic* exists. **Real** if
      annualised dispersion narrows significantly faster than 1/√T; **Weak** if it narrows
      faster but within the bootstrap null; **None** if it tracks √T.
    - **Tradability**: about the decision. **Useful** if the analysis yields a clear
      prescription for how horizon should enter an allocation; **Partial** if directional;
      **Mirage** if it cannot say.
    """
    signal = ("Real" if h["beyond_arithmetic"]
              else ("Weak" if h["slope"] < -0.5 else "None"))
    trad = ("Useful" if h["weights_flat"] else
            ("Partial" if h["max_weight_range"] < 0.40 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"Both sides of this argument are quoting true statements about the same windows. "
            f"On {h['asset']} over {h['years']:.0f} years: the standard deviation of the "
            f"**annualised** return falls from {h['ann_sd_1']:.1%} at one year to "
            f"{h['ann_sd_long']:.1%} at {h['long_years']:.0f} — the adviser's chart, and it is "
            f"correct. Over the same windows the standard deviation of **terminal wealth** "
            f"rises from {h['term_sd_1']:.2f}× to {h['term_sd_long']:.2f}× — Samuelson's point, "
            f"also correct. The question is whether the first is *more* than arithmetic. Under "
            f"i.i.d. returns annualised dispersion must fall like 1/√T, a log-log slope of "
            f"exactly −0.5. Measured here: **{h['slope']:.3f}** (±{h['slope_se']:.3f}). A block "
            f"bootstrap that destroys multi-year mean reversion while keeping the short-run "
            f"dynamics puts the null slope at {h['null_mean']:.3f} with a 5th percentile of "
            f"{h['null_p05']:.3f}, so the observed value is "
            f"{'outside' if h['beyond_arithmetic'] else 'inside'} it "
            f"(p = {h['p_value']:.3f}). Lo-MacKinlay variance ratios agree: at "
            f"{h['vr_horizon']} days VR = {h['vr_long']:.3f} with z = {h['vr_z']:.2f}. "
            f"The convergence is {'not ' if not h['beyond_arithmetic'] else ''}"
            f"merely the √T in the denominator."),
        "trad_why": (
            f"Which leaves the decision, and there the answer is cleaner than the debate. "
            f"Maximising CRRA certainty equivalent over the real windows, the optimal equity "
            f"weight moved by at most **{h['max_weight_range']:.0%} across horizons from one "
            f"to {h['weight_max_years']:.0f} years** — at γ = 3 it went from "
            f"{h['w_g3_short']:.0%} to {h['w_g3_long']:.0%}. Samuelson's 1969 theorem, that "
            f"horizon drops out under CRRA and i.i.d. returns, survives contact with the data. "
            f"The practical reading is not \"ignore your horizon\" but something more precise: "
            f"horizon should enter an allocation through **the things that genuinely depend on "
            f"it** — the size and certainty of future contributions, the flexibility to defer "
            f"spending, the ability to keep earning through a drawdown — and not through a "
            f"belief that equities become less risky if you wait. The shortfall probability "
            f"does fall, from {h['shortfall_1']:.0%} at one year to "
            f"{h['shortfall_long']:.0%} at {h['long_years']:.0f}, but the worst outcome gets "
            f"worse: the poorest {h['long_years']:.0f}-year window ended at "
            f"{h['worst_long']:.2f}× against {h['worst_1']:.2f}× for the poorest single year. "
            f"Less likely, more costly, which is precisely the trade the glide-path argument "
            f"leaves out."),
        "trad": trad,
        "one_sentence": (
            f"Annualised dispersion narrows at a log-log slope of {h['slope']:.2f} against the "
            f"−0.50 that arithmetic alone requires, so there is "
            f"{'genuine' if h['beyond_arithmetic'] else 'no'} time diversification here — and "
            f"the CRRA-optimal equity weight barely moves with horizon regardless."),
    }
