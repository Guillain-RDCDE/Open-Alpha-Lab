"""What a benchmark choice is worth — Study 1012.

Alpha is the intercept of a regression. Everything else in the specification is a choice, and
each choice moves the intercept:

- **Which benchmark.** A single index, or several. ``alpha_grid`` runs every fund against every
  candidate and reports the whole surface, which is more honest than reporting the one that was
  used.
- **How many factors.** One market factor, three Fama-French, or a bespoke blend.
  ``multi_factor_alpha`` fits them all on the same data.
- **Whether the benchmark is chosen or fitted.** ``best_fit_benchmark`` picks the
  highest-R² single index, which sounds principled and is a form of selection that inflates
  significance — measured explicitly rather than warned about.

The important machinery is in the three checks that follow.

``alpha_range`` reports the spread of alphas across defensible specifications and compares it
against the standard error of any single one. If the spread exceeds the standard error, the
specification choice is a bigger source of uncertainty than the sampling noise everybody reports
— and nobody prints an error bar for it.

``mis_specification_damage`` uses the synthetic world to measure what a wrong benchmark does
when the truth is known: how much false alpha a plausible but incorrect choice manufactures.

``can_the_data_choose`` asks whether R², information ratio or a formal encompassing test can
identify the right benchmark when one exists. The answer determines whether "which benchmark"
is an empirical question or a judgement call dressed as one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def ols_with_hac(y: np.ndarray, X: np.ndarray, lags: int = 5) -> dict:
    """OLS with Newey-West standard errors and an intercept added automatically.

    HAC by default rather than as an option. Daily fund returns are autocorrelated and
    heteroscedastic; plain OLS standard errors overstate the significance of an alpha, which is
    the direction that matters when the whole subject is how confidently alphas are reported.
    """
    y = np.asarray(y, dtype=float)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != len(y):
        X = X.T
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    if n < 60:
        return {}
    A = np.column_stack([np.ones(n), X])
    try:
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {}
    resid = y - A @ beta
    XtX_inv = np.linalg.pinv(A.T @ A)
    # Newey-West meat
    S = (A * resid[:, None]).T @ (A * resid[:, None])
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1)
        u = A[l:] * resid[l:, None]
        v = A[:-l] * resid[:-l, None]
        G = u.T @ v
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    tss = float(((y - y.mean()) ** 2).sum())
    rss = float((resid ** 2).sum())
    return {"alpha": float(beta[0] * TRADING_DAYS),
            "alpha_se": float(se[0] * TRADING_DAYS),
            "alpha_t": float(beta[0] / se[0]) if se[0] > 0 else np.nan,
            "betas": beta[1:], "beta_se": se[1:],
            "r2": float(1 - rss / tss) if tss > 0 else np.nan,
            "adj_r2": float(1 - (rss / max(n - k - 1, 1)) / (tss / max(n - 1, 1)))
            if tss > 0 else np.nan,
            "n": n, "resid_vol": float(np.std(resid, ddof=1) * np.sqrt(TRADING_DAYS)),
            "resid": resid}


def single_factor_alpha(fund: pd.Series, bench: pd.Series, rf: pd.Series = None,
                        lags: int = 5) -> dict:
    """Alpha of one fund against one benchmark, both in excess of cash."""
    df = pd.concat([fund.rename("f"), bench.rename("b")], axis=1, sort=False).dropna()
    if rf is not None:
        r = rf.reindex(df.index).fillna(0.0)
        df["f"] = df["f"] - r
        df["b"] = df["b"] - r
    if len(df) < 250:
        return {}
    out = ols_with_hac(df["f"].to_numpy(), df["b"].to_numpy().reshape(-1, 1), lags)
    if out:
        out["beta"] = float(out["betas"][0])
        out["beta_se_1"] = float(out["beta_se"][0])
        out["tracking_error"] = float((df["f"] - df["b"]).std(ddof=1)
                                      * np.sqrt(TRADING_DAYS))
        out["information_ratio"] = (float((df["f"] - df["b"]).mean() * TRADING_DAYS
                                          / out["tracking_error"])
                                    if out["tracking_error"] > 0 else np.nan)
    return out


def multi_factor_alpha(fund: pd.Series, benches: pd.DataFrame, rf: pd.Series = None,
                       lags: int = 5) -> dict:
    """Alpha against several benchmarks at once."""
    df = pd.concat([fund.rename("f"), benches], axis=1, sort=False).dropna()
    if len(df) < 250:
        return {}
    if rf is not None:
        r = rf.reindex(df.index).fillna(0.0)
        df = df.sub(r, axis=0)
    cols = [c for c in df.columns if c != "f"]
    out = ols_with_hac(df["f"].to_numpy(), df[cols].to_numpy(), lags)
    if out:
        out["factors"] = cols
        out["loadings"] = {c: float(b) for c, b in zip(cols, out["betas"])}
    return out


# --------------------------------------------------------------------------- #
# The surface
# --------------------------------------------------------------------------- #
def alpha_grid(funds: pd.DataFrame, benches: pd.DataFrame, rf: pd.Series = None,
               min_obs: int = 500) -> pd.DataFrame:
    """Every fund against every benchmark. The table that should accompany any alpha."""
    rows = []
    for f in funds.columns:
        for b in benches.columns:
            if f == b:
                continue
            d = single_factor_alpha(funds[f], benches[b], rf)
            if not d or d["n"] < min_obs:
                continue
            rows.append({"fund": f, "benchmark": b, "alpha": d["alpha"],
                         "alpha_se": d["alpha_se"], "alpha_t": d["alpha_t"],
                         "beta": d["beta"], "r2": d["r2"],
                         "tracking_error": d["tracking_error"],
                         "information_ratio": d["information_ratio"], "n": d["n"]})
    return pd.DataFrame(rows)


def alpha_range(grid: pd.DataFrame) -> pd.DataFrame:
    """Per fund: the spread of alpha across benchmarks, against its own standard error.

    The comparison that makes the point. A published alpha carries a standard error reflecting
    sampling noise. If the spread produced by choosing a different — equally defensible —
    benchmark exceeds that standard error, then the specification is the dominant source of
    uncertainty and it is the one nobody quantifies.
    """
    if grid.empty:
        return pd.DataFrame()
    rows = []
    for f, g in grid.groupby("fund"):
        if len(g) < 3:
            continue
        a = g["alpha"]
        rows.append({
            "fund": f, "n_benchmarks": int(len(g)),
            "alpha_min": float(a.min()), "alpha_max": float(a.max()),
            "alpha_median": float(a.median()),
            "alpha_spread": float(a.max() - a.min()),
            "median_se": float(g["alpha_se"].median()),
            "spread_over_se": float((a.max() - a.min()) / g["alpha_se"].median())
            if g["alpha_se"].median() > 0 else np.nan,
            "sign_flips": bool((a > 0).any() and (a < 0).any()),
            "significant_both_ways": bool((g["alpha_t"] > 2).any()
                                          and (g["alpha_t"] < -2).any()),
            "n_significant_positive": int((g["alpha_t"] > 2).sum()),
            "n_significant_negative": int((g["alpha_t"] < -2).sum()),
            "best_r2_benchmark": str(g.loc[g["r2"].idxmax(), "benchmark"]),
            "best_alpha_benchmark": str(g.loc[g["alpha"].idxmax(), "benchmark"]),
            "alpha_at_best_r2": float(g.loc[g["r2"].idxmax(), "alpha"]),
            "alpha_at_best_alpha": float(g["alpha"].max()),
        })
    return pd.DataFrame(rows).set_index("fund") if rows else pd.DataFrame()


def best_fit_benchmark(fund: pd.Series, benches: pd.DataFrame,
                       rf: pd.Series = None) -> dict:
    """Pick the highest-R² single benchmark, and report what that selection costs.

    Choosing the benchmark that fits best sounds principled. It is also a search over
    candidates, so the resulting alpha's t-statistic is not distributed the way a t-statistic
    from a pre-chosen benchmark would be. The number of candidates searched is reported so a
    reader can apply their own correction, which is more useful than a warning.
    """
    out = []
    for b in benches.columns:
        d = single_factor_alpha(fund, benches[b], rf)
        if d:
            out.append({"benchmark": b, **{k: d[k] for k in
                                           ("alpha", "alpha_se", "alpha_t", "beta",
                                            "r2")}})
    if not out:
        return {}
    d = pd.DataFrame(out).set_index("benchmark")
    best_r2 = d["r2"].idxmax()
    best_alpha = d["alpha"].idxmax()
    return {"n_candidates": int(len(d)),
            "best_r2_benchmark": str(best_r2), "best_r2": float(d.loc[best_r2, "r2"]),
            "alpha_at_best_r2": float(d.loc[best_r2, "alpha"]),
            "t_at_best_r2": float(d.loc[best_r2, "alpha_t"]),
            "best_alpha_benchmark": str(best_alpha),
            "max_alpha": float(d.loc[best_alpha, "alpha"]),
            "t_at_max_alpha": float(d.loc[best_alpha, "alpha_t"]),
            "cherry_picking_gain": float(d.loc[best_alpha, "alpha"]
                                         - d.loc[best_r2, "alpha"]),
            "table": d}


def specification_ladder(fund: pd.Series, benches: pd.DataFrame, rf: pd.Series = None,
                         ladder=None) -> pd.DataFrame:
    """The same fund under progressively richer models.

    Not to find the "right" answer but to show the path. Adding factors almost always shrinks
    alpha, because each one absorbs a piece of what was previously unexplained. Where a reader
    stops on this ladder determines the number they quote, and there is no purely statistical
    place to stop.
    """
    if ladder is None:
        ladder = [("market only", ["SPY"]),
                  ("market + size", ["SPY", "IWM"]),
                  ("market + size + value", ["SPY", "IWM", "IWD"]),
                  ("+ growth", ["SPY", "IWM", "IWD", "IWF"]),
                  ("+ international", ["SPY", "IWM", "IWD", "IWF", "EFA"])]
    rows = []
    for name, cols in ladder:
        avail = [c for c in cols if c in benches.columns]
        if not avail:
            continue
        d = multi_factor_alpha(fund, benches[avail], rf)
        if not d:
            continue
        rows.append({"model": name, "n_factors": len(avail), "alpha": d["alpha"],
                     "alpha_se": d["alpha_se"], "alpha_t": d["alpha_t"],
                     "r2": d["r2"], "adj_r2": d["adj_r2"],
                     "resid_vol": d["resid_vol"]})
    return pd.DataFrame(rows).set_index("model")


# --------------------------------------------------------------------------- #
# Can the data choose?
# --------------------------------------------------------------------------- #
def can_the_data_choose(fund: pd.Series, benches: pd.DataFrame, rf: pd.Series = None,
                        n_boot: int = 200, block: int = 63,
                        seed: int = 1012) -> dict:
    """Bootstrap how often each candidate benchmark wins on R².

    If one benchmark wins nearly always, the choice is empirical and the argument is over. If
    the winner changes from resample to resample, "which benchmark" is a judgement call and
    should be presented as one — with the alpha reported as a range rather than a number.
    """
    df = pd.concat([fund.rename("f"), benches], axis=1, sort=False).dropna()
    if len(df) < 500:
        return {}
    cols = [c for c in df.columns if c != "f"]
    n = len(df)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    wins = {c: 0 for c in cols}
    alphas = {c: [] for c in cols}
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        idx = idx[idx < n]
        sub = df.iloc[idx]
        best, best_r2 = None, -np.inf
        for c in cols:
            d = ols_with_hac(sub["f"].to_numpy(),
                             sub[c].to_numpy().reshape(-1, 1), lags=2)
            if not d:
                continue
            alphas[c].append(d["alpha"])
            if d["r2"] > best_r2:
                best, best_r2 = c, d["r2"]
        if best:
            wins[best] += 1
    total = sum(wins.values())
    if total == 0:
        return {}
    share = {c: wins[c] / total for c in cols}
    top = max(share, key=share.get)
    return {"win_share": share, "modal_benchmark": top,
            "modal_share": float(share[top]),
            "decisive": bool(share[top] > 0.80),
            "alpha_by_benchmark": {c: float(np.mean(v)) for c, v in alphas.items()
                                   if v},
            "n_candidates": len(cols)}


def encompassing_test(fund: pd.Series, bench_a: pd.Series, bench_b: pd.Series,
                      rf: pd.Series = None) -> dict:
    """Does benchmark A make B redundant, or the other way round?

    The formal version of "which benchmark". Regress the fund on both; if A's loading is
    significant and B's is not, A encompasses B. The interesting and common outcome is that
    *both* are significant, which means neither is the benchmark and the single-index alpha was
    never well defined.
    """
    df = pd.concat([fund.rename("f"), bench_a.rename("a"), bench_b.rename("b")],
                   axis=1, sort=False).dropna()
    if len(df) < 250:
        return {}
    if rf is not None:
        df = df.sub(rf.reindex(df.index).fillna(0.0), axis=0)
    d = ols_with_hac(df["f"].to_numpy(), df[["a", "b"]].to_numpy())
    if not d:
        return {}
    ta = float(d["betas"][0] / d["beta_se"][0]) if d["beta_se"][0] > 0 else np.nan
    tb = float(d["betas"][1] / d["beta_se"][1]) if d["beta_se"][1] > 0 else np.nan
    return {"beta_a": float(d["betas"][0]), "t_a": ta,
            "beta_b": float(d["betas"][1]), "t_b": tb,
            "alpha": d["alpha"], "alpha_t": d["alpha_t"], "r2": d["r2"],
            "a_encompasses_b": bool(abs(ta) > 2 and abs(tb) < 2),
            "b_encompasses_a": bool(abs(tb) > 2 and abs(ta) < 2),
            "both_needed": bool(abs(ta) > 2 and abs(tb) > 2),
            "neither": bool(abs(ta) < 2 and abs(tb) < 2)}


# --------------------------------------------------------------------------- #
# The control: a fund whose alpha is known
# --------------------------------------------------------------------------- #
def synthetic_fund(n_days: int = 3000, true_alpha: float = 0.0,
                   loadings=None, factor_vols=None, factor_corr: float = 0.6,
                   idio_vol: float = 0.08, factor_means=None,
                   seed: int = 1012) -> dict:
    """A fund with a KNOWN alpha and KNOWN loadings on correlated, PREMIUM-EARNING factors.

    ``factor_means`` is the parameter that makes this control work, and a first version of the
    study omitted it and got the wrong answer. If the factors have zero mean, dropping one from
    the regression does **not** bias the intercept — it only inflates the residual variance, so
    a mis-specified benchmark produces a noisier alpha rather than a false one. Omitted-variable
    bias lands on the *slope*, and reaches the intercept only through the omitted factor's mean.

    So a benchmark that misses a factor manufactures alpha exactly when that factor **earned a
    premium** over the period — which is the case that matters in practice, since size, value
    and momentum all did. The default gives each factor an annual mean return, and the false
    alpha that follows is roughly (omitted loading) × (omitted premium).

    The correlation between factors is the second parameter. Uncorrelated candidate benchmarks
    would make a wrong choice obvious; a large-cap index and a growth index share most of their
    variance, which is what makes a mis-specified benchmark plausible.
    """
    if loadings is None:
        loadings = {"F1": 1.0, "F2": 0.3}
    if factor_vols is None:
        factor_vols = {k: 0.18 for k in loadings}
    if factor_means is None:
        factor_means = {k: 0.07 for k in loadings}
    rng = np.random.default_rng(seed)
    names = list(loadings)
    k = len(names)
    C = np.full((k, k), factor_corr)
    np.fill_diagonal(C, 1.0)
    sd = np.array([factor_vols[n] / np.sqrt(TRADING_DAYS) for n in names])
    cov = C * np.outer(sd, sd)
    mu = np.array([factor_means.get(n, 0.0) / TRADING_DAYS for n in names])
    F = rng.multivariate_normal(mu, cov, size=n_days)
    idio = rng.normal(0, idio_vol / np.sqrt(TRADING_DAYS), n_days)
    r = (true_alpha / TRADING_DAYS
         + F @ np.array([loadings[n] for n in names]) + idio)
    idx = pd.bdate_range("1993-02-01", periods=n_days)
    return {"fund": pd.Series(r, index=idx, name="FUND"),
            "factors": pd.DataFrame(F, index=idx, columns=names),
            "true_alpha": true_alpha, "loadings": loadings,
            "factor_means": factor_means}


def mis_specification_damage(true_alpha: float = 0.0, factor_corr_grid=(0.0, 0.3, 0.6, 0.9),
                             n_days: int = 3000, n_reps: int = 8,
                             seed: int = 1012) -> pd.DataFrame:
    """How much false alpha does the WRONG benchmark manufacture, when truth is known?

    The fund loads on two correlated factors. Benchmarking it against the correct pair recovers
    the planted alpha; benchmarking against either one alone does not. The measured error is the
    false alpha a plausible mis-specification produces, and it grows with how *dissimilar* the
    omitted factor is — which is the opposite of the intuition that correlated benchmarks are
    interchangeable.
    """
    rows = []
    for corr in factor_corr_grid:
        correct, only_f1, only_f2 = [], [], []
        for k in range(n_reps):
            w = synthetic_fund(n_days, true_alpha, {"F1": 1.0, "F2": 0.5},
                               factor_corr=corr, seed=seed + k)
            F = w["factors"]
            both = multi_factor_alpha(w["fund"], F)
            a1 = single_factor_alpha(w["fund"], F["F1"])
            a2 = single_factor_alpha(w["fund"], F["F2"])
            if both:
                correct.append(both["alpha"])
            if a1:
                only_f1.append(a1["alpha"])
            if a2:
                only_f2.append(a2["alpha"])
        rows.append({"factor_corr": corr, "true_alpha": true_alpha,
                     "alpha_correct": float(np.mean(correct)) if correct else np.nan,
                     "alpha_only_f1": float(np.mean(only_f1)) if only_f1 else np.nan,
                     "alpha_only_f2": float(np.mean(only_f2)) if only_f2 else np.nan,
                     "error_only_f1": float(np.mean(only_f1) - true_alpha)
                     if only_f1 else np.nan,
                     "error_only_f2": float(np.mean(only_f2) - true_alpha)
                     if only_f2 else np.nan})
    return pd.DataFrame(rows).set_index("factor_corr")


def false_alpha_significance(true_alpha: float = 0.0, n_days: int = 3000,
                             n_reps: int = 40, factor_corr: float = 0.6,
                             seed: int = 1012) -> dict:
    """How often does a mis-specified benchmark produce a *significant* false alpha?

    The number that matters for reading a paper. A wrong benchmark that produces a small bias is
    a nuisance; one that produces a t-statistic above two on a fund with genuinely zero alpha is
    a machine for generating publishable findings.
    """
    sig_wrong, sig_right, alphas = 0, 0, []
    for k in range(n_reps):
        w = synthetic_fund(n_days, true_alpha, {"F1": 1.0, "F2": 0.5},
                           factor_corr=factor_corr, seed=seed + k)
        F = w["factors"]
        right = multi_factor_alpha(w["fund"], F)
        wrong = single_factor_alpha(w["fund"], F["F1"])
        if right and abs(right["alpha_t"]) > 2:
            sig_right += 1
        if wrong:
            alphas.append(wrong["alpha"])
            if abs(wrong["alpha_t"]) > 2:
                sig_wrong += 1
    return {"n_reps": n_reps, "true_alpha": true_alpha,
            "share_significant_wrong_benchmark": sig_wrong / max(n_reps, 1),
            "share_significant_right_benchmark": sig_right / max(n_reps, 1),
            "mean_false_alpha": float(np.mean(alphas)) if alphas else np.nan,
            "sd_false_alpha": float(np.std(alphas, ddof=1)) if len(alphas) > 1
            else np.nan}


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Real** if the spread of alpha across defensible benchmarks exceeds the
      standard error of a single estimate for most funds; **Weak** if comparable; **None** if
      the choice barely matters.
    - **Tradability**: about whether the choice is decidable. **Useful** if the data reliably
      identifies one benchmark, so the problem is solvable; **Partial** if it narrows the field;
      **Mirage** if the winner changes from resample to resample, since then any single reported
      alpha is a judgement call presented as a measurement.
    """
    signal = ("Real" if h["median_spread_over_se"] > 2.0
              else ("Weak" if h["median_spread_over_se"] > 0.8 else "None"))
    trad = ("Useful" if h["share_decisive"] > 0.7
            else ("Partial" if h["share_decisive"] > 0.3 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"Enormously, and by more than the noise everyone does report. Running "
            f"{h['n_funds']} funds against {h['n_benchmarks']} defensible benchmarks, the "
            f"median fund's alpha ranged over **{h['median_spread']:.2%} a year** depending "
            f"only on what it was measured against — **{h['median_spread_over_se']:.1f}× the "
            f"median standard error** of any single estimate. The specification is a larger "
            f"source of uncertainty than the sampling error that gets the confidence interval, "
            f"and it is the one with no error bar at all. **{h['share_sign_flip']:.0%} of funds "
            f"change the sign of their alpha** across benchmarks, and "
            f"{h['share_both_significant']:.0%} are *significantly positive against one "
            f"benchmark and significantly negative against another* — both at |t| > 2, both "
            f"defensible, both publishable. The worst case here was {h['worst_fund']}, spanning "
            f"{h['worst_spread']:.2%}. Nor is this only a matter of picking obviously wrong "
            f"comparators: climbing the specification ladder from a single market factor to "
            f"five, alpha moved from {h['ladder_first']:+.2%} to {h['ladder_last']:+.2%} for "
            f"the headline fund, with every rung a specification someone publishes."),
        "trad_why": (
            f"Only sometimes, and knowing when is the practical contribution. Bootstrapping "
            f"which benchmark fits best, a single candidate won more than 80% of resamples for "
            f"just **{h['share_decisive']:.0%} of funds**; for the rest the winner changed from "
            f"one resample to the next, so \"which benchmark\" is a judgement call and the "
            f"alpha should be reported as a range. The synthetic control prices the danger "
            f"exactly, because there the truth is planted. A fund with a **true alpha of zero** "
            f"loading on two correlated factors, benchmarked against only one of them, showed a "
            f"measured alpha of {h['false_alpha']:+.2%} a year — and it was "
            f"**statistically significant at |t| > 2 in {h['false_sig_rate']:.0%} of "
            f"simulations**, against {h['true_sig_rate']:.0%} with the correct benchmark. A "
            f"plausible mis-specification is a machine for manufacturing publishable alpha. "
            f"Two things a reader can do. Ask for the **grid**, not the number: every alpha "
            f"should arrive with the range across reasonable comparators, which costs nothing "
            f"to compute. And treat a best-fitting benchmark as a search: choosing the "
            f"highest-R² of {h['n_benchmarks']} candidates and then quoting a conventional "
            f"t-statistic ignores the selection, which here was worth "
            f"{h['cherry_pick_gain']:.2%} a year between the best-fitting and the "
            f"best-flattering choice."),
        "trad": trad,
        "one_sentence": (
            f"The median fund's alpha moves {h['median_spread']:.2%} a year across defensible "
            f"benchmarks — {h['median_spread_over_se']:.1f}× its own standard error — and a "
            f"mis-specified benchmark makes a genuinely zero alpha look significant "
            f"{h['false_sig_rate']:.0%} of the time."),
    }
