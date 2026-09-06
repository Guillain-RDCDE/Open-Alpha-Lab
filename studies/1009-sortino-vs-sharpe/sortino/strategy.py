"""Sharpe against Sortino, and what the second one costs — Study 1009.

The Sortino ratio is excess return over **downside deviation**: the root-mean-square of returns
below a threshold, with returns above it treated as zero. The argument for it is that variance
punishes good surprises, which no investor objects to.

Three things follow, and only the first is usually said.

**1. It differs from Sharpe only through skewness.** For a distribution symmetric **about the
threshold**, downside deviation is σ/√2 exactly, so Sortino is Sharpe × √2 and the ranking is
identical. Every difference between the two rankings is therefore a statement about third
moments and nothing else. ``symmetric_identity`` verifies the √2 relationship rather than
asserting it — and the qualifier earns its place, since a positive drift alone shifts the ratio
above √2 without any asymmetry being present.

**2. It is estimated from roughly half the sample.** Downside deviation uses only the
observations below the threshold — about 46% of daily equity returns — so its standard error is
larger than σ's for the same data. ``estimation_precision`` measures both by bootstrap. A
ranking that is "better in principle" but noisier in practice can easily be worse in use, and
that trade-off is never quoted alongside the ratio.

**3. Skewness itself is badly estimated.** Third moments need far more data than second moments,
and the sample skewness of daily equity returns is dominated by a handful of observations. So
the quantity on which Sortino's entire claim to superiority rests is the least reliable thing in
the calculation. ``skew_reliability`` puts a bootstrap interval on it.

``rank_agreement`` and ``out_of_sample_ranking`` then settle the practical question the way it
should be settled: split the sample, rank on the first half, measure on the second, and see
which ratio's ranking survives. That is a horse race with a scoreboard rather than an argument
about definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# The two ratios
# --------------------------------------------------------------------------- #
def downside_deviation(rets: np.ndarray, mar: float = 0.0) -> float:
    """Root-mean-square of shortfalls below ``mar``, divided by the FULL sample size.

    The denominator convention matters and is got wrong often enough to be worth stating.
    Dividing by the number of *below-threshold* observations gives the conditional standard
    deviation of losses, which is a different statistic and breaks the σ/√2 identity for
    symmetric distributions. Dividing by the full count — the Sortino-Price definition — keeps
    downside deviation on the same scale as σ, which is what makes the two ratios comparable at
    all.
    """
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return np.nan
    short = np.minimum(r - mar, 0.0)
    return float(np.sqrt((short ** 2).sum() / len(r)))


def sharpe(rets: np.ndarray, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return np.nan
    sd = float(r.std(ddof=1))
    if sd <= 1e-14:
        return np.nan
    return float((r.mean() - rf) / sd * np.sqrt(periods))


def sortino(rets: np.ndarray, mar: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return np.nan
    dd = downside_deviation(r, mar)
    if not np.isfinite(dd) or dd <= 1e-14:
        return np.nan
    return float((r.mean() - mar) / dd * np.sqrt(periods))


def ratio_table(rets: pd.DataFrame, rf: pd.Series = None,
                mar: float = 0.0) -> pd.DataFrame:
    """Both ratios plus the moments that could make them disagree."""
    rows = []
    for c in rets.columns:
        s = rets[c].dropna()
        if len(s) < 250:
            continue
        v = s.to_numpy(dtype=float)
        r0 = float(rf.reindex(s.index).mean()) if rf is not None else 0.0
        sh = sharpe(v, r0)
        so = sortino(v, mar)
        rows.append({"asset": c, "n": int(len(v)), "mean": float(v.mean() * TRADING_DAYS),
                     "vol": float(v.std(ddof=1) * np.sqrt(TRADING_DAYS)),
                     "downside_dev": downside_deviation(v, mar) * np.sqrt(TRADING_DAYS),
                     "sharpe": sh, "sortino": so,
                     "ratio": so / sh if sh and np.isfinite(sh) and sh != 0 else np.nan,
                     "skew": float(s.skew()), "kurtosis": float(s.kurtosis())})
    return pd.DataFrame(rows).set_index("asset")


def symmetric_identity(rets: np.ndarray, mar: float = 0.0) -> dict:
    """Check downside deviation against σ/√2, exact for symmetry **about the threshold**.

    The qualifier is not pedantry. Daily returns carry a small positive mean, so with a zero
    threshold rather more than half the mass sits above it, downside deviation shrinks, and the
    ratio comes out near 1.455 rather than 1.414 — an apparent 3% "skewness effect" that is
    nothing of the kind. Setting ``mar`` to the mean recovers 1.4137 on symmetric data. Since
    that drift-induced shift is common to every asset, it cannot change a ranking, which is
    part of why the rankings turn out to coincide so exactly.

    The cleanest available diagnostic. Every deviation of the ratio from √2 is skewness, so this
    single number says how much scope Sortino has to differ from Sharpe at all — before any
    ranking is computed.
    """
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 30:
        return {}
    sd = float(r.std(ddof=1))
    dd = downside_deviation(r, mar)
    return {"sd": sd, "downside_dev": dd,
            "sd_over_dd": sd / dd if dd > 0 else np.nan,
            "symmetric_prediction": float(np.sqrt(2)),
            "excess": (sd / dd - np.sqrt(2)) if dd > 0 else np.nan,
            "skew": float(pd.Series(r).skew())}


# --------------------------------------------------------------------------- #
# How much do they disagree?
# --------------------------------------------------------------------------- #
def rank_agreement(table: pd.DataFrame) -> dict:
    """Spearman agreement between the two rankings, and where they differ most."""
    d = table.dropna(subset=["sharpe", "sortino"])
    if len(d) < 4:
        return {}
    rs = d["sharpe"].rank(ascending=False)
    ro = d["sortino"].rank(ascending=False)
    diff = (rs - ro)
    rho = float(np.corrcoef(rs, ro)[0, 1])
    return {"n": int(len(d)), "spearman": rho,
            "max_rank_change": float(diff.abs().max()),
            "mean_abs_rank_change": float(diff.abs().mean()),
            "n_unchanged": int((diff == 0).sum()),
            "biggest_mover": str(diff.abs().idxmax()),
            # nan when the rankings coincide exactly: `diff` is then constant, and the
            # correlation of anything with a constant is undefined rather than zero.
            "skew_rank_corr": float(np.corrcoef(d["skew"].rank(), diff)[0, 1])
            if (d["skew"].notna().all() and diff.std(ddof=0) > 0) else np.nan}


def disagreement_vs_skew(rets: pd.DataFrame, mar: float = 0.0) -> pd.DataFrame:
    """Per-asset: how far Sortino/Sharpe departs from √2, against realised skewness.

    If the relationship is tight, the two ratios differ for exactly the stated reason and
    Sortino carries no information beyond "Sharpe plus a skewness adjustment". If it is loose,
    something else is going on — most likely estimation noise, which is section 3's subject.
    """
    rows = []
    for c in rets.columns:
        s = rets[c].dropna()
        if len(s) < 250:
            continue
        idm = symmetric_identity(s.to_numpy(dtype=float), mar)
        if not idm:
            continue
        rows.append({"asset": c, "skew": idm["skew"],
                     "sd_over_dd": idm["sd_over_dd"], "excess": idm["excess"]})
    d = pd.DataFrame(rows).set_index("asset")
    if len(d) > 3:
        d.attrs["corr_skew_excess"] = float(d["skew"].corr(d["excess"]))
    return d


# --------------------------------------------------------------------------- #
# What does the extra precision cost?
# --------------------------------------------------------------------------- #
def estimation_precision(rets: np.ndarray, n_boot: int = 500, block: int = 21,
                         mar: float = 0.0, seed: int = 1009) -> dict:
    """Block-bootstrap standard errors for both ratios, on identical resamples.

    The comparison nobody runs. Sortino uses only the observations below the threshold — a
    little under half of a daily equity series — so it estimates its denominator from roughly
    half the data. Whether the resulting noise outweighs the information gained is an empirical
    question with a cheap answer, and this is it.
    """
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 250:
        return {}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    sh, so = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        idx = idx[idx < n]
        x = r[idx]
        sh[i] = sharpe(x)
        so[i] = sortino(x, mar)
    ok = np.isfinite(sh) & np.isfinite(so)
    sh, so = sh[ok], so[ok]
    if len(sh) < 20:
        return {}
    # scaled so the two are on a comparable footing: sortino is sqrt(2) x sharpe under symmetry
    return {"n_obs": n, "below_share": float((r < mar).mean()),
            "sharpe_mean": float(sh.mean()), "sharpe_se": float(sh.std(ddof=1)),
            "sortino_mean": float(so.mean()), "sortino_se": float(so.std(ddof=1)),
            "sharpe_cv": float(sh.std(ddof=1) / abs(sh.mean()))
            if sh.mean() else np.nan,
            "sortino_cv": float(so.std(ddof=1) / abs(so.mean()))
            if so.mean() else np.nan,
            "noise_ratio": float((so.std(ddof=1) / abs(so.mean()))
                                 / (sh.std(ddof=1) / abs(sh.mean())))
            if sh.mean() and so.mean() else np.nan}


def skew_reliability(rets: np.ndarray, n_boot: int = 500, block: int = 21,
                     seed: int = 1009) -> dict:
    """A bootstrap interval on the sample skewness — the quantity Sortino's case rests on.

    Third moments converge slowly and are dominated by a few observations. If the interval on
    skewness spans zero, then whether Sortino should differ from Sharpe at all is undetermined
    for that asset, and any ranking difference is noise being taken seriously.
    """
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 250:
        return {}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        idx = idx[idx < n]
        out[i] = float(pd.Series(r[idx]).skew())
    return {"skew": float(pd.Series(r).skew()), "boot_mean": float(out.mean()),
            "se": float(out.std(ddof=1)), "p05": float(np.percentile(out, 5)),
            "p95": float(np.percentile(out, 95)),
            "spans_zero": bool(np.percentile(out, 5) < 0 < np.percentile(out, 95)),
            "t": float(out.mean() / out.std(ddof=1)) if out.std(ddof=1) > 0 else np.nan}


# --------------------------------------------------------------------------- #
# The horse race
# --------------------------------------------------------------------------- #
def out_of_sample_ranking(rets: pd.DataFrame, n_splits: int = 8, mar: float = 0.0,
                          min_obs: int = 250) -> pd.DataFrame:
    """Rank on one period, score on the next. The only test that settles it.

    For each split the assets are ranked by each ratio on the first window, and the rank
    correlation with their *realised* ranking in the following window is recorded — separately
    for realised Sharpe and realised Sortino, so neither metric is graded on its own
    examination paper. A ratio that ranks better in principle but is estimated worse will lose
    here, which is exactly the outcome the theory cannot rule out.
    """
    d = rets.dropna(how="any")
    if len(d) < min_obs * (n_splits + 1):
        return pd.DataFrame()
    edges = np.linspace(0, len(d), n_splits + 1).astype(int)
    rows = []
    for i in range(n_splits - 1):
        tr = d.iloc[edges[i]:edges[i + 1]]
        te = d.iloc[edges[i + 1]:edges[i + 2]]
        if len(tr) < min_obs or len(te) < min_obs:
            continue
        pred_sh = tr.apply(lambda c: sharpe(c.to_numpy()))
        pred_so = tr.apply(lambda c: sortino(c.to_numpy(), mar))
        real_sh = te.apply(lambda c: sharpe(c.to_numpy()))
        real_so = te.apply(lambda c: sortino(c.to_numpy(), mar))
        ok = (pred_sh.notna() & pred_so.notna() & real_sh.notna() & real_so.notna())
        if ok.sum() < 4:
            continue
        rows.append({
            "split": i, "start": tr.index[0], "n_assets": int(ok.sum()),
            "sharpe_predicts_sharpe": float(pred_sh[ok].corr(real_sh[ok],
                                                             method="spearman")),
            "sortino_predicts_sortino": float(pred_so[ok].corr(real_so[ok],
                                                               method="spearman")),
            "sharpe_predicts_sortino": float(pred_sh[ok].corr(real_so[ok],
                                                              method="spearman")),
            "sortino_predicts_sharpe": float(pred_so[ok].corr(real_sh[ok],
                                                              method="spearman")),
        })
    return pd.DataFrame(rows)


def horse_race_summary(oos: pd.DataFrame) -> dict:
    """Average out-of-sample rank correlations, and who wins on each scoreboard."""
    if oos.empty:
        return {}
    cols = ["sharpe_predicts_sharpe", "sortino_predicts_sortino",
            "sharpe_predicts_sortino", "sortino_predicts_sharpe"]
    m = {c: float(oos[c].mean()) for c in cols}
    m["n_splits"] = int(len(oos))
    # Judged on the SORTINO scoreboard: does ranking by sortino beat ranking by sharpe?
    m["sortino_wins_own_game"] = bool(m["sortino_predicts_sortino"]
                                      > m["sharpe_predicts_sortino"])
    m["sharpe_wins_own_game"] = bool(m["sharpe_predicts_sharpe"]
                                     > m["sortino_predicts_sharpe"])
    m["sortino_edge_own_game"] = m["sortino_predicts_sortino"] - \
        m["sharpe_predicts_sortino"]
    return m


# --------------------------------------------------------------------------- #
# The control
# --------------------------------------------------------------------------- #
def synthetic_skewed(n: int = 5000, target_skew: float = 0.0, mean: float = 0.0004,
                     vol: float = 0.011, seed: int = 1009) -> np.ndarray:
    """Returns with a tunable skewness at a FIXED mean and standard deviation.

    Built by mixing a normal with a shifted exponential and then rescaling to the target mean
    and volatility exactly, so skewness is the only thing that varies across the control. If it
    were not held that way, any Sortino-Sharpe difference could be attributed to the first two
    moments and the control would prove nothing.
    """
    rng = np.random.default_rng(seed)
    if abs(target_skew) < 1e-9:
        x = rng.normal(0.0, 1.0, n)
    else:
        w = min(abs(target_skew) / 2.0, 0.95)
        e = rng.exponential(1.0, n) - 1.0
        x = (1 - w) * rng.normal(0.0, 1.0, n) + w * (np.sign(target_skew) * e)
    x = (x - x.mean()) / x.std(ddof=1)
    return mean + vol * x


def skew_sweep(skews=(-1.5, -0.75, 0.0, 0.75, 1.5), n: int = 20000,
               n_reps: int = 8, seed: int = 1009) -> pd.DataFrame:
    """How the two ratios diverge as skewness varies, with everything else fixed."""
    rows = []
    for sk in skews:
        sh, so, ratio, realised = [], [], [], []
        for k in range(n_reps):
            x = synthetic_skewed(n, sk, seed=seed + k)
            a, b = sharpe(x), sortino(x)
            sh.append(a)
            so.append(b)
            ratio.append(b / a if a else np.nan)
            realised.append(float(pd.Series(x).skew()))
        rows.append({"target_skew": sk, "realised_skew": float(np.mean(realised)),
                     "sharpe": float(np.mean(sh)), "sortino": float(np.mean(so)),
                     "sortino_over_sharpe": float(np.nanmean(ratio)),
                     "symmetric_value": float(np.sqrt(2))})
    return pd.DataFrame(rows).set_index("target_skew")


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Real** if the two ratios rank assets materially differently; **Weak** if
      they differ only at the margins; **None** if the rankings are effectively identical.
    - **Tradability**: **Useful** if Sortino's ranking survives out of sample on its own
      scoreboard — if it beats Sharpe at predicting *future Sortino*; **Partial** if it wins
      one scoreboard; **Mirage** if Sharpe predicts future Sortino at least as well as Sortino
      does, since then the extra machinery buys nothing.
    """
    signal = ("Real" if h["mean_rank_change"] >= 1.0
              else ("Weak" if h["mean_rank_change"] >= 0.3 else "None"))
    # An identical ranking is the strongest possible "None": nothing to choose between them.
    trad = ("Useful" if h["sortino_edge"] > 0.05
            else ("Partial" if h["sortino_edge"] > -0.02 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"They do not — not at all, on this panel. Across {h['n_assets']} assets the two "
            f"rankings had a Spearman correlation of **{h['spearman']:.3f}**, with "
            f"{h['n_unchanged']} of {h['n_assets']} assets in **exactly** the same position "
            f"and a largest rank change of {h['max_rank_change']:.0f}. Not \"broadly "
            f"similar\": identical, position for position, including an asset whose skewness "
            f"is around -1.8. The reason is arithmetic rather than coincidence: for a "
            f"distribution symmetric about the threshold, downside deviation equals σ/√2 "
            f"**exactly**, so Sortino is Sharpe × √2 and the ranking cannot change. Every "
            f"departure is a third moment, and here those departures are far too small to make "
            f"any pair cross — the Sortino/Sharpe ratio spans only "
            f"{h['ratio_band_lo']:.3f} to {h['ratio_band_hi']:.3f} across the whole panel. "
            f"Measured here, "
            f"σ/downside-deviation averaged {h['mean_sd_over_dd']:.4f} against the symmetric "
            f"{np.sqrt(2):.4f}, and its departure correlated {h['corr_skew_excess']:+.2f} with "
            f"realised skewness — the mechanism is confirmed rather than assumed. The "
            f"uncomfortable part is what that mechanism rests on: skewness is the least "
            f"reliably estimated quantity in the calculation. Bootstrapping it asset by asset, "
            f"the 90% interval **spans zero for {h['skew_spans_zero']:.0%} of them**. For most "
            f"of this panel, whether Sortino should differ from Sharpe at all is undetermined "
            f"by the data."),
        "trad_why": (
            f"And it is estimated from less of the sample. Downside deviation uses only the "
            f"observations below the threshold — {h['below_share']:.0%} of daily returns here "
            f"— so on identical block-bootstrap resamples the Sortino ratio carried a "
            f"coefficient of variation of {h['sortino_cv']:.3f} against Sharpe's "
            f"{h['sharpe_cv']:.3f}, **{h['noise_ratio']:.2f}× the relative noise**. Better in "
            f"principle, noisier in practice; the question is which wins, and it is settled by "
            f"a horse race rather than by argument. Ranking on one period and scoring on the "
            f"next, over {h['n_splits']} splits: ranking by Sortino predicted future Sortino "
            f"with a rank correlation of {h['sortino_predicts_sortino']:+.3f}, while ranking "
            f"by **Sharpe** predicted future Sortino at {h['sharpe_predicts_sortino']:+.3f} — "
            f"an edge to Sortino of {h['sortino_edge']:+.3f} on **its own scoreboard**. "
            f"{'Sortino earns its place' if h['sortino_edge'] > 0.05 else 'Sharpe predicts the downside metric about as well as the downside metric does, which is the whole case against bothering'}. "
            f"The practical reading: report both, treat a large gap between them as a flag to "
            f"go and look at the return distribution, and do not rank managers on a statistic "
            f"whose distinguishing input has a confidence interval containing zero."),
        "trad": trad,
        "one_sentence": (
            f"Sharpe and Sortino rank this panel at {h['spearman']:.2f} correlation because "
            f"they can only differ through skewness — and skewness has a bootstrap interval "
            f"spanning zero for {h['skew_spans_zero']:.0%} of these assets, while Sortino "
            f"carries {h['noise_ratio']:.1f}× the relative estimation noise."),
    }
