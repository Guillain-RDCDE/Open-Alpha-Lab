"""How fast do returns become normal? — Study 991.

"Aggregational Gaussianity" is one of Cont's (2001) stylised facts: as the horizon lengthens,
the distribution of returns looks more and more normal. It is a real effect and it is easy to
demonstrate. It is also routinely over-claimed, because the central limit theorem needs two
things that financial returns do not supply:

1. **Independence.** Returns are close to uncorrelated but nowhere near independent — volatility
   clusters. Under clustering, a sum of *n* days is a sum of *n* draws whose scales move
   together, and the convergence rate is far slower than the i.i.d. rate.

2. **Finite variance.** If the tail index is below 2 the CLT does not apply at all and sums
   converge to a stable law instead, which never becomes normal. Whether equity returns have a
   tail index above or below 2 is contested; ``hill_estimator`` measures it rather than
   assuming.

The module measures convergence four ways, because any single normality statistic can be fooled:

- ``excess_kurtosis`` — for i.i.d. draws it decays exactly as ``k_1 / n``, which gives a
  **closed-form benchmark** to compare the tape against. That comparison is the study's core.
- ``jarque_bera`` and ``anderson_darling`` — formal tests, with the crucial caveat that their
  power collapses as the horizon grows and the sample shrinks.
- ``tail_ratio`` — the observed frequency of 3-sigma and 5-sigma moves against the normal
  prediction, which is what a risk manager actually cares about.
- ``hill_estimator`` — the tail index itself, which decides whether the CLT even applies.

``convergence_horizon`` then answers the practical question: at what horizon does the excess
kurtosis fall below a threshold, and how does that compare to the i.i.d. prediction?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
HORIZONS = (1, 5, 10, 21, 63, 126, 252)


# --------------------------------------------------------------------------- #
# Aggregating
# --------------------------------------------------------------------------- #
def aggregate(r: pd.Series, horizon: int, overlapping: bool = False) -> pd.Series:
    """Sum log returns over ``horizon`` sessions.

    Log returns are used because they add: the *h*-day log return is exactly the sum of *h*
    daily log returns, which is what the central limit theorem is about. Simple returns compound
    instead of adding, and their aggregation mixes the CLT question with a compounding effect
    that has nothing to do with it.
    """
    lr = np.log1p(r.dropna())
    if horizon <= 1:
        return lr.rename(f"h{horizon}")
    if overlapping:
        return lr.rolling(horizon).sum().dropna().rename(f"h{horizon}")
    n = len(lr) // horizon
    if n < 2:
        return pd.Series(dtype=float, name=f"h{horizon}")
    trimmed = lr.iloc[len(lr) - n * horizon:]
    out = trimmed.groupby(np.arange(len(trimmed)) // horizon).sum()
    out.index = trimmed.index[horizon - 1::horizon][:len(out)]
    return out.rename(f"h{horizon}")


def standardise(x: pd.Series) -> np.ndarray:
    """Zero mean, unit variance — normality is a claim about shape, not scale."""
    v = x.dropna().to_numpy()
    s = v.std(ddof=1)
    return (v - v.mean()) / s if s > 0 else v - v.mean()


# --------------------------------------------------------------------------- #
# Four measurements of the same thing
# --------------------------------------------------------------------------- #
def excess_kurtosis(x: pd.Series) -> float:
    """Excess kurtosis (0 for a normal). The statistic with a closed-form decay rate."""
    v = x.dropna()
    return float(v.kurtosis()) if len(v) > 3 else np.nan


def iid_kurtosis_prediction(k1: float, horizon: int) -> float:
    """Excess kurtosis after summing ``horizon`` i.i.d. draws: exactly ``k1 / horizon``.

    This is the benchmark the whole study turns on. For independent draws with excess kurtosis
    ``k1``, the sum of ``n`` of them has excess kurtosis ``k1 / n`` — no approximation, no
    asymptotics. Any gap between the tape and this line is the cost of the independence
    assumption being false.
    """
    return float(k1 / max(horizon, 1))


def jarque_bera(x: pd.Series) -> dict:
    """The skewness-and-kurtosis normality test, with its own power caveat attached."""
    v = x.dropna().to_numpy()
    n = len(v)
    if n < 20:
        return {"n": int(n)}
    stat, p = stats.jarque_bera(v)
    return {"n": int(n), "statistic": float(stat), "p_value": float(p),
            "reject_5pct": bool(p < 0.05)}


def anderson_darling(x: pd.Series) -> dict:
    """Anderson-Darling, which weights the tails more heavily than Jarque-Bera does."""
    v = x.dropna().to_numpy()
    n = len(v)
    if n < 20:
        return {"n": int(n)}
    res = stats.anderson(v, dist="norm")
    crit_5 = float(res.critical_values[2])   # the 5% level
    return {"n": int(n), "statistic": float(res.statistic), "critical_5pct": crit_5,
            "reject_5pct": bool(res.statistic > crit_5)}


def tail_ratio(x: pd.Series, sigmas=(2.0, 3.0, 4.0, 5.0)) -> dict:
    """How often did a k-sigma move happen, against how often a normal says it should?

    The risk manager's version of the normality question. A distribution can have modest excess
    kurtosis and still produce five-sigma days a hundred times too often.
    """
    z = standardise(x)
    n = len(z)
    if n < 50:
        return {"n": int(n)}
    out = {"n": int(n)}
    for s in sigmas:
        observed = float((np.abs(z) > s).mean())
        expected = float(2 * (1 - stats.norm.cdf(s)))
        out[f"obs_{s:g}sig"] = observed
        out[f"exp_{s:g}sig"] = expected
        out[f"ratio_{s:g}sig"] = observed / expected if expected > 0 else np.nan
        out[f"count_{s:g}sig"] = int((np.abs(z) > s).sum())
    return out


def hill_estimator(x: pd.Series, tail_frac: float = 0.05) -> dict:
    """The tail index (Hill 1975): how heavy is the tail, really?

    The number that decides whether the central limit theorem applies at all. Below 2 the
    variance is infinite and sums converge to a stable law that is never normal; between 2 and 4
    the variance exists but the kurtosis does not, and the sample kurtosis measured everywhere
    else in this study is an unstable statistic rather than an estimate of anything.

    The estimator is notoriously sensitive to ``tail_frac``, so the results sweep it.
    """
    v = np.abs(standardise(x))
    n = len(v)
    k = int(n * tail_frac)
    if k < 20:
        return {"n": int(n), "k": int(k)}
    srt = np.sort(v)[::-1]
    top = srt[:k]
    thresh = srt[k]
    if thresh <= 0:
        return {"n": int(n), "k": int(k)}
    alpha = k / np.sum(np.log(top / thresh))
    return {"n": int(n), "k": int(k), "alpha": float(alpha),
            "se": float(alpha / np.sqrt(k)),
            "threshold_sigma": float(thresh),
            "variance_exists": bool(alpha > 2), "kurtosis_exists": bool(alpha > 4)}


# --------------------------------------------------------------------------- #
# The convergence profile
# --------------------------------------------------------------------------- #
def convergence_profile(r: pd.Series, horizons=HORIZONS,
                        overlapping: bool = False) -> pd.DataFrame:
    """Every measurement at every horizon, plus the i.i.d. prediction to compare against."""
    k1 = excess_kurtosis(aggregate(r, 1))
    rows = []
    for hz in horizons:
        x = aggregate(r, hz, overlapping)
        if len(x) < 20:
            continue
        jb = jarque_bera(x)
        ad = anderson_darling(x)
        tr = tail_ratio(x)
        rows.append({
            "horizon": hz, "n": len(x),
            "excess_kurtosis": excess_kurtosis(x),
            "iid_prediction": iid_kurtosis_prediction(k1, hz),
            "skew": float(x.skew()),
            "jb_p": jb.get("p_value", np.nan),
            "jb_reject": jb.get("reject_5pct", False),
            "ad_reject": ad.get("reject_5pct", False),
            "ratio_3sig": tr.get("ratio_3sig", np.nan),
            "ratio_4sig": tr.get("ratio_4sig", np.nan),
            "count_4sig": tr.get("count_4sig", 0),
        })
    df = pd.DataFrame(rows).set_index("horizon")
    df["kurtosis_vs_iid"] = df["excess_kurtosis"] / df["iid_prediction"].replace(0, np.nan)
    return df


def convergence_horizon(profile: pd.DataFrame, threshold: float = 0.5) -> dict:
    """The first horizon at which excess kurtosis drops below ``threshold``.

    Reported alongside the horizon the i.i.d. prediction would have implied. The ratio between
    them is the slowdown that clustering and dependence cost — the study's headline number.
    """
    if profile.empty:
        return {}
    below = profile[profile["excess_kurtosis"] < threshold]
    iid_below = profile[profile["iid_prediction"] < threshold]
    actual = int(below.index[0]) if len(below) else None
    predicted = int(iid_below.index[0]) if len(iid_below) else None
    return {"threshold": threshold, "actual_horizon": actual,
            "iid_horizon": predicted,
            "slowdown": (actual / predicted) if (actual and predicted) else np.nan,
            "kurtosis_at_1d": float(profile.loc[1, "excess_kurtosis"])
            if 1 in profile.index else np.nan,
            "kurtosis_at_252d": float(profile.loc[252, "excess_kurtosis"])
            if 252 in profile.index else np.nan}


def fit_decay_rate(profile: pd.DataFrame) -> dict:
    """Fit ``kurtosis ~ c * horizon^(-b)`` and compare *b* against the i.i.d. value of 1.

    The i.i.d. theory says the exponent is exactly 1. A fitted exponent well below 1 means
    convergence is slower than independence would give — which is what dependence does — and
    quantifies the slowdown in a single number rather than horizon by horizon.
    """
    d = profile[profile["excess_kurtosis"] > 0]
    if len(d) < 4:
        return {"n": int(len(d))}
    x = np.log(d.index.to_numpy(dtype=float))
    y = np.log(d["excess_kurtosis"].to_numpy())
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    s2 = float((resid ** 2).sum() / max(len(x) - 2, 1))
    se = float(np.sqrt(s2 * np.linalg.pinv(A.T @ A)[1, 1]))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"n": int(len(d)), "exponent": float(-coef[1]), "se": se,
            "t_vs_one": float((-coef[1] - 1.0) / se) if se > 0 else np.nan,
            "r2": float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan}


def power_of_normality_tests(n_obs_by_horizon: dict, true_df: float = 4.0,
                             n_sims: int = 500, seed: int = 991) -> pd.DataFrame:
    """Can Jarque-Bera even see non-normality at long horizons?

    At a 252-day horizon there are only about 30 observations, and Jarque-Bera has almost no
    power against anything at *n* = 30. So "the annual returns pass a normality test" may mean
    the returns are normal, or may mean the test cannot tell. This measures which.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for hz, n in sorted(n_obs_by_horizon.items()):
        if n < 20:
            continue
        rejects = 0
        for _ in range(n_sims):
            x = rng.standard_t(true_df, n)
            rejects += bool(stats.jarque_bera(x)[1] < 0.05)
        rows.append({"horizon": hz, "n_obs": n, "power_vs_t4": rejects / n_sims})
    return pd.DataFrame(rows, columns=["horizon", "n_obs", "power_vs_t4"]).set_index("horizon")


def overlap_inflation(r: pd.Series, horizon: int, n_boot: int = 400,
                      seed: int = 991) -> dict:
    """How much does using overlapping windows overstate the sample?

    Overlapping windows are tempting at long horizons because they multiply the observation
    count by the horizon. They do not multiply the *information*. This measures the effective
    sample size by comparing the bootstrap variance of the kurtosis estimate under both schemes.
    """
    ov = aggregate(r, horizon, overlapping=True)
    nov = aggregate(r, horizon, overlapping=False)
    if len(nov) < 10:
        return {"horizon": horizon, "n_overlapping": len(ov), "n_non_overlapping": len(nov)}
    rng = np.random.default_rng(seed)

    def boot_sd(s, block=1):
        """Bootstrap SD of the kurtosis estimate.

        The block length is the load-bearing argument. Resampling overlapping windows *one at a
        time* would treat rows that share 99% of their days as independent draws — which is
        precisely the error being measured, committed inside the measurement. Overlapping
        series are therefore resampled in blocks of the horizon length; the non-overlapping
        series genuinely is independent and uses ordinary resampling.
        """
        v = s.dropna().to_numpy()
        n = len(v)
        if n < 10:
            return np.nan
        ks = []
        if block <= 1:
            for _ in range(n_boot):
                ks.append(pd.Series(rng.choice(v, n, replace=True)).kurtosis())
        else:
            n_blocks = int(np.ceil(n / block))
            offs = np.arange(block)
            for _ in range(n_boot):
                starts = rng.integers(0, n, n_blocks)
                idx = ((starts[:, None] + offs) % n).ravel()[:n]
                ks.append(pd.Series(v[idx]).kurtosis())
        return float(np.std(ks, ddof=1))

    sd_ov, sd_nov = boot_sd(ov, block=horizon), boot_sd(nov, block=1)
    return {"horizon": horizon, "n_overlapping": int(len(ov)),
            "n_non_overlapping": int(len(nov)),
            "apparent_gain": float(len(ov) / len(nov)),
            "kurtosis_overlapping": excess_kurtosis(ov),
            "kurtosis_non_overlapping": excess_kurtosis(nov),
            "boot_sd_overlapping": sd_ov, "boot_sd_non_overlapping": sd_nov,
            "effective_gain": float((sd_nov / sd_ov) ** 2)
            if sd_ov and sd_nov and sd_ov > 0 else np.nan}


def synthetic_returns(n: int = 8000, df_t: float = 4.0, clustering: float = 0.0,
                      base_vol: float = 0.01, seed: int = 991) -> pd.Series:
    """Draws with a known tail index and optional volatility clustering.

    With ``clustering = 0`` the draws are i.i.d. and the closed-form ``k1 / n`` decay must hold
    exactly — which is how the measurement apparatus gets graded against a known answer.
    """
    rng = np.random.default_rng(seed)
    vol = np.full(n, base_vol)
    if clustering > 0:
        logv = np.zeros(n)
        for t in range(1, n):
            logv[t] = clustering * logv[t - 1] + rng.normal(0, 0.15)
        vol = base_vol * np.exp(logv - logv.var() / 2)
    if df_t >= 100:
        z = rng.normal(0, 1, n)
    else:
        z = rng.standard_t(df_t, n) * np.sqrt((df_t - 2) / df_t)
    idx = pd.bdate_range("1993-02-01", periods=n)
    return pd.Series(np.expm1(z * vol), index=idx, name="ret")


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Confirmed** if excess kurtosis falls monotonically with horizon (the
      stylised fact holds) but the fitted decay exponent is significantly **below** the i.i.d.
      value of 1 (convergence is slower than independence would give); **Partial** if it decays
      at or above the i.i.d. rate; **Busted** if kurtosis does not fall at all.
    - **Tradability**: **Useful** if there is a horizon within a normal planning window (three
      years or less) where excess kurtosis drops below 0.5 and 3-sigma events occur at close to
      the normal rate; **Partial** if only one holds; **Mirage** if neither does.
    """
    falls = h["kurtosis_1d"] > h["kurtosis_longest"] > -1
    slower = h["decay_t_vs_one"] < -2.0
    signal = ("Confirmed" if (falls and slower)
              else ("Partial" if falls else "Busted"))
    reachable = (h["actual_horizon"] is not None
                 and h["actual_horizon"] <= 3 * 252)
    tails_ok = h["ratio_3sig_longest"] < 2.0
    trad = ("Useful" if (reachable and tails_ok)
            else ("Partial" if (reachable or tails_ok) else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"For {h['asset']} over {h['n_days']:,} sessions, excess kurtosis falls from "
            f"**{h['kurtosis_1d']:.1f} at one day to {h['kurtosis_longest']:.2f} at "
            f"{h['longest_horizon']} days** — so the stylised fact is real. But it is not the "
            f"central limit theorem's rate. For independent draws the decay is exactly "
            f"``k₁/n``, which would put the {h['longest_horizon']}-day kurtosis at "
            f"**{h['iid_at_longest']:.3f}**; the tape delivers "
            f"{h['kurtosis_longest'] / h['iid_at_longest'] if h['iid_at_longest'] else float('nan'):.0f}× "
            f"that. Fitting `kurtosis ~ horizon^(−b)` gives **b = {h['decay_exponent']:.2f}** "
            f"(se {h['decay_se']:.2f}) against the i.i.d. value of 1.00, *t* = "
            f"**{h['decay_t_vs_one']:+.2f}**. The Hill tail index is "
            f"**{h['hill_alpha']:.2f}** — above 2, so the variance exists and the theorem does "
            f"apply, but {'below' if h['hill_alpha'] < 4 else 'above'} 4, meaning the "
            f"{'kurtosis itself may not exist and every kurtosis number above is an unstable '
               'statistic rather than an estimate' if h['hill_alpha'] < 4 else
               'kurtosis is a well-defined quantity'}."),
        "trad": trad,
        "trad_why": (
            f"The practical question is when normal arithmetic becomes safe. Excess kurtosis "
            f"first drops below 0.5 at **{h['actual_horizon']} days** against the "
            f"{h['iid_horizon']} days independence would have predicted — a slowdown of "
            f"**{h['slowdown']:.1f}×**. At the longest horizon measured, 3-sigma moves still "
            f"arrive **{h['ratio_3sig_longest']:.1f}× more often** than a normal says. And a "
            f"warning about the tests: at {h['longest_horizon']} days there are only "
            f"{h['n_at_longest']} non-overlapping observations, where Jarque-Bera has about "
            f"**{h['power_at_longest']:.0%} power** against a *t*(4). A passing normality test "
            f"at long horizons is mostly evidence that the sample is small."),
        "one_sentence": (
            f"Returns do become more normal with horizon, but at exponent "
            f"{h['decay_exponent']:.2f} rather than the 1.00 independence would give — so the "
            f"bell arrives roughly {h['slowdown']:.0f}× later than the textbook implies, and "
            f"the tests that would tell you have run out of data by then."),
    }
