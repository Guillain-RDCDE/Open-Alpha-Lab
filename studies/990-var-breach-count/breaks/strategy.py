"""Backtesting Value-at-Risk by counting — Study 990.

VaR is the rare risk measure that makes a checkable promise. At level ``p``, the realised loss
should exceed the forecast on a fraction ``1-p`` of days. Two things follow, and standard
practice reliably gets the second one wrong:

1. **Unconditional coverage.** The breach *rate* should be ``1-p``. Kupiec (1995) gives the
   likelihood-ratio test. This is the test everybody runs.

2. **Independence.** The breaches should also be *spread out*. A model that produces exactly
   2.5 breaches a year, all in the same fortnight, is useless and passes Kupiec perfectly.
   Christoffersen (1998) tests whether a breach today predicts a breach tomorrow, and the joint
   test combines both. This is the test almost nobody runs, and it is the one that
   distinguishes a model that knows about volatility clustering from one that does not.

The five models graded here span the standard practice:

- ``historical`` — the empirical quantile of a rolling window. Assumption-free about shape,
  totally blind to current volatility.
- ``normal`` — mean plus *z* times rolling standard deviation. The one that fails hardest.
- ``student_t`` — the same with a fitted degrees-of-freedom, so the tails can be fat.
- ``ewma`` — RiskMetrics: an exponentially-weighted variance, reacting fast to new volatility.
- ``filtered_historical`` — Barone-Adesi's FHS: standardise by EWMA volatility, take the
  empirical quantile of the standardised residuals, rescale. Fat tails *and* conditioning.

The final section is about power. With 8,000 days and a 99% level you expect 80 breaches, and
the sampling noise on 80 events is large enough that a model breaching at 1.5% instead of 1.0%
is often not rejected. ``power_curve`` measures exactly how wrong a model has to be before the
tests notice — which is the number that tells you how much a "the model passed" statement is
worth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

TRADING_DAYS = 252
LEVELS = (0.95, 0.99)
MODELS = ("historical", "normal", "student_t", "ewma", "filtered_historical")


# --------------------------------------------------------------------------- #
# The five forecasters
# --------------------------------------------------------------------------- #
def var_historical(r: pd.Series, level: float = 0.99, window: int = 500) -> pd.Series:
    """Empirical quantile of the trailing window. Shifted so it uses no same-day data."""
    return (-r.rolling(window).quantile(1 - level)).shift(1).rename("var")


def var_normal(r: pd.Series, level: float = 0.99, window: int = 500) -> pd.Series:
    """Mean plus z-sigma. The standard model, and the one that under-forecasts the tail."""
    z = stats.norm.ppf(level)
    mu = r.rolling(window).mean()
    sd = r.rolling(window).std(ddof=1)
    return (-(mu - z * sd)).shift(1).rename("var")


def _fit_t_df(x: np.ndarray) -> float:
    """Fit Student-t degrees of freedom by maximum likelihood on standardised data."""
    x = x[np.isfinite(x)]
    if len(x) < 100:
        return 30.0
    s = x.std(ddof=1)
    if s <= 0:
        return 30.0
    z = (x - x.mean()) / s

    def nll(log_nu):
        nu = 2.05 + np.exp(log_nu)
        scale = np.sqrt((nu - 2) / nu)
        return -np.sum(stats.t.logpdf(z / scale, df=nu) - np.log(scale))

    try:
        res = optimize.minimize_scalar(nll, bounds=(-3.0, 4.0), method="bounded")
        return float(2.05 + np.exp(res.x))
    except Exception:
        return 30.0


def var_student_t(r: pd.Series, level: float = 0.99, window: int = 500,
                  refit: int = 63) -> pd.Series:
    """Normal's shape assumption relaxed: a fitted-t quantile times rolling volatility.

    The degrees of freedom are refitted every ``refit`` sessions rather than daily — the
    estimate barely moves day to day and refitting daily makes this the slowest model in the
    study for no gain in accuracy.
    """
    vals = pd.Series(np.nan, index=r.index)
    mu = r.rolling(window).mean()
    sd = r.rolling(window).std(ddof=1)
    nu = 30.0
    arr = r.to_numpy()
    for i in range(window, len(r)):
        if (i - window) % refit == 0:
            nu = _fit_t_df(arr[i - window:i])
        q = stats.t.ppf(1 - level, df=nu) * np.sqrt((nu - 2) / nu)
        vals.iloc[i] = -(mu.iloc[i] + q * sd.iloc[i])
    return vals.shift(1).rename("var")


def ewma_vol(r: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics exponentially-weighted volatility."""
    v = r.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(v).rename("ewma_vol")


def var_ewma(r: pd.Series, level: float = 0.99, lam: float = 0.94,
             window: int = 500) -> pd.Series:
    """RiskMetrics: a normal quantile on an exponentially-weighted variance."""
    z = stats.norm.ppf(level)
    out = (z * ewma_vol(r, lam)).shift(1).rename("var")
    out.iloc[:window] = np.nan
    return out


def var_filtered_historical(r: pd.Series, level: float = 0.99, lam: float = 0.94,
                            window: int = 500) -> pd.Series:
    """Filtered historical simulation (Barone-Adesi et al. 1999).

    Standardise returns by their EWMA volatility, take the empirical quantile of the
    *standardised* residuals over the trailing window, then rescale by today's volatility. Fat
    tails from the data and conditioning from the filter — the combination the other four
    models each get only half of.
    """
    vol = ewma_vol(r, lam)
    z = (r / vol.replace(0, np.nan))
    q = z.rolling(window).quantile(1 - level)
    return (-(q * vol)).shift(1).rename("var")


def build_var(r: pd.Series, model: str, level: float = 0.99, window: int = 500) -> pd.Series:
    """Dispatch to one of the five forecasters."""
    fns = {"historical": var_historical, "normal": var_normal,
           "student_t": var_student_t, "ewma": var_ewma,
           "filtered_historical": var_filtered_historical}
    if model not in fns:
        raise ValueError(f"unknown model {model!r}; expected one of {sorted(fns)}")
    if model in ("ewma", "filtered_historical"):
        return fns[model](r, level, window=window)
    return fns[model](r, level, window)


# --------------------------------------------------------------------------- #
# Counting, and the three tests
# --------------------------------------------------------------------------- #
def breaches(r: pd.Series, var: pd.Series) -> pd.Series:
    """1 on days the realised loss exceeded the forecast."""
    df = pd.concat([r.rename("r"), var.rename("v")], axis=1, sort=False).dropna()
    return (df["r"] < -df["v"]).astype(int).rename("breach")


def kupiec_test(b: pd.Series, level: float = 0.99) -> dict:
    """Unconditional coverage (Kupiec 1995): is the breach RATE right?

    Likelihood ratio against the binomial null, distributed chi-square with one degree of
    freedom. This is the test everyone runs, and it is blind to when the breaches happened.
    """
    n = int(len(b))
    x = int(b.sum())
    p = 1 - level
    if n < 100:
        return {"n": n, "breaches": x}
    phat = x / n
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        lr = -2 * ((n - x) * np.log((1 - p) / (1 - phat)) + x * np.log(p / phat))
    return {"n": n, "breaches": x, "expected": n * p, "rate": phat, "target": p,
            "lr": float(lr), "p_value": float(1 - stats.chi2.cdf(lr, 1)),
            "reject_5pct": bool(1 - stats.chi2.cdf(lr, 1) < 0.05)}


def christoffersen_independence(b: pd.Series) -> dict:
    """Are the breaches independent, or do they cluster? (Christoffersen 1998)

    The test everybody skips. A model can breach at exactly the promised rate and still be
    worthless if every breach arrives in the same week — which is precisely what an
    unconditional model does in a volatility cluster.
    """
    v = b.to_numpy()
    n = len(v)
    if n < 100:
        return {"n": int(n)}
    prev, cur = v[:-1], v[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return {"n": int(n), "n00": n00, "n01": n01, "n10": n10, "n11": n11,
                "lr": 0.0, "p_value": 1.0, "reject_5pct": False,
                "pi01": np.nan, "pi11": np.nan}
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def ll(p_, k, m):
        if p_ <= 0 or p_ >= 1:
            return 0.0
        return k * np.log(p_) + m * np.log(1 - p_)

    l_null = ll(pi, n01 + n11, n00 + n10)
    l_alt = ll(pi01, n01, n00) + ll(pi11, n11, n10)
    lr = -2 * (l_null - l_alt)
    return {"n": int(n), "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "pi01": float(pi01), "pi11": float(pi11), "lr": float(lr),
            "p_value": float(1 - stats.chi2.cdf(lr, 1)),
            "reject_5pct": bool(1 - stats.chi2.cdf(lr, 1) < 0.05)}


def joint_test(b: pd.Series, level: float = 0.99) -> dict:
    """Christoffersen's conditional coverage: the two LRs added, chi-square with 2 df."""
    k = kupiec_test(b, level)
    c = christoffersen_independence(b)
    if "lr" not in k or "lr" not in c:
        return {"n": k.get("n", 0)}
    lr = k["lr"] + c["lr"]
    return {"n": k["n"], "lr": float(lr), "p_value": float(1 - stats.chi2.cdf(lr, 2)),
            "reject_5pct": bool(1 - stats.chi2.cdf(lr, 2) < 0.05)}


def worst_breach_stats(r: pd.Series, var: pd.Series) -> dict:
    """How bad were the breaches when they happened?

    A model can pass every coverage test and still be dangerous if, on the days it is wrong, it
    is wrong by a factor of three. This is the expected-shortfall side of the question, which
    breach counting alone cannot see.
    """
    df = pd.concat([r.rename("r"), var.rename("v")], axis=1, sort=False).dropna()
    hit = df[df["r"] < -df["v"]]
    if len(hit) < 5:
        return {"n_breaches": int(len(hit))}
    excess = (-hit["r"] - hit["v"]) / hit["v"]
    return {"n_breaches": int(len(hit)),
            "mean_excess_pct_of_var": float(excess.mean()),
            "max_excess_pct_of_var": float(excess.max()),
            "worst_loss": float(hit["r"].min()),
            "var_on_worst_day": float(hit.loc[hit["r"].idxmin(), "v"]),
            "mean_breach_loss": float(hit["r"].mean())}


def max_consecutive(b: pd.Series) -> int:
    """The longest run of consecutive breaches — the clustering everyone can understand."""
    v = b.to_numpy()
    best = run = 0
    for x in v:
        run = run + 1 if x else 0
        best = max(best, run)
    return int(best)


def grade_model(r: pd.Series, model: str, level: float = 0.99,
                window: int = 500) -> dict:
    """One model, one asset, one level: everything in a single row."""
    var = build_var(r, model, level, window)
    b = breaches(r, var)
    k = kupiec_test(b, level)
    c = christoffersen_independence(b)
    j = joint_test(b, level)
    w = worst_breach_stats(r, var)
    return {"model": model, "level": level, "n": k.get("n", 0),
            "breaches": k.get("breaches", 0), "expected": k.get("expected", np.nan),
            "rate": k.get("rate", np.nan),
            "kupiec_p": k.get("p_value", np.nan),
            "independence_p": c.get("p_value", np.nan),
            "joint_p": j.get("p_value", np.nan),
            "max_consecutive": max_consecutive(b),
            "mean_excess": w.get("mean_excess_pct_of_var", np.nan),
            "worst_loss": w.get("worst_loss", np.nan),
            "mean_var": float(var.dropna().mean())}


def grade_all(r: pd.Series, level: float = 0.99, window: int = 500,
              models=MODELS) -> pd.DataFrame:
    """Every model on one asset."""
    return pd.DataFrame([grade_model(r, m, level, window) for m in models]).set_index("model")


# --------------------------------------------------------------------------- #
# How much is a passing grade worth?
# --------------------------------------------------------------------------- #
def power_curve(n_days: int, level: float = 0.99, true_rates=None,
                n_sims: int = 2000, seed: int = 990) -> pd.DataFrame:
    """How wrong must a model be before Kupiec notices?

    Simulate independent breaches at a *true* rate that differs from the promised one, and
    count how often the test rejects. If a model breaching at 1.5% against a 1% promise is
    caught only a third of the time, then "the model passed its backtest" is a much weaker
    statement than it sounds.
    """
    p = 1 - level
    true_rates = true_rates or (p, p * 1.25, p * 1.5, p * 2.0, p * 3.0, p * 0.5)
    rng = np.random.default_rng(seed)
    rows = []
    for tr in true_rates:
        rejects = 0
        for _ in range(n_sims):
            b = pd.Series(rng.random(n_days) < tr, dtype=int)
            k = kupiec_test(b, level)
            rejects += bool(k.get("reject_5pct", False))
        rows.append({"true_rate": tr, "ratio_to_promised": tr / p,
                     "reject_rate": rejects / n_sims,
                     "expected_breaches": n_days * tr})
    return pd.DataFrame(rows).set_index("true_rate")


def days_needed(level: float = 0.99, misstatement: float = 1.5, power: float = 0.8,
                alpha: float = 0.05) -> int:
    """Sessions required to detect a model breaching ``misstatement`` times too often.

    A normal approximation to the binomial power calculation. It is the number that should
    appear beside every VaR backtest and never does — because for a 99% model it is usually
    measured in decades.
    """
    p0 = 1 - level
    p1 = p0 * misstatement
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    num = (z_a * np.sqrt(p0 * (1 - p0)) + z_b * np.sqrt(p1 * (1 - p1))) ** 2
    return int(np.ceil(num / (p1 - p0) ** 2))


def synthetic_returns(n: int = 5000, df_t: float = 4.0, clustering: float = 0.0,
                      base_vol: float = 0.01, seed: int = 990) -> pd.Series:
    """Returns with independently controllable tail fatness and volatility clustering.

    ``df_t`` large and ``clustering`` zero gives i.i.d. normal returns, where the normal VaR
    model is *correct* and must pass every test. Lowering ``df_t`` breaks the normal model's
    shape; raising ``clustering`` breaks its conditioning. Two different failures that produce
    the same symptom in a breach count, which is why the study needs both knobs.
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
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(z * vol, index=idx, name="ret")


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Confirmed** (the promise is broken) if the *normal* model's breach rate is
      significantly above target on a majority of assets at 99% **and** the independence test
      rejects for it too — that is, both the shape and the conditioning fail; **Partial** if
      only one fails; **Busted** if the standard model is actually well calibrated.
    - **Tradability**: **Useful** if the best model passes the joint test on a majority of
      assets — meaning a practitioner has something better to switch to; **Partial** if it beats
      the normal model without passing; **Mirage** if nothing passes.
    """
    shape_fails = h["normal_reject_share"] > 0.5
    clustering_fails = h["normal_indep_reject_share"] > 0.5
    signal = ("Confirmed" if (shape_fails and clustering_fails)
              else ("Partial" if (shape_fails or clustering_fails) else "Busted"))
    if h["best_joint_pass_share"] > 0.5:
        trad = "Useful"
    elif h["best_breach_error"] < h["normal_breach_error"]:
        trad = "Partial"
    else:
        trad = "Mirage"
    return {
        "signal": signal,
        "signal_why": (
            f"At 99% confidence over {h['n_assets']} assets, the textbook **normal** VaR model "
            f"breached on **{h['normal_rate']:.2%}** of days against a promised 1.00% — "
            f"{h['normal_rate'] / 0.01:.1f}× too often — and Kupiec's coverage test rejected it "
            f"on **{h['normal_reject_share']:.0%}** of assets. Worse, and this is the part "
            f"standard practice never checks: the independence test rejected it on "
            f"**{h['normal_indep_reject_share']:.0%}**, with runs of up to "
            f"{h['normal_max_consecutive']} consecutive breaches. Those are two different "
            f"failures — the distribution's shape is wrong *and* it does not know today's "
            f"volatility — and a breach count alone cannot tell them apart. The best model here "
            f"is **{h['best_model']}** at {h['best_rate']:.2%}, passing the joint test on "
            f"{h['best_joint_pass_share']:.0%} of assets."),
        "trad": trad,
        "trad_why": (
            f"Before switching models on this evidence, note how weak the evidence can be. With "
            f"{h['typical_n']:,} sessions at 99%, a model that breaches **50% too often** "
            f"(1.5% instead of 1.0%) is caught by Kupiec only **{h['power_at_1_5x']:.0%}** of "
            f"the time; detecting that reliably needs about **{h['days_for_1_5x']:,} sessions** "
            f"— {h['days_for_1_5x'] / 252:.0f} years. On the days the models were wrong they "
            f"were wrong by a lot: the normal model's average breach overshot its own forecast "
            f"by {h['normal_mean_excess']:.0%}, and its worst day lost "
            f"{abs(h['normal_worst_loss']):.1%} against a forecast of "
            f"{h['normal_var_that_day']:.1%}. Breach counting says nothing about that, which is "
            f"the argument for expected shortfall."),
        "one_sentence": (
            f"The normal VaR model breaches {h['normal_rate'] / 0.01:.1f}× too often and its "
            f"breaches cluster, while {h['best_model']} passes the joint test on "
            f"{h['best_joint_pass_share']:.0%} of assets — but with only "
            f"{h['typical_n'] * 0.01:.0f} expected breaches to count, the test that says so "
            f"would miss a 50%-too-loose model {1 - h['power_at_1_5x']:.0%} of the time."),
    }
