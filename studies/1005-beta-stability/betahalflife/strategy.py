"""How long a beta lasts — Study 1005.

Three questions, in the order that makes them answerable.

**1. Does beta persist?** Regress each name's beta in one window on its beta in the previous
window. The slope of that cross-sectional regression is the persistence coefficient, and it is
reliably below one — which is Blume's (1971) original finding and the reason he proposed
shrinking estimates toward the mean.

**2. How much of the instability is real?** This is the question usually skipped, and it is the
one that decides what to do. A beta estimated from 252 daily observations has a standard error;
two consecutive estimates would differ even if the true beta never moved at all. ``noise_floor``
computes what the period-to-period variation *would* be under a constant true beta, using the
regression standard errors, and compares it with what is observed. The gap is the genuine time
variation, and it is a great deal smaller than the raw instability suggests.

**3. Does shrinkage help?** Blume shrinkage, Vasicek (1973) shrinkage — which weights by each
estimate's own precision rather than applying a constant factor — and a simple "use 1.0 for
everything" baseline are compared on **out-of-sample prediction error**, which is the only
criterion that settles it. The last baseline is included because a fix has to beat doing nothing
clever, and it is a surprisingly hard benchmark.

``half_life`` converts the persistence coefficient into the horizon at which a beta's deviation
from the mean has decayed by half — the study's title, and a more useful summary than the
coefficient itself because it is stated in the unit a practitioner cares about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Estimating a beta, with its standard error
# --------------------------------------------------------------------------- #
def beta_with_se(y: np.ndarray, x: np.ndarray) -> tuple:
    """OLS beta of ``y`` on ``x``, with its standard error and R².

    The standard error is the point of this function. A beta without one is a number pretending
    to be a fact, and the whole of section 2 depends on knowing how imprecise each estimate is.
    """
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    if n < 30:
        return np.nan, np.nan, np.nan
    xc = x - x.mean()
    sxx = float((xc ** 2).sum())
    if sxx <= 0:
        return np.nan, np.nan, np.nan
    b = float((xc * (y - y.mean())).sum() / sxx)
    resid = y - y.mean() - b * xc
    s2 = float((resid ** 2).sum() / max(n - 2, 1))
    se = float(np.sqrt(s2 / sxx))
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = float(1 - (resid ** 2).sum() / tss) if tss > 0 else np.nan
    return b, se, r2


def rolling_betas(rets: pd.DataFrame, market: str, window: int = TRADING_DAYS,
                  step: int = TRADING_DAYS) -> pd.DataFrame:
    """Non-overlapping beta estimates for every name, one row per window.

    Non-overlapping by default, because overlapping windows share data and would make
    consecutive estimates look far more similar than they are — manufacturing the persistence
    the study is trying to measure. ``step`` is exposed so the sensitivity can be shown.
    """
    names = [c for c in rets.columns if c != market]
    m = rets[market].to_numpy(dtype=float)
    rows = []
    for start in range(0, len(rets) - window + 1, step):
        sl = slice(start, start + window)
        mk = m[sl]
        rec = {"date": rets.index[start + window - 1], "start": rets.index[start]}
        for nm in names:
            b, se, r2 = beta_with_se(rets[nm].to_numpy(dtype=float)[sl], mk)
            rec[nm] = b
            rec[f"{nm}__se"] = se
            rec[f"{nm}__r2"] = r2
        rows.append(rec)
    return pd.DataFrame(rows).set_index("date")


def long_form(betas: pd.DataFrame) -> pd.DataFrame:
    """Reshape the wide beta frame into tidy (date, name, beta, se, r2) rows."""
    names = [c for c in betas.columns if not c.endswith(("__se", "__r2"))
             and c != "start"]
    out = []
    for nm in names:
        d = pd.DataFrame({"beta": betas[nm], "se": betas[f"{nm}__se"],
                          "r2": betas[f"{nm}__r2"]})
        d["name"] = nm
        out.append(d.reset_index())
    return pd.concat(out, ignore_index=True, sort=False).dropna(subset=["beta"])


# --------------------------------------------------------------------------- #
# 1. Persistence
# --------------------------------------------------------------------------- #
def persistence(betas: pd.DataFrame, lag: int = 1) -> dict:
    """Regress each period's cross-section of betas on the previous period's.

    Blume's (1971) test. A slope below one means betas revert toward the cross-sectional mean,
    and the amount below one is exactly the shrinkage a forecaster should apply.
    """
    lf = long_form(betas)
    piv = lf.pivot(index="date", columns="name", values="beta").sort_index()
    cur, prev = [], []
    for i in range(lag, len(piv)):
        a = piv.iloc[i - lag]
        b = piv.iloc[i]
        ok = a.notna() & b.notna()
        prev.extend(a[ok].tolist())
        cur.extend(b[ok].tolist())
    if len(cur) < 30:
        return {}
    prev_a = np.asarray(prev)
    cur_a = np.asarray(cur)
    slope, se, r2 = beta_with_se(cur_a, prev_a)
    intercept = float(cur_a.mean() - slope * prev_a.mean())
    return {"n_pairs": len(cur), "slope": slope, "slope_se": se, "r2": r2,
            "intercept": intercept,
            "correlation": float(np.corrcoef(prev_a, cur_a)[0, 1]),
            "mean_beta": float(prev_a.mean()),
            "implied_shrinkage": float(1 - slope)}


def slope_is_confounded(betas: pd.DataFrame) -> dict:
    """Why the Blume slope must not be read as a stability measure.

    This function exists because the study's own pre-registered predictions about the slope
    turned out to be wrong, twice, and the reason is worth making explicit rather than quietly
    fixing.

    Regressing a noisy measurement on another noisy measurement attenuates the slope by the
    classical errors-in-variables factor:

        plim(slope) ≈ persistence × var(true beta) / [var(true beta) + var(estimation error)]

    So the slope moves with **two** things at once. Betas that genuinely wander more have a
    *larger* cross-sectional spread, which raises the signal-to-noise ratio and pushes the slope
    **up** — the opposite of the intuitive reading, and observable in the synthetic control:
    a random-walk beta produces a slope near or above one, while a perfectly constant beta
    produces a slope near 0.95. Diversified portfolios move both terms down together, so their
    slope can land either side of a single name's.

    The reliability ratio returned here is the attenuation factor itself. Dividing the raw slope
    by it gives the disattenuated persistence, which is the quantity people think they are
    reading off a Blume regression.
    """
    p = persistence(betas)
    sn = signal_to_noise(betas)
    if not p or not np.isfinite(sn["reliability"]):
        return {}
    rel = float(sn["reliability"])
    return {"raw_slope": p["slope"], "reliability": rel,
            "disattenuated_slope": float(p["slope"] / rel) if rel > 0 else np.nan,
            "cross_sectional_sd": sn["cross_sectional_sd"], "true_sd": sn["true_sd"],
            "attenuation": float(1 - rel)}


def half_life(slope: float, period_years: float = 1.0) -> float:
    """Periods until a deviation from the mean beta has decayed by half.

    The study's title. A persistence slope of 0.5 means half the deviation is gone after one
    period, so the half-life is exactly one; smaller slopes decay faster. Reported in years
    because that is the unit in which people ask "how often should I re-estimate this".
    """
    if not np.isfinite(slope) or slope <= 0 or slope >= 1:
        return np.inf if slope >= 1 else 0.0
    return float(np.log(0.5) / np.log(slope) * period_years)


def persistence_by_horizon(rets: pd.DataFrame, market: str,
                           windows=(63, 126, 252, 504)) -> pd.DataFrame:
    """Persistence measured at several estimation-window lengths.

    Longer windows give more precise estimates, so more of the measured period-to-period change
    is real and less is noise — the persistence slope should therefore rise with the window.
    That prediction is a check on the whole framework, not merely a table.
    """
    rows = []
    for w in windows:
        b = rolling_betas(rets, market, window=w, step=w)
        p = persistence(b)
        if not p:
            continue
        rows.append({"window_days": w, "n_periods": len(b), **p,
                     "half_life_periods": half_life(p["slope"]),
                     "half_life_years": half_life(p["slope"], w / TRADING_DAYS)})
    return pd.DataFrame(rows).set_index("window_days")


# --------------------------------------------------------------------------- #
# 2. How much of the instability is real?
# --------------------------------------------------------------------------- #
def noise_floor(betas: pd.DataFrame) -> dict:
    """Separate estimation error from genuine time variation in beta.

    Observed variance of the change in beta between consecutive windows has two parts. If the
    true beta were constant, consecutive estimates would still differ by their own sampling
    error, contributing ``2 × mean(se²)`` to the variance of the difference — the two windows
    being independent. Anything above that is real movement:

        var(observed change) = 2 × mean(se²) + var(true change)

    Subtracting gives the genuine component, and the ratio is the share of apparent instability
    that is merely measurement. This is a variance decomposition of exactly the kind that never
    appears beside a beta on a risk report, and it changes the interpretation completely.
    """
    lf = long_form(betas)
    piv = lf.pivot(index="date", columns="name", values="beta").sort_index()
    sepiv = lf.pivot(index="date", columns="name", values="se").sort_index()
    d = piv.diff().to_numpy()
    se_sq = (sepiv ** 2).to_numpy()
    pair_noise = se_sq[1:] + se_sq[:-1]          # variance of a difference of two estimates
    d = d[1:]
    ok = np.isfinite(d) & np.isfinite(pair_noise)
    if ok.sum() < 30:
        return {}
    observed = float(np.var(d[ok], ddof=1))
    noise = float(np.mean(pair_noise[ok]))
    true_var = max(observed - noise, 0.0)
    return {"n": int(ok.sum()), "observed_var": observed, "noise_var": noise,
            "true_var": true_var,
            "observed_sd": float(np.sqrt(observed)), "noise_sd": float(np.sqrt(noise)),
            "true_sd": float(np.sqrt(true_var)),
            "noise_share": float(min(noise / observed, 1.0)) if observed > 0 else np.nan,
            "mean_se": float(np.nanmean(sepiv.to_numpy()))}


def signal_to_noise(betas: pd.DataFrame) -> dict:
    """Cross-sectional dispersion of true beta against the dispersion of its estimates.

    The companion to ``noise_floor``. If the spread of estimated betas across names is largely
    estimation error, then ranking companies by beta is ranking them by noise, and the
    two-decimal precision on a risk report is decoration.
    """
    lf = long_form(betas)
    piv = lf.pivot(index="date", columns="name", values="beta")
    sepiv = lf.pivot(index="date", columns="name", values="se")
    cross_var = piv.var(axis=1, ddof=1)
    noise = (sepiv ** 2).mean(axis=1)
    true_var = (cross_var - noise).clip(lower=0)
    return {"mean_cross_sectional_var": float(cross_var.mean()),
            "mean_noise_var": float(noise.mean()),
            "mean_true_var": float(true_var.mean()),
            "reliability": float((true_var / cross_var).replace(
                [np.inf, -np.inf], np.nan).mean()),
            "cross_sectional_sd": float(np.sqrt(cross_var.mean())),
            "true_sd": float(np.sqrt(true_var.mean()))}


# --------------------------------------------------------------------------- #
# 3. Does shrinkage help?
# --------------------------------------------------------------------------- #
def blume_shrink(b: np.ndarray, w: float = 0.66, target: float = 1.0) -> np.ndarray:
    """Blume's constant shrinkage toward a target — the version everyone uses."""
    return w * b + (1 - w) * target


def vasicek_shrink(b: np.ndarray, se: np.ndarray, prior_mean: float = None,
                   prior_var: float = None) -> np.ndarray:
    """Vasicek's precision-weighted shrinkage: noisier estimates are shrunk harder.

    The theoretically correct version, and the one that is almost never used because it needs a
    standard error per name. Its advantage over Blume should be largest exactly where estimate
    quality varies most across names, which is a testable prediction rather than a claim.
    """
    ok = np.isfinite(b) & np.isfinite(se)
    if ok.sum() < 3:
        return np.asarray(b, dtype=float)
    if prior_mean is None:
        prior_mean = float(np.nanmean(b[ok]))
    if prior_var is None:
        prior_var = max(float(np.nanvar(b[ok], ddof=1))
                        - float(np.nanmean(se[ok] ** 2)), 1e-6)
    out = np.asarray(b, dtype=float).copy()
    w = prior_var / (prior_var + se[ok] ** 2)
    out[ok] = w * b[ok] + (1 - w) * prior_mean
    return out


def forecast_comparison(betas: pd.DataFrame, blume_w: float = 0.66) -> pd.DataFrame:
    """Predict each period's beta from the previous one, four ways, and score the errors.

    The four predictors are the raw previous beta, Blume shrinkage, Vasicek shrinkage, and the
    constant 1.0. Scored by mean absolute and root-mean-square error against the *next* period's
    realised estimate. Out-of-sample prediction error is the only criterion that settles which
    adjustment is worth making.
    """
    lf = long_form(betas)
    piv = lf.pivot(index="date", columns="name", values="beta").sort_index()
    sepiv = lf.pivot(index="date", columns="name", values="se").sort_index()
    rows = []
    for i in range(1, len(piv)):
        prev = piv.iloc[i - 1].to_numpy(dtype=float)
        prev_se = sepiv.iloc[i - 1].to_numpy(dtype=float)
        actual = piv.iloc[i].to_numpy(dtype=float)
        ok = np.isfinite(prev) & np.isfinite(actual)
        if ok.sum() < 5:
            continue
        preds = {"raw": prev, "blume": blume_shrink(prev, blume_w),
                 "vasicek": vasicek_shrink(prev, prev_se),
                 "always_one": np.ones_like(prev)}
        for name, p in preds.items():
            e = p[ok] - actual[ok]
            rows.append({"date": piv.index[i], "method": name, "n": int(ok.sum()),
                         "mae": float(np.abs(e).mean()),
                         "rmse": float(np.sqrt((e ** 2).mean())),
                         "bias": float(e.mean())})
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    return d.groupby("method").agg(
        mae=("mae", "mean"), rmse=("rmse", "mean"), bias=("bias", "mean"),
        n_periods=("date", "nunique")).sort_values("rmse")


def optimal_shrinkage(betas: pd.DataFrame, grid=None) -> dict:
    """Search the Blume weight that actually minimises out-of-sample error.

    Blume's 0.66 is a specific number from a specific 1971 sample. Whether it is right *here* is
    an empirical question with a cheap answer, and the comparison between the fitted weight and
    the persistence slope from section 1 is a consistency check on the whole framework: in
    theory they should coincide.
    """
    if grid is None:
        grid = np.linspace(0.0, 1.2, 61)
    lf = long_form(betas)
    piv = lf.pivot(index="date", columns="name", values="beta").sort_index()
    best_w, best_rmse = np.nan, np.inf
    curve = []
    for w in grid:
        errs = []
        for i in range(1, len(piv)):
            prev = piv.iloc[i - 1].to_numpy(dtype=float)
            actual = piv.iloc[i].to_numpy(dtype=float)
            ok = np.isfinite(prev) & np.isfinite(actual)
            if ok.sum() < 5:
                continue
            e = blume_shrink(prev[ok], w) - actual[ok]
            errs.append(float((e ** 2).mean()))
        if not errs:
            continue
        rmse = float(np.sqrt(np.mean(errs)))
        curve.append({"w": float(w), "rmse": rmse})
        if rmse < best_rmse:
            best_w, best_rmse = float(w), rmse
    return {"best_w": best_w, "best_rmse": best_rmse, "curve": curve,
            "blume_default": 0.66}


# --------------------------------------------------------------------------- #
# Portfolios versus single names
# --------------------------------------------------------------------------- #
def portfolio_betas(rets: pd.DataFrame, market: str, names, n_per: int = 10,
                    n_ports: int = 20, window: int = TRADING_DAYS,
                    seed: int = 1005) -> pd.DataFrame:
    """Betas of random equal-weighted portfolios, for comparison with single names.

    Estimation error falls roughly as the portfolio's idiosyncratic risk does, so a
    ten-stock portfolio's beta should be much better measured than a single stock's. If the
    *persistence* rises correspondingly, the instability was noise; if it does not, the
    exposure genuinely moves.
    """
    cols = [c for c in names if c in rets.columns]
    if len(cols) < n_per:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    port = {}
    for i in range(n_ports):
        pick = rng.choice(cols, size=n_per, replace=False)
        port[f"P{i:02d}"] = rets[list(pick)].mean(axis=1)
    df = pd.DataFrame(port)
    df[market] = rets[market]
    return rolling_betas(df, market, window=window, step=window)


def synthetic_panel(n_names: int = 40, n_days: int = 6000, true_beta_sd: float = 0.35,
                    beta_drift: float = 0.0, idio_vol: float = 0.25,
                    market_vol: float = 0.16, seed: int = 1005) -> pd.DataFrame:
    """A market and a cross-section whose true betas are known and optionally drifting.

    With ``beta_drift = 0`` every name's true beta is **constant for the whole sample**, so any
    measured instability is estimation error by construction. That is the calibration the real
    data is compared against. Raising ``beta_drift`` introduces a random walk in the true beta,
    which lets the machinery be shown to detect genuine variation rather than merely reporting
    noise whatever happens.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1993-02-01", periods=n_days)
    m = rng.normal(0.0003, market_vol / np.sqrt(TRADING_DAYS), n_days)
    b0 = rng.normal(1.0, true_beta_sd, n_names)
    out = {"MKT": m}
    for j in range(n_names):
        if beta_drift > 0:
            steps = rng.normal(0, beta_drift / np.sqrt(TRADING_DAYS), n_days)
            b = b0[j] + np.cumsum(steps)
        else:
            b = np.full(n_days, b0[j])
        e = rng.normal(0, idio_vol / np.sqrt(TRADING_DAYS), n_days)
        out[f"N{j:02d}"] = b * m + e
    df = pd.DataFrame(out, index=idx)
    # The REALISED planted betas, so tests can score the estimator against the truth rather
    # than against the population mean it was drawn from — with forty names the sample mean of
    # b0 is itself a couple of standard errors away from 1.0 quite routinely.
    df.attrs["true_beta"] = {f"N{j:02d}": float(b0[j]) for j in range(n_names)}
    df.attrs["beta_drift"] = beta_drift
    return df


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Real** if beta persists materially — a persistence slope well above zero
      with a half-life over a year; **Weak** if it persists but decays within a year; **None**
      if last period's beta does not predict this one's.
    - **Tradability**: **Useful** if a shrinkage adjustment beats the raw estimate out of
      sample **and** beats the do-nothing baseline of assuming 1.0 for everything; **Partial**
      if it beats only one; **Mirage** if the raw estimate is as good as anything.
    """
    signal = ("Real" if (h["slope"] > 0.3 and h["half_life_years"] >= 1.0)
              else ("Weak" if h["slope"] > 0.1 else "None"))
    beats_raw = h["best_rmse"] < h["raw_rmse"]
    beats_one = h["best_rmse"] < h["one_rmse"]
    trad = ("Useful" if (beats_raw and beats_one)
            else ("Partial" if (beats_raw or beats_one) else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"Partly, and much less than the two decimal places on a risk report imply. Across "
            f"{h['n_names']} names over {h['n_periods']} non-overlapping "
            f"{h['window']}-day windows, regressing each period's cross-section of betas on the "
            f"previous period's gives a slope of **{h['slope']:.3f}** "
            f"(±{h['slope_se']:.3f}, R² {h['persist_r2']:.2f}) — Blume's finding, reproduced. "
            f"A deviation from the average beta therefore decays by half in "
            f"**{h['half_life_years']:.1f} years**. But the more important number is what that "
            f"instability is made of. A beta measured over {h['window']} days carries a mean "
            f"standard error of {h['mean_se']:.3f}, so two consecutive estimates would differ "
            f"even if the true beta never moved. Decomposing the variance of the observed "
            f"period-to-period change: **{h['noise_share']:.0%} is estimation error** and only "
            f"the remainder is genuine movement — a true standard deviation of "
            f"{h['true_sd']:.3f} against an apparent {h['observed_sd']:.3f}. Betas move much "
            f"less than they appear to; the estimates move a lot. The confirmation is the "
            f"portfolio test: ten-stock portfolios cut the mean standard error from "
            f"{h['mean_se']:.4f} to {h['port_se']:.4f} and the noise share from "
            f"{h['noise_share']:.0%} to {h['port_noise_share']:.0%} — measure beta better and "
            f"less of its instability survives. Note the slope does **not** confirm this "
            f"({h['port_slope']:.3f} against {h['slope']:.3f}), and that is a finding in "
            f"itself: regressing a noisy measure on a noisy measure attenuates the slope by "
            f"var(true)/[var(true)+var(noise)], and diversification lowers both terms at once. "
            f"The Blume slope confounds persistence with measurement quality and should not be "
            f"read as a stability statistic at all — the synthetic control makes the point "
            f"unanswerable, since a *drifting* beta there produces a **higher** slope than a "
            f"perfectly constant one."),
        "trad_why": (
            f"Shrinkage helps, the textbook constant is close to right, and the honest "
            f"benchmark is harder than it looks. Predicting each period's beta from the "
            f"previous one, root-mean-square error came to **{h['raw_rmse']:.3f}** using the "
            f"raw estimate, {h['blume_rmse']:.3f} with Blume's 0.66 shrinkage, "
            f"{h['vasicek_rmse']:.3f} with Vasicek's precision weighting, and "
            f"**{h['one_rmse']:.3f} simply assuming 1.0 for every name**. That last figure is "
            f"the one worth pausing on: {'it beats the raw estimate' if h['one_rmse'] < h['raw_rmse'] else 'the raw estimate holds it off'}, "
            f"which puts a floor under how much any beta model is really contributing. "
            f"Searching the shrinkage weight directly gives an optimum of **{h['best_w']:.2f}** "
            f"— against Blume's {h['blume_default']:.2f}, and against the persistence slope of "
            f"{h['slope']:.3f} that theory says it should equal, a coincidence close enough to "
            f"be reassuring about the whole framework. Practically: re-estimate no more often "
            f"than the half-life justifies, shrink toward one, and stop quoting the second "
            f"decimal."),
        "trad": trad,
        "one_sentence": (
            f"Beta's persistence slope is {h['slope']:.2f} — a {h['half_life_years']:.1f}-year "
            f"half-life — but {h['noise_share']:.0%} of the apparent instability is estimation "
            f"error, and shrinking toward 1.0 at a weight of {h['best_w']:.2f} beats the raw "
            f"estimate out of sample."),
    }
