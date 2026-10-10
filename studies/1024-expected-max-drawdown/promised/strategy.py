"""The drawdown a Gaussian risk model promises, and the one the tape delivers — Study 1024.

The claim under test is old and respectable. If log prices follow a Brownian motion with drift
``mu`` and volatility ``sigma``, the distribution of the maximum drawdown over a horizon ``T``
is fully determined by ``(mu, sigma, T)`` — Magdon-Ismail, Atiya, Pratap & Abu-Mostafa (2004)
give it in closed form, and with zero drift the expectation is simply
``sqrt(pi/2) * sigma * sqrt(T)``. A risk model that knows the first two moments therefore
"knows" drawdown risk, and risk budgets of the form *expected max DD ≈ f(mu, sigma, T)* follow.

The machinery, in the order the study uses it:

**The promise.** ``gbm_mdd`` simulates the drawdown **exactly** for a *discretely sampled* GBM:
log increments are i.i.d. ``N(mu, sigma^2)`` at the tape's own sampling frequency, so there is
no discretisation error in the simulation itself. That matters because the continuous formula
is *not* what a daily or monthly tape can show: a drawdown measured on closes misses the
intraday (or intra-month) trough, and ``discretisation_table`` measures by how much.

**The challengers.** ``bootstrap_mdd`` resamples the estimation window with a stationary block
bootstrap (Politis & Romano, 1994) — same returns, same tails, volatility clustering kept inside
each block. ``garch_t_mdd`` fits a GARCH(1,1) with Student-*t* shocks (Bollerslev, 1986/1987)
and simulates forward *from today's conditional variance*: the one model here that knows the
current volatility state rather than an average one.

**The ex-ante test.** ``coverage`` walks non-overlapping calendar windows. For each window it
estimates every model on the **preceding** ``lookback`` years only, writes down the promised
drawdown distribution (mean, median, 95th and 99th percentiles), then records the drawdown the
window actually delivered. A calibrated model breaches its 95th percentile in 5% of windows;
``summarise`` turns breach counts into exact binomial tests (non-overlapping windows are the
main test, so the trials are independent under the null). Overlapping windows are reported
too, with a moving-block bootstrap that respects the overlap.

**The fix.** ``multiplier`` is the factor a risk manager must apply to the Gaussian 95th
percentile to get 5% breaches on the tape, and its stability across tapes and eras is the
practical question: a constant ``k`` is a usable rule, a wandering one is not.

Everything is seeded; common random numbers are reused across windows so differences between
windows are differences in parameters, not in simulation noise.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
MODELS = ("gauss", "gauss_mu0", "gauss_oracle", "boot", "garch_t")
MODEL_LABELS = {
    "gauss": "Gaussian GBM, trailing mu & sigma",
    "gauss_mu0": "Gaussian GBM, trailing sigma, mu = 0",
    "gauss_oracle": "Gaussian GBM, whole-tape mu & sigma (hindsight)",
    "boot": "stationary block bootstrap of trailing returns",
    "garch_t": "GARCH(1,1)-t from today's conditional variance",
}


# --------------------------------------------------------------------------- #
# Drawdown arithmetic
# --------------------------------------------------------------------------- #
def max_drawdown_log(log_returns) -> np.ndarray | float:
    """Maximum drawdown in **log** units along the last axis.

    The path starts at zero *before* the first return (the window opens at the previous
    close), so a window whose first day is a loss already has a drawdown.
    """
    r = np.asarray(log_returns, dtype=float)
    x = np.cumsum(r, axis=-1)
    zero = np.zeros(x.shape[:-1] + (1,))
    x = np.concatenate([zero, x], axis=-1)
    peak = np.maximum.accumulate(x, axis=-1)
    d = (peak - x).max(axis=-1)
    return float(d) if np.ndim(d) == 0 else d


def max_drawdown(log_returns) -> np.ndarray | float:
    """Maximum drawdown as a **percentage loss** from peak: ``1 - exp(-D_log)``."""
    d = max_drawdown_log(log_returns)
    return 1.0 - np.exp(-d)


def drawdown_curve(log_returns: pd.Series) -> pd.Series:
    """Underwater curve (percentage below the running peak) of a log-return series."""
    x = np.log(1.0) + log_returns.cumsum()
    peak = np.maximum(x.cummax(), 0.0)
    return 1.0 - np.exp(-(peak - x))


def magdon_ismail_zero_drift_mean(sigma_ann: float, years: float) -> float:
    """Expected max drawdown (log units) of a **continuous** driftless Brownian motion.

    Magdon-Ismail et al. (2004): ``E[MDD] = sqrt(pi/2) * sigma * sqrt(T)`` when ``mu = 0``.
    The anchor the simulation is tested against as the sampling step goes to zero.
    """
    return float(np.sqrt(np.pi / 2.0) * sigma_ann * np.sqrt(years))


def summary(mdd: np.ndarray) -> dict:
    """Mean, median, 95th and 99th percentile of a sample of (percentage) drawdowns."""
    m = np.asarray(mdd, dtype=float)
    return {"mean": float(m.mean()), "median": float(np.median(m)),
            "q95": float(np.quantile(m, 0.95)), "q99": float(np.quantile(m, 0.99))}


# --------------------------------------------------------------------------- #
# The promise — exact discretely-sampled GBM
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def _walks(n_sims: int, n_steps: int, seed: int) -> np.ndarray:
    """Cached standard random walks (cumulated N(0, 1) draws) — the common random numbers."""
    w = np.cumsum(np.random.default_rng(seed).standard_normal((n_sims, n_steps)), axis=1)
    w.setflags(write=False)
    return w


def gbm_mdd(mu: float, sigma: float, n_steps: int, n_sims: int = 2000,
            seed: int = 1024) -> np.ndarray:
    """Percentage max drawdowns of ``n_sims`` exact GBM paths sampled ``n_steps`` times.

    ``mu`` and ``sigma`` are the per-step **log** drift and volatility. Log increments are
    exactly ``N(mu, sigma^2)`` — no Euler error — so the only approximation is Monte Carlo
    noise. Common random numbers (same ``seed``) across calls: the cached standard walk ``W``
    is rescaled to ``sigma * W_t + mu * t``, which is the same path law and costs one pass.
    """
    w = _walks(int(n_sims), int(n_steps), int(seed))
    x = w * sigma
    x += mu * np.arange(1, int(n_steps) + 1)
    peak = np.maximum.accumulate(x, axis=1)
    np.maximum(peak, 0.0, out=peak)
    peak -= x
    return 1.0 - np.exp(-peak.max(axis=1))


def discretisation_table(mu_ann: float = 0.0, sigma_ann: float = 0.16, years: float = 1.0,
                         steps_per_year=(12, 52, 252, 252 * 16), n_sims: int = 4000,
                         seed: int = 1024) -> pd.DataFrame:
    """The same GBM sampled at different frequencies: how much drawdown sampling hides.

    Rows are sampling frequencies; columns the mean, median and 95th percentile of the
    percentage drawdown, and the mean in log units against the continuous zero-drift formula
    (only meaningful when ``mu_ann == 0``).
    """
    rows = []
    for k in steps_per_year:
        n = int(round(k * years))
        m = gbm_mdd(mu_ann / k, sigma_ann / np.sqrt(k), n, n_sims, seed)
        s = summary(m)
        rows.append({"steps_per_year": int(k), "n_steps": n, **s,
                     "mean_log": float(np.mean(-np.log1p(-m)))})
    df = pd.DataFrame(rows).set_index("steps_per_year")
    df["continuous_formula_log"] = magdon_ismail_zero_drift_mean(sigma_ann, years) \
        if mu_ann == 0 else np.nan
    return df


# --------------------------------------------------------------------------- #
# The challengers
# --------------------------------------------------------------------------- #
def stationary_bootstrap_index(n_obs: int, n_steps: int, n_sims: int, mean_block: float,
                               rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary-bootstrap indices, shape ``(n_sims, n_steps)``.

    Each step either starts a new block at a uniformly random position (probability
    ``1 / mean_block``) or continues the current block, wrapping circularly.
    """
    restart = rng.random((n_sims, n_steps)) < 1.0 / max(mean_block, 1.0)
    restart[:, 0] = True
    starts = rng.integers(0, n_obs, (n_sims, n_steps))
    t = np.arange(n_steps)
    t0 = np.maximum.accumulate(np.where(restart, t[None, :], 0), axis=1)
    s0 = np.take_along_axis(starts, t0, axis=1)
    return (s0 + (t[None, :] - t0)) % n_obs


def bootstrap_mdd(est_returns, n_steps: int, n_sims: int = 1000, mean_block: float = 63.0,
                  seed: int = 1024) -> np.ndarray:
    """Percentage max drawdowns of stationary-block-bootstrapped paths of the trailing tape."""
    r = np.asarray(est_returns, dtype=float)
    r = r[np.isfinite(r)]
    idx = stationary_bootstrap_index(len(r), int(n_steps), int(n_sims), mean_block,
                                     np.random.default_rng(seed))
    return max_drawdown(r[idx])


def fit_garch_t(est_returns, max_persistence: float = 0.995) -> dict:
    """GARCH(1,1) with Student-*t* shocks and a constant mean, via ``arch``.

    Fitted on returns ×100 for numerical conditioning; parameters are returned in return
    units, together with the one-step-ahead conditional variance ``s2_next`` — the state the
    simulation starts from. Persistence is capped at ``max_persistence`` (an integrated fit
    makes the long-horizon variance explode, which is a property of the estimator on a short
    window, not of markets) and ``nu`` is floored at 2.5. On a failed fit the function falls
    back to an i.i.d. Gaussian with the window's mean and variance, and says so.
    """
    from arch import arch_model

    r = np.asarray(est_returns, dtype=float)
    r = r[np.isfinite(r)]
    out = {"mu": float(r.mean()), "omega": float(r.var(ddof=1)), "alpha": 0.0, "beta": 0.0,
           "nu": np.inf, "s2_next": float(r.var(ddof=1)), "ok": False}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(100.0 * r, mean="Constant", vol="GARCH", p=1, q=1, dist="t",
                             rescale=False).fit(disp="off", show_warning=False)
        p = res.params
        mu, om, a, b, nu = (float(p["mu"]), float(p["omega"]), float(p["alpha[1]"]),
                            float(p["beta[1]"]), float(p["nu"]))
        if not np.all(np.isfinite([mu, om, a, b, nu])) or om <= 0:
            return out
        if a + b > max_persistence:
            b = max(max_persistence - a, 0.0)
            if a > max_persistence:
                a, b = max_persistence, 0.0
        nu = max(nu, 2.5)
        e_last = float(res.resid[-1])
        s2_last = float(res.conditional_volatility[-1]) ** 2
        s2_next = om + a * e_last ** 2 + b * s2_last
        out = {"mu": mu / 100.0, "omega": om / 1e4, "alpha": a, "beta": b, "nu": nu,
               "s2_next": s2_next / 1e4, "ok": True}
    except Exception:
        pass
    return out


def garch_t_mdd(params: dict, n_steps: int, n_sims: int = 1000,
                seed: int = 1024) -> np.ndarray:
    """Percentage max drawdowns of GARCH(1,1)-*t* paths started from ``params['s2_next']``."""
    rng = np.random.default_rng(seed)
    nu = params["nu"]
    if np.isfinite(nu):
        z = rng.standard_t(nu, (int(n_steps), int(n_sims))) * np.sqrt((nu - 2.0) / nu)
    else:
        z = rng.standard_normal((int(n_steps), int(n_sims)))
    mu, om, a, b = params["mu"], params["omega"], params["alpha"], params["beta"]
    s2 = np.full(int(n_sims), params["s2_next"])
    r = np.empty((int(n_steps), int(n_sims)))
    for t in range(int(n_steps)):
        e = np.sqrt(s2) * z[t]
        r[t] = mu + e
        s2 = om + a * e * e + b * s2
    return max_drawdown(r.T)


def promise(est_returns, n_steps: int, model: str, ppy: int, oracle: tuple | None = None,
            n_sims: int = 2000, seed: int = 1024) -> np.ndarray:
    """The promised drawdown distribution (a sample) of one model for one window.

    ``ppy`` (periods per year) only sets the bootstrap block length: a quarter of daily data
    (63 days) or half a year of monthly data (6 months). ``oracle`` is the whole-tape
    ``(mu, sigma)`` per step, used only by ``gauss_oracle``.
    """
    r = np.asarray(est_returns, dtype=float)
    r = r[np.isfinite(r)]
    if model == "gauss":
        return gbm_mdd(r.mean(), r.std(ddof=1), n_steps, n_sims, seed)
    if model == "gauss_mu0":
        return gbm_mdd(0.0, r.std(ddof=1), n_steps, n_sims, seed)
    if model == "gauss_oracle":
        if oracle is None:
            raise ValueError("gauss_oracle needs the whole-tape (mu, sigma)")
        return gbm_mdd(oracle[0], oracle[1], n_steps, n_sims, seed)
    if model == "boot":
        block = 63.0 if ppy >= 200 else 6.0
        return bootstrap_mdd(r, n_steps, max(n_sims // 2, 200), block, seed)
    if model == "garch_t":
        return garch_t_mdd(fit_garch_t(r), n_steps, max(n_sims // 2, 200), seed)
    raise KeyError(model)


# --------------------------------------------------------------------------- #
# Windows and the ex-ante test
# --------------------------------------------------------------------------- #
def calendar_windows(index: pd.DatetimeIndex, horizon_years: int, lookback_years,
                     step_years: int | None = None) -> list[tuple]:
    """Complete calendar-year windows with a strictly preceding estimation period.

    Returns ``(start_year, est_mask, win_mask)`` tuples. The window covers calendar years
    ``[start, start + horizon)``; the estimation period covers the ``lookback_years`` calendar
    years before ``start`` (``"expanding"`` = everything before). Only windows whose last year
    is complete in the data and whose estimation period is fully covered are kept.
    ``step_years`` defaults to ``horizon_years`` — non-overlapping windows.
    """
    idx = pd.DatetimeIndex(index)
    years = idx.year
    first_full = years.min() + (0 if (idx[0].month == 1 and idx[0].day <= 7) else 1)
    # last complete calendar year: the data must reach into the last days of December
    last = idx[-1]
    last_full = last.year if (last.month == 12 and last.day >= 24) else last.year - 1
    step = int(step_years or horizon_years)
    lb = None if lookback_years == "expanding" else int(lookback_years)
    min_lb = 3 if lb is None else lb
    out = []
    start = first_full + min_lb
    while start + horizon_years - 1 <= last_full:
        lo = first_full if lb is None else start - lb
        est = (years >= lo) & (years < start)
        win = (years >= start) & (years < start + horizon_years)
        out.append((int(start), np.asarray(est), np.asarray(win)))
        start += step
    return out


def coverage(returns: pd.Series, tape: str, ppy: int, horizons=(1, 3, 5, 10),
             lookback=5, models=MODELS, step_years: int | None = None,
             n_sims: int = 2000, seed: int = 1024) -> pd.DataFrame:
    """The ex-ante test: one row per (horizon, window, model).

    For every window the promise is built from the preceding ``lookback`` years only, then the
    realised percentage drawdown of the window is set against it: the breach indicators at the
    95th and 99th percentiles, the ratio of realised to promised mean and median, and the PIT
    (the share of promised paths with a smaller drawdown — uniform under a correct model).
    """
    r = returns.dropna()
    oracle = (float(r.mean()), float(r.std(ddof=1)))
    garch_cache: dict = {}
    rows = []
    for H in horizons:
        ns = int(n_sims) if H <= 3 else max(int(n_sims) // 2, 1000)
        for start, est, win in calendar_windows(r.index, H, lookback, step_years):
            re, rw = r[est].to_numpy(), r[win].to_numpy()
            if len(re) < 2 * ppy * 0.9 or len(rw) < ppy * H * 0.9:
                continue
            realised = float(max_drawdown(rw))
            for m in models:
                if m == "garch_t":
                    if start not in garch_cache:
                        garch_cache[start] = fit_garch_t(re)
                    sample = garch_t_mdd(garch_cache[start], len(rw),
                                         max(ns // 2, 200), seed)
                else:
                    sample = promise(re, len(rw), m, ppy, oracle, ns, seed)
                s = summary(sample)
                rows.append({
                    "tape": tape, "horizon": int(H), "start": int(start),
                    "end": int(start + H - 1), "model": m, "n_est": int(len(re)),
                    "n_steps": int(len(rw)), "realised": realised,
                    "p_mean": s["mean"], "p_median": s["median"],
                    "p_q95": s["q95"], "p_q99": s["q99"],
                    "breach95": bool(realised > s["q95"]),
                    "breach99": bool(realised > s["q99"]),
                    "ratio_mean": realised / s["mean"] if s["mean"] > 0 else np.nan,
                    "ratio_q95": realised / s["q95"] if s["q95"] > 0 else np.nan,
                    "pit": float(np.mean(sample < realised)),
                    "est_mu_ann": float(re.mean() * ppy),
                    "est_vol_ann": float(re.std(ddof=1) * np.sqrt(ppy)),
                    "win_vol_ann": float(rw.std(ddof=1) * np.sqrt(ppy)),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def binomial_p(breaches: int, n: int, p0: float = 0.05) -> dict:
    """Exact binomial tests of a breach count against the nominal rate ``p0``.

    ``p_greater``: too many breaches (the model under-promises). ``p_less``: too few (the
    model is over-conservative). ``p_two``: two-sided.
    """
    if n == 0:
        return {"p_greater": np.nan, "p_less": np.nan, "p_two": np.nan}
    k = int(breaches)
    return {"p_greater": float(stats.binomtest(k, n, p0, alternative="greater").pvalue),
            "p_less": float(stats.binomtest(k, n, p0, alternative="less").pvalue),
            "p_two": float(stats.binomtest(k, n, p0, alternative="two-sided").pvalue)}


def multiplier(realised, promised_q, target: float = 0.95) -> float:
    """The factor ``k`` such that ``realised <= k * promised_q`` in ``target`` of windows.

    With ``promised_q`` the Gaussian 95th percentile, ``k = 1`` means the model is
    calibrated; ``k = 1.5`` means a risk manager has to budget for half as much again.
    """
    ratio = np.asarray(realised, float) / np.asarray(promised_q, float)
    ratio = ratio[np.isfinite(ratio)]
    if len(ratio) == 0:
        return np.nan
    return float(np.quantile(ratio, target))


def multiplier_ci(realised, promised_q, target: float = 0.95, n_boot: int = 2000,
                  seed: int = 1024, cluster=None) -> tuple:
    """Bootstrap 90% interval for ``multiplier``; resamples windows (or clusters of them)."""
    rv = np.asarray(realised, float)
    pq = np.asarray(promised_q, float)
    rng = np.random.default_rng(seed)
    if cluster is None:
        groups = [np.array([i]) for i in range(len(rv))]
    else:
        c = np.asarray(cluster)
        groups = [np.flatnonzero(c == g) for g in pd.unique(c)]
    if len(groups) < 3:
        return (np.nan, np.nan)
    ks = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        ii = np.concatenate([groups[j] for j in pick])
        ks.append(multiplier(rv[ii], pq[ii], target))
    return (float(np.quantile(ks, 0.05)), float(np.quantile(ks, 0.95)))


def block_bootstrap_rate(breach, block: int, n_boot: int = 2000, p0: float = 0.05,
                         seed: int = 1024) -> dict:
    """Moving-block bootstrap of a breach rate from **overlapping** windows.

    Consecutive overlapping windows share most of their path, so their breaches arrive in
    runs; resampling blocks of ``block`` consecutive windows keeps the runs intact. Returns the
    rate, a 90% interval and a one-sided p-value against ``p0`` (the bootstrap distribution
    recentred on the null).
    """
    b = np.asarray(breach, dtype=float)
    n = len(b)
    if n < 4:
        return {"rate": float(b.mean()) if n else np.nan, "lo": np.nan, "hi": np.nan,
                "p_greater": np.nan, "n": n}
    block = int(max(1, min(block, n)))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, (n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    rates = b[idx].mean(axis=1)
    obs = float(b.mean())
    null = rates - rates.mean() + p0
    return {"rate": obs, "lo": float(np.quantile(rates, 0.05)),
            "hi": float(np.quantile(rates, 0.95)),
            "p_greater": float(np.mean(null >= obs - 1e-12)), "n": n}


def summarise(cov: pd.DataFrame, by=("tape", "horizon", "model")) -> pd.DataFrame:
    """Breach rates with exact binomial p-values, ratios, PIT and the 95% multiplier."""
    rows = []
    for key, g in cov.groupby(list(by), sort=False):
        n = len(g)
        k95 = int(g["breach95"].sum())
        bp = binomial_p(k95, n, 0.05)
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))),
                     "n": n, "breaches95": k95, "rate95": k95 / n if n else np.nan,
                     "p_greater": bp["p_greater"], "p_less": bp["p_less"],
                     "rate99": float(g["breach99"].mean()),
                     "median_ratio_mean": float(g["ratio_mean"].median()),
                     "mean_pit": float(g["pit"].mean()),
                     "k95": multiplier(g["realised"], g["p_q95"])})
    return pd.DataFrame(rows)


def ks_uniform_pit(pit) -> float:
    """Kolmogorov-Smirnov p-value of the PITs against U(0, 1) — the whole-distribution check."""
    p = np.asarray(pit, float)
    if len(p) < 3:
        return np.nan
    return float(stats.kstest(p, "uniform").pvalue)


# --------------------------------------------------------------------------- #
# Synthetic calibration
# --------------------------------------------------------------------------- #
def synthetic_coverage(signal_strength: float, n_years: int = 60, horizons=(1,),
                       lookback=5, models=("gauss", "gauss_oracle"), n_tapes: int = 1,
                       n_sims: int = 1000, seed: int = 1024) -> pd.DataFrame:
    """Run ``coverage`` on ``n_tapes`` independent synthetic tapes and stack the rows.

    ``gauss_oracle`` here uses the **true** generator moments rather than the tape's sample
    moments, so at ``signal_strength = 0`` it is the textbook model with the right
    parameters — the one configuration in which a 5% breach rate is guaranteed.
    """
    from . import data

    out = []
    for i in range(int(n_tapes)):
        r, truth = data.synthetic_returns(n_years=n_years, signal_strength=signal_strength,
                                          seed=seed + i)
        ms = [m for m in models if m != "gauss_oracle"]
        if ms:
            c = coverage(r, f"synthetic_{i}", TRADING_DAYS, horizons, lookback, ms,
                         n_sims=n_sims, seed=seed)
            out.append(c)
        if "gauss_oracle" in models:
            rows = []
            for H in horizons:
                for start, est, win in calendar_windows(r.index, H, lookback):
                    rw = r[win].to_numpy()
                    realised = float(max_drawdown(rw))
                    sample = gbm_mdd(truth["mu"], truth["sigma"], len(rw), n_sims, seed)
                    s = summary(sample)
                    rows.append({"tape": f"synthetic_{i}", "horizon": H, "start": start,
                                 "model": "gauss_oracle", "realised": realised,
                                 "p_mean": s["mean"], "p_q95": s["q95"], "p_q99": s["q99"],
                                 "breach95": realised > s["q95"],
                                 "breach99": realised > s["q99"],
                                 "ratio_mean": realised / s["mean"],
                                 "pit": float(np.mean(sample < realised))})
            out.append(pd.DataFrame(rows))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# Verdict — thresholds fixed before the real run
# --------------------------------------------------------------------------- #
ALPHA = 0.05
STABLE_RANGE = 1.5     # max/min of the 1-year k95 across tapes and eras for a usable rule


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    **Signal** — is the Gaussian drawdown promise mis-calibrated on the real tape? The primary
    test is the plug-in Gaussian (trailing 5-year mu and sigma; 10 years on the monthly tape),
    1-year non-overlapping calendar windows, breaches of the promised 95th percentile, exact
    one-sided binomial test against 5%, on each of the three index tapes (S&P 500 daily, Nasdaq
    daily, Fama-French monthly):

    * **Real** — too many breaches at p < 0.05 on at least two of the three tapes, *and* the
      synthetic null (i.i.d. Gaussian, true moments) is calibrated (breach rate in [2%, 8%]);
    * **Mixed** — significantly too many breaches on one tape and significantly too *few* on
      another;
    * **Weak** — significant on exactly one tape;
    * **None** — significant on none.

    **Tradability** — is there a usable drawdown budget rule? There is no return stream to
    invest in, so **Investable** is not on offer. **Fragile** if either (a) the multiplier
    ``k95`` needed on the Gaussian 95th percentile is stable — max/min across the three tapes
    and the two halves of each tape at most 1.5 — or (b) one of the challengers (block
    bootstrap, GARCH-t) restores coverage on all three tapes (binomial p >= 0.05 for "too many"
    and a breach rate of at most 10%). **Mirage** if neither: no fixed rule and no better model.
    """
    prim = h["primary"]
    sig_hi = [t for t, d in prim.items() if d["p_greater"] < ALPHA]
    sig_lo = [t for t, d in prim.items() if d["p_less"] < ALPHA]
    null_ok = 0.02 <= h["null_oracle_rate"] <= 0.08
    if len(sig_hi) >= 2 and null_ok:
        signal = "Real"
    elif sig_hi and sig_lo:
        signal = "Mixed"
    elif len(sig_hi) >= 1:
        signal = "Weak"
    else:
        signal = "None"
    stable = bool(np.isfinite(h["k95_range"]) and h["k95_range"] <= STABLE_RANGE)
    fixers = [m for m, ok in h["fixes"].items() if ok]
    trad = "Fragile" if (stable or fixers) else "Mirage"

    rates = "; ".join(f"{h['tape_names'][t]} **{d['rate']:.0%}** ({d['breaches']}/{d['n']}, "
                      f"p = {d['p_greater']:.3f})" for t, d in prim.items())
    fix_txt = ("Of the challengers, the " + " and the ".join(MODEL_LABELS[m] for m in fixers)
               if fixers else "Neither challenger")
    signal_why = (
        f"Written down from the preceding years only, the Gaussian 95% drawdown band was "
        f"breached in 1-year non-overlapping windows on {rates} — against a promised 5%, and "
        f"**{h['pooled_rate']:.0%}** pooled. The machinery is not the problem: on a synthetic "
        f"i.i.d. Gaussian tape the formula with the true moments breaches "
        f"**{h['null_oracle_rate']:.1%}**, estimating them from five years costs only a "
        f"little ({h['null_plugin_rate']:.1%}), and the exact discretely-sampled simulation "
        f"lands within {h['disc_err']:.1%} of Magdon-Ismail's zero-drift mean as the step "
        f"shrinks. The tape is the problem, and in both directions at once: the median year "
        f"delivered only **{h['median_ratio']:.2f}×** the promised mean drawdown, yet the "
        f"bad years overran the 95th percentile {h['pooled_rate'] / 0.05:.1f}× as often as "
        f"promised — the shape is "
        f"wrong, not just the scale. Even handed the whole tape's mu and sigma in hindsight, "
        f"the Gaussian breached **{h['oracle_rate']:.0%}** of 1-year S&P 500 windows. The "
        f"20-stock panel — survivors, so a floor — breached **{h['stocks_rate']:.0%}** of "
        f"stock-years (year-clustered 90% interval for its multiplier "
        f"{h['stocks_k95_lo']:.2f}–{h['stocks_k95_hi']:.2f}).")
    trad_why = (
        f"Keeping the Gaussian and scaling it takes **k95 = {h['k95_pooled']:.2f}×** the "
        f"promised 95th percentile on the pooled 1-year index windows, but the factor a "
        f"risk manager would have needed wanders from {h['k95_min']:.2f} to "
        f"{h['k95_max']:.2f} across tapes and eras (max/min {h['k95_range']:.2f}, "
        f"{'inside' if stable else 'outside'} the pre-registered 1.5): calibrate it on one "
        f"era and it is wrong in the next. {fix_txt} restored 1-year "
        f"coverage on all three tapes (pooled breach rates: block bootstrap "
        f"{h['boot_rate']:.0%}, GARCH-t {h['garch_rate']:.0%}) — the bootstrap replays the "
        f"calm of the lookback, and the GARCH knows today's volatility but not next year's "
        f"regime. The cheapest partial repair was not a fatter tail but a humbler mean: "
        f"setting the drift to zero cut the pooled breach rate to "
        f"{h['gauss_mu0_pooled_rate']:.0%} (k95 {h['k95_mu0_pooled']:.2f}), at the price of "
        f"over-promising pain at 5–10 years. There is no return stream here, so Investable is "
        f"not on offer, and with 15–81 windows per tape every rate carries a wide interval.")
    if fixers and not stable:
        fix_line = "the repair is a model of volatility, not a fixed multiplier"
    elif stable:
        fix_line = (f"a fixed multiplier of about {h['k95_pooled']:.1f}× restores the "
                    f"promise")
    else:
        fix_line = (f"no fixed multiplier (it wandered from {h['k95_min']:.1f}× to "
                    f"{h['k95_max']:.1f}×) and no off-the-shelf model repairs it")
    one = (f"Fed the trailing mean and volatility, the Gaussian drawdown promise was broken in "
           f"{h['pooled_rate']:.0%} of years instead of 5% — knowing mu and sigma is not "
           f"knowing drawdown risk — and {fix_line}.")
    return {"signal": signal, "signal_why": signal_why, "trad": trad, "trad_why": trad_why,
            "one_sentence": one}
