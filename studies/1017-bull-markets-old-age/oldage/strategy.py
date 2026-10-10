"""Do bull markets die of old age? — the machinery of Study 1017.

The claim is a statement about a **hazard function**: the probability that a bull market ends
next month, given that it has lasted ``a`` months, rises with ``a``. Everything here exists to
measure that function honestly and to ask whether it differs from what the dating rule would
produce on a market with no memory at all.

The pipeline, in the order the study runs it:

1. **Dating.** :func:`date_cycles` applies the conventional peak/trough filter — a bull ends when
   the index falls ``down`` (default 20%) below its high; a bear ends when it rises ``up`` above
   its low. Turning points are the extrema in between. The final phase is right-censored (still
   running at the end of the tape) and the first, partial phase is dropped. An asymmetric
   ``up``/``down`` pair gives a filter in the spirit of Lunde & Timmermann (2004).
2. **Duration model.** :func:`fit_weibull` fits a two-parameter Weibull to bull durations by
   maximum likelihood with right-censoring. Shape ``k = 1`` is memoryless (an exponential: age
   is irrelevant); ``k > 1`` is "dies of old age". :func:`weibull_bootstrap` resamples cycles for
   a confidence interval. :func:`life_table` gives the non-parametric hazard by age bucket.
3. **The null that matters.** The dating rule *itself* manufactures ``k > 1``: a bull cannot be
   confirmed until prices are 20% off the low and cannot end until they are 20% off the high, so
   very young bulls almost never die. :func:`null_distribution` simulates many random walks
   (i.i.d. lognormal with the tape's drift and vol) and a GARCH(1,1)-t with the tape's fitted
   dynamics, dates them with the **same** rule, and returns the distribution of every statistic.
   The real tape's ``k`` is judged against that, never against 1.
4. **Prediction.** :func:`realtime_state` runs the same filter causally, month by month, so the
   age of the bull at ``t`` uses only data up to ``t``. :func:`predictive_regression` regresses
   the next-12-month excess log return (and forward drawdown) on that age, with Newey-West
   standard errors for the 11-month overlap — and, because age is a trending, persistent
   regressor that a HAC correction does not fully tame, the *same* regression on every null path
   supplies a simulated p-value.
5. **Tradability.** :func:`age_rule_backtest` cuts equity to 50% (rest in bills) once the bull is
   older than the median completed bull known *at that date*, with one month of lag and one-way
   costs on traded NAV, and is raced against buy-and-hold on excess-of-cash Sharpe with a
   circular block bootstrap of the difference.

:func:`verdict` stamps the result by thresholds fixed before the real tape was run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats as sps
from scipy.special import logsumexp

MONTHS = 12
DEFAULT_THRESHOLD = 0.20
HAZARD_EDGES = (0, 12, 24, 36, 60, 96, np.inf)     # bull-age buckets, months


# --------------------------------------------------------------------------- #
# 1. Dating
# --------------------------------------------------------------------------- #
def turning_points(x: np.ndarray, up: float = DEFAULT_THRESHOLD,
                   down: float | None = None) -> list[tuple[int, str, int]]:
    """Alternating peaks and troughs of a positive level series under a ±threshold filter.

    Returns a list of ``(index, kind, confirm_index)`` with ``kind`` in ``{"peak", "trough"}``.
    ``confirm_index`` is the first observation at which the turning point became knowable — the
    bar on which the index had moved ``down`` below the peak (or ``up`` above the trough). The
    gap between the two is the reason the rule is useless for real-time timing of the top.
    """
    down = up if down is None else down
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return []
    tps: list[tuple[int, str, int]] = []
    state = 0
    hi = lo = x[0]
    hi_i = lo_i = 0
    for i in range(1, n):
        v = x[i]
        if state == 0:
            if v > hi:
                hi, hi_i = v, i
            if v < lo:
                lo, lo_i = v, i
            if v >= lo * (1.0 + up) and lo_i < i:
                tps.append((lo_i, "trough", i))
                state, hi, hi_i = 1, v, i
            elif v <= hi * (1.0 - down) and hi_i < i:
                tps.append((hi_i, "peak", i))
                state, lo, lo_i = -1, v, i
        elif state == 1:
            if v > hi:
                hi, hi_i = v, i
            elif v <= hi * (1.0 - down):
                tps.append((hi_i, "peak", i))
                state, lo, lo_i = -1, v, i
        else:
            if v < lo:
                lo, lo_i = v, i
            elif v >= lo * (1.0 + up):
                tps.append((lo_i, "trough", i))
                state, hi, hi_i = 1, v, i
    # A first "turning point" on the very first observation is the edge of the sample, not a
    # market extreme: the phase it would open is left-censored with an unknown start. Drop it.
    if tps and tps[0][0] == 0:
        tps = tps[1:]
    return tps


def date_cycles(level: pd.Series, up: float = DEFAULT_THRESHOLD,
                down: float | None = None) -> pd.DataFrame:
    """Bull and bear phases of ``level`` under the ±threshold rule.

    One row per phase: ``phase`` (bull / bear), ``start`` and ``end`` dates (trough→peak for a
    bull), integer positions, ``duration`` in **periods of the input** (months on a monthly tape),
    ``amplitude`` (simple return start→end), ``confirmed`` (date the end became knowable) and
    ``censored`` (True for the last phase, still running when the tape stops — its duration is a
    lower bound). The partial phase before the first turning point is dropped: its start is
    unknown, so it cannot be aged.
    """
    x = level.to_numpy(dtype=float)
    idx = level.index
    tps = turning_points(x, up, down)
    rows = []
    for j, (i0, kind, _) in enumerate(tps):
        phase = "bull" if kind == "trough" else "bear"
        if j + 1 < len(tps):
            i1, _, c1 = tps[j + 1]
            censored = False
        else:
            i1, c1 = len(x) - 1, None
            censored = True
        if i1 <= i0:
            continue
        rows.append({"phase": phase, "start": idx[i0], "end": idx[i1],
                     "start_i": int(i0), "end_i": int(i1), "duration": int(i1 - i0),
                     "amplitude": float(x[i1] / x[i0] - 1.0),
                     "confirmed": idx[c1] if c1 is not None else pd.NaT,
                     "censored": censored})
    cols = ["phase", "start", "end", "start_i", "end_i", "duration", "amplitude",
            "confirmed", "censored"]
    return pd.DataFrame(rows, columns=cols)


def bull_durations(level: pd.Series, up: float = DEFAULT_THRESHOLD,
                   down: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(durations, censored)`` of every bull on ``level`` — the inputs to the hazard model."""
    tps = turning_points(level.to_numpy(dtype=float) if hasattr(level, "to_numpy")
                         else np.asarray(level, dtype=float), up, down)
    return _bulls_from_tps(tps, len(level))


def _bulls_from_tps(tps, n) -> tuple[np.ndarray, np.ndarray]:
    d, c = [], []
    for j, (i0, kind, _) in enumerate(tps):
        if kind != "trough":
            continue
        if j + 1 < len(tps):
            d.append(tps[j + 1][0] - i0)
            c.append(False)
        else:
            if n - 1 > i0:
                d.append(n - 1 - i0)
                c.append(True)
    return np.asarray(d, dtype=float), np.asarray(c, dtype=bool)


# --------------------------------------------------------------------------- #
# 2. Duration model
# --------------------------------------------------------------------------- #
def fit_weibull(d, censored=None) -> dict:
    """Weibull MLE with right-censoring. Hazard ``h(t) = (k/λ)(t/λ)^(k−1)``.

    ``k`` is profiled analytically: for fixed ``k`` the MLE of ``λ^k`` is ``Σ dᵢ^k / r`` with
    ``r`` the number of completed (uncensored) spells, which leaves a one-dimensional search over
    ``log k``. ``k = 1`` ⇔ exponential, memoryless; ``k > 1`` ⇔ positive duration dependence.
    Returns ``{}`` with fewer than three completed spells — too few to say anything.
    """
    d = np.asarray(d, dtype=float)
    c = np.zeros(len(d), dtype=bool) if censored is None else np.asarray(censored, bool)
    ok = d > 0
    d, c = d[ok], c[ok]
    ev = ~c
    r = int(ev.sum())
    if r < 3:
        return {}
    logd = np.log(d)
    s_log_ev = float(logd[ev].sum())
    scale = float(np.exp(logd.mean()))
    z = logd - np.log(scale)          # rescale for numerical stability

    log_scale = float(np.log(scale))

    def negll(logk):
        # profile log-likelihood: λ^k = Σ d^k / r  ⇒  Σ (d/λ)^k = r
        k = np.exp(logk)
        log_sum_dk = k * log_scale + logsumexp(k * z)
        return -(r * np.log(k) - r * (log_sum_dk - np.log(r)) + (k - 1.0) * s_log_ev - r)

    res = optimize.minimize_scalar(negll, bounds=(np.log(0.05), np.log(30.0)),
                                   method="bounded", options={"xatol": 1e-7})
    k = float(np.exp(res.x))
    lam = float(np.exp(log_scale + (logsumexp(k * z) - np.log(r)) / k))
    ll = float(r * np.log(k) - r * k * np.log(lam) + (k - 1) * s_log_ev
               - ((d / lam) ** k).sum())
    # likelihood-ratio test of k = 1 (exponential: λ = Σd / r)
    lam1 = float(d.sum() / r)
    ll1 = float(-r * np.log(lam1) - (d / lam1).sum())
    lr = 2.0 * (ll - ll1)
    p_lr = float(sps.chi2.sf(max(lr, 0.0), 1))
    return {"k": k, "lam": lam, "loglik": ll, "n": int(len(d)), "n_events": r,
            "n_censored": int(c.sum()), "lr_k1": float(lr), "p_k1": p_lr,
            "median": float(lam * np.log(2.0) ** (1.0 / k))}


def weibull_bootstrap(d, censored=None, n_boot: int = 2000, seed: int = 1017,
                      alpha: float = 0.05) -> dict:
    """Percentile CI for the Weibull shape ``k`` by resampling whole spells (cycles).

    Cycles are the unit of independence here, so they are what gets resampled — with only
    ~20 of them the interval is wide, and that width is the study's honest constraint.
    """
    d = np.asarray(d, dtype=float)
    c = np.zeros(len(d), bool) if censored is None else np.asarray(censored, bool)
    rng = np.random.default_rng(seed)
    ks = []
    n = len(d)
    for _ in range(n_boot):
        ii = rng.integers(0, n, n)
        f = fit_weibull(d[ii], c[ii])
        if f:
            ks.append(f["k"])
    ks = np.asarray(ks)
    if len(ks) == 0:
        return {}
    return {"k_lo": float(np.quantile(ks, alpha / 2)),
            "k_hi": float(np.quantile(ks, 1 - alpha / 2)),
            "k_boot_median": float(np.median(ks)),
            "share_k_gt_1": float((ks > 1).mean()), "n_boot": int(len(ks)), "ks": ks}


def life_table(d, censored=None, edges=HAZARD_EDGES) -> pd.DataFrame:
    """Non-parametric hazard of a bull ending, by age bucket (actuarial life table).

    For each bucket ``[lo, hi)`` in months: spells at risk at ``lo``, completed spells ending
    inside, censored spells leaving inside (counted as half-exposed). ``q`` is the conditional
    probability of ending inside the bucket; ``annual_hazard`` puts buckets of different widths on
    the same clock: the probability of ending within a year at that age.
    """
    d = np.asarray(d, dtype=float)
    c = np.zeros(len(d), bool) if censored is None else np.asarray(censored, bool)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        at_risk = int((d >= lo).sum())
        ev = int(((d >= lo) & (d < hi) & ~c).sum())
        cen = int(((d >= lo) & (d < hi) & c).sum())
        n_adj = at_risk - cen / 2.0
        q = ev / n_adj if n_adj > 0 else np.nan
        if np.isfinite(hi):
            width = hi - lo
            ann = 1.0 - (1.0 - q) ** (MONTHS / width) if np.isfinite(q) and q < 1 else q
        else:
            # open bucket: events per year of exposure, as a probability
            expo = float(np.clip(d[d >= lo] - lo, 0, None).sum()) / MONTHS
            ann = 1.0 - np.exp(-ev / expo) if expo > 0 else np.nan
        label = f"{int(lo // 12)}–{int(hi // 12)}y" if np.isfinite(hi) else f"{int(lo // 12)}y+"
        rows.append({"bucket": label, "lo": lo, "hi": hi, "at_risk": at_risk,
                     "ended": ev, "censored": cen, "q": q, "annual_hazard": ann})
    return pd.DataFrame(rows).set_index("bucket")


# --------------------------------------------------------------------------- #
# 4. Real-time state and prediction
# --------------------------------------------------------------------------- #
def realtime_state(level: pd.Series, up: float = DEFAULT_THRESHOLD,
                   down: float | None = None) -> pd.DataFrame:
    """The dating filter run **causally**: what an investor could know at each close.

    Columns: ``state`` (+1 confirmed bull, −1 confirmed bear, 0 not yet determined), ``age`` (periods
    since the trough that started the current confirmed bull — the trough's date is known the
    moment it is confirmed, so this is look-ahead-free; NaN outside a bull), and
    ``completed_bull`` (on the bar a bull's peak is *confirmed*, that bull's trough→peak length;
    otherwise NaN). Nothing here uses a future price.
    """
    down = up if down is None else down
    x = level.to_numpy(dtype=float)
    n = len(x)
    state = np.zeros(n, dtype=int)
    age = np.full(n, np.nan)
    done = np.full(n, np.nan)
    s = 0
    hi = lo = x[0]
    hi_i = lo_i = 0
    trough_i = None
    for i in range(1, n):
        v = x[i]
        if s == 0:
            if v > hi:
                hi, hi_i = v, i
            if v < lo:
                lo, lo_i = v, i
            if v >= lo * (1.0 + up) and lo_i < i:
                # a bull whose low is the first observation has an unknown start: no age
                s, hi, hi_i, trough_i = 1, v, i, (lo_i if lo_i > 0 else None)
            elif v <= hi * (1.0 - down) and hi_i < i:
                s, lo, lo_i = -1, v, i
        elif s == 1:
            if v > hi:
                hi, hi_i = v, i
            elif v <= hi * (1.0 - down):
                if trough_i is not None:
                    done[i] = hi_i - trough_i
                s, lo, lo_i, trough_i = -1, v, i, None
        else:
            if v < lo:
                lo, lo_i = v, i
            elif v >= lo * (1.0 + up):
                s, hi, hi_i, trough_i = 1, v, i, lo_i
        state[i] = s
        if s == 1 and trough_i is not None:
            age[i] = i - trough_i
    return pd.DataFrame({"state": state, "age": age, "completed_bull": done},
                        index=level.index)


def forward_log_return(r: pd.Series, rf: pd.Series | None = None, h: int = 12) -> pd.Series:
    """Sum of the next ``h`` log (excess) returns, months ``t+1 … t+h`` — aligned at ``t``."""
    lr = np.log1p(r.astype(float))
    if rf is not None:
        lr = lr - np.log1p(rf.reindex(r.index).astype(float))
    fwd = lr[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1)
    return fwd


def forward_drawdown(level: pd.Series, h: int = 12) -> pd.Series:
    """Worst peak-to-trough loss on the path ``t … t+h``, aligned at ``t`` (a negative number)."""
    x = level.to_numpy(dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    for t in range(n - h):
        w = x[t:t + h + 1]
        out[t] = float((w / np.maximum.accumulate(w)).min() - 1.0)
    return pd.Series(out, index=level.index)


def ols_hac(y: np.ndarray, x: np.ndarray, lags: int = 12) -> dict:
    """OLS of ``y`` on ``[1, x]`` with Newey-West (Bartlett) standard errors."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    if n < 24 or np.std(x) == 0:
        return {}
    A = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    e = y - A @ beta
    XtX_inv = np.linalg.inv(A.T @ A)
    Ae = A * e[:, None]
    S = Ae.T @ Ae
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = Ae[l:].T @ Ae[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return {"alpha": float(beta[0]), "slope": float(beta[1]), "se": float(se[1]),
            "t": float(beta[1] / se[1]) if se[1] > 0 else np.nan, "n": int(n),
            "r2": float(1 - (e @ e) / ((y - y.mean()) @ (y - y.mean())))}


def predictive_regression(r: pd.Series, rf: pd.Series | None = None, h: int = 12,
                          up: float = DEFAULT_THRESHOLD, lags: int | None = None,
                          target: str = "return") -> dict:
    """Does the (real-time) age of the bull predict the next ``h`` months?

    Sample: months in a confirmed bull. ``x`` = bull age in **years**; ``y`` = next-``h``-month
    excess log return (``target="return"``) or the worst drawdown on that path
    (``target="drawdown"``). The ``h−1`` months of overlap are handled with Newey-West at
    ``lags = h + 6`` by default (a little beyond the overlap); a non-overlapping version that keeps
    every ``h``-th observation is reported alongside. The claim predicts a **negative** slope for
    returns and a **more negative** (i.e. negative-slope) drawdown.
    """
    lags = h + 6 if lags is None else lags
    level = (1.0 + r.astype(float)).cumprod()
    st = realtime_state(level, up)
    if target == "drawdown":
        y = forward_drawdown(level, h)
    else:
        y = forward_log_return(r, rf, h)
    mask = (st["state"] == 1) & st["age"].notna() & y.notna()
    x = st["age"][mask] / MONTHS
    yy = y[mask]
    out = ols_hac(yy.to_numpy(), x.to_numpy(), lags)
    if not out:
        return {}
    pos = np.arange(len(x))
    nov = ols_hac(yy.to_numpy()[pos % h == 0], x.to_numpy()[pos % h == 0], lags=1)
    out.update({"h": h, "lags": lags, "target": target,
                "t_nonoverlap": nov.get("t", np.nan),
                "slope_nonoverlap": nov.get("slope", np.nan),
                "n_nonoverlap": nov.get("n", 0),
                "mean_age_years": float(x.mean())})
    return out


# --------------------------------------------------------------------------- #
# 3. The nulls
# --------------------------------------------------------------------------- #
def fit_garch(r: pd.Series) -> dict:
    """GARCH(1,1) with Student-t innovations on monthly log returns (``arch``), as parameters.

    Volatility clustering lengthens calm bulls and bunches crashes; a random walk with constant
    vol has neither. The GARCH null asks whether that alone reproduces whatever the real tape
    shows.
    """
    from arch import arch_model
    lr = 100.0 * np.log1p(r.astype(float).dropna())
    res = arch_model(lr, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    p = res.params
    return {"mu": float(p["mu"]) / 100.0, "omega": float(p["omega"]) / 1e4,
            "alpha": float(p["alpha[1]"]), "beta": float(p["beta[1]"]),
            "nu": float(p["nu"])}


def simulate_log_returns(kind: str, n_months: int, n_sims: int, params: dict,
                         seed: int = 1017) -> np.ndarray:
    """``(n_sims, n_months)`` monthly log returns from the ``"rw"`` or ``"garch"`` null.

    ``rw``: i.i.d. normal with ``params["mu"]``, ``params["sigma"]`` (the tape's own mean and sd of
    log returns). ``garch``: GARCH(1,1)-t with the fitted parameters, unit-variance t shocks,
    started at the unconditional variance and burnt in for 120 months.
    """
    rng = np.random.default_rng(seed)
    if kind == "rw":
        return rng.normal(params["mu"], params["sigma"], (n_sims, n_months))
    if kind != "garch":
        raise ValueError(f"unknown null {kind!r}")
    om, a, b, nu = params["omega"], params["alpha"], params["beta"], params["nu"]
    burn = 120
    T = n_months + burn
    z = rng.standard_t(nu, (n_sims, T)) * np.sqrt((nu - 2.0) / nu)
    h = np.full(n_sims, om / max(1.0 - a - b, 1e-3))
    eps_prev = np.zeros(n_sims)
    out = np.empty((n_sims, T))
    for t in range(T):
        h = om + a * eps_prev ** 2 + b * h
        eps_prev = np.sqrt(h) * z[:, t]
        out[:, t] = params["mu"] + eps_prev
    return out[:, burn:]


def path_stats(logr: np.ndarray, up: float = DEFAULT_THRESHOLD, down: float | None = None,
               h: int = 12, with_prediction: bool = True, rf=None) -> dict:
    """Every statistic the study computes on the real tape, computed on one simulated path."""
    level = np.exp(np.cumsum(logr))
    tps = turning_points(level, up, down)
    d, c = _bulls_from_tps(tps, len(level))
    f = fit_weibull(d, c)
    out = {"n_bulls": int((~c).sum()), "k": f.get("k", np.nan),
           "median_bull": float(np.median(d[~c])) if (~c).any() else np.nan,
           "mean_bull": float(np.mean(d[~c])) if (~c).any() else np.nan,
           "d": d, "c": c}
    if with_prediction:
        s = pd.Series(np.expm1(logr))
        rfs = None if rf is None else pd.Series(np.asarray(rf, dtype=float))
        pr = predictive_regression(s, rfs, h=h, up=up)
        out["pred_slope"] = pr.get("slope", np.nan)
        out["pred_t"] = pr.get("t", np.nan)
    return out


def null_distribution(r: pd.Series, kind: str = "rw", n_sims: int = 500,
                      up: float = DEFAULT_THRESHOLD, down: float | None = None,
                      h: int = 12, seed: int = 1017, with_prediction: bool = True,
                      garch_params: dict | None = None, rf: pd.Series | None = None) -> dict:
    """Date ``n_sims`` null paths with the real tape's length, drift and vol; collect statistics.

    Returns ``{"table": DataFrame (one row per path), "durations": pooled (d, c), "params"}``.
    The pooled durations give the null's life table — what "ageing" looks like when there is,
    by construction, none. Both nulls carry the tape's **own drift** (mean log return): the
    GARCH mean is reset to it, so the comparison is about dynamics, not about a different
    equity premium. If ``rf`` is given, the predictive regression on each path uses the real
    bill series, exactly as on the tape.
    """
    r = r.astype(float).dropna()
    lr = np.log1p(r).to_numpy()
    n = len(lr)
    if kind == "rw":
        params = {"mu": float(lr.mean()), "sigma": float(lr.std(ddof=1))}
    else:
        params = dict(garch_params if garch_params is not None else fit_garch(r))
        params["mu"] = float(lr.mean())
    rfa = None if rf is None else rf.reindex(r.index).fillna(0.0).to_numpy()
    paths = simulate_log_returns(kind, n, n_sims, params, seed)
    rows, D, C = [], [], []
    for p in paths:
        s = path_stats(p, up, down, h, with_prediction, rfa)
        D.append(s.pop("d"))
        C.append(s.pop("c"))
        rows.append(s)
    return {"table": pd.DataFrame(rows), "durations": (np.concatenate(D), np.concatenate(C)),
            "params": params, "kind": kind, "n_sims": n_sims}


def null_p(real: float, null: np.ndarray, side: str = "greater") -> float:
    """One-sided simulation p-value with the +1 correction (never exactly zero)."""
    v = np.asarray(null, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0 or not np.isfinite(real):
        return np.nan
    if side == "greater":
        hits = (v >= real).sum()
    else:
        hits = (v <= real).sum()
    return float((hits + 1) / (len(v) + 1))


def duration_test(r: pd.Series, up: float = DEFAULT_THRESHOLD, down: float | None = None,
                  n_sims: int = 500, kinds=("rw", "garch"), seed: int = 1017,
                  n_boot: int = 1000, with_prediction: bool = True,
                  garch_params: dict | None = None, rf: pd.Series | None = None) -> dict:
    """The whole duration-dependence test on one return series, against each null.

    Returns the real Weibull fit, its bootstrap CI, the life table, and per null: the null's
    median ``k`` and 90% band, the one-sided p-value of the real ``k`` (H₁: real bulls age
    *faster* than null bulls), and — if requested — the simulated p-value of the predictive
    slope's HAC t-stat (H₁: more negative than the null).
    """
    level = (1.0 + r.astype(float)).cumprod()
    d, c = bull_durations(level, up, down)
    fit = fit_weibull(d, c)
    boot = weibull_bootstrap(d, c, n_boot=n_boot, seed=seed) if fit else {}
    out = {"up": up, "down": up if down is None else down, "d": d, "c": c, "fit": fit,
           "boot": {k: v for k, v in boot.items() if k != "ks"}, "boot_ks": boot.get("ks"),
           "life": life_table(d, c), "n_bulls": int((~c).sum()),
           "median_bull": float(np.median(d[~c])) if (~c).any() else np.nan,
           "nulls": {}}
    pred = predictive_regression(r, rf, up=up) if with_prediction else {}
    out["pred_raw"] = pred
    for j, kind in enumerate(kinds):
        nd = null_distribution(r, kind, n_sims, up, down, seed=seed + 101 * (j + 1),
                               with_prediction=with_prediction,
                               garch_params=garch_params if kind == "garch" else None,
                               rf=rf)
        tb = nd["table"]
        entry = {"k_null_median": float(tb["k"].median()),
                 "k_null_lo": float(tb["k"].quantile(0.05)),
                 "k_null_hi": float(tb["k"].quantile(0.95)),
                 "p_k": null_p(fit.get("k", np.nan), tb["k"], "greater"),
                 "n_bulls_null_median": float(tb["n_bulls"].median()),
                 "median_bull_null": float(tb["median_bull"].median()),
                 "life": life_table(*nd["durations"]),
                 "table": tb, "params": nd["params"]}
        if with_prediction and "pred_t" in tb:
            entry["p_pred"] = null_p(pred.get("t", np.nan), tb["pred_t"], "less")
            entry["pred_t_null_lo"] = float(tb["pred_t"].quantile(0.05))
            entry["pred_t_null_median"] = float(tb["pred_t"].median())
        out["nulls"][kind] = entry
    return out


# --------------------------------------------------------------------------- #
# 5. Tradability
# --------------------------------------------------------------------------- #
def age_rule_weights(level: pd.Series, up: float = DEFAULT_THRESHOLD, low_weight: float = 0.5,
                     min_completed: int = 3, prior_durations=None) -> pd.DataFrame:
    """Target equity weight at each close: ``low_weight`` once the bull is older than the median
    completed bull **known at that close**, else 1.0.

    The median is expanding and causal: a bull's length only enters it on the bar its peak was
    confirmed. ``prior_durations`` seeds that list (e.g. bulls dated on an earlier, separate tape);
    with fewer than ``min_completed`` known bulls the rule stays fully invested.
    """
    st = realtime_state(level, up)
    known = list(prior_durations) if prior_durations is not None else []
    med = np.full(len(st), np.nan)
    w = np.ones(len(st))
    age = st["age"].to_numpy()
    done = st["completed_bull"].to_numpy()
    for i in range(len(st)):
        if np.isfinite(done[i]):
            known.append(done[i])
        if len(known) >= min_completed:
            med[i] = float(np.median(known))
            if np.isfinite(age[i]) and age[i] > med[i]:
                w[i] = low_weight
    return pd.DataFrame({"weight": w, "median_age": med, "age": st["age"],
                         "state": st["state"]}, index=level.index)


def age_rule_backtest(r: pd.Series, rf: pd.Series, weights: pd.Series,
                      cost_bps: float = 10.0) -> pd.DataFrame:
    """Book the rule with **one period of lag**: the weight set at the close of ``t`` earns
    period ``t+1``'s return — one ``shift(1)``, applied once, here and nowhere else.

    Cash earns ``rf``. Costs are ``cost_bps`` one-way × traded NAV, where traded NAV is the gap
    between the new target weight and the weight the book had **drifted** to. Returns gross and
    net returns, the buy-and-hold leg, and each leg's excess over cash.
    """
    r = r.astype(float)
    rf = rf.reindex(r.index).astype(float).fillna(0.0)
    w = weights.reindex(r.index).astype(float).shift(1).fillna(1.0)
    gross = w * r + (1.0 - w) * rf
    drifted = (w * (1.0 + r) / (1.0 + gross)).shift(1).fillna(1.0)
    turnover = (w - drifted).abs()
    turnover.iloc[0] = abs(w.iloc[0] - 1.0)
    net = gross - turnover * cost_bps / 1e4
    return pd.DataFrame({"weight": w, "gross": gross, "net": net, "bh": r, "rf": rf,
                         "turnover": turnover, "gross_x": gross - rf, "net_x": net - rf,
                         "bh_x": r - rf})


def sharpe(x, periods: int = MONTHS) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan
    return float(x.mean() / x.std(ddof=1) * np.sqrt(periods))


def sharpe_diff_bootstrap(a, b, periods: int = MONTHS, block: int = 12, n_boot: int = 2000,
                          seed: int = 1017) -> dict:
    """Circular block bootstrap of ``Sharpe(a) − Sharpe(b)`` on paired excess returns.

    Pairs are resampled together (the two legs are ~95% correlated, which is what makes a small
    Sharpe gap testable at all); blocks preserve volatility clustering. ``p`` is one-sided for
    ``a`` beating ``b``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        st = rng.integers(0, n, nb)
        ii = ((st[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        diffs[i] = sharpe(a[ii], periods) - sharpe(b[ii], periods)
    d0 = sharpe(a, periods) - sharpe(b, periods)
    return {"diff": float(d0), "lo": float(np.quantile(diffs, 0.025)),
            "hi": float(np.quantile(diffs, 0.975)),
            "p": float(((diffs <= 0).sum() + 1) / (n_boot + 1))}


def summarize_backtest(bt: pd.DataFrame, periods: int = MONTHS) -> dict:
    """Excess-of-cash Sharpe for every leg, terminal wealth, vol, max drawdown, turnover."""
    def mdd(x):
        lv = (1.0 + x).cumprod()
        return float((lv / lv.cummax() - 1.0).min())
    yrs = len(bt) / periods
    return {
        "sharpe_bh": sharpe(bt["bh_x"], periods), "sharpe_gross": sharpe(bt["gross_x"], periods),
        "sharpe_net": sharpe(bt["net_x"], periods),
        "tw_bh": float((1.0 + bt["bh"]).prod()), "tw_gross": float((1.0 + bt["gross"]).prod()),
        "tw_net": float((1.0 + bt["net"]).prod()),
        "cagr_bh": float((1.0 + bt["bh"]).prod() ** (1 / yrs) - 1),
        "cagr_net": float((1.0 + bt["net"]).prod() ** (1 / yrs) - 1),
        "vol_bh": float(bt["bh"].std() * np.sqrt(periods)),
        "vol_net": float(bt["net"].std() * np.sqrt(periods)),
        "mdd_bh": mdd(bt["bh"]), "mdd_net": mdd(bt["net"]),
        "avg_weight": float(bt["weight"].mean()),
        "share_derisked": float((bt["weight"] < 1).mean()),
        "turnover_yr": float(bt["turnover"].sum() / yrs),
        "n_periods": int(len(bt)),
    }


def constant_mix(r: pd.Series, rf: pd.Series, w: float) -> pd.Series:
    """A monthly-rebalanced constant mix at weight ``w`` (costless; the exposure-matched foil)."""
    rf = rf.reindex(r.index).fillna(0.0)
    return w * r + (1.0 - w) * rf


# --------------------------------------------------------------------------- #
# 6. The verdict — thresholds fixed before the real tape was run
# --------------------------------------------------------------------------- #
P_REAL = 0.05
P_WEAK = 0.20


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    **Signal** — is there *excess* duration dependence, beyond what the dating rule manufactures?

    - ``p_dd`` = the larger (more conservative) of the random-walk and GARCH one-sided p-values for
      the real Weibull ``k`` exceeding the null's, on the main tape (FF monthly, ±20%).
    - ``p_pred`` = the larger of the two simulated p-values for the predictive slope of the
      next-12-month return on bull age being more negative than the null's; the regression
      counts only if its HAC t is also ≤ −2.
    - **Real** if ``p_dd < 0.05``, or ``p_pred < 0.05`` with HAC t ≤ −2.
    - **Weak** if either is below 0.20 (a lean the tape cannot certify).
    - **None** otherwise. A raw ``k > 1`` alone earns nothing: random walks produce it too.

    **Tradability** — the de-risk-old-bulls rule, net of costs, excess-vs-excess Sharpe vs
    buy-and-hold on the main tape (``sharpe_diff``, block-bootstrap one-sided ``p_sharpe``):

    - **Investable** if Signal is Real, ``p_sharpe < 0.05`` and the S&P daily cross-check also
      improves (``sp_sharpe_diff > 0``).
    - **Fragile** if ``sharpe_diff > 0`` and ``p_sharpe < 0.20``.
    - **Mirage** otherwise.
    """
    p_dd = max(h["p_k_rw"], h["p_k_garch"])
    p_pred = max(h["p_pred_rw"], h["p_pred_garch"])
    pred_ok = (h["pred_t"] <= -2.0) and (p_pred < P_REAL)
    if p_dd < P_REAL or pred_ok:
        signal = "Real"
    elif p_dd < P_WEAK or p_pred < P_WEAK:
        signal = "Weak"
    else:
        signal = "None"
    if (signal == "Real" and h["sharpe_diff"] > 0 and h["p_sharpe"] < P_REAL
            and h.get("sp_sharpe_diff", -1.0) > 0):
        trad = "Investable"
    elif h["sharpe_diff"] > 0 and h["p_sharpe"] < P_WEAK:
        trad = "Fragile"
    else:
        trad = "Mirage"

    k_txt = (f"Dated with the conventional ±20% rule on {h['n_years']:.0f} years of "
             f"total-return data, the {h['n_bulls']} completed US bull markets have a Weibull "
             f"shape of **k = {h['k']:.2f}** (cycle-bootstrap 95% CI "
             f"{h['k_lo']:.2f}–{h['k_hi']:.2f}); k > 1 is the textbook signature of "
             f"\"dying of old age\". But the ruler produces that by itself. The same rule applied to "
             f"random walks with the tape's own drift and volatility gives a median **k = "
             f"{h['k_null_rw']:.2f}** (GARCH: {h['k_null_garch']:.2f}), and the real bulls sit "
             f"at one-sided **p = {h['p_k_rw']:.2f}** against the random walk and "
             f"**p = {h['p_k_garch']:.2f}** against GARCH. ")
    pred_txt = (f"Bull age does not forecast the next year either: the slope of the "
                f"next-12-month excess log return on age is {h['pred_slope']:+.2%} per year of "
                f"age (Newey-West t = **{h['pred_t']:+.2f}**, simulated p = {p_pred:.2f}). ")
    sens_txt = h.get("sens_txt", "")
    if signal == "Real":
        why = k_txt + pred_txt + "Real bulls age faster than random-walk bulls. " + sens_txt
    elif signal == "Weak":
        why = (k_txt + pred_txt + "There is a lean in the claim's direction that the tape "
               "cannot certify. " + sens_txt)
    else:
        why = (k_txt + pred_txt + "Real bulls age like random-walk bulls: the apparent "
               "senescence is an artefact of the dating rule. " + sens_txt)
    trad_why = (f"Cutting equity to 50% once a bull outlives the median completed bull known "
                f"at the time (one month lag, {h['cost_bps']:.0f} bp one-way, bills as cash) "
                f"changes the excess-of-cash Sharpe from {h['sharpe_bh']:.2f} (buy-and-hold) to "
                f"{h['sharpe_net']:.2f} net — a gap of {h['sharpe_diff']:+.2f} "
                f"(block-bootstrap 95% CI {h['sharpe_lo']:+.2f} to {h['sharpe_hi']:+.2f}, "
                f"p = {h['p_sharpe']:.2f}) — while terminal wealth on $1 goes from "
                f"${h['tw_bh']:,.0f} to ${h['tw_net']:,.0f}: the rule spends "
                f"{h['share_derisked']:.0%} of months half in bills and simply forgoes the "
                f"equity premium in old bulls that kept running. On the out-of-sample S&P 500 "
                f"daily tape (price-only, 1990–2018) the Sharpe gap is "
                f"{h['sp_sharpe_diff']:+.2f}. ")
    trad_why += {"Investable": "It survives costs and both tapes.",
                 "Fragile": "On paper it holds up; the evidence is thin.",
                 "Mirage": "Nothing to harvest: de-risking old bulls is market timing on noise."
                 }[trad]
    one = (f"Bull markets only look like they die of old age (Weibull k = {h['k']:.2f}) "
           f"because a ±20% ruler builds ageing into any series — random walks dated the same "
           f"way age at least as fast (p = {p_dd:.2f}), bull age does not forecast next year's return "
           f"(t = {h['pred_t']:+.2f}), and selling half of every old bull ended with "
           f"{h['tw_bh'] / max(h['tw_net'], 1e-9):.1f}× less money than buy-and-hold.")
    if signal != "None":
        one = (f"Bull markets show Weibull k = {h['k']:.2f}; against random-walk bulls dated "
               f"the same way the excess ageing has p = {p_dd:.2f}, and the de-risk rule's net "
               f"Sharpe gap is {h['sharpe_diff']:+.2f}.")
    return {"signal": signal, "signal_why": why, "trad": trad, "trad_why": trad_why,
            "one_sentence": one, "p_dd": p_dd, "p_pred": p_pred}
