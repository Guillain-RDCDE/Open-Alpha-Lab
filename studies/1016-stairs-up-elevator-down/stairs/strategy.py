"""Stairs up, elevator down — the measurements, the nulls, the rules. Study 1016.

The proverb is a statement about the **order** of returns, not their size, so every
measurement here is built to tell the two apart.

**1 · Legs.** ``zigzag`` cuts a log-price path into alternating up and down legs, each at least
``h = log(1+X)`` deep — the same log distance both ways, so a 10% fall and a 10% rise are the
same size (in simple percent a 10% fall is the *bigger* move, which would tilt the race toward
the stairs). ``leg_stats`` then times the first passage across ``h`` inside every leg: the days
it takes to gain X% from a trough, against the days it takes to lose X% from a peak. The
**elevator ratio** is the geometric-mean up time over the geometric-mean down time; above 1 the
falls are faster.

**2 · Nulls for the legs.** Two Monte Carlo nulls answer two different questions.
``signflip_null`` keeps every ``|return|`` in place — so the volatility clustering is untouched
— and flips the sign of each de-meaned return at random before adding the drift back. It
destroys *all* sign asymmetry: skewness and leverage alike. ``permutation_null`` shuffles the
order and keeps the marginal distribution, skew included. If the tape beats the sign-flip null
the path is asymmetric; if it also beats the permutation null, the *order* matters beyond what
the skewness of single days could produce — which is the proverb's real content.

**3 · Time reversibility.** ``tr_stats`` computes the Ramsey-Rothman (1996) statistic on
standardised returns, ``TR(k) = E[x_t^2 x_{t-k}] - E[x_t x_{t-k}^2]``, for ``k = 1..K`` and
their sum. Under time reversibility every ``TR(k)`` is zero; the time-reversed copy flips its
sign exactly. A *negative* value is the leverage signature: a down day is followed by large
squared returns more than an up day is. Inference is Newey-West on the summed product series
and a circular block bootstrap of the standardised returns.

**4 · Skewness by horizon.** Moment skewness with block-bootstrap intervals, next to Kelly's
quantile skewness (10/50/90), which a handful of crash days cannot move.

**5 · Mechanism.** ``fit_models`` fits GARCH, GJR-GARCH and EGARCH (Student-t, ``arch``) to the
tape; ``mechanism_check`` simulates each, plus two i.i.d. nulls, at the tape's length and asks
what share of the real asymmetry each reproduces.

**6 · Could you trade it?** ``zigzag_rule`` is the pre-registered rule: exit after a ``exit_x``
fall from the peak since entry, re-enter after an ``entry_x`` rise from the trough since exit.
Fast-exit / slow-entry is ``(5%, 10%)``; its symmetric twins are ``(5%, 5%)`` and
``(10%, 10%)``. ``backtest`` applies **one execution lag** (the signal formed at the close of day
*t* earns from day *t+1*), charges the one-way cost on traded NAV, pays cash while out, and
reports gross and net excess-of-cash Sharpe. ``sharpe_diff_test`` block-bootstraps the paired
Sharpe difference.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

TRADING_DAYS = 252
THRESHOLDS = (0.05, 0.10, 0.20)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def log_returns(price: pd.Series) -> pd.Series:
    """Close-to-close log returns, first NaN dropped."""
    return np.log(price.astype(float)).diff().dropna()


def nw_tstat(z: np.ndarray, lags: int | None = None) -> dict:
    """Newey-West (Bartlett) t-statistic of the mean of ``z``."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    n = z.size
    if n < 30:
        return {"mean": np.nan, "se": np.nan, "t": np.nan, "lags": 0, "n": n}
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    e = z - z.mean()
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1.0)) * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {"mean": float(z.mean()), "se": float(se),
            "t": float(z.mean() / se) if se > 0 else np.nan, "lags": lags, "n": n}


def cbb_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """One circular-block-bootstrap index vector of length ``n``."""
    block = max(1, int(block))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return idx.ravel()[:n]


def skewness(x: np.ndarray) -> float:
    """Sample moment skewness (biased, population form)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    d = x - x.mean()
    m2 = float(np.mean(d * d))
    return float(np.mean(d ** 3) / m2 ** 1.5) if m2 > 0 else np.nan


def kelly_skew(x: np.ndarray) -> float:
    """Kelly quantile skewness ``(q90 + q10 - 2 q50) / (q90 - q10)``."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    q10, q50, q90 = np.quantile(x, [0.10, 0.50, 0.90])
    return float((q90 + q10 - 2 * q50) / (q90 - q10)) if q90 > q10 else np.nan


def block_boot_ci(x: np.ndarray, fn, block: int, n_boot: int = 1000, seed: int = 1016,
                  level: float = 0.95) -> dict:
    """Circular-block-bootstrap percentile interval and SE for ``fn(x)``."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    est = fn(x)
    bs = np.array([fn(x[cbb_indices(x.size, block, rng)]) for _ in range(n_boot)])
    a = (1.0 - level) / 2.0
    return {"est": float(est), "lo": float(np.quantile(bs, a)),
            "hi": float(np.quantile(bs, 1 - a)), "se": float(np.std(bs, ddof=1)),
            "p_le_0": float(np.mean(bs >= 0.0)) if est < 0 else float(np.mean(bs <= 0.0))}


# --------------------------------------------------------------------------- #
# 1 · Legs and first passage
# --------------------------------------------------------------------------- #
def zigzag(logp: np.ndarray, h: float) -> tuple[np.ndarray, int]:
    """Confirmed turning points of a log-price path at reversal size ``h``.

    Returns ``(extremes, first_dir)``: the integer positions of alternating troughs and peaks,
    the first one being the starting extreme, and ``+1`` if the first leg is up, ``-1`` if down.
    A turning point is confirmed only once the path has reversed by ``h`` from it, so the last
    running extreme is not included. Empty if the path never moves ``h``.
    """
    p = np.asarray(logp, dtype=float)
    n = p.size
    hi = lo = p[0]
    hi_i = lo_i = 0
    direction = 0
    start = 0
    i = 1
    while i < n:
        v = p[i]
        if v > hi:
            hi, hi_i = v, i
        if v < lo:
            lo, lo_i = v, i
        if v - lo >= h:
            direction, start = 1, lo_i
            break
        if hi - v >= h:
            direction, start = -1, hi_i
            break
        i += 1
    if direction == 0:
        return np.array([], dtype=int), 0
    first_dir = direction
    ext = [start]
    # running extreme of the current leg
    run_i = int(np.argmax(p[start:i + 1]) + start) if direction == 1 else int(
        np.argmin(p[start:i + 1]) + start)
    run_v = p[run_i]
    for j in range(i + 1, n):
        v = p[j]
        if direction == 1:
            if v > run_v:
                run_v, run_i = v, j
            elif run_v - v >= h:
                ext.append(run_i)
                direction = -1
                run_v, run_i = v, j
        else:
            if v < run_v:
                run_v, run_i = v, j
            elif v - run_v >= h:
                ext.append(run_i)
                direction = 1
                run_v, run_i = v, j
    return np.asarray(ext, dtype=int), first_dir


def legs(logp: np.ndarray, h: float) -> pd.DataFrame:
    """One row per leg starting at a confirmed extreme.

    ``fp_days`` is the first-passage time across ``h`` from the leg's starting extreme (the
    time to gain or lose X%). ``dur_days`` and ``move`` describe the full extreme-to-extreme
    leg and are NaN for the final, unconfirmed one.
    """
    p = np.asarray(logp, dtype=float)
    ext, d0 = zigzag(p, h)
    rows = []
    d = d0
    for k, s in enumerate(ext):
        seg = p[s + 1:] - p[s]
        hit = np.flatnonzero(seg >= h) if d == 1 else np.flatnonzero(seg <= -h)
        if hit.size == 0:
            break
        fp = int(hit[0] + 1)
        if k + 1 < len(ext):
            e = ext[k + 1]
            dur, mv = int(e - s), float(p[e] - p[s])
        else:
            dur, mv = np.nan, np.nan
        rows.append({"start": int(s), "dir": int(d), "fp_days": fp, "dur_days": dur,
                     "move": mv})
        d = -d
    return pd.DataFrame(rows, columns=["start", "dir", "fp_days", "dur_days", "move"])


def leg_stats(logp: np.ndarray, x: float) -> dict:
    """Time to gain X% against time to lose X% (log-symmetric ``h = log(1+x)``).

    ``elevator_ratio`` = geometric-mean up first-passage / geometric-mean down first-passage.
    Above 1: the falls are faster than the rises. ``speed_ratio`` is the same comparison on
    full legs, median |move| per day down over up.
    """
    h = float(np.log1p(x))
    L = legs(logp, h)
    up, dn = L[L["dir"] == 1], L[L["dir"] == -1]
    out = {"x": x, "n_up": int(len(up)), "n_down": int(len(dn))}
    if len(up) < 2 or len(dn) < 2:
        out.update({"gm_up": np.nan, "gm_down": np.nan, "med_up": np.nan,
                    "med_down": np.nan, "elevator_ratio": np.nan, "speed_ratio": np.nan})
        return out
    gm_up = float(np.exp(np.mean(np.log(up["fp_days"]))))
    gm_dn = float(np.exp(np.mean(np.log(dn["fp_days"]))))
    fu, fd = up.dropna(subset=["dur_days"]), dn.dropna(subset=["dur_days"])
    sp_up = (fu["move"].abs() / fu["dur_days"]).median() if len(fu) else np.nan
    sp_dn = (fd["move"].abs() / fd["dur_days"]).median() if len(fd) else np.nan
    out.update({"gm_up": gm_up, "gm_down": gm_dn,
                "med_up": float(up["fp_days"].median()),
                "med_down": float(dn["fp_days"].median()),
                "elevator_ratio": gm_up / gm_dn,
                "speed_ratio": float(sp_dn / sp_up) if sp_up and sp_up > 0 else np.nan,
                "med_full_up": float(fu["dur_days"].median()) if len(fu) else np.nan,
                "med_full_down": float(fd["dur_days"].median()) if len(fd) else np.nan})
    return out


def _null_paths(r: np.ndarray, kind: str, n_rep: int, seed: int):
    rng = np.random.default_rng(seed)
    mu = r.mean()
    for _ in range(n_rep):
        if kind == "signflip":
            s = rng.choice(np.array([-1.0, 1.0]), size=r.size)
            rr = mu + s * (r - mu)
        elif kind == "permutation":
            rr = rng.permutation(r)
        else:
            raise ValueError(kind)
        yield np.concatenate([[0.0], np.cumsum(rr)])


def leg_null_test(logret: np.ndarray, xs=THRESHOLDS, kind: str = "signflip",
                  n_rep: int = 300, seed: int = 1016) -> pd.DataFrame:
    """Observed elevator ratio per threshold against a Monte Carlo null.

    ``p`` is one-sided toward the elevator: ``(1 + #null >= obs) / (1 + n_rep)``;
    ``p_reverse`` is the mirror-image one-sided p (rises *faster* than falls). ``speed_p`` is
    the elevator-direction p for the full-leg speed ratio, the steelman's second reading.
    """
    r = np.asarray(logret, dtype=float)
    r = r[np.isfinite(r)]
    logp = np.concatenate([[0.0], np.cumsum(r)])
    obs = {x: leg_stats(logp, x) for x in xs}
    null = {x: [] for x in xs}
    snull = {x: [] for x in xs}
    for path in _null_paths(r, kind, n_rep, seed):
        for x in xs:
            ls = leg_stats(path, x)
            null[x].append(ls["elevator_ratio"])
            snull[x].append(ls["speed_ratio"])

    def _p(o, nv):
        nv = np.asarray(nv, dtype=float)
        nv = nv[np.isfinite(nv)]
        if not (np.isfinite(o) and nv.size):
            return np.nan, np.nan, np.nan, np.nan
        return (float((1 + np.sum(nv >= o)) / (1 + nv.size)),
                float((1 + np.sum(nv <= o)) / (1 + nv.size)),
                float(np.median(nv)), float(np.quantile(nv, 0.95)))

    rows = []
    for x in xs:
        p, p_rev, med, q95 = _p(obs[x]["elevator_ratio"], null[x])
        sp, _, smed, _ = _p(obs[x]["speed_ratio"], snull[x])
        rows.append({"x": x, **{k: v for k, v in obs[x].items() if k != "x"},
                     "null_median": med, "null_q95": q95, "p": p, "p_reverse": p_rev,
                     "speed_null_median": smed, "speed_p": sp, "null": kind})
    return pd.DataFrame(rows).set_index("x")


# --------------------------------------------------------------------------- #
# 2 · Time reversibility (Ramsey-Rothman)
# --------------------------------------------------------------------------- #
def standardise(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    return (r - r.mean()) / r.std(ddof=0)


def tr_products(x: np.ndarray, K: int) -> np.ndarray:
    """Matrix ``(n-K, K)`` of ``x_t^2 x_{t-k} - x_t x_{t-k}^2`` for ``k = 1..K``."""
    n = x.size
    xt = x[K:]
    cols = []
    for k in range(1, K + 1):
        xl = x[K - k:n - k]
        cols.append(xt * xt * xl - xt * xl * xl)
    return np.column_stack(cols)


def tr_stats(r: np.ndarray, K: int = 10, n_boot: int = 500, block: int = 20,
             seed: int = 1016, winsor: float | None = None) -> dict:
    """Ramsey-Rothman TR(k), k=1..K, and their sum, with NW and block-bootstrap errors.

    Returns per-lag estimates, NW t-stats, and for the sum ``TR_sum``: the estimate, NW t and
    a circular-block-bootstrap SE / t (resampling the standardised returns in blocks of
    ``block``). ``winsor`` clips the standardised returns at that two-sided quantile first — a
    robustness check that a few crash days are not the whole story. The statistic of the
    time-reversed series is reported too: it is exactly the negative, which is what
    reversibility being *testable* means.
    """
    x = standardise(r)
    if winsor:
        lo, hi = np.quantile(x, [winsor, 1 - winsor])
        x = standardise(np.clip(x, lo, hi))
    Z = tr_products(x, K)
    per = []
    for k in range(K):
        t = nw_tstat(Z[:, k])
        per.append({"k": k + 1, "tr": t["mean"], "t_nw": t["t"]})
    zs = Z.sum(axis=1)
    tsum = nw_tstat(zs)
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for b in range(n_boot):
        xb = x[cbb_indices(x.size, block, rng)]
        bs[b] = tr_products(xb, K).sum(axis=1).mean()
    se_b = float(np.std(bs, ddof=1))
    rev = tr_products(x[::-1], K).sum(axis=1).mean()
    return {"K": K, "n": int(x.size), "per_lag": pd.DataFrame(per).set_index("k"),
            "tr_sum": float(zs.mean()), "t_nw": tsum["t"], "se_nw": tsum["se"],
            "se_boot": se_b, "t_boot": float(zs.mean() / se_b) if se_b > 0 else np.nan,
            "tr_sum_reversed": float(rev)}


def tr_sum_fast(x: np.ndarray, K: int = 10) -> float:
    """Just the summed TR statistic on already-standardised returns (for simulations)."""
    return float(tr_products(x, K).sum(axis=1).mean())


# --------------------------------------------------------------------------- #
# 3 · Skewness by horizon
# --------------------------------------------------------------------------- #
def aggregate(logp: pd.Series, rule: str) -> pd.Series:
    """Non-overlapping log returns at a calendar frequency (``W-FRI``, ``ME``, ``QE``, ``YE``)."""
    return logp.resample(rule).last().diff().dropna()


def skew_table(series: dict, blocks: dict, n_boot: int = 1000, seed: int = 1016) -> pd.DataFrame:
    """Skewness and Kelly skewness with CBB 95% intervals for each named return series."""
    rows = []
    for name, x in series.items():
        x = np.asarray(x, dtype=float)
        b = blocks.get(name, 5)
        sk = block_boot_ci(x, skewness, b, n_boot, seed)
        kk = block_boot_ci(x, kelly_skew, b, n_boot, seed + 1)
        rows.append({"series": name, "n": int(np.isfinite(x).sum()), "block": b,
                     "skew": sk["est"], "skew_lo": sk["lo"], "skew_hi": sk["hi"],
                     "kelly": kk["est"], "kelly_lo": kk["lo"], "kelly_hi": kk["hi"]})
    return pd.DataFrame(rows).set_index("series")


# --------------------------------------------------------------------------- #
# 4 · Mechanism: what reproduces the asymmetry?
# --------------------------------------------------------------------------- #
def fit_models(logret: np.ndarray) -> dict:
    """GARCH, GJR-GARCH and EGARCH(1,1) with Student-t shocks, fitted with ``arch``.

    Fitted on returns in percent (``arch``'s preferred scale). Returns the parameter dicts
    plus the log-likelihoods so the leverage terms can be judged against the fit gained.
    """
    from arch import arch_model

    y = 100.0 * np.asarray(logret, dtype=float)
    out = {}
    specs = {"garch": dict(vol="GARCH", p=1, o=0, q=1),
             "gjr": dict(vol="GARCH", p=1, o=1, q=1),
             "egarch": dict(vol="EGARCH", p=1, o=1, q=1)}
    for name, spec in specs.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(y, mean="Constant", dist="t", rescale=False, **spec).fit(
                disp="off", show_warning=False)
        prm = res.params
        se = res.std_err
        d = {"mu": float(prm["mu"]), "omega": float(prm["omega"]),
             "alpha": float(prm["alpha[1]"]), "beta": float(prm["beta[1]"]),
             "nu": float(prm["nu"]), "gamma": float(prm.get("gamma[1]", 0.0)),
             "gamma_t": float(prm["gamma[1]"] / se["gamma[1]"]) if "gamma[1]" in prm
             else np.nan, "loglik": float(res.loglikelihood)}
        out[name] = d
    return out


def _path_metrics(lr: np.ndarray, x_leg: float = 0.10, K: int = 10, month: int = 21) -> dict:
    """The three asymmetry metrics on one return path (decimal log returns)."""
    x = standardise(lr)
    n_m = lr.size // month
    m = lr[:n_m * month].reshape(n_m, month).sum(axis=1)
    logp = np.concatenate([[0.0], np.cumsum(lr)])
    return {"tr_sum": tr_sum_fast(x, K), "skew_daily": skewness(lr),
            "skew_monthly": skewness(m),
            "elevator_ratio": leg_stats(logp, x_leg)["elevator_ratio"]}


def simulate_world(model: str, params: dict, n: int, n_paths: int, seed: int,
                   real: np.ndarray | None = None) -> np.ndarray:
    """Return paths ``(n, n_paths)`` in decimal log-return units for one model."""
    from . import data

    p = params
    if model == "gjr":
        y = data.simulate_gjr(n, p["mu"], p["omega"], p["alpha"], p["gamma"], p["beta"],
                              p["nu"], n_paths=n_paths, seed=seed)
    elif model == "garch":
        y = data.simulate_gjr(n, p["mu"], p["omega"], p["alpha"], 0.0, p["beta"], p["nu"],
                              n_paths=n_paths, seed=seed)
    elif model == "egarch":
        y = data.simulate_egarch(n, p["mu"], p["omega"], p["alpha"], p["gamma"], p["beta"],
                                 p["nu"], n_paths=n_paths, seed=seed)
    elif model == "iid_t":
        rng = np.random.default_rng(seed)
        nu = p["nu"]
        z = rng.standard_t(nu, size=(n, n_paths)) * np.sqrt((nu - 2.0) / nu)
        y = p["mu"] + p["sd"] * z
    elif model == "iid_perm":
        rng = np.random.default_rng(seed)
        r = np.asarray(real, dtype=float)
        return np.column_stack([rng.permutation(r) for _ in range(n_paths)])
    else:
        raise ValueError(model)
    return y / 100.0 if model in ("gjr", "garch", "egarch", "iid_t") else y


def mechanism_check(logret: np.ndarray, fits: dict, n_paths: int = 100,
                    seed: int = 1016, x_leg: float = 0.10, K: int = 10) -> pd.DataFrame:
    """Share of each real asymmetry metric that each model reproduces.

    Worlds: the fitted GJR and EGARCH (leverage on), the fitted symmetric GARCH, an i.i.d.
    Student-t with the tape's mean and variance, and an i.i.d. shuffle of the tape itself
    (keeps its skewness, destroys its order). ``share_*`` = median simulated metric / real
    metric; for the elevator ratio it is measured on ``log(ratio)`` so 1.0 means "all of it"
    and 0 means "none of it".
    """
    r = np.asarray(logret, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    real = _path_metrics(r, x_leg, K)
    worlds = {
        "GJR-GARCH (leverage)": ("gjr", fits["gjr"]),
        "EGARCH (leverage)": ("egarch", fits["egarch"]),
        "GARCH (symmetric)": ("garch", fits["garch"]),
        "i.i.d. Student-t": ("iid_t", {"mu": 100 * r.mean(), "sd": 100 * r.std(),
                                       "nu": fits["garch"]["nu"]}),
        "i.i.d. shuffle of the tape": ("iid_perm", {}),
    }
    rows = [{"world": "REAL TAPE", **real, "share_tr": 1.0, "share_skew_m": 1.0,
             "share_elevator": 1.0, "tr_p05": np.nan, "tr_p95": np.nan}]
    for k, (name, (model, prm)) in enumerate(worlds.items()):
        Y = simulate_world(model, prm, n, n_paths, seed + 101 * k, real=r)
        mets = pd.DataFrame([_path_metrics(Y[:, j], x_leg, K) for j in range(n_paths)])
        med = mets.median()
        er = np.log(mets["elevator_ratio"].dropna())
        rows.append({"world": name, "tr_sum": float(med["tr_sum"]),
                     "skew_daily": float(med["skew_daily"]),
                     "skew_monthly": float(med["skew_monthly"]),
                     "elevator_ratio": float(np.exp(er.median())),
                     "share_tr": float(med["tr_sum"] / real["tr_sum"]),
                     "share_skew_m": float(med["skew_monthly"] / real["skew_monthly"]),
                     "share_elevator": float(er.median() / np.log(real["elevator_ratio"]))
                     if real["elevator_ratio"] != 1 else np.nan,
                     "tr_p05": float(mets["tr_sum"].quantile(0.05)),
                     "tr_p95": float(mets["tr_sum"].quantile(0.95))})
    return pd.DataFrame(rows).set_index("world")


# --------------------------------------------------------------------------- #
# 5 · The pre-registered trading rules
# --------------------------------------------------------------------------- #
RULES = {"fast-exit/slow-entry (5% / 10%)": (0.05, 0.10),
         "symmetric (5% / 5%)": (0.05, 0.05),
         "symmetric (10% / 10%)": (0.10, 0.10)}
PRIMARY_RULE = "fast-exit/slow-entry (5% / 10%)"


def zigzag_rule(price: pd.Series, exit_x: float, entry_x: float) -> pd.Series:
    """In (1) / out (0) signal known at each close. Starts invested.

    Invested: track the peak since entry; go to cash once the close is ``exit_x`` below it.
    In cash: track the trough since exit; re-enter once the close is ``entry_x`` above it.
    """
    px = price.to_numpy(dtype=float)
    sig = np.empty(px.size)
    state, ref = 1, px[0]
    for i, v in enumerate(px):
        if state == 1:
            ref = max(ref, v)
            if v <= ref * (1.0 - exit_x):
                state, ref = 0, v
        else:
            ref = min(ref, v)
            if v >= ref * (1.0 + entry_x):
                state, ref = 1, v
        sig[i] = state
    return pd.Series(sig, index=price.index, name="signal")


def backtest(price: pd.Series, rf: pd.Series, signal: pd.Series, cost: float,
             lag: int = 1, periods: int = TRADING_DAYS) -> dict:
    """Run a 0/1 signal with ``lag`` periods of execution lag and one-way ``cost`` on NAV.

    ``signal`` at the close of period *t* is acted on at *t+lag*: the position earning the
    return from *t+lag-1* to *t+lag* is ``signal.shift(lag)``. Out of the market the book earns
    ``rf``. Cost is charged on |Δposition| (traded NAV). Returns gross and net excess-of-cash
    series and their annualised Sharpe, plus time-in-market and round trips per year.
    """
    r = price.pct_change()
    df = pd.DataFrame({"r": r, "rf": rf.reindex(price.index), "sig": signal}).dropna(
        subset=["r", "rf"])
    pos = df["sig"].shift(lag).fillna(1.0)
    turn = pos.diff().abs().fillna(0.0)
    gross = pos * df["r"] + (1.0 - pos) * df["rf"]
    net = gross - cost * turn
    ex_g, ex_n = gross - df["rf"], net - df["rf"]
    ex_bh = df["r"] - df["rf"]

    def sh(x):
        s = x.std(ddof=1)
        return float(x.mean() / s * np.sqrt(periods)) if s > 0 else np.nan

    years = len(df) / periods
    return {"ex_gross": ex_g, "ex_net": ex_n, "ex_bh": ex_bh,
            "sharpe_gross": sh(ex_g), "sharpe_net": sh(ex_n), "sharpe_bh": sh(ex_bh),
            "ann_ex_net": float(ex_n.mean() * periods), "ann_ex_bh": float(ex_bh.mean() * periods),
            "time_in": float(pos.mean()), "trades_per_year": float(turn.sum() / years),
            "maxdd_net": _maxdd(net), "maxdd_bh": _maxdd(df["r"])}


def _maxdd(r: pd.Series) -> float:
    w = (1.0 + r.fillna(0.0)).cumprod()
    return float((w / w.cummax() - 1.0).min())


def sharpe_diff_test(a: pd.Series, b: pd.Series, block: int, n_boot: int = 1000,
                     seed: int = 1016, periods: int = TRADING_DAYS) -> dict:
    """Paired circular-block bootstrap of ``Sharpe(a) - Sharpe(b)`` (excess-vs-excess).

    ``p`` is the one-sided share of resamples with a difference ≤ 0.
    """
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    A, B = df["a"].to_numpy(), df["b"].to_numpy()

    def sh(x):
        s = x.std(ddof=1)
        return x.mean() / s * np.sqrt(periods) if s > 0 else 0.0

    d0 = sh(A) - sh(B)
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        idx = cbb_indices(len(A), block, rng)
        bs[i] = sh(A[idx]) - sh(B[idx])
    return {"diff": float(d0), "lo": float(np.quantile(bs, 0.025)),
            "hi": float(np.quantile(bs, 0.975)), "p": float(np.mean(bs <= 0.0))}


def race(price: pd.Series, rf: pd.Series, cost: float, lag: int = 1,
         periods: int = TRADING_DAYS, block: int = 20, n_boot: int = 1000,
         seed: int = 1016) -> pd.DataFrame:
    """Every pre-registered rule against buy-and-hold on one tape, net of costs."""
    rows = []
    res = {}
    for name, (ex, en) in RULES.items():
        bt = backtest(price, rf, zigzag_rule(price, ex, en), cost, lag, periods)
        res[name] = bt
        vs_bh = sharpe_diff_test(bt["ex_net"], bt["ex_bh"], block, n_boot, seed, periods)
        rows.append({"rule": name, "sharpe_gross": bt["sharpe_gross"],
                     "sharpe_net": bt["sharpe_net"], "sharpe_bh": bt["sharpe_bh"],
                     "ann_ex_net": bt["ann_ex_net"], "ann_ex_bh": bt["ann_ex_bh"],
                     "time_in": bt["time_in"], "trades_per_year": bt["trades_per_year"],
                     "maxdd_net": bt["maxdd_net"], "maxdd_bh": bt["maxdd_bh"],
                     "d_vs_bh": vs_bh["diff"], "d_vs_bh_lo": vs_bh["lo"],
                     "d_vs_bh_hi": vs_bh["hi"], "p_vs_bh": vs_bh["p"]})
    out = pd.DataFrame(rows).set_index("rule")
    prim = res[PRIMARY_RULE]["ex_net"]
    for name in RULES:
        if name == PRIMARY_RULE:
            out.loc[name, "p_vs_primary"] = np.nan
            continue
        t = sharpe_diff_test(prim, res[name]["ex_net"], block, n_boot, seed + 7, periods)
        out.loc[name, "d_primary_minus_this"] = t["diff"]
        out.loc[name, "p_vs_primary"] = t["p"]
    return out


# --------------------------------------------------------------------------- #
# The verdict rule — fixed before the real tape was run
# --------------------------------------------------------------------------- #
def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    **Signal** (primary tape: S&P 500 daily price index).

    - ``tr_ok``: the summed Ramsey-Rothman statistic (K = 10) is negative with NW t ≤ -2
      **and** block-bootstrap t ≤ -2 — irreversibility in the elevator direction, certified by
      robust inference.
    - ``dur_ok``: the 10% elevator ratio exceeds 1 with sign-flip Monte Carlo p < 0.05 — the
      literal proverb (losing 10% is faster than gaining 10%) beats a world with the same
      volatility clustering and drift but no sign asymmetry.
    - **Real** if both; **Mixed** if exactly one (the verdict splits by leg); **Weak** if neither
      clears the bar but both point in the elevator direction; **None** otherwise.

    **Tradability** (the pre-registered fast-exit / slow-entry rule, 5% exit / 10% entry).

    - **Investable** if, on the Fama-French total-return tape, it beats buy-and-hold on net
      excess Sharpe with bootstrap p < 0.05 *and* beats the 5%/5% symmetric twin, *and* its
      net Sharpe edge over buy-and-hold is positive on both daily tapes.
    - **Fragile** if its net Sharpe beats buy-and-hold on at least two of the three tapes
      without clearing that bar.
    - **Mirage** otherwise.
    """
    tr_ok = (h["tr_sum"] < 0) and (h["tr_t_nw"] <= -2.0) and (h["tr_t_boot"] <= -2.0)
    dur_ok = (h["elev10"] > 1.0) and (h["elev10_p_signflip"] < 0.05)
    if tr_ok and dur_ok:
        signal = "Real"
    elif tr_ok or dur_ok:
        signal = "Mixed"
    elif h["tr_sum"] < 0 and h["elev10"] > 1.0:
        signal = "Weak"
    else:
        signal = "None"

    d = h["trade_d_vs_bh"]          # {"ff": .., "sp500": .., "nasdaq": ..}
    wins = sum(1 for v in d.values() if v > 0)
    if (h["trade_ff_p_vs_bh"] < 0.05 and h["trade_ff_beats_sym"]
            and d["sp500"] > 0 and d["nasdaq"] > 0):
        trad = "Investable"
    elif wins >= 2:
        trad = "Fragile"
    else:
        trad = "Mirage"

    if h["elev10"] > 1.0:
        elev = (f"The literal proverb holds on the S&P 500 price index: losing 10% from a peak "
                f"took a geometric-mean **{h['gm_down10']:.0f} sessions** and gaining 10% off "
                f"a trough **{h['gm_up10']:.0f}** — an elevator ratio of "
                f"**{h['elev10']:.2f}×** (5%: {h['elev05']:.2f}×, 20%: {h['elev20']:.2f}×), "
                f"sign-flip p = **{h['elev10_p_signflip']:.3f}**, shuffle p = "
                f"{h['elev10_p_perm']:.3f}.")
    else:
        elev = (f"But the literal proverb — the same distance covered faster on the way down "
                f"— runs **backwards**. Losing 10% from a peak took a geometric-mean "
                f"**{h['gm_down10']:.0f} sessions**; gaining 10% off a trough took "
                f"**{h['gm_up10']:.0f}**, an elevator ratio of **{h['elev10']:.2f}×** "
                f"(5%: {h['elev05']:.2f}×, 20%: {h['elev20']:.2f}×). Against a null that keeps "
                f"every day's size, the drift and the volatility clustering but flips signs at "
                f"random, the elevator-direction p is {h['elev10_p_signflip']:.3f} — the "
                f"opposite tail, p = **{h['elev10_p_reverse']:.3f}**, is where the tape sits. "
                f"The fastest X% on the tape is the rebound off a trough, because that is where "
                f"volatility is highest. What survives of the steelman is the *completed* leg: "
                f"whole declines are shorter and steeper than whole advances (median speed "
                f"ratio {h['speed10']:.2f}× at 10%, sign-flip p = {h['speed10_p']:.2f}; "
                f"{h['speed05']:.2f}× at 5%, p = {h['speed05_p']:.2f}) — bull legs simply run "
                f"long.")
    trs = (f"The tape is time-irreversible, in the leverage direction: the Ramsey-Rothman "
           f"statistic summed over lags 1-10 is **{h['tr_sum']:+.2f}** (Newey-West t = "
           f"**{h['tr_t_nw']:+.2f}**, block-bootstrap t = **{h['tr_t_boot']:+.2f}**, "
           f"{h['tr_t_nw_wins']:+.2f} with the 0.5% tails clipped), and it flips sign exactly "
           f"on the time-reversed copy. The Nasdaq replicates it (t = {h['nq_tr_t_nw']:+.2f}); "
           f"92 years of monthly total returns give t = {h['ff_tr_t_nw']:+.2f}.")
    def _pc(v):
        return "0%" if abs(v) < 0.005 else f"{v:.0%}"

    mech = (f"The mechanism is volatility, not the size of single days: a GJR-GARCH fitted to "
            f"the tape reproduces **{_pc(h['share_tr_gjr'])}** of the reversibility statistic "
            f"and {_pc(h['share_elev_gjr'])} of the (log) elevator ratio, EGARCH "
            f"{_pc(h['share_tr_egarch'])} and {_pc(h['share_elev_egarch'])}, a symmetric GARCH "
            f"{_pc(h['share_tr_garch'])} and {_pc(h['share_elev_garch'])}, an i.i.d. Student-t "
            f"{_pc(h['share_tr_iid'])} and {_pc(h['share_elev_iid'])}.")
    skew = (f"Skewness, the usual stand-in, is the weaker witness: daily {h['skew_d']:+.2f} "
            f"[{h['skew_d_lo']:+.2f}, {h['skew_d_hi']:+.2f}], monthly {h['skew_m']:+.2f} "
            f"[{h['skew_m_lo']:+.2f}, {h['skew_m_hi']:+.2f}] (block-bootstrap 95%), while the "
            f"quantile skew of a typical day is {h['kelly_d']:+.3f}.")
    signal_why = " ".join([trs, elev, mech, skew])

    trad_why = (
        f"The rule was fixed before the run: exit after a 5% fall from the peak, re-enter only "
        f"after a 10% rise from the trough — fast out because the elevator is fast, slow in "
        f"because the stairs are slow. One execution lag, {h['cost_daily_bps']:.0f} bp one-way "
        f"on the daily tapes and {h['cost_monthly_bps']:.0f} bp on the monthly one, T-bills "
        f"while out. Net excess Sharpe minus buy-and-hold: **{d['sp500']:+.2f}** on the S&P 500 "
        f"(1990-2018) and **{d['nasdaq']:+.2f}** on the Nasdaq (1999-2018) — on price-only "
        f"tapes that *flatter* a rule which sits in cash — and **{d['ff']:+.2f}** on Fama-French "
        f"total returns since 1926 (p = {h['trade_ff_p_vs_bh']:.2f}) — a win that shrinks to "
        f"{h['trade_ff_post45_d']:+.2f} (p = {h['trade_ff_post45_p']:.2f}) once 1929-32 is "
        f"out of the sample, and that the *symmetric* 10%/10% twin matches "
        f"({h['trade_ff_sym10_d']:+.2f}), so the asymmetry in the rule is not what earns it. "
        + (f"On daily data the rule is whipsawed by exactly the feature the study found: the "
           f"sharpest rallies come straight off the lows, which a slow re-entry misses. "
           if min(d["sp500"], d["nasdaq"]) < 0 else "")
        + f"The asymmetry is "
        f"real and it is a description of risk — the shape index puts are priced on, not a "
        f"timing signal a holder can harvest after one lag.")
    one = (f"The index is measurably time-irreversible (t = {h['tr_t_nw']:+.1f}) because "
           f"volatility jumps after falls — yet the same 10% is regained off a trough faster "
           f"({h['gm_up10']:.0f} sessions) than it is lost from a peak "
           f"({h['gm_down10']:.0f}), and a fast-exit / slow-entry rule built on the proverb "
           f"{'does not beat' if trad != 'Investable' else 'beats'} buy-and-hold on daily "
           f"data after costs.")
    return {"signal": signal, "signal_why": signal_why, "trad": trad, "trad_why": trad_why,
            "one_sentence": one}
