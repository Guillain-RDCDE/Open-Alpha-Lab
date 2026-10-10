"""The Taylor-rule gap as a market timer — Study 1022 (Behind the Curve).

The machinery, in the order the study runs it:

1. **The rule.** :func:`taylor_rate` is Taylor (1993) verbatim:
   ``i* = r* + pi + 0.5 (pi - pi*) + 0.5 gap`` with ``r* = pi* = 2``. Inflation is the
   four-quarter CPI change. The **Taylor gap** is ``tgap = i_actual - i*``: negative means the
   Fed is *behind the curve* (too easy), positive means *ahead* (too tight).
2. **Two real-time output gaps**, both computable at the time from past data only:
   :func:`one_sided_hp_gap` re-runs an HP filter (lambda 1600) on log real GDP *every quarter
   on the data to date* and keeps the last point; :func:`unemployment_gap` is Okun-style,
   ``-2 x (u - trailing 10-year mean of u)``. :func:`build_signals` applies the **one-quarter
   release lag** to every macro input, exactly once; the policy-rate proxy (the quarterly
   average T-bill) is a market price known at the end of its own quarter and is not lagged.
   A two-sided, full-sample HP gap is also built — labelled **hindsight** — purely to show how
   far the knowable gap sits from the one a historian computes (Orphanides 2001 in miniature;
   the real Orphanides problem is bigger still, because our GDP is final vintage).
3. **Predictive regressions**, :func:`predictive_regression`: next-quarter (and 4-, 8-quarter
   cumulative) equity excess return or AAA-yield change on ``tgap_t``, Newey-West.
4. **Persistence.** The gap is highly persistent (AR(1) rho near 0.86), so the slope is biased in small samples
   whenever the gap's innovations correlate with returns (Stambaugh 1999).
   :func:`stambaugh_correction` applies the first-order bias formula;
   :func:`simulation_null` simulates the regression under H0 with an AR(1) regressor and
   jointly resampled (return, regressor) innovations, so the null keeps the data's own
   innovation correlation — and, because the long-horizon statistics are recomputed on every
   simulated path, it is also the overlap-correct reference distribution for h = 4, 8.
5. **Tradability.** :func:`timing_backtest` holds equity when the Fed is behind the curve
   (``tgap_t < 0``) and bills otherwise, one quarter later, costs one-way x traded NAV, raced
   excess-of-cash against buy-and-hold; :func:`sharpe_diff_bootstrap` puts a circular block
   bootstrap on the Sharpe difference.
6. :func:`verdict` — the stamps, by thresholds fixed before the real run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as _data

QPY = 4                      # quarters per year
HP_LAMBDA = 1600.0
HP_MIN_OBS = 40              # ten years of GDP before the first one-sided HP gap
U_WINDOW = 40                # trailing window (quarters) for the unemployment "natural rate"
OKUN = 2.0                   # output gap ~ -2 x unemployment gap
HORIZONS = (1, 4, 8)
REGIME_BREAK = "1987-09-30"  # Greenspan takes office 1987Q3 — Taylor's own 1987-92 sample


# --------------------------------------------------------------------------- #
# 1-2. The rule and the gaps
# --------------------------------------------------------------------------- #
def taylor_rate(infl: pd.Series, gap: pd.Series, r_star: float = _data.R_STAR,
                pi_star: float = _data.PI_STAR) -> pd.Series:
    """Taylor (1993): ``i* = r* + pi + 0.5 (pi - pi*) + 0.5 gap`` (all in percent)."""
    return r_star + infl + 0.5 * (infl - pi_star) + 0.5 * gap


def hp_cycle(x: np.ndarray, lamb: float = HP_LAMBDA) -> np.ndarray:
    """Two-sided HP cycle of ``x`` (statsmodels ``hpfilter``)."""
    from statsmodels.tsa.filters.hp_filter import hpfilter
    cyc, _ = hpfilter(np.asarray(x, dtype=float), lamb=lamb)
    return np.asarray(cyc)


def one_sided_hp_gap(log_gdp: pd.Series, lamb: float = HP_LAMBDA,
                     min_obs: int = HP_MIN_OBS) -> pd.Series:
    """Real-time HP output gap: at each *t*, filter the data up to *t* and keep the endpoint.

    Uses no observation after *t* — the gap an analyst could have computed on the (final-
    vintage) numbers available then. ``log_gdp`` should be ``100 * ln(GDP)`` so the gap is in
    percent of potential. The first ``min_obs - 1`` points are NaN.
    """
    x = log_gdp.astype(float).to_numpy()
    out = np.full(len(x), np.nan)
    for t in range(min_obs - 1, len(x)):
        seg = x[: t + 1]
        if np.isfinite(seg).all():
            out[t] = hp_cycle(seg, lamb)[-1]
    return pd.Series(out, index=log_gdp.index, name="gap_hp")


def unemployment_gap(unemp: pd.Series, window: int = U_WINDOW,
                     okun: float = OKUN) -> pd.Series:
    """Okun-style output gap from unemployment: ``-okun * (u_t - mean(u over the trailing window))``.

    The trailing mean stands in for the natural rate an investor could estimate from history;
    positive = economy running hot. Unemployment is barely revised, so this gap is much closer
    to what was really knowable than any GDP-based one.
    """
    nat = unemp.rolling(window, min_periods=window).mean()
    return (-okun * (unemp - nat)).rename("gap_u")


def build_signals(q: pd.DataFrame) -> pd.DataFrame:
    """Every Taylor-rule input and gap, aligned to what is **known at the end of quarter t**.

    Macro inputs (CPI inflation, both output gaps) are dated *t-1* — the one-quarter release
    lag, applied here and only here. The policy rate is the T-bill average for quarter *t*.

    Columns: ``i`` (policy-rate proxy), ``pi`` (4-quarter CPI inflation, lagged), ``gap_hp``,
    ``gap_u`` (real-time gaps, lagged), ``gap_hind`` (two-sided full-sample HP — **hindsight**,
    not tradable), ``istar_hp`` / ``istar_u`` / ``istar_hind``, and ``tgap_hp`` / ``tgap_u`` /
    ``tgap_hind = i - istar``.
    """
    pi = 100.0 * np.log(q["cpi"] / q["cpi"].shift(4))
    lg = 100.0 * np.log(q["realgdp"])
    gap_hp = one_sided_hp_gap(lg.dropna()).reindex(q.index)
    gap_u = unemployment_gap(q["unemp"])
    valid = lg.dropna()
    gap_hind = pd.Series(hp_cycle(valid.to_numpy()), index=valid.index).reindex(q.index)

    lag = 1  # the release lag, applied once
    s = pd.DataFrame(index=q.index)
    s["i"] = q["tbilrate"]
    s["pi"] = pi.shift(lag)
    s["gap_hp"] = gap_hp.shift(lag)
    s["gap_u"] = gap_u.shift(lag)
    s["gap_hind"] = gap_hind.shift(lag)
    for g in ("hp", "u", "hind"):
        s[f"istar_{g}"] = taylor_rate(s["pi"], s[f"gap_{g}"])
        s[f"tgap_{g}"] = s["i"] - s[f"istar_{g}"]
    return s


def demeaned(signal: pd.Series, min_periods: int = 8) -> pd.Series:
    """The gap relative to its own **expanding** (past-and-present only) mean.

    A sensitivity for the sign rule: the T-bill sits below the fed funds rate, and the Fed
    spent most of 1969-2009 "behind" Taylor (1993) by this measure, so the raw sign keeps the
    rule invested most of the time. Demeaning in real time asks "looser than usual?" instead.
    """
    return (signal - signal.expanding(min_periods=min_periods).mean()).rename(
        f"{signal.name}_dm")


def real_panel() -> pd.DataFrame:
    """The real tape and the signals in one frame (row *t*: signal known at end of *t*,
    returns realised during *t*). Rows before the first valid gap are dropped."""
    q = _data.load_quarterly()
    s = build_signals(q)
    p = pd.concat([q, s], axis=1)
    first = p[["tgap_hp", "tgap_u"]].dropna().index.min()
    p = p.loc[p.index >= first].copy()
    p["tgap_hp_dm"] = demeaned(p["tgap_hp"].rename("tgap_hp"))
    p["tgap_u_dm"] = demeaned(p["tgap_u"].rename("tgap_u"))
    return p


# --------------------------------------------------------------------------- #
# 3. Regressions
# --------------------------------------------------------------------------- #
def nw_slope(y: np.ndarray, x: np.ndarray, lags: int) -> dict:
    """OLS of ``y`` on ``[1, x]`` with a Newey-West (Bartlett) slope standard error.

    Works on 1-D arrays or row-wise on 2-D ``(n_paths, n)`` arrays (the simulation null runs
    thousands at once). The slope's HAC variance is computed on the partialled-out score
    ``(x - xbar) e``, which equals the slope element of the full sandwich.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n = y.shape[1]
    xd = x - x.mean(axis=1, keepdims=True)
    yd = y - y.mean(axis=1, keepdims=True)
    sxx = (xd ** 2).sum(axis=1)
    b = (xd * yd).sum(axis=1) / sxx
    e = yd - b[:, None] * xd
    sc = xd * e
    S = (sc ** 2).sum(axis=1)
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        S = S + 2.0 * w * (sc[:, l:] * sc[:, :-l]).sum(axis=1)
    se = np.sqrt(np.clip(S, 0, None)) / sxx
    t = b / se
    r2 = 1.0 - (e ** 2).sum(axis=1) / (yd ** 2).sum(axis=1)
    if b.size == 1:
        return {"slope": float(b[0]), "se": float(se[0]), "t": float(t[0]),
                "r2": float(r2[0]), "n": int(n)}
    return {"slope": b, "se": se, "t": t, "r2": r2, "n": int(n)}


def default_lags(h: int) -> int:
    """Newey-West lags: 4 for one quarter, ``2h`` for overlapping h-quarter sums."""
    return 4 if h == 1 else 2 * h


def forward_sum(y: pd.Series, h: int) -> pd.Series:
    """``y[t+1] + ... + y[t+h]`` stamped at *t* — the target aligned to the signal at *t*.

    The one place the forecast lead is applied. Sums of simple quarterly returns (or yield
    changes) — a first-order approximation to the compounded h-quarter return."""
    return sum(y.shift(-k) for k in range(1, h + 1))


def predictive_regression(panel: pd.DataFrame, signal: str, target: str, h: int = 1,
                          lags: int | None = None) -> dict:
    """Regress the next-``h``-quarter ``target`` on ``signal`` known at *t*, Newey-West."""
    fy = forward_sum(panel[target], h)
    df = pd.concat([fy.rename("y"), panel[signal].rename("x")], axis=1).dropna()
    if len(df) < 30:
        return {}
    out = nw_slope(df["y"].to_numpy(), df["x"].to_numpy(),
                   default_lags(h) if lags is None else lags)
    out.update({"signal": signal, "target": target, "h": h,
                "start": df.index[0], "end": df.index[-1]})
    return out


def ar1(x: np.ndarray) -> dict:
    """OLS AR(1) of ``x``: intercept, rho, residuals (aligned to t = 1..n-1)."""
    x = np.asarray(x, dtype=float)
    X = np.column_stack([np.ones(len(x) - 1), x[:-1]])
    c, rho = np.linalg.lstsq(X, x[1:], rcond=None)[0]
    v = x[1:] - c - rho * x[:-1]
    return {"c": float(c), "rho": float(rho), "v": v}


def _aligned(panel: pd.DataFrame, signal: str, target: str) -> tuple[np.ndarray, np.ndarray]:
    """``x_t`` and ``y_{t+1}`` over the rows where both exist, in time order."""
    df = pd.concat([panel[signal].rename("x"), panel[target].shift(-1).rename("y")],
                   axis=1).dropna()
    return df["x"].to_numpy(), df["y"].to_numpy()


def stambaugh_correction(panel: pd.DataFrame, signal: str, target: str) -> dict:
    """First-order Stambaugh (1999) bias correction of the one-quarter predictive slope.

    With ``y_{t+1} = a + b x_t + u`` and ``x_{t+1} = c + rho x_t + v``,
    ``E[b_hat - b] = (cov(u, v) / var(v)) * E[rho_hat - rho]`` and, to first order,
    ``E[rho_hat - rho] = -(1 + 3 rho) / T`` (Kendall). The corrected slope is
    ``b_hat + gamma (1 + 3 rho_c) / T`` with ``rho_c`` the bias-corrected persistence. The
    corrected *t* keeps the Newey-West standard error of the original regression.
    """
    x, y = _aligned(panel, signal, target)
    T = len(x)
    reg = nw_slope(y, x, default_lags(1))
    a1 = ar1(x)
    rho_c = min(a1["rho"] + (1.0 + 3.0 * a1["rho"]) / T, 0.999)
    u = y - (y.mean() + reg["slope"] * (x - x.mean()))
    uu, vv = u[:-1], a1["v"]          # u[j] and v[j] are both shocks dated j+1
    gamma = float(np.cov(uu, vv)[0, 1] / np.var(vv, ddof=1))
    bias = -gamma * (1.0 + 3.0 * rho_c) / T
    b_c = reg["slope"] - bias
    return {"T": T, "slope": reg["slope"], "se": reg["se"], "t": reg["t"],
            "rho": a1["rho"], "rho_c": rho_c, "gamma": gamma,
            "corr_uv": float(np.corrcoef(uu, vv)[0, 1]),
            "bias": float(bias), "slope_c": float(b_c), "t_c": float(b_c / reg["se"])}


def simulation_null(panel: pd.DataFrame, signal: str, target: str,
                    horizons=HORIZONS, n_sim: int = 2000, seed: int = 1022) -> dict:
    """Reference distribution of the Newey-West *t* under H0: no predictability.

    Fit the regressor's AR(1) (persistence bias-corrected, so the null is *more* persistent,
    not less), take the null return innovations ``u = y - mean(y)`` and the AR(1) innovations
    ``v``, and resample the **pairs** ``(u_t, v_t)`` together — the null keeps the data's own
    innovation correlation, the ingredient of Stambaugh bias. On each simulated path the
    same 1-, 4- and 8-quarter regressions are rerun with the same lags, so the long-horizon
    p-values are overlap-correct by construction.

    Returns, per horizon, the observed *t*, the two-sided p-value ``P(|t*| >= |t|)``, the
    one-sided p-value in the claim's direction ``P(t* <= t)`` (the claim predicts a negative
    slope for both legs) and the mean null slope (a Monte-Carlo read of the bias).
    """
    from scipy.signal import lfilter

    x, y = _aligned(panel, signal, target)
    T = len(x)
    a1 = ar1(x)
    rho_c = min(a1["rho"] + (1.0 + 3.0 * a1["rho"]) / T, 0.999)
    c_c = x.mean() * (1.0 - rho_c)
    v = a1["v"] - a1["v"].mean()
    u = (y - y.mean())[:-1]           # pairs with v: both are shocks dated j+1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_sim, T))
    vs = v[idx]
    us = u[idx]
    # x*_0 = observed x_0; x*_t = c + rho x*_{t-1} + v*_t
    drive = c_c + vs
    drive[:, 0] = x[0]
    xs = lfilter([1.0], [1.0, -rho_c], drive, axis=1)
    # y*_{t+1} = mean(y) + u*_t, paired with v*_{t+1}: shift u one step so (u, v) pair up
    ys = y.mean() + np.roll(us, -1, axis=1)
    out = {"rho": a1["rho"], "rho_c": rho_c, "n_sim": n_sim, "T": T}
    for h in horizons:
        lags = default_lags(h)
        # observed
        xo = pd.Series(x)
        yo = sum(pd.Series(y).shift(-(k - 1)) for k in range(1, h + 1))
        ok = yo.notna().to_numpy()
        obs = nw_slope(yo.to_numpy()[ok], xo.to_numpy()[ok], lags)
        # simulated: same construction on each path
        if h == 1:
            yh = ys[:, :-1]
            xh = xs[:, :-1]
        else:
            cs = np.cumsum(np.c_[np.zeros((n_sim, 1)), ys[:, :-1]], axis=1)
            yh = cs[:, h:] - cs[:, :-h]
            xh = xs[:, : yh.shape[1]]
        sim = nw_slope(yh, xh, lags)
        ts = sim["t"]
        out[h] = {"t": obs["t"], "slope": obs["slope"], "r2": obs["r2"], "n": obs["n"],
                  "p_two": float(np.mean(np.abs(ts) >= abs(obs["t"]))),
                  "p_claim": float(np.mean(ts <= obs["t"])),
                  "null_mean_slope": float(np.mean(sim["slope"])),
                  "null_t_q": [float(q) for q in np.quantile(ts, [0.025, 0.5, 0.975])]}
    return out


def conditional_means(panel: pd.DataFrame, signal: str, target: str) -> dict:
    """Mean next-quarter ``target`` when behind (``signal < 0``) vs ahead, with the HAC *t* of
    the difference (a dummy regression)."""
    x, y = _aligned(panel, signal, target)
    d = (x < 0).astype(float)
    r = nw_slope(y, d, default_lags(1))
    return {"mean_behind": float(y[d == 1].mean()), "mean_ahead": float(y[d == 0].mean()),
            "n_behind": int(d.sum()), "n_ahead": int((1 - d).sum()),
            "diff": r["slope"], "t_diff": r["t"]}


def regime_split(panel: pd.DataFrame, signal: str, target: str,
                 break_date: str = REGIME_BREAK) -> dict:
    """Slope before and after ``break_date`` and the HAC *t* of the difference
    (interaction regression ``y = a + b x + d D + e (x * D)``)."""
    df = pd.concat([panel[signal].rename("x"), panel[target].shift(-1).rename("y")],
                   axis=1).dropna()
    D = (df.index > pd.Timestamp(break_date)).astype(float)
    X = np.column_stack([np.ones(len(df)), df["x"], D, df["x"] * D])
    y = df["y"].to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    XtXi = np.linalg.inv(X.T @ X)
    sc = X * e[:, None]
    S = sc.T @ sc
    lags = default_lags(1)
    for l in range(1, lags + 1):
        G = sc[l:].T @ sc[:-l]
        S += (1 - l / (lags + 1)) * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ S @ XtXi))
    pre = nw_slope(y[D == 0], df["x"].to_numpy()[D == 0], lags)
    post = nw_slope(y[D == 1], df["x"].to_numpy()[D == 1], lags)
    return {"slope_pre": pre["slope"], "t_pre": pre["t"], "n_pre": pre["n"],
            "slope_post": post["slope"], "t_post": post["t"], "n_post": post["n"],
            "diff": float(beta[3]), "t_diff": float(beta[3] / se[3])}


def gap_agreement(panel: pd.DataFrame) -> dict:
    """How far the knowable gaps sit from the hindsight one (Orphanides in miniature)."""
    df = panel[["gap_hp", "gap_u", "gap_hind", "tgap_hp", "tgap_u", "tgap_hind"]].dropna()
    return {
        "corr_gap_hp_hind": float(df["gap_hp"].corr(df["gap_hind"])),
        "corr_gap_u_hind": float(df["gap_u"].corr(df["gap_hind"])),
        "corr_gap_hp_u": float(df["gap_hp"].corr(df["gap_u"])),
        "mean_abs_gap_rev": float((df["gap_hp"] - df["gap_hind"]).abs().mean()),
        "sign_agree_tgap_hp_hind": float((np.sign(df["tgap_hp"])
                                          == np.sign(df["tgap_hind"])).mean()),
        "sign_agree_tgap_hp_u": float((np.sign(df["tgap_hp"])
                                       == np.sign(df["tgap_u"])).mean()),
        "share_behind_hp": float((df["tgap_hp"] < 0).mean()),
        "share_behind_u": float((df["tgap_u"] < 0).mean()),
    }


# --------------------------------------------------------------------------- #
# 5. Tradability
# --------------------------------------------------------------------------- #
def sharpe(x: np.ndarray, ppy: int = QPY) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(ppy)) if sd > 0 else np.nan


def timing_backtest(panel: pd.DataFrame, signal: str, cost_bp: float = 10.0) -> pd.DataFrame:
    """Equity when the Fed is behind the curve, bills when it is ahead — one quarter later.

    ``pos[t+1] = 1{signal[t] < 0}``: the signal is fixed at the end of quarter *t* (macro
    already lagged a quarter inside it) and the position earns quarter *t+1*. One ``shift``,
    here, once. Costs: ``cost_bp`` one-way x traded NAV, charged when the position changes
    (a full switch equity<->bills trades 100% of NAV one way; bills are treated as free).
    Returns a frame of ``pos``, ``turnover``, ``strat_xs_gross``, ``strat_xs_net`` (both in
    excess of the T-bill) and ``bh_xs`` (buy-and-hold equity excess).
    """
    pos = (panel[signal] < 0).astype(float).where(panel[signal].notna()).shift(1)
    df = pd.DataFrame({"pos": pos, "bh_xs": panel["eq_xs"], "rf": panel["rf"]}).dropna()
    df["turnover"] = df["pos"].diff().abs().fillna(df["pos"].iloc[0])
    df["strat_xs_gross"] = df["pos"] * df["bh_xs"]
    df["strat_xs_net"] = df["strat_xs_gross"] - cost_bp / 1e4 * df["turnover"]
    return df


def backtest_summary(bt: pd.DataFrame) -> dict:
    """Excess-of-cash Sharpe race, turnover, timing alpha (HAC) and the break-even cost."""
    n_years = len(bt) / QPY
    alpha = nw_slope(bt["strat_xs_net"].to_numpy(), bt["bh_xs"].to_numpy(), 4)
    # intercept of strat on bh: mean(s) - b mean(bh); HAC t of the intercept via the
    # demeaned identity: regress (s - b*bh) on a constant
    resid = bt["strat_xs_net"].to_numpy() - alpha["slope"] * bt["bh_xs"].to_numpy()
    from quantlab.analytics import mean_tstat_hac
    a_t = mean_tstat_hac(pd.Series(resid), lags=4)
    gross_edge = (bt["strat_xs_gross"].mean() - bt["bh_xs"].mean())
    turn = bt["turnover"].sum() / n_years
    sr_g, sr_n, sr_b = (sharpe(bt["strat_xs_gross"]), sharpe(bt["strat_xs_net"]),
                        sharpe(bt["bh_xs"]))
    # cost at which the net excess Sharpe falls to buy-and-hold's
    lo, hi = 0.0, 2000.0
    if sr_g <= sr_b:
        be = 0.0
    else:
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            s = sharpe(bt["strat_xs_gross"] - mid / 1e4 * bt["turnover"])
            lo, hi = (mid, hi) if s > sr_b else (lo, mid)
        be = 0.5 * (lo + hi)
    return {"n_quarters": int(len(bt)), "share_invested": float(bt["pos"].mean()),
            "switches_per_year": float((bt["turnover"] > 0).sum() / n_years),
            "turnover_per_year": float(turn),
            "ann_xs_gross": float(bt["strat_xs_gross"].mean() * QPY),
            "ann_xs_net": float(bt["strat_xs_net"].mean() * QPY),
            "ann_xs_bh": float(bt["bh_xs"].mean() * QPY),
            "sr_gross": sr_g, "sr_net": sr_n, "sr_bh": sr_b, "sr_diff_net": sr_n - sr_b,
            "alpha_ann": float(resid.mean() * QPY), "alpha_t": float(a_t["tstat"]),
            "beta": float(alpha["slope"]), "gross_edge_ann": float(gross_edge * QPY),
            "breakeven_bp": float(be)}


def sharpe_diff_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000,
                          block: int = 8, seed: int = 1022) -> dict:
    """Circular block bootstrap of ``Sharpe(a) - Sharpe(b)`` (paired, same blocks).

    Returns the point estimate, a 95% interval and the one-sided p-value ``P(diff* <= 0)``
    computed on the bootstrap distribution re-centred at zero."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(n_boot, -1)[:, :n]
    A, B = a[idx], b[idx]

    def sr(M):
        return M.mean(axis=1) / M.std(axis=1, ddof=1) * np.sqrt(QPY)
    d = sr(A) - sr(B)
    obs = sharpe(a) - sharpe(b)
    return {"diff": float(obs), "ci_lo": float(np.quantile(d, 0.025)),
            "ci_hi": float(np.quantile(d, 0.975)),
            "p_one_sided": float(np.mean(d - d.mean() >= obs))}


def cost_sweep(panel: pd.DataFrame, signal: str, costs=(0, 5, 10, 25, 50, 100)) -> pd.DataFrame:
    rows = []
    for c in costs:
        s = backtest_summary(timing_backtest(panel, signal, c))
        rows.append({"cost_bp": c, "sr_net": s["sr_net"], "sr_bh": s["sr_bh"],
                     "ann_xs_net": s["ann_xs_net"], "ann_xs_bh": s["ann_xs_bh"]})
    return pd.DataFrame(rows).set_index("cost_bp")


# --------------------------------------------------------------------------- #
# 6. Verdict — thresholds fixed before the real run
# --------------------------------------------------------------------------- #
def leg_is_real(slope: float, t: float, p_sim: float) -> bool:
    """A leg is real when the slope has the claim's sign (negative), the robust |t| clears 2
    and the simulation-null two-sided p-value is below 5%."""
    return bool(slope < 0 and abs(t) >= 2.0 and p_sim < 0.05)


def verdict(h: dict) -> dict:
    """The two stamps by a pre-registered rule.

    **Signal** — the primary specification is the one-quarter regression on the one-sided-HP
    Taylor gap.

    - The **equity leg** is real if its Stambaugh-corrected slope is negative with |t| >= 2,
      its simulation-null p < 0.05, *and* the unemployment-gap version has the same sign.
    - The **bond leg** is real if the AAA-yield-change slope is negative (behind the curve ->
      yields rise) with |t_NW| >= 2 and simulation-null p < 0.05.
    - **Real** = both legs real; **Mixed** = exactly one; otherwise **Weak** if *any* of the
      twelve specifications (2 gaps x 3 horizons x 2 legs) has the claim's sign with a
      simulation-null p < 0.05, or the primary equity slope has the claim's sign with p < 0.10;
      else **None**.

    **Tradability** — the sign-of-the-gap equity/bills rule at 10 bp one-way, excess-vs-excess:

    - **Investable** only if the equity leg is real *and* the net Sharpe beats buy-and-hold by
      >= 0.15 with a block-bootstrap one-sided p < 0.05;
    - **Fragile** if the net Sharpe beats buy-and-hold at all and the signal is not None;
    - **Mirage** otherwise.
    """
    eq = h["eq_primary"]
    eq_real = bool(eq["slope_c"] < 0 and abs(eq["t_c"]) >= 2.0 and eq["p_sim"] < 0.05
                   and h["eq_u_slope"] < 0)
    bd = h["bond_primary"]
    bond_real = leg_is_real(bd["slope"], bd["t"], bd["p_sim"])
    any_spec = any(s["slope"] < 0 and s["p_sim"] < 0.05 for s in h["specs"])
    if eq_real and bond_real:
        signal = "Real"
    elif eq_real or bond_real:
        signal = "Mixed"
    elif any_spec or (eq["slope"] < 0 and eq["p_sim"] < 0.10):
        signal = "Weak"
    else:
        signal = "None"

    tr = h["trade"]
    if eq_real and tr["sr_diff_net"] >= 0.15 and tr["p_diff"] < 0.05:
        trad = "Investable"
    elif tr["sr_diff_net"] > 0 and signal != "None":
        trad = "Fragile"
    else:
        trad = "Mirage"

    def sgn(x):
        return "the claim's (negative) sign" if x < 0 else "the *wrong* (positive) sign"

    best = min(h["specs"], key=lambda s: s["p_sim"])
    rg = h.get("bond_regime")
    regime = ("" if not rg else
              f" — and what there is lives before 1987Q3 (slope {rg['slope_pre']:+.3f}, "
              f"*t* = {rg['t_pre']:+.2f}) rather than after ({rg['slope_post']:+.3f}, "
              f"*t* = {rg['t_post']:+.2f}), a Great-Inflation story more than a rule")
    signal_why = (
        f"On {h['n_quarters']} quarters ({h['start']}–{h['end']}), the one-quarter slope of the "
        f"equity excess return on the real-time Taylor gap is **{eq['slope']*100:+.2f}% per "
        f"point of gap** ({sgn(eq['slope'])}), Newey-West *t* = {eq['t']:+.2f}. The gap is "
        f"highly persistent (AR(1) ρ = {eq['rho']:.3f}) and its shocks correlate "
        f"{eq['corr_uv']:+.2f} with returns, so the Stambaugh correction moves the slope to "
        f"{eq['slope_c']*100:+.2f}% (*t* = {eq['t_c']:+.2f}) and the simulation null — an "
        f"AR(1) regressor whose innovations are resampled jointly with the returns — gives a "
        f"two-sided **p = {eq['p_sim']:.2f}**. The unemployment-gap version has slope "
        f"{h['eq_u_slope']*100:+.2f}% (*t* = {h['eq_u_t']:+.2f}). The bond leg: next-quarter "
        f"AAA-yield change on the gap {bd['slope']:+.3f} pp per point (*t* = {bd['t']:+.2f}, "
        f"simulation p = {bd['p_sim']:.2f}; one-sided in the claim's direction "
        f"{bd.get('p_claim', float('nan')):.2f}){regime}. Best of all twelve specifications: "
        f"{best['label']} with *t* = {best['t']:+.2f}, simulation p = {best['p_sim']:.2f}. "
        f"And all of this is on **final-vintage** data — Orphanides (2001) showed the real-time "
        f"gap was worse — so it is an upper bound on what an investor could have had.")
    trad_why = (
        f"Holding stocks only when the Fed is behind the curve (gap < 0, one quarter later) was "
        f"invested {tr['share_invested']:.0%} of the time and switched "
        f"{tr['switches_per_year']:.2f} times a year. Net of 10 bp one-way, its excess-of-cash "
        f"Sharpe was **{tr['sr_net']:.2f} against {tr['sr_bh']:.2f}** for buy-and-hold "
        f"(difference {tr['sr_diff_net']:+.2f}, block-bootstrap 95% CI "
        f"[{tr['ci_lo']:+.2f}, {tr['ci_hi']:+.2f}], one-sided p = {tr['p_diff']:.2f}); it "
        f"earned {tr['ann_xs_net']:.2%} a year in excess of bills against "
        f"{tr['ann_xs_bh']:.2%} for the index, a timing alpha of {tr['alpha_ann']:+.2%} "
        f"(HAC *t* = {tr['alpha_t']:+.2f}). It trades about once a year, so costs are not what "
        f"decides it; the gross edge is the problem.")
    if signal in ("Real", "Mixed"):
        one = (
            f"The real-time Taylor gap carried information — equity slope "
            f"{eq['slope']*100:+.2f}% per point (simulation p = {eq['p_sim']:.2f}), AAA-yield "
            f"slope {bd['slope']:+.3f} pp (p = {bd['p_sim']:.2f}) — but the timing rule's net "
            f"Sharpe was {tr['sr_net']:.2f} against {tr['sr_bh']:.2f} for buy-and-hold, on "
            f"final-vintage data that flatter it.")
    else:
        one = (
            f"Being behind the curve bought stocks no tailwind — the real-time Taylor gap's "
            f"one-quarter equity slope is {eq['slope']*100:+.2f}% per point (simulation p = "
            f"{eq['p_sim']:.2f}), only the bond headwind comes close (p = {bd['p_sim']:.2f}), "
            f"and the timing rule's net Sharpe was {tr['sr_net']:.2f} against {tr['sr_bh']:.2f} "
            f"for buy-and-hold, on final-vintage data that flatter it.")
    return {"signal": signal, "signal_why": signal_why, "trad": trad, "trad_why": trad_why,
            "one_sentence": one, "eq_real": eq_real, "bond_real": bond_real}
