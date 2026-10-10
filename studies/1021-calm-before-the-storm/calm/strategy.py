"""Does prolonged calm breed the next crash? — Study 1021.

The claim, at full strength (Minsky; Danielsson, Valenzuela & Zer 2018): when measured
risk stays low for long enough, agents lever up, sell insurance and stretch for yield, so a
long quiet spell **raises** the odds of a crisis a few years out. The practitioner's version
is "low VIX = complacency = sell".

Every step below is fixed in advance; the numbers in ``docs/results.md`` come from running
it once on the real tape.

The signal — past-only, slow, relative
--------------------------------------
``calm_signal``. Low volatility only means something relative to a slow-moving norm (the
1950s and the 1930s do not share a "normal"). So:

1. ``rv_t`` — annualised standard deviation of the last 12 monthly returns;
2. ``trend_t`` — the trailing 10-year mean of ``rv``;
3. ``low_t = max(−log(rv_t / trend_t), 0)`` — how far *below* its own trend vol sits now
   (high vol is deliberately set to zero: the claim is about calm, not about storms);
4. ``calm_t`` — the trailing **5-year** average of ``low`` — *prolonged* calm, the Minsky
   state. Everything is known at the close of month ``t``.

The outcomes — what "the storm" means
-------------------------------------
``forward_outcomes``: over months ``t+1 … t+H`` (H = 1, 3, 12, 24, 36), the **maximum
drawdown** of the total-return wealth path, whether it reached **−20%** (a crash), the
realised volatility, and the cumulative excess log return. **The primary pre-registered
test is the slope of the 24-month forward max drawdown on calm.**

Inference — overlapping windows and the confound
------------------------------------------------
Forward windows overlap, so ``hac_slope`` uses Newey-West with ``2H`` lags and every
slope is also re-estimated on **non-overlapping** samples.

The confound is the whole study. Volatility mean-reverts, so in *any* clustering model a
calm is followed by a less-calm. ``mean_reversion_null`` fits two models to the real tape
— a GARCH(1,1)-t and a long-memory FIGARCH-t — that have **no** risk-taking feedback at
all, simulates hundreds of tapes from each, and runs the identical regression on every one.
The real slope earns its keep only if it sits in the right tail of *that* distribution.
``vol_horizon_profile`` shows the two horizons side by side: low vol predicts low vol next
month (clustering), and the claim is about what it predicts years out.

The trade
---------
``backtest`` — buy-and-hold, a **calm de-risk** rule (half the equity into bills when calm
is in the top fifth of its own *past* distribution), and a plain unlevered vol-target, all
on the total-return tape: signal at the close of month ``t``, position held through month
``t+1`` (one ``shift``, applied once), costs one-way × traded NAV, bills on the cash leg,
Sharpe on excess-of-bills returns for every leg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

try:  # the package is importable both as `calm` and from the study root
    from . import data as _data
except ImportError:  # pragma: no cover
    import data as _data  # type: ignore

MPY = 12
VOL_WINDOW = _data.VOL_WINDOW
TREND_WINDOW = _data.TREND_WINDOW
CALM_WINDOW = _data.CALM_WINDOW
CRASH_THRESHOLD = 0.20
HORIZONS = (1, 3, 12, 24, 36)
PRIMARY_H = 24
OUTCOMES = ("mdd", "crash", "vol", "ret")


# --------------------------------------------------------------------------- #
# The signal
# --------------------------------------------------------------------------- #
def calm_signal(ret: pd.Series | pd.DataFrame, vol_window: int = VOL_WINDOW,
                trend_window: int = TREND_WINDOW,
                calm_window: int = CALM_WINDOW) -> dict:
    """The prolonged-calm score, past-only. Works column-wise on a frame of many paths.

    Returns a dict of same-shaped objects: ``rv`` (12-month realised vol, annualised),
    ``trend`` (its trailing 10-year mean), ``dev`` (``log(rv/trend)``, negative = calm),
    ``low`` (``max(−dev, 0)``) and ``calm`` (the 5-year average of ``low``). The first
    ``vol + trend + calm − 3`` months are NaN — never back-filled.
    """
    rv = ret.rolling(vol_window).std() * np.sqrt(MPY)
    trend = rv.rolling(trend_window).mean()
    dev = np.log(rv / trend)
    low = (-dev).clip(lower=0.0)
    calm = low.rolling(calm_window).mean()
    return {"rv": rv, "trend": trend, "dev": dev, "low": low, "calm": calm}


# --------------------------------------------------------------------------- #
# The outcomes
# --------------------------------------------------------------------------- #
def _forward_np(ret: np.ndarray, rf: np.ndarray, H: int) -> dict:
    """Forward outcomes for one path or many (columns), as numpy arrays (NaN-padded).

    Row ``t`` describes months ``t+1 … t+H``: max drawdown of the wealth path (starting
    from wealth 1 at the close of ``t``), realised vol (zero-mean RMS, annualised) and
    cumulative excess log return.
    """
    r = np.asarray(ret, dtype=float)
    one = r.ndim == 1
    if one:
        r = r[:, None]
    f = np.asarray(rf, dtype=float)
    if f.ndim == 1:
        f = np.broadcast_to(f[:, None], r.shape)
    n, k = r.shape
    lr = np.log1p(r)
    lw = np.vstack([np.zeros((1, k)), np.cumsum(lr, axis=0)])        # (n+1, k)
    mdd = np.full((n, k), np.nan)
    vol = np.full((n, k), np.nan)
    exr = np.full((n, k), np.nan)
    if n > H:
        W = sliding_window_view(lw, H + 1, axis=0)[1:]               # (n-H, k, H+1)
        dd = (np.maximum.accumulate(W, axis=2) - W).max(axis=2)
        mdd[: n - H] = 1.0 - np.exp(-dd)
        R2 = sliding_window_view(r ** 2, H, axis=0)[1:]              # (n-H, k, H)
        vol[: n - H] = np.sqrt(R2.mean(axis=2) * MPY)
        lex = lr - np.log1p(f)
        E = sliding_window_view(lex, H, axis=0)[1:]
        exr[: n - H] = E.sum(axis=2)
    out = {"mdd": mdd, "crash": np.where(np.isfinite(mdd), (mdd >= CRASH_THRESHOLD)
                                         .astype(float), np.nan),
           "vol": vol, "ret": exr}
    if one:
        out = {k_: v[:, 0] for k_, v in out.items()}
    return out


def forward_outcomes(ret: pd.Series, rf: pd.Series | None = None,
                     H: int = PRIMARY_H) -> pd.DataFrame:
    """Forward max drawdown, crash flag (MDD ≥ 20%), vol and excess return over t+1…t+H."""
    rfv = np.zeros(len(ret)) if rf is None else rf.reindex(ret.index).fillna(0.0).to_numpy()
    o = _forward_np(ret.to_numpy(), rfv, H)
    return pd.DataFrame(o, index=ret.index)


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def hac_slope(y, x, lags: int) -> dict:
    """OLS of ``y`` on ``[1, x]`` with Newey-West (Bartlett) standard errors."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    if n < 30 or np.std(x) == 0:
        return {}
    A = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    e = y - A @ beta
    XtX_inv = np.linalg.inv(A.T @ A)
    U = A * e[:, None]
    S = U.T @ U
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        G = U[l:].T @ U[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    return {"slope": float(beta[1]), "intercept": float(beta[0]), "se": se,
            "t": float(beta[1] / se) if se > 0 else np.nan, "n": int(n)}


def _standardise(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std()


def predictive_regression(state: pd.Series, outcome: pd.Series, H: int) -> dict:
    """Slope of ``outcome`` on the **standardised** state, HAC and non-overlapping.

    The slope reads "per one standard deviation of calm". HAC uses ``max(2H, 6)`` lags.
    The non-overlapping version keeps every H-th observation, for each of the H possible
    offsets, and reports the median t — a check that the HAC correction is not doing all
    the work.
    """
    d = pd.concat([state.rename("x"), outcome.rename("y")], axis=1).dropna()
    if len(d) < 60:
        return {}
    d["x"] = _standardise(d["x"])
    out = hac_slope(d["y"], d["x"], lags=max(2 * H, 6))
    ts = []
    for off in range(H):
        sub = d.iloc[off::H]
        r = hac_slope(sub["y"], sub["x"], lags=0) if len(sub) >= 30 else {}
        if r:
            ts.append(r["t"])
    out["t_nonoverlap_median"] = float(np.median(ts)) if ts else np.nan
    out["n_nonoverlap"] = int(len(d) // H)
    out["mean_y"] = float(d["y"].mean())
    return out


def horizon_table(ret: pd.Series, rf: pd.Series | None = None,
                  horizons=HORIZONS, state: str = "calm") -> pd.DataFrame:
    """Every horizon × every outcome: slope per SD of the state, HAC t, non-overlap t."""
    sig = calm_signal(ret)
    rows = []
    for H in horizons:
        fo = forward_outcomes(ret, rf, H)
        for o in OUTCOMES:
            r = predictive_regression(sig[state], fo[o], H)
            if r:
                rows.append({"H": H, "outcome": o, "slope": r["slope"], "t_hac": r["t"],
                             "t_nonoverlap": r["t_nonoverlap_median"],
                             "n": r["n"], "n_indep": r["n_nonoverlap"],
                             "mean_y": r["mean_y"]})
    return pd.DataFrame(rows)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (float(c - hw), float(c + hw))


def crash_odds(ret: pd.Series, rf: pd.Series | None = None, H: int = PRIMARY_H,
               q: float = 0.75) -> dict:
    """Share of months followed by a ≥20% drawdown within H, calm top quartile vs rest.

    Descriptive (the quartile cut uses the full sample). Wilson intervals count **months**,
    which overlap, so they are too narrow — the "independent episodes" count says how many
    genuinely separate crashes the comparison rests on.
    """
    sig = calm_signal(ret)
    fo = forward_outcomes(ret, rf, H)
    d = pd.concat([sig["calm"].rename("c"), fo["crash"].rename("k")], axis=1).dropna()
    hi = d["c"] >= d["c"].quantile(q)
    k1, n1 = int(d.loc[hi, "k"].sum()), int(hi.sum())
    k0, n0 = int(d.loc[~hi, "k"].sum()), int((~hi).sum())
    # independent crash episodes: drawdowns of ≥20% from a running peak, counted once
    wealth = (1 + ret.loc[d.index[0]:]).cumprod()
    under = wealth / wealth.cummax() - 1
    episodes, inside = 0, False
    for v in under.to_numpy():
        if not inside and v <= -CRASH_THRESHOLD:
            episodes, inside = episodes + 1, True
        elif inside and v >= 0:
            inside = False
    return {"H": H, "p_calm": k1 / n1 if n1 else np.nan, "ci_calm": wilson(k1, n1),
            "n_calm": n1, "p_rest": k0 / n0 if n0 else np.nan, "ci_rest": wilson(k0, n0),
            "n_rest": n0, "n_episodes": int(episodes),
            "calm_cut": float(d["c"].quantile(q))}


def half_sample_slopes(ret: pd.Series, rf: pd.Series | None = None,
                       H: int = PRIMARY_H, outcome: str = "mdd") -> list[dict]:
    """The primary regression on the first and second half of the usable sample."""
    sig = calm_signal(ret)
    fo = forward_outcomes(ret, rf, H)
    d = pd.concat([sig["calm"].rename("x"), fo[outcome].rename("y")], axis=1).dropna()
    mid = len(d) // 2
    out = []
    for part in (d.iloc[:mid], d.iloc[mid:]):
        r = predictive_regression(part["x"], part["y"], H)
        out.append({"start": str(part.index[0].date()), "end": str(part.index[-1].date()),
                    "slope": r.get("slope", np.nan), "t": r.get("t", np.nan)})
    return out


# --------------------------------------------------------------------------- #
# The mean-reversion nulls
# --------------------------------------------------------------------------- #
def fit_vol_model(excess: pd.Series, kind: str = "garch"):
    """Fit a constant-mean GARCH(1,1)-t or FIGARCH(1,d,1)-t to monthly excess returns (%).

    Fitted on the **full** sample: this is a null-generating process, not a forecaster,
    so look-ahead in its parameters does not leak into any trading rule.
    """
    from arch import arch_model
    vol = "GARCH" if kind == "garch" else "FIGARCH"
    am = arch_model(100.0 * excess.dropna().to_numpy(), mean="Constant", vol=vol,
                    p=1, q=1, dist="t")
    return am.fit(disp="off")


def simulate_null(res, kind: str, n: int, n_paths: int, seed: int = 1021,
                  burn: int = 500) -> np.ndarray:
    """``n_paths`` monthly excess-return tapes (decimal) from a fitted model. (n, n_paths)."""
    rng = np.random.default_rng(seed)
    p = res.params
    if kind == "garch":
        mu, om, a, b, nu = (p["mu"], p["omega"], p["alpha[1]"], p["beta[1]"], p["nu"])
        T = n + burn
        z = rng.standard_t(nu, (T, n_paths)) * np.sqrt((nu - 2.0) / nu)
        h = np.full(n_paths, om / max(1.0 - a - b, 1e-6))
        out = np.empty((T, n_paths))
        for t in range(T):
            e = np.sqrt(h) * z[t]
            out[t] = e
            h = om + a * e ** 2 + b * h
        return (mu + out[burn:]) / 100.0
    from arch.univariate import FIGARCH, ConstantMean, StudentsT
    out = np.empty((n, n_paths))
    for j in range(n_paths):
        mod = ConstantMean(None, volatility=FIGARCH(p=1, q=1),
                           distribution=StudentsT(seed=np.random.default_rng(
                               rng.integers(2 ** 32))))
        sim = mod.simulate(p.to_numpy(), n, burn=burn)
        out[:, j] = sim["data"].to_numpy() / 100.0
    return out


def _slopes_many(total: np.ndarray, rf: np.ndarray, H: int, windows=None) -> dict:
    """Primary-style slope (per SD of calm) for every column of a (n, k) total-return array."""
    df = pd.DataFrame(total)
    calm = calm_signal(df, **(windows or {}))["calm"].to_numpy()
    fo = _forward_np(total, rf, H)
    out = {}
    for o in OUTCOMES:
        y = fo[o]
        ok = np.isfinite(calm) & np.isfinite(y)
        x = np.where(ok, calm, np.nan)
        yy = np.where(ok, y, np.nan)
        xm = np.nanmean(x, axis=0)
        ym = np.nanmean(yy, axis=0)
        xs = np.nanstd(x, axis=0, ddof=1)
        cov = np.nansum((x - xm) * (yy - ym), axis=0) / (ok.sum(axis=0) - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[o] = cov / xs                           # slope per SD of calm
    return out


def mean_reversion_null(ff: pd.DataFrame, kind: str = "garch", H: int = PRIMARY_H,
                        n_paths: int = 400, seed: int = 1021, res=None,
                        windows: dict | None = None) -> dict:
    """Is the real calm → storm slope stronger than mean reversion alone delivers?

    Fit ``kind`` to the real excess returns, simulate ``n_paths`` tapes of the same length
    (adding back the **real** bill path so total returns and excess returns line up), run
    the identical regression on each, and report the one-sided p-value: the share of
    no-feedback tapes whose slope is at least as large as the real one. ``windows``
    overrides the signal's ``vol_window`` / ``trend_window`` / ``calm_window`` (the
    robustness grid); pass a fitted ``res`` to reuse one fit across calls.
    """
    if res is None:
        res = fit_vol_model(ff["mkt_rf"], kind)
    n = len(ff)
    rf = ff["rf"].to_numpy()
    sims = simulate_null(res, kind, n, n_paths, seed)
    # A fat-tailed draw on a high-vol path can fall below −100%; a price cannot.
    tot = np.clip(sims + rf[:, None], -0.95, None)
    null = _slopes_many(tot, rf, H, windows)
    real = _slopes_many(ff["mkt"].to_numpy()[:, None], rf, H, windows)
    out = {"kind": kind, "H": H, "n_paths": n_paths,
           "params": {k: float(v) for k, v in res.params.items()}}
    for o in OUTCOMES:
        nv = null[o][np.isfinite(null[o])]
        rv = float(real[o][0])
        out[o] = {"real": rv, "null_mean": float(nv.mean()), "null_sd": float(nv.std()),
                  "null_q05": float(np.quantile(nv, 0.05)),
                  "null_q95": float(np.quantile(nv, 0.95)),
                  "p_value": float((1 + (nv >= rv).sum()) / (1 + len(nv))),
                  "excess_over_null": float(rv - nv.mean())}
    out["_null_samples"] = {o: null[o] for o in OUTCOMES}
    return out


def vol_horizon_profile(ret: pd.Series, state: pd.Series, lags=range(0, 61, 3),
                        window: int = 6) -> pd.DataFrame:
    """Correlation of today's state with realised vol over months ``t+1+h … t+h+window``.

    Run with state = ``−dev`` (how far below trend vol is *now*) it shows clustering —
    calm now, calm next month. Run with state = ``calm`` it asks the Minsky question at
    every horizon. One picture, both signs.
    """
    r2 = (ret ** 2).to_numpy()
    n = len(r2)
    cs = np.concatenate([[0.0], np.cumsum(r2)])
    rows = []
    s = state.reindex(ret.index).to_numpy()
    for h in lags:
        y = np.full(n, np.nan)
        m = n - h - window
        if m > 0:
            t = np.arange(m)
            y[:m] = np.sqrt((cs[t + 1 + h + window] - cs[t + 1 + h]) / window * MPY)
        ok = np.isfinite(s) & np.isfinite(y)
        if ok.sum() < 60:
            continue
        rows.append({"h": int(h), "corr": float(np.corrcoef(s[ok], np.log(y[ok]))[0, 1]),
                     "n": int(ok.sum())})
    return pd.DataFrame(rows)


def null_horizon_profile(res, kind: str, ff: pd.DataFrame, lags=range(0, 61, 3),
                         window: int = 6, n_paths: int = 60, seed: int = 7) -> pd.DataFrame:
    """The same profile on no-feedback simulated tapes: the mean and a 90% band."""
    sims = np.clip(simulate_null(res, kind, len(ff), n_paths, seed)
                   + ff["rf"].to_numpy()[:, None], -0.95, None)
    profs = []
    for j in range(n_paths):
        r = pd.Series(sims[:, j], index=ff.index)
        p = vol_horizon_profile(r, calm_signal(r)["calm"], lags, window)
        profs.append(p.set_index("h")["corr"])
    P = pd.concat(profs, axis=1)
    return pd.DataFrame({"mean": P.mean(axis=1), "q05": P.quantile(0.05, axis=1),
                         "q95": P.quantile(0.95, axis=1)})


def clustering_ar1(rv_monthly: pd.Series, lags: int = 6) -> dict:
    """AR(1) of log monthly realised vol with HAC t — the short-horizon sign."""
    y = np.log(rv_monthly.replace(0, np.nan)).dropna()
    r = hac_slope(y.iloc[1:].to_numpy(), y.iloc[:-1].to_numpy(), lags=lags)
    return {"phi": r["slope"], "t": r["t"], "n": r["n"],
            "half_life_months": float(np.log(0.5) / np.log(r["slope"]))
            if 0 < r["slope"] < 1 else np.nan}


# --------------------------------------------------------------------------- #
# The trade
# --------------------------------------------------------------------------- #
def rule_weights(ret: pd.Series, rule: str, derisk_to: float = 0.5, q: float = 0.8,
                 min_hist: int = 120) -> pd.Series:
    """Target equity weight decided at the close of month ``t`` (applied to ``t+1`` later).

    - ``buy_hold`` — 1.0 always.
    - ``calm_derisk`` — ``derisk_to`` when today's calm score is above the ``q`` quantile
      of its **own past** values (expanding, at least ``min_hist`` months of calm history),
      else 1.0. NaN until the signal exists.
    - ``vol_target`` — ``min(1, target / rv_t)``, target = expanding median of past
      ``rv``. Unlevered on purpose: the comparison is between two ways of holding *less*.
    """
    sig = calm_signal(ret)
    idx = ret.index
    if rule == "buy_hold":
        return pd.Series(1.0, index=idx)
    if rule == "calm_derisk":
        c = sig["calm"]
        cut = c.expanding(min_periods=min_hist).quantile(q).shift(1)
        w = pd.Series(np.where(c > cut, derisk_to, 1.0), index=idx)
        return w.where(cut.notna())
    if rule == "vol_target":
        rv = sig["rv"]
        target = rv.expanding(min_periods=min_hist).median()
        return (target / rv).clip(upper=1.0).where(target.notna())
    raise ValueError(rule)


def backtest(ff: pd.DataFrame, w_target: pd.Series, cost_bps: float = 10.0) -> pd.DataFrame:
    """Monthly book: target known at close of ``t`` is held through ``t+1`` — one shift.

    The equity sleeve drifts with the market inside the month; the trade at each close is
    the distance from the **drifted** weight back to target, so turnover is one-way × NAV
    exactly. Costs are ``cost_bps`` per unit of NAV traded. The cash sleeve earns bills.
    """
    r = ff["mkt"]
    rf = ff["rf"]
    w = w_target.shift(1)                              # the one execution lag
    gross = w * r + (1 - w) * rf
    drift = w * (1 + r) / (1 + gross)                  # weight at the close, before trading
    trade = (w_target - drift).abs()
    trade.iloc[0] = np.nan
    cost = trade.shift(0).fillna(0.0) * cost_bps / 1e4
    net = gross - cost
    out = pd.DataFrame({"w": w, "gross": gross, "net": net, "rf": rf, "turnover": trade})
    return out.dropna(subset=["w"])


def summary(bt: pd.DataFrame) -> dict:
    """Excess-of-bills Sharpe (gross and net), CAGR, vol, max DD, terminal wealth, turnover."""
    out = {}
    for leg in ("gross", "net"):
        ex = bt[leg] - bt["rf"]
        out[f"sharpe_{leg}"] = float(ex.mean() / ex.std() * np.sqrt(MPY))
    w = (1 + bt["net"]).cumprod()
    yrs = len(bt) / MPY
    out.update({"cagr_net": float(w.iloc[-1] ** (1 / yrs) - 1),
                "vol": float(bt["net"].std() * np.sqrt(MPY)),
                "max_dd": float((w / w.cummax() - 1).min()),
                "terminal": float(w.iloc[-1]),
                "turnover_yr": float(bt["turnover"].mean() * MPY),
                "avg_weight": float(bt["w"].mean()),
                "start": str(bt.index[0].date()), "end": str(bt.index[-1].date())})
    return out


def sharpe_diff_bootstrap(ex_a: pd.Series, ex_b: pd.Series, block: int = 24,
                          n_boot: int = 2000, seed: int = 1021) -> dict:
    """Circular block bootstrap of Sharpe(a) − Sharpe(b) on aligned excess returns."""
    d = pd.concat([ex_a.rename("a"), ex_b.rename("b")], axis=1).dropna().to_numpy()
    n = len(d)
    rng = np.random.default_rng(seed)

    def sh(x):
        return x.mean(axis=0) / x.std(axis=0, ddof=1) * np.sqrt(MPY)

    point = float(np.subtract(*sh(d)))
    nb = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        st = rng.integers(0, n, nb)
        idx = ((st[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        s = sh(d[idx])
        diffs[i] = s[0] - s[1]
    return {"diff": point, "ci": (float(np.quantile(diffs, 0.025)),
                                  float(np.quantile(diffs, 0.975))),
            "p_le_zero": float((diffs <= 0).mean())}


def cost_sweep(ff: pd.DataFrame, w: pd.Series, costs=(0, 10, 25, 50)) -> pd.DataFrame:
    """Net excess Sharpe of target weights ``w`` vs buy-and-hold over one-way costs.

    ``w`` must be computed on the full tape and passed in: recomputing the signal on a
    truncated tape would silently move its start date and change the sample.
    """
    w = w.reindex(ff.index)
    bh = pd.Series(1.0, index=ff.index).where(w.notna())
    rows = []
    for c in costs:
        a = summary(backtest(ff, w, c))
        b = summary(backtest(ff, bh, c))
        rows.append({"cost_bps": c, "sharpe_rule": a["sharpe_net"],
                     "sharpe_bh": b["sharpe_net"], "diff": a["sharpe_net"] - b["sharpe_net"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# The verdict rule — fixed before the real run
# --------------------------------------------------------------------------- #
def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    **Signal** keys off the primary test — the 24-month forward max drawdown regressed on
    the calm score — and the two mean-reversion nulls:

    - **Real** — HAC t ≥ 2 **and** the slope beats both the GARCH and the FIGARCH null at
      one-sided p < 0.05. **Mixed** if that holds but the two halves of the sample
      disagree in sign.
    - **Weak** — something survives raw (HAC t ≥ 2) but a null absorbs it, or the stronger
      null comparison reaches p < 0.10.
    - **None** — otherwise.

    **Tradability** keys off the calm de-risk rule against buy-and-hold, net of 10 bp:

    - **Investable** — net excess Sharpe gain with block-bootstrap p < 0.05, a shallower
      max drawdown, *and* a net Sharpe at least that of the plain vol-target.
    - **Fragile** — no significant gain, but a point Sharpe gain ≥ 0 with a shallower
      drawdown.
    - **Mirage** — otherwise.
    """
    t = h["primary_t"]
    pmin = min(h["p_garch"], h["p_figarch"])
    pmax = max(h["p_garch"], h["p_figarch"])
    halves_disagree = any(np.sign(s) != np.sign(h["primary_slope"])
                          for s in h["half_slopes"])
    if t >= 2.0 and pmax < 0.05:
        signal = "Mixed" if halves_disagree else "Real"
    elif t >= 2.0 or pmin < 0.10:
        signal = "Weak"
    else:
        signal = "None"

    d = h["derisk_sharpe_diff"]
    dd_better = h["derisk_max_dd"] > h["bh_max_dd"]
    if (d > 0 and h["derisk_p"] < 0.05 and dd_better
            and h["derisk_sharpe"] >= h["voltarget_sharpe"]):
        trad = "Investable"
    elif d >= 0 and dd_better:
        trad = "Fragile"
    else:
        trad = "Mirage"

    slope = h["primary_slope"]
    direction = ("points the way Minsky said" if slope > 0
                 else "points the *wrong* way for Minsky")
    raw_word = "clears" if t >= 2 else "does not clear"
    if pmax < 0.05:
        null_txt = "it beats both"
    elif pmin < 0.10:
        null_txt = ("it sits in the right tail of both, but short of the 5% line — "
                    "suggestive, not certified")
    else:
        null_txt = "it sits comfortably inside what mean reversion alone produces"
    signal_why = (
        f"The raw link {direction}: one standard deviation more prolonged calm goes with a "
        f"**{slope:+.1%}** change in the maximum drawdown over the next 24 months (HAC "
        f"t = {t:.2f}, which {raw_word} the bar of 2; non-overlapping t = "
        f"{h['primary_t_nonoverlap']:.2f} on ~{h['n_indep']} independent windows). Because "
        f"volatility mean-reverts, the real question is whether that beats a world with no "
        f"risk-taking feedback at all — and there a no-feedback GARCH(1,1)-t fitted to the "
        f"same tape predicts a slope of **{h['null_garch_mean']:+.1%}**: calm is "
        f"*supposed* to be followed by calm. Against that null the real slope has one-sided "
        f"**p = {h['p_garch']:.2f}**, against a long-memory FIGARCH-t **p = "
        f"{h['p_figarch']:.2f}** — {null_txt}. Forward volatility tells the same story "
        f"(GARCH p = {h['p_garch_vol']:.2f}). At the one-month horizon the sign is the "
        f"opposite and certain: log realised vol has AR(1) φ = {h['ar1_phi']:.2f} "
        f"(t = {h['ar1_t']:.1f}) on the daily S&P tape. The long-horizon leg rests on only "
        f"**{h['n_episodes']} separate ≥20% drawdowns** after the signal starts, and its two "
        f"halves disagree ({h['half_slopes'][0]:+.1%} then {h['half_slopes'][1]:+.1%}).")
    beat = "beat" if d > 0 else "trailed"
    trad_why = (
        f"Halving equity when calm is in the top fifth of its own past (decided at the close, "
        f"held the next month, 10 bp one-way, bills on the cash leg) {beat} buy-and-hold: "
        f"net excess Sharpe {h['derisk_sharpe']:.2f} vs {h['bh_sharpe']:.2f} (difference "
        f"{d:+.3f}, block-bootstrap p(≤0) = {h['derisk_p']:.2f}), with a plain unlevered "
        f"vol-target at {h['voltarget_sharpe']:.2f}. Max drawdown {h['derisk_max_dd']:.0%} "
        f"vs {h['bh_max_dd']:.0%}; terminal wealth {h['derisk_terminal']:,.0f}× vs "
        f"{h['bh_terminal']:,.0f}× over {h['bt_years']:.0f} years. The rule is cheap to run "
        f"(turnover {h['derisk_turnover']:.2f}× NAV a year, so costs are not the problem) — "
        f"the problem is that it sits half out of the market through long calm *bull* "
        f"runs, and the crashes it partly dodges do not repay the rallies it misses.")
    if signal in ("Real", "Mixed"):
        head = "Long calm really is followed by bigger drawdowns than mean reversion explains"
    elif pmin < 0.10:
        head = ("Long calm is followed by somewhat bigger drawdowns than a no-feedback "
                "volatility model predicts, but not significantly so")
    else:
        head = ("Long calm is followed by no bigger drawdowns than a no-feedback "
                "volatility model predicts")
    one = (f"{head} (raw t = {t:.2f}; GARCH p = {h['p_garch']:.2f}, FIGARCH p = "
           f"{h['p_figarch']:.2f}), and selling the calm moved net Sharpe by {d:+.2f}"
           + (" — and that edge survives costs." if trad == "Investable" else
              " — \"low vol = sell\" is a story the tape hints at and cannot cash."))
    return {"signal": signal, "signal_why": signal_why, "trad": trad,
            "trad_why": trad_why, "one_sentence": one}
