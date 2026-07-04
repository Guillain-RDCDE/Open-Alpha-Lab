"""Strategy + stats for Study 633 — BTC Vol Targeting.

The tested rule, exactly one execution lag:

    at the close of day t-1, estimate BTC's realized volatility over the trailing
    ``window`` days (rolling std of daily returns x sqrt(365) — BTC trades every
    calendar day); hold, for day t,

        w_t = min( cap,  target_vol / RV_{t-1}(window) ),      cap = 1.5

    i.e. the weight earning day t's return uses ONLY information through the close
    of day t-1 (a plain ``shift(1)`` — the single documented execution lag). The
    un-invested fraction sits in cash earning 0% (a stablecoin balance — documented,
    and conservative *against* the overlay). Headline: target 30%, window 30d.

Races are excess-vs-excess with rf = 0 (the same 0% cash assumption on both legs, so
the convention cancels in every comparison). Costs are one-way bps x |Δw| x NAV per
daily rebalance; leverage above 1 pays a retail margin-borrow spread on max(w-1, 0).

Inference (the desk bar):
  * **HAC (Newey-West) t on the daily log-wealth growth difference**
    ``d_t = log(1+r_strat) - log(1+r_bh)`` — the "same ride?" test on terminal wealth.
  * **HAC alpha regression** of strategy-on-B&H daily returns (Moreira-Muir style) —
    the "is there risk-adjusted alpha?" test.
  * A shuffled-vol-signal placebo averaged over >= 20 seeds (single-seed baselines
    are banned on this desk).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365
CAP = 1.5
TARGET = 0.30          # headline: 30% annualised vol target
WINDOW = 30            # headline: 30-day realized-vol window


# --------------------------------------------------------------------------- #
# The vol-targeting weight rule (past-only, one clean lag)
# --------------------------------------------------------------------------- #
def realized_vol(ret: pd.Series, window: int = WINDOW,
                 ann: int = DAYS_PER_YEAR) -> pd.Series:
    """Trailing annualised realized vol: rolling std (ddof=1) x sqrt(ann)."""
    return ret.rolling(window).std(ddof=1) * np.sqrt(ann)


def vol_target_weights(ret: pd.Series, target: float = TARGET, window: int = WINDOW,
                       cap: float = CAP, ann: int = DAYS_PER_YEAR) -> pd.Series:
    """w_t = min(cap, target / RV_{t-1}) — the weight for day t uses only returns
    through the close of day t-1 (``shift(1)``: the single execution lag).
    NaN during the burn-in (first ``window`` days)."""
    rv = realized_vol(ret, window=window, ann=ann)
    return (target / rv).clip(upper=cap).shift(1)


def run_overlay(ret: pd.Series, target: float = TARGET, window: int = WINDOW,
                cap: float = CAP, cost_bps: float = 0.0,
                borrow_spread_ann: float = 0.0, ann: int = DAYS_PER_YEAR,
                weights: pd.Series | None = None) -> dict:
    """Vol-targeted vs buy-and-hold daily returns on the SAME post-burn-in sample.

    Costs: one-way ``cost_bps`` x |w_t - w_{t-1}| x NAV per daily rebalance (the initial
    entry trade is excluded — both legs pay it identically), plus a retail
    ``borrow_spread_ann`` charged on the levered fraction max(w-1, 0). Cash earns 0%.
    """
    w = vol_target_weights(ret, target=target, window=window, cap=cap, ann=ann) \
        if weights is None else weights
    ok = w.notna() & ret.notna()
    wv, rv_ = w[ok], ret[ok]
    dw = wv.diff().abs()
    dw.iloc[0] = 0.0                                   # entry trade excluded (both legs pay it)
    cost = cost_bps / 1e4 * dw
    borrow = borrow_spread_ann / ann * (wv - 1.0).clip(lower=0.0)
    strat = wv * rv_ - cost - borrow
    return {
        "strat": strat, "bh": rv_, "w": wv,
        "avg_w": float(wv.mean()), "share_levered": float((wv > 1.0).mean()),
        "share_capped": float((wv >= cap - 1e-12).mean()),
        "avg_turnover": float(dw.mean()),              # |Δw| per day
        "turnover_ann": float(dw.mean()) * ann,        # x NAV per year
    }


# --------------------------------------------------------------------------- #
# Inference — HAC (Newey-West, Bartlett kernel)
# --------------------------------------------------------------------------- #
def nw_lags(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_tstat(x: np.ndarray, lags: int | None = None) -> dict:
    """HAC (Newey-West) t of the mean of a serially-correlated series."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if lags is None:
        lags = nw_lags(n)
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1.0)) * (float(e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(lrv, 0.0) / n)
    return {"mean": mu, "t": mu / se if se > 0 else np.nan, "n": n, "lags": lags}


def hac_alpha(y: np.ndarray, x: np.ndarray, lags: int | None = None,
              ann: int = DAYS_PER_YEAR) -> dict:
    """OLS ``y = a + b x`` with Newey-West (Bartlett) HAC standard errors.

    Returns annualised alpha (x ann, in %), HAC t of alpha, beta, appraisal ratio.
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    if lags is None:
        lags = nw_lags(n)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    Xu = X * e[:, None]
    S = Xu.T @ Xu
    for k in range(1, lags + 1):
        wgt = 1.0 - k / (lags + 1.0)
        G = Xu[k:].T @ Xu[:-k]
        S += wgt * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se_a = float(np.sqrt(max(V[0, 0], 0.0)))
    resid_sd = float(e.std(ddof=2))
    alpha_d = float(b[0])
    return {
        "alpha_ann_pct": alpha_d * ann * 100.0,
        "t_alpha": alpha_d / se_a if se_a > 0 else np.nan,
        "beta": float(b[1]),
        "appraisal": alpha_d / resid_sd * np.sqrt(ann) if resid_sd > 0 else np.nan,
        "n": n, "lags": lags,
    }


# --------------------------------------------------------------------------- #
# Performance summary (rf = 0 documented; ann = 365 for BTC calendar-daily bars)
# --------------------------------------------------------------------------- #
def max_drawdown(r: pd.Series | np.ndarray) -> float:
    nav = np.cumprod(1.0 + np.asarray(r, float))
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


def perf(r: pd.Series | np.ndarray, ann: int = DAYS_PER_YEAR) -> dict:
    """CAGR, annualised vol, Sharpe (rf = 0), max drawdown, terminal wealth multiple."""
    x = np.asarray(r, float)
    x = x[np.isfinite(x)]
    n = len(x)
    lg = np.log1p(x)
    cagr = float(np.exp(lg.mean() * ann) - 1.0)
    vol = float(x.std(ddof=1) * np.sqrt(ann))
    return {
        "cagr_pct": cagr * 100.0,
        "vol_ann_pct": vol * 100.0,
        "sharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(ann)) if x.std(ddof=1) > 0 else np.nan,
        "maxdd_pct": max_drawdown(x) * 100.0,
        "wealth_mult": float(np.exp(lg.sum())),
        "n": n, "years": n / ann,
    }


def race(ret: pd.Series, target: float = TARGET, window: int = WINDOW,
         cap: float = CAP, cost_bps: float = 0.0, borrow_spread_ann: float = 0.0,
         ann: int = DAYS_PER_YEAR) -> dict:
    """The full head-to-head: perf of both legs + HAC t on the log-growth gap +
    HAC alpha of strategy-on-B&H, on the same post-burn-in sample."""
    ov = run_overlay(ret, target=target, window=window, cap=cap, cost_bps=cost_bps,
                     borrow_spread_ann=borrow_spread_ann, ann=ann)
    ps, pb = perf(ov["strat"], ann=ann), perf(ov["bh"], ann=ann)
    d = np.log1p(ov["strat"].values) - np.log1p(ov["bh"].values)
    hd = hac_tstat(d)
    reg = hac_alpha(ov["strat"].values, ov["bh"].values, ann=ann)
    return {
        "strat": ps, "bh": pb,
        "growth_gap_ann_pct": hd["mean"] * ann * 100.0,   # strat - B&H, log-growth/yr
        "t_growth_gap": hd["t"],
        "alpha_ann_pct": reg["alpha_ann_pct"], "t_alpha": reg["t_alpha"],
        "beta": reg["beta"], "appraisal": reg["appraisal"],
        "avg_w": ov["avg_w"], "share_levered": ov["share_levered"],
        "share_capped": ov["share_capped"], "turnover_ann": ov["turnover_ann"],
        "n_days": len(ov["strat"]),
    }


# --------------------------------------------------------------------------- #
# Grid — is the result parameter-robust?
# --------------------------------------------------------------------------- #
def grid(ret: pd.Series, targets=(0.20, 0.30, 0.50), windows=(20, 30, 60),
         cap: float = CAP, ann: int = DAYS_PER_YEAR) -> list[dict]:
    out = []
    for tg in targets:
        for wd in windows:
            r = race(ret, target=tg, window=wd, cap=cap, ann=ann)
            out.append({"target": tg, "window": wd,
                        "sharpe": r["strat"]["sharpe"], "sharpe_bh": r["bh"]["sharpe"],
                        "maxdd_pct": r["strat"]["maxdd_pct"],
                        "maxdd_bh_pct": r["bh"]["maxdd_pct"],
                        "cagr_pct": r["strat"]["cagr_pct"],
                        "cagr_bh_pct": r["bh"]["cagr_pct"],
                        "t_alpha": r["t_alpha"], "alpha_ann_pct": r["alpha_ann_pct"],
                        "t_growth_gap": r["t_growth_gap"],
                        "growth_gap_ann_pct": r["growth_gap_ann_pct"]})
    return out


# --------------------------------------------------------------------------- #
# Placebo — shuffle the vol signal (>= 20 seeds, averaged; desk rule)
# --------------------------------------------------------------------------- #
def placebo_shuffle(ret: pd.Series, target: float = TARGET, window: int = WINDOW,
                    cap: float = CAP, n_seeds: int = 200, base_seed: int = 633,
                    ann: int = DAYS_PER_YEAR) -> dict:
    """Destroy the vol-timing information: permute the trailing-RV series across days
    (same marginal weight distribution, no alignment with tomorrow's risk), rebuild the
    overlay per seed, and collect the HAC alpha t and the max drawdown. Reports the
    placebo mean/sd, ``p_alpha = Pr[shuffled alpha >= observed]`` and
    ``p_dd = Pr[shuffled maxDD at least as shallow as observed]``."""
    rv = realized_vol(ret, window=window, ann=ann)
    obs = race(ret, target=target, window=window, cap=cap, ann=ann)
    ts, alphas, dds = [], [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        rv_sh = pd.Series(rng.permutation(rv.dropna().values), index=rv.dropna().index)
        w = (target / rv_sh).clip(upper=cap).shift(1).reindex(ret.index)
        ov = run_overlay(ret, weights=w, ann=ann)
        reg = hac_alpha(ov["strat"].values, ov["bh"].values, ann=ann)
        ts.append(reg["t_alpha"])
        alphas.append(reg["alpha_ann_pct"])
        dds.append(perf(ov["strat"], ann=ann)["maxdd_pct"])
    ts, alphas, dds = np.array(ts), np.array(alphas), np.array(dds)
    return {
        "obs_alpha_ann_pct": obs["alpha_ann_pct"], "obs_t": obs["t_alpha"],
        "obs_maxdd_pct": obs["strat"]["maxdd_pct"],
        "placebo_mean_t": float(ts.mean()), "placebo_sd_t": float(ts.std(ddof=1)),
        "placebo_mean_alpha": float(alphas.mean()),
        "placebo_mean_maxdd_pct": float(dds.mean()),
        "p_alpha": float((alphas >= obs["alpha_ann_pct"]).mean()),
        "p_dd": float((dds >= obs["strat"]["maxdd_pct"]).mean()),
        "n_seeds": n_seeds,
        "alpha_draws": alphas, "dd_draws": dds,      # for the notebook histograms
    }


# --------------------------------------------------------------------------- #
# Crash-window drawdowns (the heart-attack ledger)
# --------------------------------------------------------------------------- #
CRASH_WINDOWS = {
    "2018 bear": ("2017-12-01", "2018-12-31"),
    "COVID 2020": ("2020-02-01", "2020-04-30"),
    "2021-22 bear": ("2021-11-01", "2022-12-31"),
}


def crash_table(ret: pd.Series, target: float = TARGET, window: int = WINDOW,
                cap: float = CAP, ann: int = DAYS_PER_YEAR) -> dict:
    """Max drawdown of the vol-targeted vs B&H NAV inside each named crash window +
    the full sample (NAVs rebased inside each window)."""
    ov = run_overlay(ret, target=target, window=window, cap=cap, ann=ann)
    out = {}
    for name, (a, b) in CRASH_WINDOWS.items():
        m = (ov["strat"].index >= pd.Timestamp(a)) & (ov["strat"].index <= pd.Timestamp(b))
        if m.sum() < 30:
            continue
        out[name] = {"strat": max_drawdown(ov["strat"][m]) * 100.0,
                     "bh": max_drawdown(ov["bh"][m]) * 100.0}
    out["full sample"] = {"strat": max_drawdown(ov["strat"]) * 100.0,
                          "bh": max_drawdown(ov["bh"]) * 100.0}
    return out


def vol_tracking(ret: pd.Series, target: float = TARGET, window: int = WINDOW,
                 cap: float = CAP, ann: int = DAYS_PER_YEAR) -> dict:
    """Does the overlay actually deliver the promised constant vol? Rolling 30d
    realized vol of the STRATEGY: median + share of days inside a +/-10 vol-pt band."""
    ov = run_overlay(ret, target=target, window=window, cap=cap, ann=ann)
    roll = ov["strat"].rolling(30).std(ddof=1) * np.sqrt(ann)
    roll = roll.dropna()
    band = float(((roll >= target - 0.10) & (roll <= target + 0.10)).mean())
    roll_bh = (ov["bh"].rolling(30).std(ddof=1) * np.sqrt(ann)).dropna()
    return {
        "median_roll_vol_pct": float(roll.median()) * 100.0,
        "p10_pct": float(roll.quantile(0.10)) * 100.0,
        "p90_pct": float(roll.quantile(0.90)) * 100.0,
        "share_in_band": band,
        "bh_median_roll_vol_pct": float(roll_bh.median()) * 100.0,
        "bh_p90_pct": float(roll_bh.quantile(0.90)) * 100.0,
    }


# --------------------------------------------------------------------------- #
# Synthetic harness — averaged over seeds (desk rule)
# --------------------------------------------------------------------------- #
def synthetic_check(disconnect: float, n_seeds: int = 20, n_days: int = 4000,
                    target: float = TARGET, window: int = WINDOW, cap: float = CAP,
                    base_seed: int = 633, ann: int = DAYS_PER_YEAR) -> dict:
    """Run the full rule on ``n_seeds`` independent synthetic worlds and average the
    HAC alpha t (the desk rule: random baselines are never single-seed)."""
    from . import data as d
    ts, alphas = [], []
    for s in range(n_seeds):
        r, _ = d.synthetic_world(n_days=n_days, disconnect=disconnect,
                                 seed=base_seed + 1000 * s)
        ser = pd.Series(r)
        res = race(ser, target=target, window=window, cap=cap, ann=ann)
        ts.append(res["t_alpha"])
        alphas.append(res["alpha_ann_pct"])
    ts, alphas = np.array(ts), np.array(alphas)
    return {
        "mean_t": float(ts.mean()), "sd_t": float(ts.std(ddof=1)),
        "mean_alpha_ann_pct": float(alphas.mean()),
        "share_t_ge_2": float((ts >= 2.0).mean()), "n_seeds": n_seeds,
    }
