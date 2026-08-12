"""Strategy + inference for Study 898 — Managed-Vol Equity.

The tested rule, exactly one execution lag:

    at the close of day t-1, estimate SPY's realized volatility over the trailing
    ``window`` days (rolling std of daily returns x sqrt(252)); hold, for day t,

        w_t = min( cap,  target_vol / RV_{t-1}(window) )

    of SPY and the rest in bills (BIL). The weight earning day t's return uses ONLY
    information through the close of day t-1 (a plain ``shift(1)`` — the single
    documented execution lag). Headline: target 12%, window 21d, cap 2.0x.

Everything is measured **excess of cash (BIL)** on both legs, so the managed book's
excess return is simply ``w_t * (SPY − BIL)_t`` and the buy-and-hold leg is
``(SPY − BIL)_t``. Because a *constant* scale leaves the Sharpe ratio unchanged, any
Sharpe advantage of the managed book over buy-and-hold is, by construction, the
**leverage-timing** component — which is exactly the thing we decompose and test.

Costs: one-way ``cost_bps`` x |Δw| x NAV per daily rebalance (entry trade excluded — both
legs pay it identically); the levered fraction max(w-1, 0) pays a retail ``borrow_spread``
over the bill rate.

Inference (the desk bar):
  * **HAC (Newey-West) alpha regression** of managed-excess on B&H-excess (Moreira-Muir
    style) — the "did the Sharpe rise?" test; alpha > 0 with t >= 2 is the real edge.
  * **HAC t on the daily excess-return difference** ``managed_excess − BH_excess``.
  * The **leverage-timing decomposition**: mean(managed_excess) = beta*mean(BH_excess)
    (the average-leverage / exposure term) + alpha (the timing term).
  * A **bootstrap CI** on the Sharpe advantage (managed − B&H).
  * A **shuffled-vol-signal placebo** averaged over >= 20 seeds (single-seed baselines are
    banned on this desk): same weight distribution, timing alignment destroyed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 252
CAP = 2.0
TARGET = 0.12          # headline: 12% annualised vol target
WINDOW = 21            # headline: 21-day (one month) realized-vol window


# --------------------------------------------------------------------------- #
# The vol-targeting weight rule (past-only, one clean lag)
# --------------------------------------------------------------------------- #
def realized_vol(ret: pd.Series, window: int = WINDOW,
                 ann: int = DAYS_PER_YEAR) -> pd.Series:
    """Trailing annualised realized vol: rolling std (ddof=1) x sqrt(ann)."""
    return ret.rolling(window).std(ddof=1) * np.sqrt(ann)


def vol_target_weights(spy: pd.Series, target: float = TARGET, window: int = WINDOW,
                       cap: float = CAP, ann: int = DAYS_PER_YEAR) -> pd.Series:
    """w_t = min(cap, target / RV_{t-1}) — the weight for day t uses only SPY returns
    through the close of day t-1 (``shift(1)``: the single execution lag).
    NaN during the burn-in (first ``window`` days)."""
    rv = realized_vol(spy, window=window, ann=ann)
    return (target / rv).clip(upper=cap).shift(1)


def run_overlay(ex: pd.DataFrame, target: float = TARGET, window: int = WINDOW,
                cap: float = CAP, cost_bps: float = 0.0,
                borrow_spread_ann: float = 0.0, ann: int = DAYS_PER_YEAR,
                weights: pd.Series | None = None) -> dict:
    """Managed-vol vs buy-and-hold **excess-of-cash** daily returns, same sample.

    ``ex`` must carry columns ``spy`` (SPY simple return), ``spy_excess`` (SPY − cash).
    The managed book holds ``w`` in SPY and ``1 − w`` in bills, so its excess-of-cash
    return is ``w * spy_excess``. Costs: one-way ``cost_bps`` x |Δw| x NAV per daily
    rebalance (entry trade excluded); ``borrow_spread_ann`` charged on max(w-1, 0).
    """
    w = vol_target_weights(ex["spy"], target=target, window=window, cap=cap, ann=ann) \
        if weights is None else weights
    ok = w.notna() & ex["spy_excess"].notna()
    wv = w[ok]
    se = ex["spy_excess"][ok]
    dw = wv.diff().abs()
    dw.iloc[0] = 0.0                                   # entry trade excluded (both legs pay it)
    cost = cost_bps / 1e4 * dw
    borrow = borrow_spread_ann / ann * (wv - 1.0).clip(lower=0.0)
    strat = wv * se - cost - borrow
    return {
        "strat": strat, "bh": se, "w": wv,
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


def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = nw_lags(n)
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        var += 2.0 * w * (float(u[k:] @ u[:-k]) / n)
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def hac_alpha(y: np.ndarray, x: np.ndarray, lags: int | None = None,
              ann: int = DAYS_PER_YEAR) -> dict:
    """OLS ``y = a + b x`` with Newey-West (Bartlett) HAC standard errors.

    ``y`` = managed excess return, ``x`` = buy-and-hold excess return. Returns annualised
    alpha (x ann, in %), HAC t of alpha, beta, appraisal ratio (= Sharpe improvement).
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
        "alpha_bps": alpha_d * 1e4,
        "t_alpha": alpha_d / se_a if se_a > 0 else np.nan,
        "beta": float(b[1]),
        "appraisal": alpha_d / resid_sd * np.sqrt(ann) if resid_sd > 0 else np.nan,
        "n": n, "lags": lags,
    }


# --------------------------------------------------------------------------- #
# Performance summary (excess-of-cash; ann = 252)
# --------------------------------------------------------------------------- #
def max_drawdown(r: pd.Series | np.ndarray) -> float:
    nav = np.cumprod(1.0 + np.asarray(r, float))
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


def perf(r: pd.Series | np.ndarray, ann: int = DAYS_PER_YEAR) -> dict:
    """Excess CAGR, annualised vol, Sharpe (excess-of-cash), max drawdown, wealth mult."""
    x = np.asarray(r, float)
    x = x[np.isfinite(x)]
    lg = np.log1p(x)
    cagr = float(np.exp(lg.mean() * ann) - 1.0)
    vol = float(x.std(ddof=1) * np.sqrt(ann))
    return {
        "cagr_pct": cagr * 100.0,
        "vol_ann_pct": vol * 100.0,
        "sharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(ann)) if x.std(ddof=1) > 0 else np.nan,
        "maxdd_pct": max_drawdown(x) * 100.0,
        "wealth_mult": float(np.exp(lg.sum())),
        "n": len(x), "years": len(x) / ann,
    }


# --------------------------------------------------------------------------- #
# Sharpe advantage bootstrap CI (block bootstrap, paired)
# --------------------------------------------------------------------------- #
def sharpe_gap_bootstrap(strat: np.ndarray, bh: np.ndarray, n_boot: int = 2000,
                         block: int | None = None, alpha: float = 0.05,
                         seed: int = 898, ann: int = DAYS_PER_YEAR) -> dict:
    """Circular block-bootstrap CI for the excess-of-cash Sharpe ADVANTAGE
    (managed − B&H). Pairs are resampled together (same block indices) so the
    dependence between the two legs is preserved."""
    s = np.asarray(strat, float)
    b = np.asarray(bh, float)
    ok = np.isfinite(s) & np.isfinite(b)
    s, b = s[ok], b[ok]
    n = len(s)
    if block is None:
        block = max(1, round(n ** (1.0 / 3.0)))
    rng = np.random.default_rng(seed)
    annf = np.sqrt(ann)

    def _sh(a):
        sd = a.std(ddof=1)
        return a.mean() / sd * annf if sd > 0 else np.nan

    point = _sh(s) - _sh(b)
    n_blocks = int(np.ceil(n / block))
    offs = np.arange(block)
    gaps = np.full(n_boot, np.nan)
    for i in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offs[None, :]) % n).ravel()[:n]
        gaps[i] = _sh(s[idx]) - _sh(b[idx])
    valid = gaps[np.isfinite(gaps)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "sharpe_strat": float(_sh(s)), "sharpe_bh": float(_sh(b)),
        "gap": float(point), "ci_low": float(lo), "ci_high": float(hi),
        "frac_negative": float((valid < 0).mean()), "block": int(block),
        "n_boot_valid": int(valid.size),
    }


# --------------------------------------------------------------------------- #
# The full head-to-head + leverage-timing decomposition
# --------------------------------------------------------------------------- #
def race(ex: pd.DataFrame, target: float = TARGET, window: int = WINDOW, cap: float = CAP,
         cost_bps: float = 0.0, borrow_spread_ann: float = 0.0,
         ann: int = DAYS_PER_YEAR) -> dict:
    """Managed-vol vs buy-and-hold, both excess-of-cash: perf of each leg, the HAC alpha
    regression (Moreira-Muir), the HAC t on the excess-return difference, and the
    leverage-timing decomposition."""
    ov = run_overlay(ex, target=target, window=window, cap=cap, cost_bps=cost_bps,
                     borrow_spread_ann=borrow_spread_ann, ann=ann)
    ps, pb = perf(ov["strat"], ann=ann), perf(ov["bh"], ann=ann)
    reg = hac_alpha(ov["strat"].values, ov["bh"].values, ann=ann)
    diff = ov["strat"].values - ov["bh"].values
    t_diff = newey_west_t(diff)
    # leverage-timing decomposition of the managed mean excess return:
    #   mean(managed) = beta * mean(BH)  [exposure/leverage term]  +  alpha  [timing term]
    mean_bh = float(ov["bh"].mean())
    exposure_bps = reg["beta"] * mean_bh * 1e4
    return {
        "strat": ps, "bh": pb,
        "sharpe_gap": ps["sharpe"] - pb["sharpe"],
        "alpha_ann_pct": reg["alpha_ann_pct"], "alpha_bps": reg["alpha_bps"],
        "t_alpha": reg["t_alpha"], "beta": reg["beta"], "appraisal": reg["appraisal"],
        "mean_diff_bps": float(np.nanmean(diff)) * 1e4, "t_diff": t_diff,
        "exposure_bps": exposure_bps, "timing_bps": reg["alpha_bps"],
        "avg_w": ov["avg_w"], "share_levered": ov["share_levered"],
        "share_capped": ov["share_capped"], "turnover_ann": ov["turnover_ann"],
        "n_days": len(ov["strat"]),
    }


# --------------------------------------------------------------------------- #
# Grid — is the result parameter-robust?
# --------------------------------------------------------------------------- #
def grid(ex: pd.DataFrame, targets=(0.10, 0.12, 0.15), windows=(21, 42, 63),
         cap: float = CAP, ann: int = DAYS_PER_YEAR) -> list[dict]:
    out = []
    for tg in targets:
        for wd in windows:
            r = race(ex, target=tg, window=wd, cap=cap, ann=ann)
            out.append({"target": tg, "window": wd,
                        "sharpe": r["strat"]["sharpe"], "sharpe_bh": r["bh"]["sharpe"],
                        "sharpe_gap": r["sharpe_gap"],
                        "maxdd_pct": r["strat"]["maxdd_pct"],
                        "maxdd_bh_pct": r["bh"]["maxdd_pct"],
                        "t_alpha": r["t_alpha"], "alpha_ann_pct": r["alpha_ann_pct"],
                        "avg_w": r["avg_w"]})
    return out


# --------------------------------------------------------------------------- #
# Era cut
# --------------------------------------------------------------------------- #
def era_cut(ex: pd.DataFrame, split: str = "2016-01-01", target: float = TARGET,
            window: int = WINDOW, cap: float = CAP, ann: int = DAYS_PER_YEAR) -> dict:
    """The headline alpha regression on each half of the sample (era robustness)."""
    ov = run_overlay(ex, target=target, window=window, cap=cap, ann=ann)
    s = pd.Timestamp(split)
    out = {}
    for lbl, mask in (("early", ov["strat"].index < s), ("late", ov["strat"].index >= s)):
        y, x = ov["strat"][mask].values, ov["bh"][mask].values
        reg = hac_alpha(y, x, ann=ann)
        ps, pb = perf(y, ann=ann), perf(x, ann=ann)
        out[lbl] = {"n": len(y), "alpha_ann_pct": reg["alpha_ann_pct"],
                    "t_alpha": reg["t_alpha"], "sharpe": ps["sharpe"],
                    "sharpe_bh": pb["sharpe"], "maxdd_pct": ps["maxdd_pct"],
                    "maxdd_bh_pct": pb["maxdd_pct"]}
    return out


# --------------------------------------------------------------------------- #
# Crash-window drawdowns (the heart-attack ledger)
# --------------------------------------------------------------------------- #
CRASH_WINDOWS = {
    "GFC 2008-09": ("2007-10-01", "2009-06-30"),
    "2018 Q4": ("2018-09-01", "2018-12-31"),
    "COVID 2020": ("2020-02-01", "2020-04-30"),
    "2022 bear": ("2022-01-01", "2022-12-31"),
}


def crash_table(ex: pd.DataFrame, target: float = TARGET, window: int = WINDOW,
                cap: float = CAP, ann: int = DAYS_PER_YEAR) -> dict:
    """Max drawdown of the managed vs B&H excess-of-cash NAV inside each named crash
    window + the full sample (NAVs rebased inside each window)."""
    ov = run_overlay(ex, target=target, window=window, cap=cap, ann=ann)
    out = {}
    for name, (a, b) in CRASH_WINDOWS.items():
        m = (ov["strat"].index >= pd.Timestamp(a)) & (ov["strat"].index <= pd.Timestamp(b))
        if m.sum() < 20:
            continue
        out[name] = {"strat": max_drawdown(ov["strat"][m]) * 100.0,
                     "bh": max_drawdown(ov["bh"][m]) * 100.0}
    out["full sample"] = {"strat": max_drawdown(ov["strat"]) * 100.0,
                          "bh": max_drawdown(ov["bh"]) * 100.0}
    return out


def vol_tracking(ex: pd.DataFrame, target: float = TARGET, window: int = WINDOW,
                 cap: float = CAP, ann: int = DAYS_PER_YEAR) -> dict:
    """Does the overlay actually deliver the promised constant vol? Rolling 21d realized
    vol of the STRATEGY excess return: median + share of days inside a +/-5 vol-pt band."""
    ov = run_overlay(ex, target=target, window=window, cap=cap, ann=ann)
    roll = (ov["strat"].rolling(21).std(ddof=1) * np.sqrt(ann)).dropna()
    band = float(((roll >= target - 0.05) & (roll <= target + 0.05)).mean())
    roll_bh = (ov["bh"].rolling(21).std(ddof=1) * np.sqrt(ann)).dropna()
    return {
        "median_roll_vol_pct": float(roll.median()) * 100.0,
        "p10_pct": float(roll.quantile(0.10)) * 100.0,
        "p90_pct": float(roll.quantile(0.90)) * 100.0,
        "share_in_band": band,
        "bh_median_roll_vol_pct": float(roll_bh.median()) * 100.0,
        "bh_p90_pct": float(roll_bh.quantile(0.90)) * 100.0,
    }


# --------------------------------------------------------------------------- #
# Placebo — shuffle the vol signal (>= 20 seeds, averaged; desk rule)
# --------------------------------------------------------------------------- #
def placebo_shuffle(ex: pd.DataFrame, target: float = TARGET, window: int = WINDOW,
                    cap: float = CAP, n_seeds: int = 200, base_seed: int = 898,
                    ann: int = DAYS_PER_YEAR) -> dict:
    """Destroy the vol-timing information: permute the trailing-RV series across days
    (same marginal weight distribution, no alignment with tomorrow's risk), rebuild the
    overlay per seed, and collect the HAC alpha t and Sharpe gap. Reports the placebo
    mean/sd, ``p_alpha = Pr[shuffled alpha >= observed]`` and
    ``p_dd = Pr[shuffled maxDD at least as shallow as observed]``."""
    rv = realized_vol(ex["spy"], window=window, ann=ann)
    obs = race(ex, target=target, window=window, cap=cap, ann=ann)
    rv_clean = rv.dropna()
    alphas, ts, sh_gaps, dds = [], [], [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        rv_sh = pd.Series(rng.permutation(rv_clean.values), index=rv_clean.index)
        w = (target / rv_sh).clip(upper=cap).shift(1).reindex(ex.index)
        ov = run_overlay(ex, weights=w, ann=ann)
        reg = hac_alpha(ov["strat"].values, ov["bh"].values, ann=ann)
        alphas.append(reg["alpha_ann_pct"]); ts.append(reg["t_alpha"])
        sh_gaps.append(perf(ov["strat"], ann=ann)["sharpe"] - perf(ov["bh"], ann=ann)["sharpe"])
        dds.append(perf(ov["strat"], ann=ann)["maxdd_pct"])
    alphas, ts, sh_gaps, dds = map(np.array, (alphas, ts, sh_gaps, dds))
    return {
        "obs_alpha_ann_pct": obs["alpha_ann_pct"], "obs_t": obs["t_alpha"],
        "obs_sharpe_gap": obs["sharpe_gap"], "obs_maxdd_pct": obs["strat"]["maxdd_pct"],
        "placebo_mean_alpha": float(alphas.mean()), "placebo_sd_alpha": float(alphas.std(ddof=1)),
        "placebo_mean_t": float(ts.mean()),
        "placebo_mean_sh_gap": float(sh_gaps.mean()),
        "placebo_mean_maxdd_pct": float(dds.mean()),
        "p_alpha": float((alphas >= obs["alpha_ann_pct"]).mean()),
        "p_sh_gap": float((sh_gaps >= obs["sharpe_gap"]).mean()),
        "p_dd": float((dds >= obs["strat"]["maxdd_pct"]).mean()),
        "n_seeds": n_seeds, "alpha_draws": alphas, "sh_gap_draws": sh_gaps,
    }


# --------------------------------------------------------------------------- #
# Synthetic harness — averaged over seeds (desk rule)
# --------------------------------------------------------------------------- #
def synthetic_check(disconnect: float, n_seeds: int = 20, n_days: int = 4000,
                    target: float = TARGET, window: int = WINDOW, cap: float = CAP,
                    base_seed: int = 898, ann: int = DAYS_PER_YEAR) -> dict:
    """Run the full rule on ``n_seeds`` independent synthetic worlds and average the HAC
    alpha t (the desk rule: random baselines are never single-seed). The synthetic cash
    leg is 0%, so the world's excess returns feed both ``spy`` and ``spy_excess``."""
    from . import data as d
    ts, alphas = [], []
    for s in range(n_seeds):
        r, _ = d.synthetic_world(n_days=n_days, disconnect=disconnect,
                                 seed=base_seed + 1000 * s)
        ser = pd.Series(r)
        ex = pd.DataFrame({"spy": ser, "cash": 0.0, "spy_excess": ser})
        res = race(ex, target=target, window=window, cap=cap, ann=ann)
        ts.append(res["t_alpha"]); alphas.append(res["alpha_ann_pct"])
    ts, alphas = np.array(ts), np.array(alphas)
    return {
        "mean_t": float(ts.mean()), "sd_t": float(ts.std(ddof=1)),
        "mean_alpha_ann_pct": float(alphas.mean()),
        "share_t_ge_2": float((ts >= 2.0).mean()), "n_seeds": n_seeds,
    }


def synthetic_detect(disconnect: float, seed: int = 898, n_days: int = 4000,
                     target: float = TARGET, window: int = WINDOW,
                     cap: float = CAP, ann: int = DAYS_PER_YEAR) -> dict:
    """Single-world detector (used by the fast notebook/live cells)."""
    from . import data as d
    r, _ = d.synthetic_world(n_days=n_days, disconnect=disconnect, seed=seed)
    ser = pd.Series(r)
    ex = pd.DataFrame({"spy": ser, "cash": 0.0, "spy_excess": ser})
    res = race(ex, target=target, window=window, cap=cap, ann=ann)
    return {"t_alpha": res["t_alpha"], "alpha_ann_pct": res["alpha_ann_pct"],
            "sharpe_gap": res["sharpe_gap"], "n_days": res["n_days"]}
