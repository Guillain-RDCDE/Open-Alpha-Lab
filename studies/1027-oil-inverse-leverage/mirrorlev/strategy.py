"""The mirror test — Study 1027.

One sign convention runs through this whole module, because the two GARCH families print
their asymmetry with opposite signs and it is easy to get lost:

    **inverse score > 0  ⇔  rallies raise volatility more than falls (the oil claim)**
    **inverse score < 0  ⇔  falls raise volatility more than rallies (equity leverage)**

Four independent looks at the same question, then the money question:

1. ``fit_asymmetry`` — GJR-GARCH(1,1,1) and EGARCH(1,1,1) through the ``arch`` package,
   Student-t innovations, **robust (sandwich) standard errors**. GJR's ``gamma`` is the extra
   variance a *down* shock carries (equity > 0); EGARCH's ``gamma`` is the signed-shock term
   (equity < 0). Both are mapped onto the inverse score above.
2. ``forward_vol_regression`` — model-free. Tomorrow-onward realised volatility over the next
   ``h`` days regressed on today's up-move ``r+`` and down-move ``|r-|`` separately, with the
   trailing realised vol as a control (volatility clusters regardless of sign). The test is on
   ``b_up - b_down``; standard errors are Newey-West with ``h`` lags because consecutive
   forward windows overlap. ``sign_correlation`` adds the raw corr(r_t, RV_{t+1..t+h}) with a
   circular block-bootstrap interval.
3. ``era_table`` / ``rolling_asymmetry`` — is the mirror a property of oil, or of an era?
   Disjoint eras are compared with a z-test on the difference of their GJR gammas (disjoint
   samples, so the two estimates are close to independent), and a rolling three-year GJR traces
   the path.
4. ``skew_table`` — return skewness with a block-bootstrap interval. An asset whose volatility
   rises in rallies should be *less* negatively skewed than one whose volatility rises in falls.
5. ``vol_target_overlay`` — the consequence. A thermostat that scales exposure by
   ``target / trailing vol`` de-risks after whatever raised the volatility. On equities that is
   after falls; if the mirror is real, on oil it is after rallies. **One execution lag**: the
   weight computed from returns through the close of day *t* earns day *t+1*'s return — one
   ``shift(1)``, applied once, here and nowhere else. Costs are one-way × |Δw| × NAV.

``verdict`` applies the stamps by a rule written before the real tape was run.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# 1. Parametric asymmetry
# --------------------------------------------------------------------------- #
def fit_asymmetry(r: pd.Series, model: str = "gjr", dist: str = "t") -> dict:
    """Fit a GJR-GARCH(1,1,1) or EGARCH(1,1,1) to daily log returns ``r`` (decimal).

    Returns the raw asymmetry coefficient, its robust SE and t, and the **inverse score**
    ``inv`` / ``inv_t`` (positive = volatility rises more after rallies). For GJR,
    ``inv = -gamma``; for EGARCH, ``inv = +gamma``. Also returns the down/up impact ratio
    implied by the fit (GJR: (alpha+gamma)/alpha; EGARCH: exp-scale ratio of the news-impact
    slopes), persistence, the log-likelihood and the optimiser's convergence flag.
    """
    from arch import arch_model

    x = 100.0 * pd.Series(r).dropna().astype(float)
    if len(x) < 250:
        return {}
    vol = "GARCH" if model == "gjr" else "EGARCH"
    am = arch_model(x, mean="Constant", vol=vol, p=1, o=1, q=1, dist=dist, rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", cov_type="robust", options={"maxiter": 500})
    p, se = res.params, res.std_err
    g, g_se = float(p["gamma[1]"]), float(se["gamma[1]"])
    a = float(p["alpha[1]"])
    b = float(p["beta[1]"])
    if model == "gjr":
        inv, persistence = -g, a + 0.5 * g + b
        ratio = (a + g) / a if a > 1e-8 else np.inf
    else:
        inv, persistence = g, b
        # EGARCH news impact on log-variance: down slope (gamma - alpha)… expressed as
        # |slope_down| / |slope_up| = (alpha - gamma) / (alpha + gamma)
        ratio = (a - g) / (a + g) if abs(a + g) > 1e-8 else np.inf
    t = g / g_se if g_se > 0 else np.nan
    return {"model": model, "gamma": g, "gamma_se": g_se, "gamma_t": float(t),
            "inv": float(inv), "inv_t": float(-t if model == "gjr" else t),
            "alpha": a, "beta": b, "persistence": float(persistence),
            "down_up_ratio": float(ratio), "nobs": int(res.nobs),
            "loglik": float(res.loglikelihood),
            "converged": bool(res.convergence_flag == 0)}


# --------------------------------------------------------------------------- #
# 2. Model-free asymmetry
# --------------------------------------------------------------------------- #
def realised_vol(r: pd.Series, window: int) -> pd.Series:
    """Trailing annualised realised vol over ``window`` days ending at (and including) t."""
    return np.sqrt(TRADING_DAYS * (r ** 2).rolling(window).mean())


def forward_vol(r: pd.Series, h: int) -> pd.Series:
    """Annualised realised vol over days t+1 … t+h, stamped at t (strictly future)."""
    return realised_vol(r, h).shift(-h)


def _hac_ols(y: np.ndarray, X: np.ndarray, lags: int):
    import statsmodels.api as sm

    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def forward_vol_regression(r: pd.Series, h: int = 21, control_window: int = 21) -> dict:
    """RV(t+1..t+h) = a + b_up·r⁺_t + b_dn·|r⁻_t| + c·RV(t-w+1..t) + e, Newey-West(h).

    ``b_up - b_dn`` is the model-free inverse score: positive means an up-move of a given size
    is followed by more volatility than a down-move of the same size, after controlling for
    the volatility already in place. Slopes are in "annualised vol points per unit of daily
    return". Returns the slopes, the difference, its HAC t and the sample size.
    """
    r = pd.Series(r).astype(float)
    df = pd.DataFrame({"fwd": forward_vol(r, h),
                       "up": r.clip(lower=0.0), "dn": (-r).clip(lower=0.0),
                       "past": realised_vol(r, control_window)}).dropna()
    if len(df) < 200:
        return {}
    fit = _hac_ols(df["fwd"].to_numpy(), df[["up", "dn", "past"]].to_numpy(), lags=h)
    b = fit.params
    cov = fit.cov_params()
    diff = b[1] - b[2]
    var = cov[1, 1] + cov[2, 2] - 2 * cov[1, 2]
    se = float(np.sqrt(max(var, 0.0)))
    return {"h": h, "b_up": float(b[1]), "b_dn": float(b[2]),
            "t_up": float(fit.tvalues[1]), "t_dn": float(fit.tvalues[2]),
            "diff": float(diff), "diff_se": se,
            "diff_t": float(diff / se) if se > 0 else np.nan, "n": int(len(df))}


def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block-bootstrap index vector of length n."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return idx.ravel()[:n]


def sign_correlation(r: pd.Series, h: int = 21, n_boot: int = 500, block: int | None = None,
                     seed: int = 1027) -> dict:
    """corr(r_t, RV_{t+1..t+h}) with a circular block-bootstrap 95% interval and p-value.

    Positive = rallies precede higher volatility (the oil claim). Pairs are resampled in
    blocks of ``max(h, 21)`` days so the overlap between consecutive forward windows is
    preserved. Also returns the |r|-matched up/down ratio of forward vol (within |r| quintiles,
    averaged), a descriptive number that is immune to "down days are bigger".
    """
    r = pd.Series(r).astype(float)
    df = pd.DataFrame({"r": r, "fwd": forward_vol(r, h)}).dropna()
    x, y = df["r"].to_numpy(), df["fwd"].to_numpy()
    rho = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    blk = block or max(h, 21)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = _block_indices(len(x), blk, rng)
        boots[i] = np.corrcoef(x[idx], y[idx])[0, 1]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    q = pd.qcut(np.abs(x), 5, labels=False, duplicates="drop")
    ratios = []
    for k in np.unique(q):
        m = q == k
        up, dn = y[m & (x > 0)], y[m & (x < 0)]
        if len(up) > 10 and len(dn) > 10:
            ratios.append(up.mean() / dn.mean())
    return {"h": h, "corr": rho, "lo": float(lo), "hi": float(hi), "p": min(1.0, max(p, 1.0 / n_boot)),
            "matched_up_over_down": float(np.mean(ratios)) if ratios else np.nan,
            "n": int(len(x))}


# --------------------------------------------------------------------------- #
# 3. Regimes
# --------------------------------------------------------------------------- #
def era_table(r: pd.Series, eras: dict[str, tuple[str, str]], h: int = 21) -> pd.DataFrame:
    """GJR, EGARCH and the model-free regression on each era ``{label: (start, end)}``."""
    rows = []
    for lab, (a, b) in eras.items():
        x = r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]
        g = fit_asymmetry(x, "gjr")
        e = fit_asymmetry(x, "egarch")
        m = forward_vol_regression(x, h)
        rows.append({"era": lab, "start": str(x.index[0].date()), "end": str(x.index[-1].date()),
                     "n": int(len(x)), "gjr_gamma": g["gamma"], "gjr_se": g["gamma_se"],
                     "gjr_inv_t": g["inv_t"], "egarch_gamma": e["gamma"],
                     "egarch_inv_t": e["inv_t"], "mf_diff": m["diff"], "mf_t": m["diff_t"],
                     "ann_vol": float(x.std() * np.sqrt(TRADING_DAYS))})
    return pd.DataFrame(rows).set_index("era")


def era_difference(tbl: pd.DataFrame, a: str, b: str) -> dict:
    """z-test that the GJR gamma differs between two disjoint eras (≈ independent fits)."""
    ga, gb = tbl.loc[a, "gjr_gamma"], tbl.loc[b, "gjr_gamma"]
    sa, sb = tbl.loc[a, "gjr_se"], tbl.loc[b, "gjr_se"]
    se = float(np.sqrt(sa ** 2 + sb ** 2))
    z = float((gb - ga) / se) if se > 0 else np.nan
    return {"a": a, "b": b, "gamma_a": float(ga), "gamma_b": float(gb), "diff": float(gb - ga),
            "se": se, "z": z}


def rolling_asymmetry(r: pd.Series, window: int = 756, step: int = 63,
                      model: str = "gjr") -> pd.DataFrame:
    """Rolling GJR (or EGARCH) inverse score on ``window``-day windows every ``step`` days."""
    r = pd.Series(r).dropna()
    rows = []
    for end in range(window, len(r) + 1, step):
        x = r.iloc[end - window:end]
        f = fit_asymmetry(x, model)
        if not f:
            continue
        rows.append({"date": x.index[-1], "inv": f["inv"], "inv_t": f["inv_t"],
                     "converged": f["converged"]})
    return pd.DataFrame(rows).set_index("date")


# --------------------------------------------------------------------------- #
# 4. Skewness
# --------------------------------------------------------------------------- #
def skew_ci(r: pd.Series, n_boot: int = 500, block: int = 21, seed: int = 1027) -> dict:
    """Sample skewness with a circular block-bootstrap 95% interval.

    Also returns a 5/95 quantile skewness, ``(q95 + q05 - 2·q50) / (q95 - q05)``, which a
    single extreme day (WTI's −33% on 17 January 1991) cannot drive.
    """
    from scipy.stats import skew

    x = pd.Series(r).dropna().to_numpy()
    s = float(skew(x))
    rng = np.random.default_rng(seed)
    b = np.array([skew(x[_block_indices(len(x), block, rng)]) for _ in range(n_boot)])
    lo, hi = np.quantile(b, [0.025, 0.975])
    q05, q50, q95 = np.quantile(x, [0.05, 0.50, 0.95])
    qs = float((q95 + q05 - 2 * q50) / (q95 - q05)) if q95 > q05 else np.nan
    return {"skew": s, "lo": float(lo), "hi": float(hi), "quantile_skew": qs,
            "n": int(len(x))}


def skew_difference(r1: pd.Series, r2: pd.Series, n_boot: int = 500, block: int = 21,
                    seed: int = 1027) -> dict:
    """skew(r1) - skew(r2) on common dates, paired block bootstrap (95% CI, two-sided p)."""
    from scipy.stats import skew

    df = pd.concat([r1, r2], axis=1, join="inner").dropna()
    a, b = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    d = float(skew(a) - skew(b))
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        idx = _block_indices(len(a), block, rng)
        bs[i] = skew(a[idx]) - skew(b[idx])
    lo, hi = np.quantile(bs, [0.025, 0.975])
    p = float(2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return {"diff": d, "lo": float(lo), "hi": float(hi), "p": min(1.0, max(p, 1.0 / n_boot)),
            "n": int(len(a))}


# --------------------------------------------------------------------------- #
# 5. The vol-target overlay
# --------------------------------------------------------------------------- #
def sharpe(x: pd.Series | np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan


def max_drawdown(x: pd.Series) -> float:
    w = np.exp(np.log1p(pd.Series(x).fillna(0.0)).cumsum())
    return float((w / w.cummax() - 1.0).min())


def vol_target_weights(r: pd.Series, window: int = 21, cap: float = 2.0,
                       min_history: int = 252) -> pd.Series:
    """Weight decided at the close of t from returns through t (NOT yet lagged).

    ``w_t = min(cap, target_t / RV21_t)`` where ``target_t`` is the **expanding mean** of the
    trailing realised vol up to t — a past-only normaliser, so average exposure sits near one
    without peeking at the full-sample volatility. NaN until ``min_history`` days exist.
    """
    rv = realised_vol(r, window)
    target = rv.expanding(min_periods=min_history).mean()
    w = (target / rv).clip(upper=cap)
    return w.rename("w")


def vol_target_overlay(r: pd.Series, rf: pd.Series | None = None, window: int = 21,
                       cap: float = 2.0, cost_bps: float = 5.0, n_boot: int = 500,
                       block: int = 21, seed: int = 1027) -> dict:
    """Vol-target overlay vs buy-and-hold, excess-of-cash on both legs, gross and net.

    ``r`` are daily **simple** returns. If ``rf`` is given the asset's excess return is
    ``r - rf``; if not, ``r`` is treated as already an excess return (the convention for the
    WTI spot proxy: a fully collateralised futures position whose collateral earns the bill
    rate). The overlay's excess return on day t+1 is ``w_t · excess_{t+1}`` — the weight is
    shifted exactly once, here. Cost on day t+1 is ``cost_bps/1e4 · |w_t - w_{t-1}|``
    (one-way × traded NAV). The Sharpe gain carries a paired circular block-bootstrap 95% CI.

    Also measures *what the thermostat reacts to*: the correlation between the trailing 21-day
    return and the weight. On an equity-leverage asset it is positive (falls → vol → smaller
    weight); if the mirror holds, on oil it is negative (rallies → vol → smaller weight).
    """
    r = pd.Series(r).astype(float)
    ex = r - rf.reindex(r.index) if rf is not None else r.copy()
    w_dec = vol_target_weights(np.log1p(r), window=window, cap=cap)
    w = w_dec.shift(1)                                    # the one execution lag
    turn = w.diff().abs()
    gross = w * ex
    net = gross - cost_bps / 1e4 * turn
    df = pd.DataFrame({"bh": ex, "gross": gross, "net": net, "w": w, "turn": turn}).dropna()
    past21 = r.rolling(21).sum()
    react = pd.concat([past21, w_dec], axis=1).dropna()
    rng = np.random.default_rng(seed)
    a, b = df["net"].to_numpy(), df["bh"].to_numpy()
    gains = np.empty(n_boot)
    for i in range(n_boot):
        idx = _block_indices(len(a), block, rng)
        gains[i] = sharpe(a[idx]) - sharpe(b[idx])
    lo, hi = np.quantile(gains, [0.025, 0.975])
    sr_bh, sr_g, sr_n = sharpe(df["bh"]), sharpe(df["gross"]), sharpe(df["net"])
    turn_ann = float(df["turn"].mean() * TRADING_DAYS)
    gain_gross = sr_g - sr_bh
    return {
        "start": str(df.index[0].date()), "end": str(df.index[-1].date()), "n": int(len(df)),
        "sharpe_bh": sr_bh, "sharpe_gross": sr_g, "sharpe_net": sr_n,
        "gain_gross": gain_gross, "gain_net": sr_n - sr_bh,
        "gain_lo": float(lo), "gain_hi": float(hi),
        "vol_bh": float(df["bh"].std() * np.sqrt(TRADING_DAYS)),
        "vol_net": float(df["net"].std() * np.sqrt(TRADING_DAYS)),
        "mdd_bh": max_drawdown(df["bh"]), "mdd_net": max_drawdown(df["net"]),
        "mean_w": float(df["w"].mean()), "turnover_ann": turn_ann,
        "cost_bps": cost_bps,
        "breakeven_bps": _breakeven_bps(df) if gain_gross > 0 else 0.0,
        "react_corr": float(react.corr().iloc[0, 1]),
        "series": df,
    }


def _breakeven_bps(df: pd.DataFrame) -> float:
    """One-way cost (bps) at which the net Sharpe falls to buy-and-hold's."""
    target = sharpe(df["bh"])
    lo, hi = 0.0, 1000.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if sharpe(df["gross"] - mid / 1e4 * df["turn"]) > target:
            lo = mid
        else:
            hi = mid
    return float(lo)


def cost_sweep(r: pd.Series, rf: pd.Series | None = None,
               costs=(0.0, 2.0, 5.0, 10.0, 20.0, 50.0)) -> pd.DataFrame:
    """Net Sharpe gain of the overlay across one-way cost levels (no bootstrap)."""
    rows = []
    for c in costs:
        o = vol_target_overlay(r, rf, cost_bps=c, n_boot=2)
        rows.append({"cost_bps": c, "sharpe_net": o["sharpe_net"], "gain_net": o["gain_net"]})
    return pd.DataFrame(rows).set_index("cost_bps")


# --------------------------------------------------------------------------- #
# Monthly cross-check
# --------------------------------------------------------------------------- #
def monthly_asymmetry(px_m: pd.Series, lags: int = 3) -> dict:
    """|r_{m+1}| = a + b_up·r⁺_m + b_dn·|r⁻_m| + c·|r_m-ish trailing|, HAC(``lags``).

    On a monthly-average tape a one-month-ahead absolute return stands in for next-month
    volatility. Positive ``b_up - b_dn`` = the oil claim. Low power, direction only.
    """
    import statsmodels.api as sm  # noqa: F401

    r = np.log(px_m.astype(float)).diff().dropna()
    df = pd.DataFrame({"fwd": r.abs().shift(-1), "up": r.clip(lower=0), "dn": (-r).clip(lower=0),
                       "past": r.abs().rolling(3).mean()}).dropna()
    fit = _hac_ols(df["fwd"].to_numpy(), df[["up", "dn", "past"]].to_numpy(), lags=lags)
    b, cov = fit.params, fit.cov_params()
    diff = b[1] - b[2]
    se = float(np.sqrt(max(cov[1, 1] + cov[2, 2] - 2 * cov[1, 2], 0.0)))
    return {"b_up": float(b[1]), "b_dn": float(b[2]), "diff": float(diff), "diff_se": se,
            "diff_t": float(diff / se) if se > 0 else np.nan, "n": int(len(df))}


# --------------------------------------------------------------------------- #
# The verdict — thresholds fixed before the real tape was run
# --------------------------------------------------------------------------- #
T_BAR = 2.0


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    Inputs (all on the **inverse score** convention: positive = the oil claim):

    - ``wti_gjr_inv_t`` — full-sample GJR robust t on WTI;
    - ``wti_mf_t`` — full-sample model-free HAC t of ``b_up - b_dn`` at h = 21;
    - ``era_inv_t`` — list of per-era GJR robust t's; ``era_diff_z`` — z of the era difference;
      ``era_inv`` — list of per-era point estimates;
    - ``ov_wti_gain_net``, ``ov_wti_mdd_bh``, ``ov_wti_mdd_net`` — the WTI overlay.

    **Signal**

    - **Mixed** if the asymmetry *genuinely splits by era*: some era has inverse t ≥ 2 while
      another has inverse t ≤ -2, or some era has inverse t ≥ 2, the era difference has
      |z| ≥ 2, and another era's point estimate is equity-signed.
    - else **Real** if both the GJR t and the model-free t are ≥ 2 on the full WTI sample;
    - else **Weak** if either is ≥ 2, or some era alone clears 2;
    - else **None** (no inverse effect, or the equity sign).

    **Tradability** — never Investable: the tape is a spot price nobody can hold, and any
    real implementation adds a roll the spot ignores.

    - **Fragile** if the vol-target overlay on WTI beats buy-and-hold on net Sharpe *and* has
      the shallower drawdown — a risk tool that works on oil whatever the asymmetry;
    - else **Mirage**.
    """
    g, m = h["wti_gjr_inv_t"], h["wti_mf_t"]
    et = list(h["era_inv_t"])
    ep = list(h["era_inv"])
    z = h["era_diff_z"]
    split = ((max(et) >= T_BAR and min(et) <= -T_BAR)
             or (max(et) >= T_BAR and abs(z) >= T_BAR and min(ep) < 0))
    if split:
        signal = "Mixed"
    elif g >= T_BAR and m >= T_BAR:
        signal = "Real"
    elif g >= T_BAR or m >= T_BAR or max(et) >= T_BAR:
        signal = "Weak"
    else:
        signal = "None"
    ok = (h["ov_wti_gain_net"] > 0) and (h["ov_wti_mdd_net"] > h["ov_wti_mdd_bh"])
    trad = "Fragile" if ok else "Mirage"

    era_txt = "; ".join(f"{lab} t = {t:+.2f}" for lab, t in zip(h["era_labels"], et))
    if signal == "Mixed":
        sig_head = ("The mirror is real in one era and not in another, so it is a regime, "
                    "not a property of oil.")
    elif signal == "Real":
        sig_head = "On the full 1986-2018 WTI tape the mirror shows up in both lenses."
    elif signal == "Weak":
        sig_head = "Only one lens sees the mirror at the bar."
    else:
        sig_head = "Neither lens sees the mirror on this tape."
    signal_why = (
        f"{sig_head} Full-sample WTI spot: GJR inverse score t = **{g:+.2f}** "
        f"(EGARCH {h['wti_egarch_inv_t']:+.2f}), model-free forward-vol regression "
        f"t = **{m:+.2f}** at 21 days ({h['wti_mf5_t']:+.2f} at 5); the S&P 500 mirror reads "
        f"GJR t = {h['sp_gjr_inv_t']:+.2f}, the textbook equity sign. By era: {era_txt}; the "
        f"era difference in GJR gamma has z = **{z:+.2f}**. Monthly-average Brent cross-check "
        f"t = {h['brent_mf_t']:+.2f} (direction only). Positive = volatility rises after rallies.")
    if "roll_share_inverse" in h:
        signal_why += (
            f" A rolling three-year GJR (descriptive, overlapping windows) clears +2 in only "
            f"{h['roll_share_inverse']:.0%} of windows — none ending after "
            f"{str(h['roll_last_inverse'])[:4]} — against {h['roll_share_equity']:.0%} at -2 or "
            f"below.")
    trad_why = (
        f"Capped by the tape: WTI here is a **spot** price, not an investable futures return. "
        f"On that spot proxy, a 21-day vol-target overlay (one lag, "
        f"{h['ov_wti_cost_bps']:.0f} bp one-way) moves the excess Sharpe from "
        f"{h['ov_wti_sharpe_bh']:.2f} to {h['ov_wti_sharpe_net']:.2f} net "
        f"(gain {h['ov_wti_gain_net']:+.2f}, block-bootstrap 95% CI "
        f"[{h['ov_wti_gain_lo']:+.2f}, {h['ov_wti_gain_hi']:+.2f}]) and the drawdown from "
        f"{h['ov_wti_mdd_bh']:.0%} to {h['ov_wti_mdd_net']:.0%}; the same overlay on the S&P "
        f"price index gains {h['ov_sp_gain_net']:+.2f} "
        f"([{h['ov_sp_gain_lo']:+.2f}, {h['ov_sp_gain_hi']:+.2f}]). The thermostat's "
        f"correlation with the trailing month's return is {h['ov_wti_react']:+.2f} on oil vs "
        f"{h['ov_sp_react']:+.2f} on the S&P — what it actually de-risks after.")
    one = h.get("one_sentence_override") or _one_sentence(signal, trad, h)
    return {"signal": signal, "signal_why": signal_why, "trad": trad, "trad_why": trad_why,
            "one_sentence": one}


def _one_sentence(signal: str, trad: str, h: dict) -> str:
    if signal == "Mixed":
        et = list(h["era_inv_t"])
        hi_lab = h["era_labels"][int(np.argmax(et))]
        lo_lab = h["era_labels"][int(np.argmin(et))]
        s = (f"Oil's volatility rose with its rallies in {hi_lab} (GJR t = {max(et):+.1f}) "
             f"but not in {lo_lab} (t = {min(et):+.1f}) — the mirror is a regime, not a "
             f"law of oil")
    elif signal == "Real":
        s = (f"Oil really is equity in a mirror on 1986-2018 WTI spot (GJR t = "
             f"{h['wti_gjr_inv_t']:+.1f}, model-free t = {h['wti_mf_t']:+.1f})")
    elif signal == "Weak":
        s = (f"Oil leans towards the mirror (GJR t = {h['wti_gjr_inv_t']:+.1f}, model-free "
             f"t = {h['wti_mf_t']:+.1f}) but not in both lenses at the bar")
    elif h["wti_gjr_inv_t"] <= -T_BAR:
        s = (f"On 1986-2018 WTI spot oil is not equity in a mirror but a fainter copy — its "
             f"volatility also rises more after falls (GJR t = {h['wti_gjr_inv_t']:+.1f}), a "
             f"tilt that appeared after 2008 (era z = {h['era_diff_z']:+.1f}) while the mirror "
             f"never cleared the bar in any era")
    else:
        s = (f"On 1986-2018 WTI spot the mirror is not there (GJR t = "
             f"{h['wti_gjr_inv_t']:+.1f}, model-free t = {h['wti_mf_t']:+.1f})")
    if trad == "Fragile":
        s += (f", and a vol-target overlay still helps oil (net Sharpe gain "
              f"{h['ov_wti_gain_net']:+.2f}) on a spot tape you cannot hold.")
    else:
        s += (f"; and a vol-target overlay does not pay on oil (net Sharpe gain "
              f"{h['ov_wti_gain_net']:+.2f}) on a spot tape you cannot hold anyway.")
    return s
