"""Do correlations really go to one in a crisis? — Study 1018.

The claim has three layers, and this module measures each separately, because they are routinely
collapsed into one:

1. **The naive number.** Average pairwise correlation of the 20 stocks on crisis days versus calm
   days. Six regime definitions (:data:`REGIMES`), four of them *past-only* — the label for day t
   uses nothing later than t-1 — and two *contemporaneous* ones (the month's own realised
   volatility; the day's own big drop) because that is how the folklore is usually measured.

2. **The artefact.** Selecting a subsample by the size of its realisations raises a measured
   correlation even when the true one never moves (Boyer, Gibson & Loretan 1999), and a rise in
   the market's variance raises every stock's correlation with it at a constant beta (Forbes &
   Rigobon 2002). Two benchmarks price this:

   * **Constant-correlation GARCH** (:func:`fit_ccc_garch`, :func:`simulate_ccc_garch`) — the
     tape's own volatility dynamics, asset by asset, driven by Gaussian shocks with a *constant*
     correlation matrix (Bollerslev 1990). The identical regime procedure is run on simulated
     paths; whatever "crisis − calm" it reports is pure artefact. ``genuine = real − artefact``,
     with a circular block-bootstrap interval.
   * **Constant beta / Forbes–Rigobon** (:func:`fr_adjust`, :func:`constant_beta_benchmark`) —
     the calm-period betas and idiosyncratic variances held fixed, only the market's variance
     allowed to change. This answers a *different* question — did the transmission change, or
     did the common factor just get louder? — and its answer must not be confused with "the
     correlation increase is fake": a louder common factor is exactly what hurts a holder.

3. **The holder's arithmetic** (:func:`portfolio_stats`). An equal-weight 20-stock book: realised
   crisis volatility against the volatility predicted by the *calm* correlation matrix with each
   stock's volatility scaled up to its crisis level — i.e. "vol rose, correlations didn't".
   The ratio is the part of the pain that is due to correlation, and the diversification ratio
   (Choueifaty & Coignard 2008) says how much of the free lunch is left.

Exceedance correlations (Longin & Solnik 2001; Ang & Chen 2002) close the loop on asymmetry:
correlation conditional on *both* returns being beyond a threshold, downside versus upside,
against the curve a bivariate normal with the same unconditional correlation would draw — which
is *not* flat: it decays toward zero in both tails, so a rising empirical curve is never compared
to a constant.

Nothing in this study is a trading rule; no execution lag or cost applies. The verdict
(:func:`verdict`) is fixed in code and unit-tested in both directions before the real run.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MIN_OBS = 40

# --------------------------------------------------------------------------- #
# Regimes — the market state variable
# --------------------------------------------------------------------------- #
REGIMES = {
    "past21": {"label": "Past 21-day S&P vol: top decile vs bottom half", "past_only": True},
    "past63": {"label": "Past 63-day S&P vol: top decile vs bottom half", "past_only": True},
    "garch": {"label": "GARCH(1,1) one-step vol forecast: top decile vs bottom half",
              "past_only": True},
    "drawdown": {"label": "S&P drawdown at t-1 worse than -20% vs better than -5%",
                 "past_only": True},
    "month_vol": {"label": "Month's own realised vol: top-decile months vs bottom half",
                  "past_only": False},
    "big_down": {"label": "Day's own S&P return in bottom 5% vs abs. return below median",
                 "past_only": False},
}
HEADLINE = "past21"
FAST_REGIMES = ("past21", "past63", "drawdown", "month_vol", "big_down")

CRISIS_Q = 0.90
CALM_Q = 0.50

EPISODES = (
    ("1998 LTCM / Russia", "1998-08-01", "1998-10-15"),
    ("2001 September 11", "2001-09-17", "2001-10-31"),
    ("2002 bear-market low", "2002-07-01", "2002-10-31"),
    ("2008-09 Lehman / GFC", "2008-09-15", "2009-03-31"),
    ("2010 flash crash / euro", "2010-05-01", "2010-07-31"),
    ("2011 US downgrade", "2011-08-01", "2011-10-31"),
    ("2015 China devaluation", "2015-08-17", "2015-09-30"),
    ("2018 Q4 sell-off", "2018-10-01", "2018-12-31"),
    ("2020 COVID crash", "2020-02-20", "2020-04-30"),
    ("2022 rate shock", "2022-01-03", "2022-10-31"),
)


def _quantile_labels(x: pd.Series, crisis_q: float = CRISIS_Q,
                     calm_q: float = CALM_Q) -> pd.Series:
    """1 where ``x`` is in its top ``1-crisis_q``, 0 where in its bottom ``calm_q``, else NaN."""
    ok = x.dropna()
    hi, lo = ok.quantile(crisis_q), ok.quantile(calm_q)
    lab = pd.Series(np.nan, index=x.index)
    lab[x >= hi] = 1.0
    lab[x <= lo] = 0.0
    return lab


def garch_vol(r: pd.Series) -> pd.Series:
    """GARCH(1,1) conditional volatility — for day t it uses returns up to t-1 only.

    Parameters are estimated on the full sample (a mild look-ahead in the *parameters*, not in
    the filtered volatility); stated wherever the regime is reported.
    """
    from arch import arch_model
    x = 100.0 * r.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(x, mean="Constant", vol="GARCH", p=1, q=1,
                         dist="normal").fit(disp="off")
    return (res.conditional_volatility / 100.0).reindex(r.index)


def regime_labels(market: pd.Series, kind: str) -> pd.Series:
    """Crisis (1) / calm (0) / neither (NaN) labels for each day, from the market return alone.

    Past-only kinds use information through t-1 for the label of day t. The two
    contemporaneous kinds deliberately do not — they are the folklore's own measurement, and the
    artefact benchmark shows what that costs.
    """
    m = market.astype(float)
    if kind == "past21":
        return _quantile_labels(m.rolling(21).std().shift(1))
    if kind == "past63":
        return _quantile_labels(m.rolling(63).std().shift(1))
    if kind == "garch":
        return _quantile_labels(garch_vol(m))
    if kind == "drawdown":
        lvl = (1.0 + m.fillna(0.0)).cumprod()
        dd = (lvl / lvl.cummax() - 1.0).shift(1)
        lab = pd.Series(np.nan, index=m.index)
        lab[dd <= -0.20] = 1.0
        lab[dd > -0.05] = 0.0
        return lab
    if kind == "month_vol":
        per = m.index.to_period("M")
        g = m.groupby(per)
        months = g.std().where(g.count() >= 10)
        mv = pd.Series(months.reindex(per).to_numpy(), index=m.index)
        months = months.dropna()
        hi, lo = months.quantile(CRISIS_Q), months.quantile(CALM_Q)
        lab = pd.Series(np.nan, index=m.index)
        lab[mv >= hi] = 1.0
        lab[mv <= lo] = 0.0
        return lab
    if kind == "big_down":
        lab = pd.Series(np.nan, index=m.index)
        lab[m <= m.quantile(0.05)] = 1.0
        lab[m.abs() <= m.abs().median()] = 0.0
        return lab
    raise KeyError(f"unknown regime {kind!r}; one of {sorted(REGIMES)}")


def all_labels(market: pd.Series, kinds=tuple(REGIMES)) -> dict:
    return {k: regime_labels(market, k) for k in kinds}


# --------------------------------------------------------------------------- #
# Correlation measurement
# --------------------------------------------------------------------------- #
def corr_matrix(X: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix of the columns of ``X`` (rows = days)."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    sd = np.sqrt((Xc ** 2).sum(axis=0))
    sd[sd == 0] = np.nan
    return (Xc.T @ Xc) / np.outer(sd, sd)


def avg_offdiag(C: np.ndarray) -> float:
    k = C.shape[0]
    iu = np.triu_indices(k, 1)
    return float(np.nanmean(C[iu]))


def avg_pairwise_corr(R) -> float:
    """Mean off-diagonal Pearson correlation of a returns panel (NaN if too few days)."""
    X = np.asarray(R, dtype=float)
    if X.shape[0] < MIN_OBS:
        return np.nan
    return avg_offdiag(corr_matrix(X))


def regime_corr(R: pd.DataFrame, labels: pd.Series) -> dict:
    """Average pairwise correlation on crisis days and on calm days, and the gap."""
    lab = labels.reindex(R.index).to_numpy()
    X = R.to_numpy(dtype=float)
    hi, lo = X[lab == 1.0], X[lab == 0.0]
    c_hi, c_lo = avg_pairwise_corr(hi), avg_pairwise_corr(lo)
    return {"calm": c_lo, "crisis": c_hi, "diff": c_hi - c_lo,
            "n_calm": int(len(lo)), "n_crisis": int(len(hi))}


def rolling_avg_corr(R: pd.DataFrame, window: int = 63, step: int = 5) -> pd.Series:
    """Average pairwise correlation over trailing ``window`` days, sampled every ``step``."""
    X = R.to_numpy(dtype=float)
    out, idx = [], []
    for end in range(window, len(X) + 1, step):
        out.append(avg_pairwise_corr(X[end - window:end]))
        idx.append(R.index[end - 1])
    return pd.Series(out, index=pd.DatetimeIndex(idx), name=f"avg_corr_{window}d")


# --------------------------------------------------------------------------- #
# The artefact, in closed form (Boyer-Gibson-Loretan / Forbes-Rigobon)
# --------------------------------------------------------------------------- #
def bgl_conditional_corr(rho: float, var_ratio: float) -> float:
    """Correlation measured in a subsample where Var(x) is ``var_ratio`` times its full value.

    For ``y = beta * x + e`` with ``e`` independent of ``x`` and the subsample chosen on ``x``
    alone, the true linear relationship never changes, yet the measured correlation becomes
    ``rho * sqrt(k) / sqrt(1 + rho^2 (k - 1))`` with ``k = var_ratio`` (Boyer, Gibson & Loretan
    1999). Picking high-variance days (k > 1) raises it; picking quiet days lowers it.
    """
    k = float(var_ratio)
    return float(rho * np.sqrt(k) / np.sqrt(1.0 + rho ** 2 * (k - 1.0)))


def fr_adjust(rho, delta):
    """Forbes & Rigobon (2002) heteroskedasticity-adjusted correlation.

    ``rho* = rho / sqrt(1 + delta * (1 - rho^2))`` where ``delta = Var_crisis(x)/Var_calm(x) - 1``
    for the conditioning (source) variable. The exact inverse of :func:`bgl_conditional_corr`:
    it returns the correlation the crisis subsample *would* show if the market had stayed at its
    calm variance and nothing else had changed.
    """
    rho = np.asarray(rho, dtype=float)
    return rho / np.sqrt(1.0 + delta * (1.0 - rho ** 2))


def fr_panel(R: pd.DataFrame, market: pd.Series, labels: pd.Series) -> dict:
    """FR adjustment on the crisis subsample, pair by pair and stock-by-market.

    ``delta`` comes from the market's own variance ratio. For stock-vs-market correlations the
    formula is exact under a constant-beta model; for stock-vs-stock pairs it is the standard
    empirical-contagion usage (the market as the common source). Both are reported.
    """
    lab = labels.reindex(R.index).to_numpy()
    m = market.reindex(R.index).to_numpy(dtype=float)
    X = R.to_numpy(dtype=float)
    hi, lo = lab == 1.0, lab == 0.0
    delta = float(np.var(m[hi], ddof=1) / np.var(m[lo], ddof=1) - 1.0)
    C_hi = corr_matrix(X[hi])
    iu = np.triu_indices(X.shape[1], 1)
    pairs_adj = fr_adjust(C_hi[iu], delta)
    XM_hi = np.column_stack([X[hi], m[hi]])
    XM_lo = np.column_stack([X[lo], m[lo]])
    cm_hi = corr_matrix(XM_hi)[-1, :-1]
    cm_lo = corr_matrix(XM_lo)[-1, :-1]
    return {"delta": delta, "var_ratio": delta + 1.0,
            "crisis_raw": float(np.nanmean(C_hi[iu])),
            "crisis_fr": float(np.nanmean(pairs_adj)),
            "calm_raw": avg_pairwise_corr(X[lo]),
            "mkt_corr_calm": float(np.mean(cm_lo)), "mkt_corr_crisis": float(np.mean(cm_hi)),
            "mkt_corr_crisis_fr": float(np.mean(fr_adjust(cm_hi, delta)))}


def constant_beta_benchmark(R: pd.DataFrame, market: pd.Series, labels: pd.Series) -> dict:
    """What a one-factor model with CALM betas and CALM idiosyncratic variance implies in crisis.

    Only the market's variance is moved to its crisis value. If the real crisis correlation
    matches this, the whole rise is "the common factor got louder" (Forbes–Rigobon
    interdependence); above it, betas rose or idiosyncratic risk shrank (contagion in their
    sense); below it, idiosyncratic risk rose too, which is the usual finding for single stocks.
    """
    lab = labels.reindex(R.index).to_numpy()
    m = market.reindex(R.index).to_numpy(dtype=float)
    X = R.to_numpy(dtype=float)
    lo, hi = lab == 0.0, lab == 1.0
    mc = m[lo] - m[lo].mean()
    Xc = X[lo] - X[lo].mean(axis=0)
    v_lo = float(mc @ mc / (lo.sum() - 1))
    beta = (Xc.T @ mc) / (mc @ mc)
    resid = Xc - np.outer(mc, beta)
    s2 = resid.var(axis=0, ddof=2)
    v_hi = float(np.var(m[hi], ddof=1))
    iu = np.triu_indices(X.shape[1], 1)

    def implied(v):
        cov = np.outer(beta, beta) * v
        sd = np.sqrt(beta ** 2 * v + s2)
        return float(np.mean((cov / np.outer(sd, sd))[iu]))

    return {"implied_calm": implied(v_lo), "implied_crisis": implied(v_hi),
            "implied_diff": implied(v_hi) - implied(v_lo), "var_ratio": v_hi / v_lo,
            "beta_mean": float(beta.mean()),
            "idio_vol_calm": float(np.sqrt(s2.mean() * TRADING_DAYS))}


def idio_vol_ratio(R: pd.DataFrame, market: pd.Series, labels: pd.Series) -> dict:
    """How much the market's variance and the stocks' residual variance each rose in crisis.

    Correlation rises iff the common part grows faster than the stock-specific part. Betas are
    re-estimated within each regime so the residual is the regime's own.
    """
    lab = labels.reindex(R.index).to_numpy()
    m = market.reindex(R.index).to_numpy(dtype=float)
    X = R.to_numpy(dtype=float)
    out = {}
    for name, msk in (("calm", lab == 0.0), ("crisis", lab == 1.0)):
        mc = m[msk] - m[msk].mean()
        Xc = X[msk] - X[msk].mean(axis=0)
        beta = (Xc.T @ mc) / (mc @ mc)
        resid = Xc - np.outer(mc, beta)
        out[name] = {"mkt_vol": float(mc.std(ddof=1) * np.sqrt(TRADING_DAYS)),
                     "idio_vol": float(np.sqrt(resid.var(axis=0, ddof=2).mean()
                                               * TRADING_DAYS)),
                     "beta": float(beta.mean())}
    out["mkt_vol_ratio"] = out["crisis"]["mkt_vol"] / out["calm"]["mkt_vol"]
    out["idio_vol_ratio"] = out["crisis"]["idio_vol"] / out["calm"]["idio_vol"]
    out["beta_ratio"] = out["crisis"]["beta"] / out["calm"]["beta"]
    return out


# --------------------------------------------------------------------------- #
# Constant-correlation GARCH — the exact simulation benchmark
# --------------------------------------------------------------------------- #
def _nearest_corr(C: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    C = (C + C.T) / 2.0
    w, V = np.linalg.eigh(C)
    w = np.clip(w, floor, None)
    C2 = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(C2))
    C2 = C2 / np.outer(d, d)
    np.fill_diagonal(C2, 1.0)
    return C2


def fit_ccc_garch(R: pd.DataFrame, market: pd.Series) -> dict:
    """Fit a GARCH(1,1) to each stock and to the market; correlate the standardised residuals.

    The resulting model has every series' own volatility clustering, persistence and spikes,
    and one *constant* correlation matrix linking their shocks (Bollerslev 1990). By
    construction its conditional correlation never moves, which is what makes it the null for
    "correlation rises in crises".
    """
    from arch import arch_model
    X = pd.concat([R, market.rename("__MKT__")], axis=1, sort=False).dropna()
    cols = list(X.columns)
    params, Z = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in cols:
            res = arch_model(100.0 * X[c], mean="Constant", vol="GARCH", p=1, q=1,
                             dist="normal").fit(disp="off")
            p = res.params
            a, b = float(p["alpha[1]"]), float(p["beta[1]"])
            om = float(p["omega"])
            uv = om / (1.0 - a - b) if a + b < 0.999 else float((100 * X[c]).var())
            params.append({"mu": float(p["mu"]), "omega": om, "alpha": a, "beta": b,
                           "uncond_var": uv})
            Z.append(np.asarray(res.std_resid))
    Z = np.column_stack(Z)
    Z = Z[np.isfinite(Z).all(axis=1)]
    Rz = _nearest_corr(np.corrcoef(Z.T))
    return {"cols": cols[:-1], "params": params, "corr": Rz, "index": X.index,
            "avg_corr_stocks": avg_offdiag(Rz[:-1, :-1])}


def simulate_ccc_garch(fit: dict, n_days: int | None = None, seed: int = 1018,
                       burn: int = 250) -> tuple[pd.DataFrame, pd.Series]:
    """One path from the fitted constant-correlation GARCH: ``(stocks, market)`` returns."""
    rng = np.random.default_rng(seed)
    idx = fit["index"] if n_days is None else pd.bdate_range("2000-01-03", periods=n_days)
    n = len(idx)
    k = len(fit["params"])
    L = np.linalg.cholesky(fit["corr"])
    Z = rng.standard_normal((n + burn, k)) @ L.T
    mu = np.array([p["mu"] for p in fit["params"]])
    om = np.array([p["omega"] for p in fit["params"]])
    al = np.array([p["alpha"] for p in fit["params"]])
    be = np.array([p["beta"] for p in fit["params"]])
    s2 = np.array([p["uncond_var"] for p in fit["params"]])
    eps_prev = np.sqrt(s2) * Z[0]
    out = np.empty((n + burn, k))
    out[0] = eps_prev
    for t in range(1, n + burn):
        s2 = om + al * eps_prev ** 2 + be * s2
        eps_prev = np.sqrt(s2) * Z[t]
        out[t] = eps_prev
    out = (out[burn:] + mu) / 100.0
    stocks = pd.DataFrame(out[:, :-1], index=idx, columns=fit["cols"])
    mkt = pd.Series(out[:, -1], index=idx, name="MKT")
    return stocks, mkt


def artefact_benchmark(fit: dict, kinds=tuple(REGIMES), n_sims: int = 40,
                       seed: int = 1018, exceed: bool = True) -> dict:
    """Run the identical regime procedure on ``n_sims`` constant-correlation paths.

    Returns per-regime means and standard deviations of calm, crisis and crisis−calm average
    correlation, of the portfolio shortfall, and (optionally) of the exceedance asymmetry —
    i.e. everything the procedure reports in a world where nothing genuine can be found.
    """
    rows = {k: {"calm": [], "crisis": [], "diff": [], "shortfall": [], "dr_gap": []}
            for k in kinds}
    asym, curves = [], []
    for s in range(n_sims):
        Rs, ms = simulate_ccc_garch(fit, seed=seed + s)
        for k in kinds:
            lab = regime_labels(ms, k)
            rc = regime_corr(Rs, lab)
            ps = portfolio_stats(Rs, lab)
            rows[k]["calm"].append(rc["calm"])
            rows[k]["crisis"].append(rc["crisis"])
            rows[k]["diff"].append(rc["diff"])
            rows[k]["shortfall"].append(ps["shortfall"])
            rows[k]["dr_gap"].append(ps["dr_calm"] - ps["dr_crisis"])
        if exceed:
            cur = exceedance_panel(Rs, ms)
            curves.append(cur)
            asym.append(asymmetry(cur))
    out = {}
    for k, d in rows.items():
        out[k] = {f"{m}_mean": float(np.nanmean(v)) for m, v in d.items()}
        out[k].update({f"{m}_sd": float(np.nanstd(v, ddof=1)) for m, v in d.items()})
    res = {"regimes": out, "n_sims": n_sims}
    if exceed:
        res["asym_mean"] = float(np.mean(asym))
        res["asym_sd"] = float(np.std(asym, ddof=1))
        res["curve"] = pd.concat(curves, axis=1).mean(axis=1)
    return res


# --------------------------------------------------------------------------- #
# The diversified holder
# --------------------------------------------------------------------------- #
def portfolio_stats(R, labels, w=None) -> dict:
    """Equal-weight (daily-rebalanced) book: crisis vol, calm vol, and crisis vol predicted from
    the CALM correlation matrix with each stock's vol scaled to its crisis level.

    ``shortfall = realised / predicted - 1`` is the part of crisis risk owed to correlation
    rather than to volatility. ``dr`` is the diversification ratio ``w'sigma / sigma_p``: 1 means
    no diversification at all, ``sqrt(N)`` is the ceiling for N uncorrelated equal-vol assets.
    """
    X = np.asarray(R, dtype=float)
    lab = np.asarray(labels.reindex(R.index) if isinstance(labels, pd.Series) else labels,
                     dtype=float)
    k = X.shape[1]
    w = np.full(k, 1.0 / k) if w is None else np.asarray(w, dtype=float)
    hi, lo = X[lab == 1.0], X[lab == 0.0]
    if len(hi) < MIN_OBS or len(lo) < MIN_OBS:
        return {k_: np.nan for k_ in ("vol_calm", "vol_crisis", "vol_pred", "shortfall",
                                      "dr_calm", "dr_crisis", "stock_vol_ratio",
                                      "port_vol_ratio")}
    a = np.sqrt(TRADING_DAYS)
    s_hi, s_lo = hi.std(axis=0, ddof=1), lo.std(axis=0, ddof=1)
    vp_hi, vp_lo = float((hi @ w).std(ddof=1)), float((lo @ w).std(ddof=1))
    C_lo = corr_matrix(lo)
    pred = float(np.sqrt((w * s_hi) @ C_lo @ (w * s_hi)))
    return {"vol_calm": vp_lo * a, "vol_crisis": vp_hi * a, "vol_pred": pred * a,
            "shortfall": vp_hi / pred - 1.0,
            "dr_calm": float(w @ s_lo) / vp_lo, "dr_crisis": float(w @ s_hi) / vp_hi,
            "stock_vol_ratio": float(np.mean(s_hi) / np.mean(s_lo)),
            "port_vol_ratio": vp_hi / vp_lo}


def episode_table(R: pd.DataFrame, episodes=EPISODES, ref_days: int = 252,
                  gap: int = 21) -> pd.DataFrame:
    """Named crisis windows (chosen with hindsight — descriptive only).

    The reference is the ``ref_days`` window ending ``gap`` trading days before the episode,
    i.e. what a holder could have estimated beforehand. ``ratio`` is realised episode vol over
    the vol predicted from the pre-episode correlation matrix with the episode's own stock
    volatilities plugged in.
    """
    rows = []
    X = R.to_numpy(dtype=float)
    k = X.shape[1]
    w = np.full(k, 1.0 / k)
    for name, a, b in episodes:
        msk = (R.index >= pd.Timestamp(a)) & (R.index <= pd.Timestamp(b))
        if msk.sum() < 15:
            continue
        i0 = int(np.argmax(msk))
        if i0 - gap - ref_days < 0:
            continue
        ref = X[i0 - gap - ref_days:i0 - gap]
        win = X[msk]
        s_w, s_r = win.std(axis=0, ddof=1), ref.std(axis=0, ddof=1)
        C_r = corr_matrix(ref)
        vp = float((win @ w).std(ddof=1))
        pred = float(np.sqrt((w * s_w) @ C_r @ (w * s_w)))
        rows.append({"episode": name, "start": a, "end": b, "days": int(msk.sum()),
                     "corr_before": avg_offdiag(C_r), "corr_during": avg_offdiag(corr_matrix(win)),
                     "vol_during": vp * np.sqrt(TRADING_DAYS),
                     "vol_pred": pred * np.sqrt(TRADING_DAYS), "ratio": vp / pred,
                     "dr_before": float(w @ s_r) / float((ref @ w).std(ddof=1)),
                     "dr_during": float(w @ s_w) / vp})
    return pd.DataFrame(rows).set_index("episode") if rows else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Exceedance correlations (Longin-Solnik 2001; Ang-Chen 2002)
# --------------------------------------------------------------------------- #
THETAS = (-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
ASYM_THETAS = (0.5, 1.0, 1.5)
BOOT_THETAS = (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)   # all the bootstrap needs for asymmetry()


def _corr2(x, y) -> float:
    if len(x) < 20:
        return np.nan
    xc, yc = x - x.mean(), y - y.mean()
    d = np.sqrt((xc @ xc) * (yc @ yc))
    return float(xc @ yc / d) if d > 0 else np.nan


def exceedance_corr(x, y, thetas=THETAS) -> pd.Series:
    """Correlation of ``x`` and ``y`` on days when BOTH are beyond ``theta`` standard deviations.

    Below ``theta`` for negative thresholds, above it for positive ones. Each series is
    standardised by its own sample mean and standard deviation. Fewer than 20 joint
    exceedances → NaN rather than a number built on a handful of days.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    zx = (x - x.mean()) / x.std(ddof=1)
    zy = (y - y.mean()) / y.std(ddof=1)
    out = {}
    for th in thetas:
        m = (zx < th) & (zy < th) if th < 0 else (zx > th) & (zy > th)
        out[th] = _corr2(zx[m], zy[m])
    return pd.Series(out, name="exceedance")


def exceedance_matrix(X, y, thetas=THETAS) -> np.ndarray:
    """Vectorised :func:`exceedance_corr` of every column of ``X`` against ``y``.

    Returns a ``len(thetas) x k`` array; NaN where fewer than 20 joint exceedances exist.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ZX = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    zy = ((y - y.mean()) / y.std(ddof=1))[:, None]
    out = np.full((len(thetas), X.shape[1]), np.nan)
    for i, th in enumerate(thetas):
        M = ((ZX < th) & (zy < th)) if th < 0 else ((ZX > th) & (zy > th))
        n = M.sum(axis=0).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            mx = (ZX * M).sum(axis=0) / n
            my = (zy * M).sum(axis=0) / n
            sxy = (ZX * zy * M).sum(axis=0) / n - mx * my
            sxx = (ZX ** 2 * M).sum(axis=0) / n - mx ** 2
            syy = (zy ** 2 * M).sum(axis=0) / n - my ** 2
            r = sxy / np.sqrt(sxx * syy)
        r[n < 20] = np.nan
        out[i] = r
    return out


def exceedance_panel(R, market, thetas=THETAS) -> pd.Series:
    """Average stock-vs-market exceedance curve across the panel."""
    if isinstance(R, pd.DataFrame):
        m = market.reindex(R.index).to_numpy(dtype=float)
        X = R.to_numpy(dtype=float)
    else:
        X, m = np.asarray(R, dtype=float), np.asarray(market, dtype=float)
    E = exceedance_matrix(X, m, thetas)
    return pd.Series(np.nanmean(E, axis=1), index=list(thetas), name="exceedance")


def gaussian_exceedance(rho: float, thetas=THETAS, n: int = 400_000,
                        seed: int = 1018) -> pd.Series:
    """The exceedance curve of a bivariate normal with correlation ``rho`` (simulated, exact
    to Monte-Carlo error). Symmetric, and *falling* toward zero in both tails."""
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(n)
    return exceedance_corr(z1, z2, thetas)


def gaussian_panel_curve(R: pd.DataFrame, market: pd.Series, thetas=THETAS) -> pd.Series:
    """The Gaussian benchmark curve for the panel: each stock's own full-sample correlation
    with the market, pushed through :func:`gaussian_exceedance`, then averaged."""
    m = market.reindex(R.index)
    rhos = [float(np.corrcoef(R[c], m)[0, 1]) for c in R.columns]
    curves = [gaussian_exceedance(r, thetas, n=200_000, seed=1018 + i)
              for i, r in enumerate(rhos)]
    return pd.concat(curves, axis=1).mean(axis=1)


def asymmetry(curve: pd.Series, thetas=ASYM_THETAS) -> float:
    """Mean over ``thetas`` of downside minus upside exceedance correlation (0 for any
    elliptical distribution, by symmetry)."""
    return float(np.nanmean([curve.get(-t, np.nan) - curve.get(t, np.nan) for t in thetas]))


def ang_chen_h(curve: pd.Series, gauss: pd.Series, side: str = "down") -> float:
    """Ang & Chen (2002) H: root-mean-square distance between the empirical and the Gaussian
    exceedance curve on one side."""
    th = [t for t in curve.index if (t < 0 if side == "down" else t > 0)]
    d = (curve[th] - gauss.reindex(th)).dropna()
    return float(np.sqrt((d ** 2).mean())) if len(d) else np.nan


def devolatilise(R: pd.DataFrame, market: pd.Series, fit: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Divide every series by its own fitted GARCH volatility (the CCC fit's filter)."""
    X = pd.concat([R, market.rename("__MKT__")], axis=1, sort=False).dropna()
    Z = {}
    for c, p in zip(list(X.columns), fit["params"]):
        x = 100.0 * X[c].to_numpy() - p["mu"]
        s2 = np.empty(len(x))
        s2[0] = p["uncond_var"]
        for t in range(1, len(x)):
            s2[t] = p["omega"] + p["alpha"] * x[t - 1] ** 2 + p["beta"] * s2[t - 1]
        Z[c] = x / np.sqrt(s2)
    Zdf = pd.DataFrame(Z, index=X.index)
    return Zdf[list(R.columns)], Zdf["__MKT__"]


# --------------------------------------------------------------------------- #
# Circular block bootstrap
# --------------------------------------------------------------------------- #
def circular_block_indices(n: int, block: int, rng) -> np.ndarray:
    """One circular block-bootstrap resample of ``range(n)`` (Politis & Romano 1992)."""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=nb)
    return ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]


def bootstrap(R: pd.DataFrame, market: pd.Series, labels: dict, n_boot: int = 400,
              block: int = 63, seed: int = 1018, exceed: bool = True,
              Z: tuple | None = None) -> dict:
    """Resample whole days in circular blocks, carrying each day's regime label with it.

    Labels are attached to the rows (they were computed from the original tape), so a
    resample re-weights which crisis and calm days are seen but never relabels a day — the
    variability measured is the sampling variability of the conditional correlations, which is
    the question. 63-day blocks keep the volatility clustering intact.
    """
    rng = np.random.default_rng(seed)
    X = R.to_numpy(dtype=float)
    m = market.reindex(R.index).to_numpy(dtype=float)
    L = {k: v.reindex(R.index).to_numpy(dtype=float) for k, v in labels.items()}
    if Z is not None:
        ZX = Z[0].reindex(R.index).to_numpy(dtype=float)
        Zm = Z[1].reindex(R.index).to_numpy(dtype=float)
    n = len(X)
    w_eq = np.full(X.shape[1], 1.0 / X.shape[1])
    out = {k: {"diff": [], "shortfall": [], "dr_gap": [], "dr_crisis": []} for k in L}
    out["_asym"], out["_asym_devol"] = [], []
    for _ in range(n_boot):
        ii = circular_block_indices(n, block, rng)
        Xb = X[ii]
        for k, lab in L.items():
            lb = lab[ii]
            hi, lo = Xb[lb == 1.0], Xb[lb == 0.0]
            if len(hi) < MIN_OBS or len(lo) < MIN_OBS:
                for m_ in out[k]:
                    out[k][m_].append(np.nan)
                continue
            # one covariance per subsample serves both the correlation gap and the book
            C_h, C_l = np.cov(hi, rowvar=False), np.cov(lo, rowvar=False)
            s_h, s_l = np.sqrt(np.diag(C_h)), np.sqrt(np.diag(C_l))
            R_h, R_l = C_h / np.outer(s_h, s_h), C_l / np.outer(s_l, s_l)
            out[k]["diff"].append(avg_offdiag(R_h) - avg_offdiag(R_l))
            vp_h, vp_l = np.sqrt(w_eq @ C_h @ w_eq), np.sqrt(w_eq @ C_l @ w_eq)
            pred = np.sqrt((w_eq * s_h) @ R_l @ (w_eq * s_h))
            out[k]["shortfall"].append(vp_h / pred - 1.0)
            out[k]["dr_gap"].append((w_eq @ s_l) / vp_l - (w_eq @ s_h) / vp_h)
            out[k]["dr_crisis"].append((w_eq @ s_h) / vp_h)
        if exceed:
            mb = m[ii]
            out["_asym"].append(asymmetry(exceedance_panel(Xb, mb, BOOT_THETAS)))
            if Z is not None:
                zb, zmb = ZX[ii], Zm[ii]
                ok = np.isfinite(zmb)
                out["_asym_devol"].append(
                    asymmetry(exceedance_panel(zb[ok], zmb[ok], BOOT_THETAS)))
    res = {}
    for k, d in out.items():
        if isinstance(d, dict):
            res[k] = {m_: np.asarray(v, dtype=float) for m_, v in d.items()}
        else:
            res[k] = np.asarray(d, dtype=float)
    return res


def ci(a, centre_shift: float = 0.0, level: float = 0.95) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)] - centre_shift
    q = (1.0 - level) / 2.0
    return float(np.quantile(a, q)), float(np.quantile(a, 1.0 - q))


def boot_p_positive(a, centre_shift: float = 0.0) -> float:
    """One-sided bootstrap p-value for "the quantity is > 0": share of replicates ≤ 0."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)] - centre_shift
    return float(max((a <= 0).mean(), 1.0 / (len(a) + 1)))


# --------------------------------------------------------------------------- #
# The detector — what tests run on the synthetic world
# --------------------------------------------------------------------------- #
def crisis_correlation_test(R: pd.DataFrame, market: pd.Series, regime: str = HEADLINE,
                            n_sims: int = 12, n_boot: int = 150, block: int = 63,
                            seed: int = 1018) -> dict:
    """Real crisis−calm gap, the constant-correlation artefact, the genuine part and its CI."""
    lab = regime_labels(market, regime)
    rc = regime_corr(R, lab)
    fit = fit_ccc_garch(R, market)
    art = artefact_benchmark(fit, kinds=(regime,), n_sims=n_sims, seed=seed, exceed=False)
    a = art["regimes"][regime]["diff_mean"]
    bt = bootstrap(R, market, {regime: lab}, n_boot=n_boot, block=block, seed=seed,
                   exceed=False)
    lo, hi = ci(bt[regime]["diff"], centre_shift=a)
    return {"regime": regime, "calm": rc["calm"], "crisis": rc["crisis"], "diff": rc["diff"],
            "artefact": a, "genuine": rc["diff"] - a, "ci_lo": lo, "ci_hi": hi,
            "p": boot_p_positive(bt[regime]["diff"], centre_shift=a),
            "significant": bool(lo > 0)}


# --------------------------------------------------------------------------- #
# Verdict — thresholds fixed before the real run
# --------------------------------------------------------------------------- #
def verdict(h: dict) -> dict:
    """The two stamps by a pre-registered rule.

    **Signal** — is there a *genuine* crisis rise in correlation, i.e. beyond what the identical
    procedure reports under constant correlation?

    - ``Real``  : headline (past-only 21-day vol) genuine gap has a block-bootstrap 95% CI
      entirely above zero **and** the genuine gap is positive in at least half of the regime
      definitions.
    - ``Mixed`` : the headline CI is above zero but fewer than half the definitions agree.
    - ``Weak``  : positive point estimate, CI touching zero.
    - ``None``  : no positive genuine gap.

    **Tradability** — does the genuine part hurt a diversified holder? Measured as the
    equal-weight book's crisis volatility over the volatility predicted from calm correlations
    with crisis volatilities plugged in (net of the same-procedure artefact).

    - ``Fragile`` : that shortfall is ≥ 5% with a bootstrap CI above zero — the holder really is
      hurt beyond the volatility increase, but there is no free trade in knowing it.
    - ``Mirage``  : otherwise — "diversification fails" adds nothing to "volatility rose".
    - ``Investable`` is unreachable by design: nothing here is a trading rule, and a risk-model
      correction is not a return stream that survives costs and capacity.
    """
    if h["genuine_lo"] > 0:
        signal = "Real" if h["share_regimes_genuine_pos"] >= 0.5 else "Mixed"
    elif h["genuine"] > 0:
        signal = "Weak"
    else:
        signal = "None"
    trad = "Fragile" if (h["shortfall_lo"] > 0 and h["shortfall"] >= 0.05) else "Mirage"

    if h["fr_crisis"] <= h["naive_calm"]:
        fr_txt = (f"The Forbes–Rigobon adjustment takes the crisis figure to "
                  f"{h['fr_crisis']:.2f}, *below* the calm {h['naive_calm']:.2f}: the whole rise "
                  f"is accounted for by a louder common factor (market variance "
                  f"×{h['var_ratio']:.1f}, stock-specific volatility only "
                  f"×{h['idio_ratio']:.2f}). In their vocabulary that is interdependence, not "
                  f"contagion — and it is still exactly what a diversified holder feels.")
    else:
        fr_txt = (f"The Forbes–Rigobon adjustment takes the crisis figure to "
                  f"{h['fr_crisis']:.2f} against a calm {h['naive_calm']:.2f}: part of the rise "
                  f"survives even after the market's ×{h['var_ratio']:.1f} variance increase is "
                  f"taken out — contagion in their sense, on top of a louder common factor.")
    sig_core = (
        f"Measured the naive way, the past-only crisis regime lifts average pairwise "
        f"correlation from {h['naive_calm']:.2f} to {h['naive_crisis']:.2f}, and the identical "
        f"procedure run on {h['n_sims']} constant-correlation GARCH paths reports "
        f"{h['artefact']:+.3f} — so the genuine rise is **{h['genuine']:+.3f}** (circular "
        f"block-bootstrap 95% CI [{h['genuine_lo']:+.3f}, {h['genuine_hi']:+.3f}], one-sided "
        f"p {h['genuine_p']:.3f}), positive in {h['n_regimes_genuine_pos']} of "
        f"{h['n_regimes']} regime definitions. The statistical artefact is real but lives where "
        f"the folklore measures: conditioning on the month's own volatility, constant "
        f"correlation alone manufactures {h['contemp_artefact']:+.3f} of the "
        f"{h['contemp_naive']:+.3f} naive gap. Correlations do **not** go to one: the crisis "
        f"average is {h['naive_crisis']:.2f}, and the highest 63-day reading in "
        f"{h['n_years']:.0f} years was {h['max_rolling_corr']:.2f}. {fr_txt} In the tails, "
        f"exceedance correlations sit far above the Gaussian benchmark on *both* sides "
        f"(±1σ: {h['exc_down_1']:.2f} down, {h['exc_up_1']:.2f} up, Gaussian "
        f"{h['exc_gauss_1']:.2f}); the downside-minus-upside asymmetry is "
        f"{h['asym']:+.3f} on raw returns (CI [{h['asym_lo']:+.3f}, {h['asym_hi']:+.3f}]) and "
        f"{h['asym_devol']:+.3f} on GARCH-devolatilised shocks (CI "
        f"[{h['asym_devol_lo']:+.3f}, {h['asym_devol_hi']:+.3f}]). Survivor panel of 20 large "
        f"caps — the caveats reason out the direction of that bias.")
    dr_loss = (h["dr_calm"] - h["dr_crisis"]) / max(h["dr_calm"] - 1.0, 1e-9)
    if trad == "Fragile":
        trad_core = (
            f"It hurts, but it does not make diversification fail. In the headline crisis "
            f"regime the equal-weight 20-stock book ran {h['vol_crisis']:.1%} annualised "
            f"volatility against {h['vol_pred']:.1%} predicted from calm correlations with "
            f"crisis volatilities plugged in — a correlation shortfall of "
            f"**{h['shortfall']:+.1%}** net of the {h['shortfall_artefact']:+.1%} the same "
            f"procedure shows under constant correlation (CI [{h['shortfall_lo']:+.1%}, "
            f"{h['shortfall_hi']:+.1%}]). The diversification ratio fell from "
            f"{h['dr_calm']:.2f} to {h['dr_crisis']:.2f}: {dr_loss:.0%} of the free lunch's "
            f"excess over 1 is gone in a crisis, the rest is still there. In "
            f"{h['n_episodes_under']} of the {h['n_episodes']} named episodes, a holder who "
            f"had scaled the pre-crisis covariance by the crisis volatilities would still have under-forecast risk "
            f"(median ratio {h['episode_ratio_median']:.2f}×). The practical consequence is "
            f"a risk-model correction — stress correlations, not just volatilities — and within "
            f"an equity-only book no number of extra names removes it. Nothing here is a trade, "
            f"so nothing can be Investable.")
    else:
        trad_core = (
            f"No. In the headline crisis regime the equal-weight 20-stock book ran "
            f"{h['vol_crisis']:.1%} volatility against {h['vol_pred']:.1%} predicted from calm "
            f"correlations with crisis volatilities plugged in — a shortfall of "
            f"{h['shortfall']:+.1%} (CI [{h['shortfall_lo']:+.1%}, {h['shortfall_hi']:+.1%}]), "
            f"too small or too uncertain to say diversification failed beyond the volatility "
            f"increase. Diversification ratio {h['dr_calm']:.2f} calm vs {h['dr_crisis']:.2f} "
            f"crisis.")
    if signal in ("Real", "Mixed"):
        one = (f"Correlations genuinely rise in crises but nowhere near one — "
               f"{h['naive_calm']:.2f} calm to {h['naive_crisis']:.2f} in high-volatility "
               f"markets, with essentially none of it a statistical artefact when the regime is "
               f"defined from past data — and that makes a diversified stock book "
               f"{h['shortfall']:.0%} riskier than volatility alone predicts, without ever "
               f"making diversification fail.")
    else:
        one = (f"Once the conditioning artefact is subtracted, the crisis rise in correlation "
               f"({h['genuine']:+.2f}) cannot be told apart from zero on this tape.")
    return {"signal": signal, "signal_why": sig_core, "trad": trad, "trad_why": trad_core,
            "one_sentence": one}
