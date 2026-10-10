"""Gibson vs Fisher — the machinery of Study 1026.

The claim is a statement about **levels**: yields co-move with the price level. Levels of
trending series are the most dangerous objects in econometrics — two independent random walks
correlate at ±0.5 more often than not, and a regression of one on the other "finds" a
relationship with a t-statistic in the dozens (Granger & Newbold 1974). So the study runs the
claim through three gates, each of which a spurious correlation fails:

1. **Levels, honestly labelled.** ``level_correlations`` reports the correlations the claim is
   built on — yields against the raw and the detrended price level, and against inflation —
   as *descriptions*, never as evidence. ``unit_root_table`` (ADF + KPSS) says why.
2. **Cointegration and differences.** ``engle_granger`` asks whether the yield and the price gap
   share a common stochastic trend (a real long-run relation leaves a stationary residual; a
   spurious one does not). ``diff_horse_race`` regresses *changes* in yields on changes in the
   price gap and changes in inflation, jointly, with Newey-West errors — the gap must earn its
   coefficient against Fisher's variable, not instead of it.
3. **Out of sample.** ``oos_forecast`` forecasts the next quarter's / year's yield change from
   the current yield plus the (publication-lagged) price gap, on an expanding window, and
   ``clark_west`` tests the improvement over the yield-only model (a nested comparison, so the
   Clark & West 2007 adjustment is the right one). ``timing_backtest`` turns the forecast into
   a duration overlay on a 20-year AAA par bond and charges costs.

``gibson_legs`` packages gates 2-3 into three pass/fail **legs** for each side (Gibson: the gap;
Fisher: inflation), and ``verdict`` maps legs to stamps by a rule written before the real run.

Conventions. Yields in percent. The one execution lag: a forecast formed with information
available at the end of month *t* sets the duration held over month *t+1* — one shift, applied
once in ``timing_backtest``. On top of it, every price-based regressor is lagged by one period
(``add_publication_lag``) because CPI for month *t* is published in month *t+1*; the yield
itself is observed at *t*.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Small OLS with Newey-West — numpy only, so the tests stay fast and deterministic
# --------------------------------------------------------------------------- #


def hac_ols(y, X, lags: int = 12, add_const: bool = True) -> dict:
    """OLS with Newey-West (Bartlett) standard errors.

    Returns ``coef``, ``se``, ``t`` (arrays, intercept first when ``add_const``), ``n``, ``r2``
    and ``resid``. Rows with any NaN are dropped. Empty dict if fewer than 20 usable rows.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    n = len(y)
    if n < 20:
        return {}
    A = np.column_stack([np.ones(n), X]) if add_const else X
    XtX_inv = np.linalg.pinv(A.T @ A)
    beta = XtX_inv @ A.T @ y
    resid = y - A @ beta
    Z = A * resid[:, None]
    S = Z.T @ Z
    for lag in range(1, int(lags) + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = Z[lag:].T @ Z[:-lag]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    tss = float(((y - y.mean()) ** 2).sum())
    return {"coef": beta, "se": se, "t": t, "n": int(n),
            "r2": float(1.0 - (resid ** 2).sum() / tss) if tss > 0 else np.nan,
            "resid": resid}


def hac_mean(x, lags: int = 12) -> dict:
    """Mean of a series with a Newey-West t-statistic (NaNs dropped)."""
    x = pd.Series(np.asarray(x, dtype=float)).dropna().to_numpy()
    if len(x) < 20:
        return {"mean": np.nan, "se": np.nan, "t": np.nan, "n": int(len(x))}
    d = hac_ols(x, np.ones((len(x), 1)), lags=lags, add_const=False)
    return {"mean": float(d["coef"][0]), "se": float(d["se"][0]),
            "t": float(d["t"][0]), "n": d["n"]}


# --------------------------------------------------------------------------- #
# 1 · Levels and unit roots
# --------------------------------------------------------------------------- #
def level_correlations(df: pd.DataFrame, ycol: str, cols=("logp", "gap_exp", "gap_rol", "pi"),
                       start=None, end=None) -> dict:
    """Pearson correlations of the yield level with each price variable (descriptive only)."""
    d = df.loc[start:end] if (start or end) else df
    out = {}
    for c in cols:
        if c in d.columns:
            pair = d[[ycol, c]].dropna()
            out[c] = float(pair[ycol].corr(pair[c])) if len(pair) > 10 else np.nan
    return out


def unit_root_table(series: dict) -> pd.DataFrame:
    """ADF (H0: unit root) and KPSS (H0: stationary) for each series, constant only.

    A series reads **I(1)-like** when ADF fails to reject *and* KPSS rejects — both tests
    pointing the same way. That label is what licenses the spurious-regression warning.
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    rows = []
    for name, s in series.items():
        x = pd.Series(s).dropna().to_numpy(dtype=float)
        if len(x) < 30:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf = adfuller(x, regression="c", autolag="AIC")
            kp = kpss(x, regression="c", nlags="auto")
        rows.append({"series": name, "n": len(x), "adf_stat": float(adf[0]),
                     "adf_p": float(adf[1]), "kpss_stat": float(kp[0]),
                     "kpss_p": float(kp[1]),
                     "i1_like": bool(adf[1] > 0.05 and kp[1] < 0.05)})
    return pd.DataFrame(rows).set_index("series")


def engle_granger(y: pd.Series, x: pd.Series) -> dict:
    """Engle-Granger two-step test of **no** cointegration between ``y`` and ``x``.

    Small p ⇒ the residual of ``y`` on ``x`` is stationary ⇒ a genuine long-run relation.
    Also returns the levels slope and its (meaningless-if-spurious) naive OLS t, which is
    exactly the number Granger & Newbold warned about.
    """
    from statsmodels.tsa.stattools import coint
    pair = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(pair) < 40:
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, _ = coint(pair["y"], pair["x"], trend="c", autolag="aic")
    naive = hac_ols(pair["y"].to_numpy(), pair["x"].to_numpy(), lags=0)
    return {"eg_stat": float(stat), "eg_p": float(p), "slope": float(naive["coef"][1]),
            "naive_t": float(naive["t"][1]), "r2": naive["r2"], "n": int(len(pair))}


# --------------------------------------------------------------------------- #
# 2 · Differences: the horse race
# --------------------------------------------------------------------------- #
def diff_horse_race(df: pd.DataFrame, ycol: str, gapcol: str = "gap_exp",
                    picol: str = "pi", lags: int = 12, start=None, end=None) -> dict:
    """ΔY on Δgap and Δπ jointly (and each alone), Newey-West.

    Differencing removes the common trend that makes level correlations spurious. If the price
    level carries information Fisher's variable does not, ``b_gap`` survives with inflation in
    the regression. Returns slopes and HAC t-stats for the joint and single regressions.
    """
    d = df[[ycol, gapcol, picol]].diff()
    if start or end:
        d = d.loc[start:end]
    d = d.dropna()
    if len(d) < 30:
        return {}
    y = d[ycol].to_numpy()
    j = hac_ols(y, d[[gapcol, picol]].to_numpy(), lags=lags)
    g = hac_ols(y, d[[gapcol]].to_numpy(), lags=lags)
    f = hac_ols(y, d[[picol]].to_numpy(), lags=lags)
    return {"n": j["n"], "b_gap": float(j["coef"][1]), "t_gap": float(j["t"][1]),
            "b_pi": float(j["coef"][2]), "t_pi": float(j["t"][2]), "r2": j["r2"],
            "b_gap_alone": float(g["coef"][1]), "t_gap_alone": float(g["t"][1]),
            "r2_gap_alone": g["r2"],
            "b_pi_alone": float(f["coef"][1]), "t_pi_alone": float(f["t"][1]),
            "r2_pi_alone": f["r2"]}


def regime_difference(df: pd.DataFrame, ycol: str, regime_a, regime_b,
                      gapcol: str = "gap_exp", picol: str = "pi", lags: int = 12) -> dict:
    """Is the gap slope different between two regimes? Interaction regression, Newey-West.

    ΔY = a + b·Δgap + c·Δπ + d·D + e·(D·Δgap) + f·(D·Δπ), D = 1 in regime B. ``e`` is the
    difference in the gap slope (B − A) and its HAC t is the test of the difference.
    """
    d = df[[ycol, gapcol, picol]].diff().dropna()
    a0, a1 = pd.Timestamp(regime_a[0]), pd.Timestamp(regime_a[1])
    b0, b1 = pd.Timestamp(regime_b[0]), pd.Timestamp(regime_b[1])
    ina = (d.index >= a0) & (d.index <= a1)
    inb = (d.index >= b0) & (d.index <= b1)
    d = d[ina | inb]
    D = ((d.index >= b0) & (d.index <= b1)).astype(float)
    X = np.column_stack([d[gapcol], d[picol], D, D * d[gapcol], D * d[picol]])
    r = hac_ols(d[ycol].to_numpy(), X, lags=lags)
    if not r:
        return {}
    return {"diff_gap": float(r["coef"][4]), "t_diff_gap": float(r["t"][4]),
            "diff_pi": float(r["coef"][5]), "t_diff_pi": float(r["t"][5]), "n": r["n"]}


# --------------------------------------------------------------------------- #
# 3 · Out of sample
# --------------------------------------------------------------------------- #
def add_publication_lag(df: pd.DataFrame, cols=("gap_exp", "gap_rol", "pi", "logp"),
                        lag: int = 1) -> pd.DataFrame:
    """Shift price-based columns by ``lag`` periods: CPI for *t* is only known in *t+1*."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].shift(lag)
    return out


def oos_forecast(df: pd.DataFrame, ycol: str, xcols, h: int, min_train: int = 120) -> pd.DataFrame:
    """Expanding-window forecasts of the h-period yield change Y[t+h] − Y[t].

    At each origin *t* the model is fitted on pairs (X[s], Y[s+h] − Y[s]) with s + h ≤ t —
    outcomes already realised at *t* — and evaluated at X[t]. Three nested models, all on the
    same rows: ``mean`` (prevailing mean change), ``base`` (yield level only: mean reversion),
    ``full`` (yield level + ``xcols``). Columns: ``actual``, ``f_mean``, ``f_base``, ``f_full``.
    """
    xcols = list(xcols)
    y = df[ycol].to_numpy(dtype=float)
    n = len(y)
    target = np.full(n, np.nan)
    target[:n - h] = y[h:] - y[:n - h]
    Xb = np.column_stack([np.ones(n), y])
    Xf = np.column_stack([Xb] + [df[c].to_numpy(dtype=float) for c in xcols])
    ok_f = np.isfinite(Xf).all(axis=1)
    rows = []
    for t in range(n):
        if not ok_f[t]:
            continue
        last = t - h                                   # last origin whose outcome is known
        if last < 0:
            continue
        train = np.arange(0, last + 1)
        train = train[ok_f[train] & np.isfinite(target[train])]
        if len(train) < min_train:
            continue
        tt = target[train]
        bb = np.linalg.lstsq(Xb[train], tt, rcond=None)[0]
        bf = np.linalg.lstsq(Xf[train], tt, rcond=None)[0]
        rows.append((df.index[t], target[t], float(tt.mean()),
                     float(Xb[t] @ bb), float(Xf[t] @ bf)))
    out = pd.DataFrame(rows, columns=["date", "actual", "f_mean", "f_base", "f_full"])
    return out.set_index("date")


def clark_west(actual, f_small, f_big, lags: int) -> dict:
    """Clark & West (2007) test that the bigger nested model forecasts better.

    adj_t = e_small² − (e_big² − (f_small − f_big)²); a HAC t on its mean, one-sided.
    """
    from scipy.stats import norm
    a, s, b = (np.asarray(v, dtype=float) for v in (actual, f_small, f_big))
    ok = np.isfinite(a) & np.isfinite(s) & np.isfinite(b)
    a, s, b = a[ok], s[ok], b[ok]
    if len(a) < 30:
        return {"cw_t": np.nan, "cw_p": np.nan, "oos_r2": np.nan, "n": int(len(a))}
    adj = (a - s) ** 2 - ((a - b) ** 2 - (s - b) ** 2)
    m = hac_mean(adj, lags)
    oos_r2 = 1.0 - ((a - b) ** 2).sum() / ((a - s) ** 2).sum()
    return {"cw_t": m["t"], "cw_p": float(1.0 - norm.cdf(m["t"])), "oos_r2": float(oos_r2),
            "n": int(len(a))}


def oos_table(df: pd.DataFrame, ycol: str, h: int, min_train: int,
              variables=("gap_exp", "pi")) -> tuple[pd.DataFrame, dict]:
    """Clark-West and OOS R² for each variable added to the yield-only model, at horizon h."""
    rows, fc = [], {}
    for v in variables:
        f = oos_forecast(df, ycol, [v], h, min_train)
        if f.empty:
            continue
        cw = clark_west(f["actual"], f["f_base"], f["f_full"], lags=max(h, 1) + 2)
        cm = clark_west(f["actual"], f["f_mean"], f["f_base"], lags=max(h, 1) + 2)
        rows.append({"variable": v, "h": h, "n": cw["n"], "oos_r2_vs_yield_only": cw["oos_r2"],
                     "cw_t": cw["cw_t"], "cw_p": cw["cw_p"],
                     "yield_only_oos_r2_vs_mean": cm["oos_r2"],
                     "first": f.index[0], "last": f.index[-1]})
        fc[v] = f
    return pd.DataFrame(rows).set_index("variable"), fc


def adaptive_expectation(pi_1m: pd.Series, halflife: float) -> pd.Series:
    """Fisher with a long memory: expected inflation as an exponentially-weighted average of
    past one-period inflation, half-life ``halflife`` periods. Past-only (``adjust=False``).

    This is Irving Fisher's own answer to Gibson (1930): if investors form inflation
    expectations from a long distributed lag of past inflation, the yield tracks a smoothed
    *cumulation* of inflation — which looks, on a chart, exactly like the price level relative
    to its trend. The gap and this variable are the two readings of the same correlation.
    """
    lam = 1.0 - 0.5 ** (1.0 / float(halflife))
    return pi_1m.ewm(alpha=lam, adjust=False).mean().rename(f"ewma{int(halflife)}")


# --------------------------------------------------------------------------- #
# 4 · A duration overlay on a 20-year AAA par bond
# --------------------------------------------------------------------------- #
def par_bond_duration_convexity(y_pct, maturity: float = 20.0, freq: int = 2):
    """Modified duration and convexity of a par bond at yield ``y_pct`` (percent, annual).

    Closed forms for a par bond (coupon = yield): D_mod = (1 − v^N) / y with v = 1/(1 + y/f),
    N = f·maturity; convexity by numerical second difference of the price.
    """
    y = np.maximum(np.asarray(y_pct, dtype=float), 0.25) / 100.0
    N = freq * maturity

    def price(c, yy):
        v = 1.0 / (1.0 + yy / freq)
        return (c / freq) * (1.0 - v ** N) / (yy / freq) + v ** N

    v = 1.0 / (1.0 + y / freq)
    D = (1.0 - v ** N) / y
    dy = 1e-4
    C = (price(y, y + dy) + price(y, y - dy) - 2.0 * price(y, y)) / (dy ** 2 * price(y, y))
    return D, C


def bond_returns(yields: pd.Series, rf: pd.Series, periods_per_year: int = 12,
                 maturity: float = 20.0) -> pd.DataFrame:
    """Monthly holding-period return of a constant-maturity 20-year AAA par bond.

    The **duration approximation**, stated: r[t+1] ≈ Y[t]/12 − D[t]·ΔY + ½·C[t]·ΔY², with D, C
    the modified duration and convexity of a 20-year semi-annual par bond at Y[t] and ΔY the
    change in the Moody's AAA yield (decimal). Carry plus price change; no roll-down (a
    constant-maturity index has none to speak of at 20 years). Total return, **nominal**.
    Excess return subtracts ``rf`` (the one-month T-bill, decimal per period).
    """
    y = yields.astype(float) / 100.0
    D, C = par_bond_duration_convexity(yields.shift(1), maturity)
    dy = y.diff()
    r = y.shift(1) / periods_per_year - D * dy + 0.5 * C * dy ** 2
    out = pd.DataFrame({"bond": r, "rf": rf.reindex(yields.index)})
    out["bond_ex"] = out["bond"] - out["rf"]
    out["duration"] = D
    return out


def timing_positions(forecast: pd.DataFrame, col: str = "f_full", lo: float = 0.5,
                     hi: float = 1.5) -> pd.Series:
    """Duration weight decided at the end of each month from the forecast.

    Long duration (``hi``) when the model forecasts a yield change **below** the prevailing
    mean change (yields falling more, or rising less, than usual), short duration (``lo``)
    otherwise. Demeaning against the prevailing mean keeps the rule from simply learning that
    yields fell 1982-2020.
    """
    sig = forecast[col] < forecast["f_mean"]
    return pd.Series(np.where(sig, hi, lo), index=forecast.index, name="w")


def timing_backtest(bond: pd.DataFrame, w: pd.Series, cost_bps: float = 5.0,
                    periods_per_year: int = 12) -> pd.DataFrame:
    """Overlay P&L with **one** execution lag: weight set at *t* earns the bond excess of *t+1*.

    Returns, per period: ``pos`` (weight held), ``gross`` = pos·bond_ex, ``turnover`` = |Δpos|
    (one-way, fraction of NAV), ``net`` = gross − cost·turnover, and ``const`` = 1·bond_ex
    (constant duration, the benchmark, never trades after inception). All are **excess of
    cash**, so the Sharpe race is excess-vs-excess.
    """
    pos = w.reindex(bond.index).shift(1)
    d = pd.DataFrame({"pos": pos, "bond_ex": bond["bond_ex"]}).dropna()
    d["turnover"] = d["pos"].diff().abs().fillna(0.0)
    d["gross"] = d["pos"] * d["bond_ex"]
    d["net"] = d["gross"] - cost_bps / 1e4 * d["turnover"]
    d["const"] = d["bond_ex"]
    return d


def perf(x: pd.Series, periods_per_year: int = 12) -> dict:
    x = x.dropna()
    if len(x) < 12:
        return {"ann_mean": np.nan, "ann_vol": np.nan, "sharpe": np.nan}
    m = x.mean() * periods_per_year
    v = x.std(ddof=1) * np.sqrt(periods_per_year)
    return {"ann_mean": float(m), "ann_vol": float(v), "sharpe": float(m / v) if v > 0 else np.nan}


def timing_summary(bt: pd.DataFrame, periods_per_year: int = 12, lags: int = 12,
                   splits=None) -> dict:
    """Gross / net / constant performance and the HAC test of (net − constant)."""
    out = {"n": int(len(bt)), "first": bt.index[0], "last": bt.index[-1],
           "turnover_ann": float(bt["turnover"].mean() * periods_per_year),
           "gross": perf(bt["gross"], periods_per_year), "net": perf(bt["net"], periods_per_year),
           "const": perf(bt["const"], periods_per_year)}
    diff = bt["net"] - bt["const"]
    m = hac_mean(diff, lags)
    out["diff_net_ann"] = m["mean"] * periods_per_year
    out["diff_net_t"] = m["t"]
    gd = hac_mean(bt["gross"] - bt["const"], lags)
    out["diff_gross_ann"] = gd["mean"] * periods_per_year
    out["diff_gross_t"] = gd["t"]
    turn = bt["turnover"].mean()
    out["breakeven_bps"] = float(gd["mean"] / turn * 1e4) if turn > 0 else np.nan
    out["mean_pos"] = float(bt["pos"].mean())
    reg = hac_ols(bt["net"].to_numpy(), bt[["const"]].to_numpy(), lags=lags)
    out["alpha_ann"] = float(reg["coef"][0] * periods_per_year)
    out["alpha_t"] = float(reg["t"][0])
    out["beta_to_const"] = float(reg["coef"][1])
    out["subs"] = {}
    for name, (a, b) in (splits or {}).items():
        sub = diff.loc[a:b]
        if len(sub) >= 24:
            ms = hac_mean(sub, lags)
            out["subs"][name] = {"diff_net_ann": ms["mean"] * periods_per_year, "t": ms["t"],
                                 "n": int(len(sub))}
    return out


# --------------------------------------------------------------------------- #
# The legs, and the verdict
# --------------------------------------------------------------------------- #
def gibson_legs(df: pd.DataFrame, ycol: str = "aaa", gapcol: str = "gap_exp",
                picol: str = "pi", h: int = 12, min_train: int = 120, lags: int = 12,
                pub_lag: int = 1, alpha: float = 0.05) -> dict:
    """Three pass/fail legs for each side of the argument, on one tape.

    For the **Gibson** side (variable = price gap) and the **Fisher** side (inflation):

    - **L1 cointegration** — Engle-Granger p < ``alpha`` for the yield on the variable.
    - **L2 differences** — HAC t ≥ 2 on the variable's slope in the *joint* ΔY regression.
    - **L3 out of sample** — Clark-West one-sided p < ``alpha`` for adding the
      (publication-lagged) variable to the yield-only forecast of the h-period yield change.

    A spurious levels relation fails L1 and L2; a relation that exists but cannot be used
    fails L3.
    """
    eg_g = engle_granger(df[ycol], df[gapcol])
    eg_f = engle_granger(df[ycol], df[picol])
    hr = diff_horse_race(df, ycol, gapcol, picol, lags=lags)
    lagged = add_publication_lag(df, cols=(gapcol, picol), lag=pub_lag)
    tbl, _ = oos_table(lagged, ycol, h, min_train, variables=(gapcol, picol))
    g = {"L1": bool(eg_g.get("eg_p", 1.0) < alpha),
         "L2": bool(hr.get("t_gap", 0.0) >= 2.0),
         "L3": bool(tbl.loc[gapcol, "cw_p"] < alpha) if gapcol in tbl.index else False}
    f = {"L1": bool(eg_f.get("eg_p", 1.0) < alpha),
         "L2": bool(hr.get("t_pi", 0.0) >= 2.0),
         "L3": bool(tbl.loc[picol, "cw_p"] < alpha) if picol in tbl.index else False}
    return {"gibson": g, "fisher": f, "eg_gap": eg_g, "eg_pi": eg_f, "horse_race": hr,
            "oos": tbl, "n_gibson": int(sum(g.values())), "n_fisher": int(sum(f.values())),
            "gibson_strong": bool(g["L2"] and (g["L1"] or g["L3"])),
            "fisher_strong": bool(f["L2"] and (f["L1"] or f["L3"]))}


def signal_stamp(us_legs: dict, de_legs: dict) -> str:
    """Pre-registered Signal rule for the **Gibson** claim (US primary, Germany check).

    - **Real**: the US tape is strong (L2 and one of L1/L3) **and** Germany passes ≥ 1 leg.
    - **Mixed**: one tape strong, the other passes nothing.
    - **Weak**: at least one Gibson leg passes somewhere, short of the above.
    - **None**: no Gibson leg passes on either tape.
    """
    us_strong = bool(us_legs["L2"] and (us_legs["L1"] or us_legs["L3"]))
    de_strong = bool(de_legs["L2"] and (de_legs["L1"] or de_legs["L3"]))
    us_any, de_any = any(us_legs.values()), any(de_legs.values())
    if us_strong and de_any:
        return "Real"
    if (us_strong and not de_any) or (de_strong and not us_any):
        return "Mixed"
    if us_any or de_any:
        return "Weak"
    return "None"


def trad_stamp(t: dict, signal: str) -> str:
    """Pre-registered Tradability rule for the gap-driven duration overlay (net of costs).

    - **Investable**: net-minus-constant HAC t ≥ 2, positive in *both* regimes, beats the
      yield-only timer (credit belongs to the price level, not to yield mean reversion), and a
      Real or Mixed signal behind it.
    - **Fragile**: positive net edge over constant duration *and* a higher net Sharpe.
    - **Mirage**: anything else.
    """
    subs = t.get("subs", {})
    both = len(subs) >= 2 and all(v["diff_net_ann"] > 0 for v in subs.values())
    if (t["diff_net_t"] >= 2.0 and both and t.get("beats_yield_only", False)
            and signal in ("Real", "Mixed")):
        return "Investable"
    if t["diff_net_ann"] > 0 and t["net"]["sharpe"] > t["const"]["sharpe"]:
        return "Fragile"
    return "Mirage"


def verdict(h: dict) -> dict:
    """Stamps by the rules in :func:`signal_stamp` and :func:`trad_stamp`, plus the prose.

    ``h`` needs the leg dicts ``us_gibson`` / ``de_gibson`` / ``us_fisher`` / ``de_fisher``,
    ``timing`` (a :func:`timing_summary` dict carrying ``beats_yield_only`` and
    ``diff_vs_yield_only_ann``) and the headline numbers quoted in the prose: ``corr_logp``,
    ``corr_gap``, ``corr_pi``, ``corr_ewma``, ``t_gap``, ``t_pi``, ``eg_p_gap``, ``eg_p_pi``,
    ``eg_p_ewma``, ``ewma_hl``, ``t_gap_vs_ewma``, ``cw_p_gap``, ``cw_p_pi``, ``cw_p_gap_rol``,
    ``de_t_gap``, ``de_t_pi``, ``de_eg_p_gap``, ``gi_corr_logp``, ``dis_corr_logp``,
    ``cost_bps``.
    """
    signal = signal_stamp(h["us_gibson"], h["de_gibson"])
    t = h["timing"]
    trad = trad_stamp(t, signal)

    def legs_txt(d):
        return ", ".join(k for k, v in d.items() if v) or "none"

    subs = t.get("subs", {})
    sub_txt = "; ".join(f"{k} {v['diff_net_ann']:+.2%} (t {v['t']:+.1f})"
                        for k, v in subs.items())
    signal_why = (
        f"Split by tape. **US (core CPI, 1957-2018): the price gap wins the horse race against "
        f"12-month inflation.** In first differences, with both in the regression, the gap's "
        f"Newey-West t is **{h['t_gap']:+.2f}** and inflation's **{h['t_pi']:+.2f}**; out of "
        f"sample the (publication-lagged) gap improves a yield-only forecast of the next year's "
        f"yield change at Clark-West p = {h['cw_p_gap']:.3f} (inflation p = {h['cw_p_pi']:.3f}). "
        f"US Gibson legs passed: {legs_txt(h['us_gibson'])}. **Germany (1972-98): nothing** — "
        f"gap t {h['de_t_gap']:+.2f}, inflation t {h['de_t_pi']:+.2f}, legs passed: "
        f"{legs_txt(h['de_gibson'])}. Three things keep this from Real. (1) *No cointegration*: "
        f"the yield and the price gap do not share a stochastic trend (Engle-Granger p = "
        f"{h['eg_p_gap']:.2f}), so the famous level correlation ({h['corr_gap']:+.2f} with the gap; "
        f"the raw log price level flips from {h['gi_corr_logp']:+.2f} in 1965-81 to "
        f"{h['dis_corr_logp']:+.2f} in 1982-2018) is two trends, not a law. (2) *Fisher with a "
        f"long memory explains it better*: an adaptive inflation expectation (exponential "
        f"average, {h['ewma_hl']:.0f}-month half-life) correlates {h['corr_ewma']:+.2f} with "
        f"the yield and **is** cointegrated with it (p = {h['eg_p_ewma']:.3f}); in differences "
        f"the two are near-identical and the gap's edge over it shrinks to t "
        f"{h['t_gap_vs_ewma']:+.2f} — Irving Fisher's own 1930 resolution of the paradox, "
        f"visible on a modern tape. (3) *Specification*: detrend the price level over a rolling "
        f"10-year window instead of the full history and the out-of-sample edge disappears "
        f"(Clark-West p = {h['cw_p_gap_rol']:.2f}). The tapes hold no price index before 1957, "
        f"so the gold-standard paradox itself is untested here.")
    trad_why = (
        f"As a duration overlay on a 20-year AAA par bond (1.5× duration when the expanding-"
        f"window forecast of next year's yield change is below its prevailing mean, else 0.5×; "
        f"CPI lagged a month, one execution lag, {h['cost_bps']:.0f} bp one-way), the gap timer "
        f"earned **{t['diff_net_ann']:+.2%} a year net over constant duration (HAC t "
        f"{t['diff_net_t']:+.2f})** with net excess Sharpe {t['net']['sharpe']:.2f} vs "
        f"{t['const']['sharpe']:.2f}. But it trades {t['turnover_ann']:.2f}× NAV a year and sat at "
        f"{t['mean_pos']:.2f}× duration on average — it is essentially one regime call, long "
        f"duration through the 1982-2018 bond bull; by regime: {sub_txt}. Its alpha after "
        f"regressing on constant duration is {t['alpha_ann']:+.2%} (t {t['alpha_t']:+.2f}). It "
        f"{'beats' if t.get('beats_yield_only') else 'does not beat'} a yield-only timer "
        f"({t.get('diff_vs_yield_only_ann', np.nan):+.2%} a year). Costs are not the problem "
        f"(break-even {t['breakeven_bps']:.0f} bp); one secular trade and zero in the Great "
        f"Inflation is.")
    if signal in ("Real", "Mixed"):
        mid = ("the detrended price level beats 12-month inflation on the US tape but not in "
               "Germany, and it is not cointegrated with yields while long-memory inflation "
               "expectations are")
    else:
        mid = "the price level adds nothing reliable beyond inflation"
    one = (f"Under fiat money Gibson's paradox survives only as a shadow of Fisher: {mid}; the "
           f"duration timer it implies earns {t['diff_net_ann']:+.2%} a year net (t "
           f"{t['diff_net_t']:+.1f}) from what is really one long bet on the 1982-2018 bond bull.")
    return {"signal": signal, "signal_why": signal_why, "trad": trad, "trad_why": trad_why,
            "one_sentence": one}
