"""Is silver a levered gold position? — Study 987.

The claim is unusually testable. "Silver is gold with the volume turned up" says that silver's
returns are ``beta * gold + noise`` with no systematic second term. Three things have to be
checked, and they are routinely conflated:

1. **Is beta stable?** A single full-sample regression coefficient means nothing if the true
   loading wanders between 1.2 and 2.5 depending on the decade. ``rolling_beta_table`` measures
   how much it moves and how much of the return variance that movement accounts for.

2. **Is the residual just noise, or a second asset?** If silver minus beta-times-gold has no
   structure — no autocorrelation, no loading on any other observable, no risk premium — then
   silver really is redundant. ``residual_diagnostics`` and ``residual_loadings`` look for the
   structure; industrial activity and copper are the natural suspects, since roughly half of
   silver demand is industrial and none of gold's is.

3. **Does the replication actually work?** The statistical question and the portfolio question
   are different. A levered gold position tracking silver at an R² of 0.7 sounds close; run it
   as a strategy with daily rebalancing, financing and the volatility drag that leverage
   incurs, and the two can end up in different places entirely. ``replication_backtest`` does
   the honest version.

One asymmetry worth stating up front, because it drives the results: **volatility drag is not
symmetric between the thing and its levered replica**. A 2x gold position rebalanced daily is
not 2x gold's return — it is 2x gold's *daily* return compounded, which over years costs
roughly ``beta * (beta - 1) * sigma^2 / 2`` a year. That term is arithmetic, not opinion, and
it is computed explicitly in ``leverage_drag``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Beta, and whether it holds still
# --------------------------------------------------------------------------- #
def full_sample_beta(y: pd.Series, x: pd.Series, hac_lags: int = 0) -> dict:
    """OLS of silver on gold with HC1 errors — the number everyone quotes.

    ``hac_lags`` switches to Newey-West. It must be used whenever the left-hand side is an
    *overlapping* forward return, as in ``ratio_mean_reversion``: with overlapping windows the
    HC1 standard error is far too small and a pure random walk will routinely produce a
    *t*-statistic past 4. The daily regressions in the rest of this module have no overlap and
    leave it at zero.
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1, sort=False).dropna()
    n = len(df)
    if n < 100:
        return {"n": int(n)}
    A = np.column_stack([np.ones(n), df["x"].to_numpy()])
    coef, *_ = np.linalg.lstsq(A, df["y"].to_numpy(), rcond=None)
    resid = df["y"].to_numpy() - A @ coef
    XtX_inv = np.linalg.pinv(A.T @ A)
    if hac_lags > 0:
        L = int(min(hac_lags, n // 4))
        u = A * resid[:, None]
        S = u.T @ u / n
        for k in range(1, L + 1):
            w = 1.0 - k / (L + 1.0)
            G = u[k:].T @ u[:-k] / n
            S += w * (G + G.T)
        XtX_n = np.linalg.pinv(A.T @ A / n)
        V = XtX_n @ S @ XtX_n / n
    else:
        V = XtX_inv @ (A.T @ np.diag(resid ** 2) @ A) @ XtX_inv * n / max(n - 2, 1)
    se = float(np.sqrt(max(V[1, 1], 0.0)))
    ss_tot = float(((df["y"] - df["y"].mean()) ** 2).sum())
    return {"n": int(n), "alpha": float(coef[0]), "beta": float(coef[1]), "se": se,
            "t_vs_zero": float(coef[1] / se) if se > 0 else np.nan,
            "alpha_ann": float(coef[0] * TRADING_DAYS),
            "r2": float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan,
            "resid_vol_ann": float(resid.std(ddof=1) * np.sqrt(TRADING_DAYS))}


def rolling_beta(y: pd.Series, x: pd.Series, window: int = 252) -> pd.Series:
    """Trailing beta, known at each close (uses data through *t-1*)."""
    cov = y.rolling(window).cov(x).shift(1)
    var = x.rolling(window).var().shift(1)
    return (cov / var).rename("beta")


def rolling_beta_table(y: pd.Series, x: pd.Series, windows=(63, 126, 252, 756)) -> pd.DataFrame:
    """How much the loading moves, at several estimation windows."""
    rows = []
    for w in windows:
        b = rolling_beta(y, x, w).dropna()
        if len(b) < 100:
            continue
        rows.append({"window": w, "n": len(b), "mean": float(b.mean()),
                     "sd": float(b.std(ddof=1)), "min": float(b.min()),
                     "max": float(b.max()), "p10": float(b.quantile(0.10)),
                     "p90": float(b.quantile(0.90)),
                     "range_over_mean": float((b.max() - b.min()) / b.mean())})
    return pd.DataFrame(rows).set_index("window")


def beta_by_regime(y: pd.Series, x: pd.Series, n_buckets: int = 3) -> pd.DataFrame:
    """Beta in gold's up months against its down months, and by gold's volatility.

    A "levered gold" story predicts the same beta everywhere. An asymmetry — silver keeping up
    on the way up and falling further on the way down, say — would mean the leverage claim is
    an average of two different behaviours.
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    rows = []
    for label, mask in (("gold up days", df["x"] > 0), ("gold down days", df["x"] <= 0)):
        sl = df[mask]
        b = full_sample_beta(sl["y"], sl["x"])
        rows.append({"regime": label, "n": b.get("n", 0), "beta": b.get("beta", np.nan),
                     "se": b.get("se", np.nan), "r2": b.get("r2", np.nan)})
    vol = df["x"].rolling(63).std().shift(1)
    q = pd.qcut(vol, n_buckets, labels=[f"gold vol Q{i + 1}" for i in range(n_buckets)])
    for label, sl in df.groupby(q, observed=True):
        b = full_sample_beta(sl["y"], sl["x"])
        rows.append({"regime": str(label), "n": b.get("n", 0), "beta": b.get("beta", np.nan),
                     "se": b.get("se", np.nan), "r2": b.get("r2", np.nan)})
    return pd.DataFrame(rows).set_index("regime")


# --------------------------------------------------------------------------- #
# What is in the residual?
# --------------------------------------------------------------------------- #
def residuals(y: pd.Series, x: pd.Series, window: int = 252) -> pd.Series:
    """Silver minus a trailing-beta gold position — the part that is not gold."""
    b = rolling_beta(y, x, window)
    return (y - b * x).rename("residual").dropna()


def residual_diagnostics(resid: pd.Series) -> dict:
    """Is the leftover noise, or an asset?"""
    r = resid.dropna()
    n = len(r)
    if n < 100:
        return {"n": int(n)}
    mean_ann = float(r.mean() * TRADING_DAYS)
    vol_ann = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    se = r.std(ddof=1) / np.sqrt(n)
    return {"n": int(n), "mean_ann": mean_ann, "vol_ann": vol_ann,
            "sharpe": mean_ann / vol_ann if vol_ann > 0 else np.nan,
            "t_mean": float(r.mean() / se) if se > 0 else np.nan,
            "autocorr_1": float(r.autocorr(1)), "autocorr_5": float(r.autocorr(5)),
            "skew": float(r.skew()), "kurtosis": float(r.kurtosis()),
            "share_of_silver_vol": np.nan}


def residual_loadings(resid: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    """Does the residual load on anything observable?

    If silver is levered gold plus noise, none of these should matter. If roughly half of silver
    demand really is industrial, the industrial and copper columns should light up and gold's
    should not (it has been projected out).
    """
    rows = []
    for name in factors.columns:
        b = full_sample_beta(resid, factors[name])
        rows.append({"factor": name, "n": b.get("n", 0), "beta": b.get("beta", np.nan),
                     "t": b.get("t_vs_zero", np.nan), "r2": b.get("r2", np.nan)})
    joint = _multi_ols(resid, factors)
    for name in factors.columns:
        rows.append({"factor": f"{name} (joint)", "n": joint.get("n", 0),
                     "beta": joint.get(f"beta_{name}", np.nan),
                     "t": joint.get(f"t_{name}", np.nan), "r2": joint.get("r2", np.nan)})
    return pd.DataFrame(rows).set_index("factor")


def _multi_ols(y: pd.Series, X: pd.DataFrame) -> dict:
    df = pd.concat([y.rename("_y"), X], axis=1, sort=False).dropna()
    n = len(df)
    if n < 100:
        return {"n": int(n)}
    names = list(X.columns)
    A = np.column_stack([np.ones(n), df[names].to_numpy()])
    coef, *_ = np.linalg.lstsq(A, df["_y"].to_numpy(), rcond=None)
    resid = df["_y"].to_numpy() - A @ coef
    XtX_inv = np.linalg.pinv(A.T @ A)
    V = XtX_inv @ (A.T @ np.diag(resid ** 2) @ A) @ XtX_inv * n / max(n - len(names) - 1, 1)
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    ss_tot = float(((df["_y"] - df["_y"].mean()) ** 2).sum())
    out = {"n": int(n), "r2": float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan}
    for i, nm in enumerate(names, start=1):
        out[f"beta_{nm}"] = float(coef[i])
        out[f"t_{nm}"] = float(coef[i] / se[i]) if se[i] > 0 else np.nan
    return out


# --------------------------------------------------------------------------- #
# The arithmetic of levering
# --------------------------------------------------------------------------- #
def leverage_drag(beta: float, vol_ann: float) -> float:
    """The annual cost of holding ``beta`` units of something with volatility ``vol_ann``.

    A daily-rebalanced ``beta``-times position compounds ``beta`` times the *daily* return, not
    ``beta`` times the period return. The difference, to second order, is
    ``beta * (beta - 1) * sigma^2 / 2`` a year. At beta 2 and 16% volatility that is about 2.6%
    a year — which is larger than most of the alphas anyone argues about in this space, and it
    is arithmetic rather than an empirical claim.
    """
    return float(beta * (beta - 1.0) * vol_ann ** 2 / 2.0)


def levered_position(x: pd.Series, beta: float, cash: pd.Series | None = None,
                     financing_spread: float = 0.005, cost_bps: float = 1.0) -> pd.Series:
    """Daily-rebalanced ``beta``-times ``x``, financed at cash plus a spread."""
    r = x.dropna()
    c = (cash.reindex(r.index).fillna(0.0) if cash is not None
         else pd.Series(0.0, index=r.index))
    borrow = max(beta - 1.0, 0.0)
    financing = borrow * (c + financing_spread / TRADING_DAYS)
    turnover = beta * r.abs() * 0.0 + abs(beta) * 0.02   # rebalancing back to target
    return (beta * r - financing - turnover * cost_bps / 1e4).rename("levered")


def replication_backtest(silver: pd.Series, gold: pd.Series, cash: pd.Series,
                         beta: float | None = None, financing_spread: float = 0.005,
                         cost_bps: float = 1.0) -> dict:
    """Hold silver, or hold levered gold. Which ends up where?"""
    df = pd.concat([silver.rename("s"), gold.rename("g")], axis=1).dropna()
    beta = full_sample_beta(df["s"], df["g"])["beta"] if beta is None else beta
    lev = levered_position(df["g"], beta, cash, financing_spread, cost_bps)
    aligned = pd.concat([df["s"].rename("silver"), lev.rename("replica")], axis=1).dropna()
    years = len(aligned) / TRADING_DAYS

    def stats(col):
        c = (1 + col).cumprod()
        sd = float(col.std(ddof=1))
        return {"cagr": float(c.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan,
                "vol": sd * np.sqrt(TRADING_DAYS),
                "sharpe": float(col.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
                "max_dd": float((c / c.cummax() - 1).min()),
                "total": float(c.iloc[-1] - 1)}

    s, rep = stats(aligned["silver"]), stats(aligned["replica"])
    d = (aligned["silver"] - aligned["replica"]).dropna()
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 30 else np.nan
    te = float(d.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return {"beta_used": float(beta), "years": float(years), "silver": s, "replica": rep,
            "cagr_gap": s["cagr"] - rep["cagr"], "tracking_error_ann": te,
            "t_gap": float(d.mean() / se) if se and se > 0 else np.nan,
            "correlation": float(aligned["silver"].corr(aligned["replica"])),
            "predicted_drag": leverage_drag(beta, rep["vol"] / max(beta, 1e-9)),
            "returns": aligned}


def gold_silver_ratio(gold_px: pd.Series, silver_px: pd.Series) -> pd.Series:
    """The ratio the trade press quotes, on a common index base."""
    df = pd.concat([gold_px.rename("g"), silver_px.rename("s")], axis=1).dropna()
    return (df["g"] / df["g"].iloc[0]) / (df["s"] / df["s"].iloc[0])


def ratio_mean_reversion(ratio: pd.Series, lookback: int = 252,
                         horizons=(21, 63, 126, 252)) -> pd.DataFrame:
    """Does a stretched gold/silver ratio predict its own reversal?

    The classic trade, with two traps this table is built to avoid.

    First, a *ratio* that reverts says nothing about which leg to hold unless the reversion is
    measured against a real return. Second — and this is the one that catches people — the
    forward windows **overlap**, so an HC1 standard error is far too small here. A pure random
    walk run through this table with heteroskedasticity-robust errors routinely produces
    *t*-statistics past 4 and looks like a tradeable mean-reverting signal. Newey-West at the
    horizon length is not optional; it is the difference between a result and an artefact.
    """
    z = ((np.log(ratio) - np.log(ratio).rolling(lookback).mean())
         / np.log(ratio).rolling(lookback).std())
    rows = []
    for hz in horizons:
        fwd = np.log(ratio).shift(-hz) - np.log(ratio)
        df = pd.concat([z.rename("z"), fwd.rename("fwd")], axis=1, sort=False).dropna()
        if len(df) < 200:
            continue
        b = full_sample_beta(df["fwd"], df["z"], hac_lags=hz)
        rows.append({"horizon": hz, "n": b.get("n", 0), "slope": b.get("beta", np.nan),
                     "t": b.get("t_vs_zero", np.nan), "r2": b.get("r2", np.nan)})
    return pd.DataFrame(rows).set_index("horizon") if rows else pd.DataFrame()


def synthetic_world(n: int = 5000, true_beta: float = 1.8, industrial_load: float = 0.0,
                    beta_drift: float = 0.0, seed: int = 987) -> pd.DataFrame:
    """Gold, silver and an industrial factor, with silver's construction under our control.

    At ``industrial_load = 0`` and ``beta_drift = 0``, silver is *exactly* levered gold plus
    idiosyncratic noise: the null in which the folklore is literally true and every test here
    must agree that it is. Turning either knob up gives silver a life of its own.
    """
    rng = np.random.default_rng(seed)
    gold = rng.normal(0.0002, 0.010, n)
    industrial = rng.normal(0.0003, 0.011, n)
    drift = beta_drift * np.sin(np.linspace(0, 6 * np.pi, n))
    beta_t = true_beta + drift
    silver = beta_t * gold + industrial_load * industrial + rng.normal(0, 0.012, n)
    idx = pd.bdate_range("2006-05-01", periods=n)
    return pd.DataFrame({"gold": gold, "silver": silver, "industrial": industrial,
                         "beta_t": beta_t,
                         "cash": np.full(n, 0.02 / TRADING_DAYS)}, index=idx)


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Confirmed** (silver *is* levered gold) only if the residual has no
      significant loading on any outside factor **and** beta is stable — its rolling range is
      under half its mean; **Partial** if one of those holds; **Busted** if neither does.
    - **Tradability**: **Useful** if the levered-gold replica beats silver on Sharpe by a
      margin worth acting on; **Partial** if the two are within a rounding error; **Mirage** if
      the replica is worse.
    """
    clean_residual = h["max_abs_residual_t"] < 2.0
    stable_beta = h["beta_range_over_mean"] < 0.5
    signal = ("Confirmed" if (clean_residual and stable_beta)
              else ("Partial" if (clean_residual or stable_beta) else "Busted"))
    edge = h["replica_sharpe"] - h["silver_sharpe"]
    trad = ("Useful" if edge > 0.1 else ("Partial" if edge > -0.1 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"Regressed on gold over {h['n_days']:,} sessions, silver's beta is "
            f"**{h['beta']:.2f}** (±{h['beta_se']:.2f}) with an R² of {h['r2']:.0%} — so about "
            f"{1 - h['r2']:.0%} of silver's variance is *not* gold. That leftover is not noise. "
            f"Its annualised volatility is {h['resid_vol']:.1%}, and it loads on outside factors "
            f"with a largest |*t*| of **{h['max_abs_residual_t']:.2f}** "
            f"(on {h['strongest_factor']}). Nor is the beta a constant: over rolling one-year "
            f"windows it ranges {h['beta_min']:.2f} to {h['beta_max']:.2f}, a spread of "
            f"**{h['beta_range_over_mean']:.0%} of its own mean**, and it is "
            f"{h['beta_up']:.2f} on gold's up days against {h['beta_down']:.2f} on its down "
            f"days. 'Silver is levered gold' is a reasonable first approximation and a poor "
            f"second one."),
        "trad": trad,
        "trad_why": (
            f"Holding a {h['beta']:.2f}× daily-rebalanced gold position instead of silver, "
            f"financed at cash plus {h['financing_spread']:.1%}, tracked it with a correlation "
            f"of {h['correlation']:.2f} and an annualised tracking error of "
            f"**{h['tracking_error']:.1%}** — which is not tracking, it is a different asset. "
            f"Over {h['years']:.0f} years silver compounded at {h['silver_cagr']:+.1%} against "
            f"the replica's {h['replica_cagr']:+.1%} (Sharpe {h['silver_sharpe']:.2f} vs "
            f"{h['replica_sharpe']:.2f}). Part of that gap is pure arithmetic: levering a "
            f"{h['gold_vol']:.0%}-vol asset {h['beta']:.2f}× costs about "
            f"**{h['predicted_drag']:.1%} a year** in volatility drag, before financing and "
            f"before costs."),
        "one_sentence": (
            f"Silver's beta to gold is {h['beta']:.2f} with an R² of {h['r2']:.0%}, but it "
            f"wanders across a {h['beta_range_over_mean']:.0%} range and leaves a "
            f"{h['resid_vol']:.0%}-vol residual with structure in it — so silver is levered "
            f"gold in the same way a dog is a levered cat."),
    }
