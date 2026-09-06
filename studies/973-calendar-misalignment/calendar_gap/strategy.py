"""Non-synchronous trading and the estimates it bends — Study 973.

The mechanism, in one sentence: when two assets do not incorporate the same information at the
same moment, the covariance between their *same-day* returns misses the part of the common
shock that one of them will only reflect tomorrow — so measured correlation and beta are biased
toward zero, and the bias grows with how much of the day the two markets did not share.

Four estimators, in increasing order of how much they admit that:

1. ``same_day_correlation`` — the naive one. Everything downstream of a correlation matrix in
   a multi-asset portfolio is built on it.
2. ``dimson_beta`` — Dimson (1979): regress on the market's lead, contemporaneous and lagged
   returns and **sum the coefficients**. The standard aggregated-coefficients fix.
3. ``scholes_williams_beta`` — Scholes & Williams (1977): the same idea derived from the
   covariance structure, normalised by the market's own autocorrelation.
4. ``aggregated_correlation`` — the blunt fix that needs no model: measure at a lower
   frequency (weekly, monthly). If the bias is a timing artefact it must shrink as the
   measurement interval grows past the mismatch, and that convergence is the cleanest
   evidence that the effect is timing rather than economics.

Two things this study is careful about, because they are where the naive version of it goes
wrong:

- **All five tapes are New-York-listed ETFs.** They quote in the same timezone on the same
  exchange calendar, so there is no *quote* asynchrony at all. What remains is the mismatch in
  the **underlying** markets' hours, which is the honest version of the question — and it means
  the result cannot be dismissed as a data-alignment problem, because the data is aligned.
- **A same-market control (IWM against SPY)** runs through every table. Any bias the machinery
  reports there is machinery, not economics.

``portfolio_impact`` closes the loop: build a minimum-variance portfolio from the naive
correlation matrix and from the lower-frequency one, and compare the volatility each *promised*
with the volatility each delivered. A correlation matrix biased toward zero tells an optimiser
that diversification is cheaper than it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
FREQS = {"daily": 1, "weekly": 5, "biweekly": 10, "monthly": 21}


def to_returns(prices: pd.DataFrame, step: int = 1) -> pd.DataFrame:
    """Non-overlapping ``step``-day returns (step = 1 is the daily panel)."""
    p = prices.dropna(how="all")
    if step > 1:
        p = p.iloc[::step]
    return p.pct_change().dropna(how="all")


def same_day_correlation(rets: pd.DataFrame) -> pd.DataFrame:
    """The naive correlation matrix — the thing every risk system actually uses."""
    return rets.corr()


def lead_lag_profile(rets: pd.DataFrame, a: str, b: str, max_lag: int = 3) -> pd.Series:
    """corr(a_t, b_{t+k}) for k in [-max_lag, max_lag] — where the shared information sits.

    A symmetric profile peaked at zero is synchronous trading. A profile whose mass sits at
    positive k (the foreign asset following the US one) is the signature of non-synchronous
    information, and its shape is what the Dimson correction sums up.
    """
    out = {}
    for k in range(-max_lag, max_lag + 1):
        x, y = rets[a], rets[b].shift(-k)
        pair = pd.concat([x, y], axis=1).dropna()
        out[k] = float(pair.iloc[:, 0].corr(pair.iloc[:, 1])) if len(pair) > 30 else np.nan
    return pd.Series(out, name=f"corr({a}_t, {b}_t+k)")


def ols_beta(y: pd.Series, x: pd.Series) -> float:
    """Univariate OLS slope, NaN-safe."""
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 30:
        return np.nan
    xv = df.iloc[:, 1].to_numpy()
    v = xv.var(ddof=1)
    return float(np.cov(xv, df.iloc[:, 0].to_numpy(), ddof=1)[0, 1] / v) if v > 0 else np.nan


def dimson_beta(asset: pd.Series, market: pd.Series, n_lags: int = 1) -> dict:
    """Dimson (1979) aggregated-coefficients beta: sum of lead, contemporaneous and lag slopes.

    Multiple regression on the market's returns at ``t-n_lags .. t+n_lags``; the beta is the
    sum of the coefficients. With ``n_lags = 0`` it reduces to the ordinary beta, which is how
    the size of the correction is read off.
    """
    # Named explicitly rather than by signed offset: "lag1" is YESTERDAY's market return, and
    # a positive coefficient on it is the whole signature of non-synchronous trading. Signed
    # keys invite exactly the off-by-one that this study is about.
    cols = {"contemp": market}
    for k in range(1, n_lags + 1):
        cols[f"lag{k}"] = market.shift(k)
        cols[f"lead{k}"] = market.shift(-k)
    df = pd.concat([asset.rename("y"), pd.DataFrame(cols)], axis=1).dropna()
    if len(df) < 50:
        return {"beta": np.nan, "n": int(len(df))}
    X = np.column_stack([np.ones(len(df)), df.drop(columns="y").to_numpy()])
    coef, *_ = np.linalg.lstsq(X, df["y"].to_numpy(), rcond=None)
    parts = dict(zip(df.drop(columns="y").columns, coef[1:]))
    return {"beta": float(sum(coef[1:])), "alpha": float(coef[0]), "n": int(len(df)),
            **{k: float(v) for k, v in parts.items()}}


def scholes_williams_beta(asset: pd.Series, market: pd.Series) -> dict:
    """Scholes-Williams (1977) beta: (lag + contemporaneous + lead) / (1 + 2 rho_market)."""
    b_minus = ols_beta(asset, market.shift(1))
    b_zero = ols_beta(asset, market)
    b_plus = ols_beta(asset, market.shift(-1))
    rho = float(market.dropna().autocorr(1))
    beta = (b_minus + b_zero + b_plus) / (1 + 2 * rho) if np.isfinite(rho) else np.nan
    return {"beta": float(beta), "beta_lag": b_minus, "beta_0": b_zero, "beta_lead": b_plus,
            "rho_market": rho}


def aggregated_correlation(prices: pd.DataFrame, a: str, b: str,
                           freqs=FREQS) -> pd.DataFrame:
    """Correlation of ``a`` and ``b`` measured at several sampling intervals."""
    rows = []
    for name, step in freqs.items():
        r = to_returns(prices[[a, b]], step).dropna()
        if len(r) < 40:
            continue
        rows.append({"frequency": name, "step_days": step, "n": int(len(r)),
                     "correlation": float(r[a].corr(r[b]))})
    return pd.DataFrame(rows).set_index("frequency")


def bias_table(prices: pd.DataFrame, market: str, assets, freqs=FREQS) -> pd.DataFrame:
    """Everything, per asset: naive and corrected betas, and correlation across frequencies."""
    rets = to_returns(prices)
    rows = []
    for a in assets:
        agg = aggregated_correlation(prices, a, market, freqs)
        naive_beta = ols_beta(rets[a], rets[market])
        d = dimson_beta(rets[a], rets[market], n_lags=1)
        sw = scholes_williams_beta(rets[a], rets[market])
        row = {"asset": a, "corr_daily": float(rets[a].corr(rets[market])),
               "beta_naive": naive_beta, "beta_dimson": d["beta"],
               "beta_sw": sw["beta"], "lead_coef": d.get("lead1", np.nan),
               "lag_coef": d.get("lag1", np.nan)}
        for f in agg.index:
            row[f"corr_{f}"] = float(agg.loc[f, "correlation"])
        row["corr_lift"] = row.get("corr_monthly", np.nan) - row["corr_daily"]
        row["beta_lift"] = d["beta"] - naive_beta if np.isfinite(naive_beta) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("asset")


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Global minimum-variance weights with a small ridge for numerical safety."""
    n = cov.shape[0]
    inv = np.linalg.pinv(cov + np.eye(n) * 1e-12)
    w = inv @ np.ones(n)
    return w / w.sum()


def portfolio_impact(prices: pd.DataFrame, assets, step_estimate: int = 1,
                     step_truth: int = 21) -> dict:
    """Build a minimum-variance book on one estimation frequency; price it on another.

    The optimiser is given the covariance matrix measured at ``step_estimate`` days (the naive
    daily one, usually) and the resulting weights are then evaluated against the covariance
    measured at ``step_truth`` days, which is the horizon a holder actually cares about and
    the one the timing bias has washed out of. The gap between the promised and the realised
    volatility is what a downward-biased correlation matrix costs.
    """
    cols = list(assets)
    r_est = to_returns(prices[cols], step_estimate).dropna()
    r_true = to_returns(prices[cols], step_truth).dropna()
    cov_est = np.cov(r_est.to_numpy().T, ddof=1) * (TRADING_DAYS / step_estimate)
    cov_true = np.cov(r_true.to_numpy().T, ddof=1) * (TRADING_DAYS / step_truth)
    w_est = min_variance_weights(cov_est)
    w_true = min_variance_weights(cov_true)
    promised = float(np.sqrt(w_est @ cov_est @ w_est))
    delivered = float(np.sqrt(w_est @ cov_true @ w_est))
    best = float(np.sqrt(w_true @ cov_true @ w_true))
    return {"promised_vol": promised, "delivered_vol": delivered, "best_possible_vol": best,
            "understatement": delivered / promised - 1.0,
            "cost_of_bad_matrix": delivered / best - 1.0,
            "weights_estimated": dict(zip(cols, w_est.tolist())),
            "weights_truth": dict(zip(cols, w_true.tolist())),
            "max_weight_gap": float(np.max(np.abs(w_est - w_true)))}


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Real** if the daily-to-monthly correlation lift exceeds 0.10 on at least
      two foreign tapes **and** stays below 0.05 on the same-market control; **Weak** if the
      lift is there without the control staying clean; **None** otherwise.
    - **Usefulness**: **Useful** if the minimum-variance book built on the biased matrix
      understates its own volatility by more than 5%; **Fragile** above 1%; **Mirage** below.
    """
    real = h["n_big_lifts"] >= 2 and abs(h["control_lift"]) < 0.05
    signal = "Real" if real else ("Weak" if h["n_big_lifts"] >= 1 else "None")
    u = abs(h["understatement"])
    trad = "Useful" if u >= 0.05 else ("Fragile" if u >= 0.01 else "Mirage")
    return {
        "signal": signal,
        "signal_why": (
            f"Correlation with the US market rises from **{h['worst_daily_corr']:.2f}** at "
            f"daily frequency to **{h['worst_monthly_corr']:.2f}** at monthly on "
            f"{h['worst_lift_asset']} — a lift of **{h['worst_lift']:+.2f}** — and "
            f"**{h['n_big_lifts']} of {h['n_foreign']}** foreign tapes lift by more than 0.10. "
            f"The same-market control ({h['control_asset']}) lifts by "
            f"**{h['control_lift']:+.2f}**, which is the machinery's own noise floor. Dimson's "
            f"correction recovers most of it without changing frequency: beta on "
            f"{h['worst_lift_asset']} goes from {h['worst_beta_naive']:.2f} to "
            f"**{h['worst_beta_dimson']:.2f}**, and the lagged US coefficient — the smoking "
            f"gun — is {h['worst_lag_coef']:+.2f}."),
        "trad": trad,
        "trad_why": (
            f"A minimum-variance book built on the **daily** covariance matrix promises "
            f"{h['promised_vol']:.2%} annualised and delivers **{h['delivered_vol']:.2%}** at "
            f"the monthly horizon — it understates its own risk by "
            f"**{h['understatement']:+.1%}** — and its weights differ from the ones the "
            f"unbiased matrix would choose by up to {h['max_weight_gap']:.0%}. The fix costs "
            f"nothing: measure at a lower frequency, or add one lead and one lag."),
        "one_sentence": (
            f"Nothing is wrong with the data — the tapes are all New-York-listed and close at "
            f"the same minute — but the *markets underneath* keep different hours, and that "
            f"alone drags a daily correlation down by up to {abs(h['worst_lift']):.2f}, biases "
            f"beta by {h['worst_beta_dimson'] - h['worst_beta_naive']:+.2f} and lets a "
            f"minimum-variance optimiser understate its own volatility by "
            f"{h['understatement']:+.0%}."),
    }
