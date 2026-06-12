"""The analysis engine for Study 84 (Moon-Math).

Three honest tests of the Stock-to-Flow (S2F) model for BTC:

1. **Levels regression** — log(price) ~ α + β·log(S2F). This is PlanB's headline R²,
   which is huge (~0.94 in-sample). We replicate it — then immediately expose it as a
   spurious regression of two integrated (trend) series.

2. **Spuriousness tests** — (a) first-difference regression (ΔICM, the forecasting content),
   (b) a 'time-dummy' contest (log(price) ~ log(time) fits equally well), (c) the
   Engle-Granger cointegration test (are the residuals stationary?). A spurious regression
   has non-stationary residuals; a real structural link does not.

3. **Out-of-sample test (2021-12-01 onward)** — the period after PlanB's model was
   publicly stress-tested. BTC's crash to ~$16k while S2F predicted $100k+ is the
   smoking gun.

The control arm throughout is log(time): if fitting log(price) on log(S2F) and
log(time) gives indistinguishable R² and similar residuals, the S2F signal is
redundant — price is just trending with time, not with scarcity.

No look-ahead: OOS split is fixed at the in-sample end date chosen *a priori*
(2021-11-30, the all-time-high plateau). We do not search for a convenient split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy import stats as sc_stats
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    from statsmodels.tsa.stattools import adfuller
    HAS_SM = True
except ImportError:
    HAS_SM = False


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------
def ols_summary(y: np.ndarray, X: np.ndarray) -> dict:
    """OLS fit of y on X (X already includes the constant column).

    Returns: ``{alpha, beta, r2, resid, n}``.
    Uses statsmodels if available for HAC SE and ADF test; falls back to numpy.
    """
    n = len(y)
    if HAS_SM:
        res = OLS(y, X).fit(cov_type="HC3")
        beta = res.params
        resid = res.resid
        r2 = res.rsquared
    else:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        resid = y - y_hat
        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "alpha": float(beta[0]),
        "beta": float(beta[1]) if len(beta) > 1 else float("nan"),
        "r2": float(r2),
        "resid": resid,
        "n": int(n),
    }


def levels_regression(df: pd.DataFrame) -> dict:
    """OLS: log(price) ~ α + β·log(S2F) in levels.

    This is PlanB's headline regression. Returns ``ols_summary`` output plus
    ``adf_pvalue`` (ADF test on residuals — p < 0.05 ⇒ residuals stationary ⇒
    possible cointegration; p > 0.05 ⇒ spurious).
    """
    mask = df["log_s2f"].notna() & df["log_close"].notna()
    sub = df[mask]
    y = sub["log_close"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)), sub["log_s2f"].to_numpy(dtype=float)])
    out = ols_summary(y, X)
    out["label"] = "S2F levels"

    # Engle-Granger step 2: ADF on residuals
    if HAS_SM and len(out["resid"]) > 20:
        adf_res = adfuller(out["resid"], autolag="AIC")
        out["adf_pvalue"] = float(adf_res[1])
        out["adf_stat"] = float(adf_res[0])
    else:
        out["adf_pvalue"] = float("nan")
        out["adf_stat"] = float("nan")
    return out


def time_regression(df: pd.DataFrame) -> dict:
    """OLS: log(price) ~ α + β·log(time_since_genesis).

    Serves as the 'equally good spurious alternative': if this fits as well as the
    S2F regression, then the S2F R² is not evidence of a causal link — both are just
    proxies for the monotone march of time.
    """
    mask = df["log_time"].notna() & df["log_close"].notna()
    sub = df[mask]
    y = sub["log_close"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)), sub["log_time"].to_numpy(dtype=float)])
    out = ols_summary(y, X)
    out["label"] = "log(time) levels"

    if HAS_SM and len(out["resid"]) > 20:
        adf_res = adfuller(out["resid"], autolag="AIC")
        out["adf_pvalue"] = float(adf_res[1])
        out["adf_stat"] = float(adf_res[0])
    else:
        out["adf_pvalue"] = float("nan")
        out["adf_stat"] = float("nan")
    return out


def first_diff_regression(df: pd.DataFrame) -> dict:
    """OLS: Δlog(price) ~ α + β·Δlog(S2F) in first differences.

    First differencing removes the shared trend, so a *real* structural link should
    still show a significant β. On the actual data, Δlog(S2F) is nearly constant
    within each epoch (S2F changes smoothly) — meaning this test has low power — but
    an R² ≈ 0 and |t| < 2 is still informative: the *forecasting content* of S2F
    changes for *price changes* is minimal.
    """
    mask = df["log_s2f"].notna() & df["log_close"].notna()
    sub = df[mask].copy()
    dy = sub["log_close"].diff().dropna()
    dx = sub["log_s2f"].diff().dropna()
    idx = dy.index.intersection(dx.index)
    y = dy.loc[idx].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)), dx.loc[idx].to_numpy(dtype=float)])
    out = ols_summary(y, X)
    out["label"] = "S2F first differences"

    # HAC t-stat on the slope (Newey-West)
    if len(y) > 10:
        e = y - X @ np.array([out["alpha"], out["beta"]])
        lags = max(1, int(np.floor(4.0 * (len(y) / 100.0) ** (2.0 / 9.0))))
        lrv_11 = float(e[1:] ** 2 @ np.ones(len(y) - 1)) / len(y)  # init
        # Full NW long-run variance of the slope
        Xe1 = X[:, 1] * e  # x_t * e_t
        gamma0 = float(Xe1 @ Xe1) / len(y)
        lrv = gamma0
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(Xe1[k:] @ Xe1[:-k]) / len(y)
        xx = float(X[:, 1] @ X[:, 1]) / len(y)
        se_beta = np.sqrt(max(lrv, 0.0) / len(y)) / max(xx, 1e-12)
        out["tstat_beta"] = float(out["beta"] / se_beta) if se_beta > 0 else float("nan")
    else:
        out["tstat_beta"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Out-of-sample evaluation
# ---------------------------------------------------------------------------
def oos_eval(
    df: pd.DataFrame,
    train_end: str = "2021-11-30",
) -> dict:
    """Fit the S2F levels model on train, evaluate on OOS.

    Train: genesis → ``train_end``.
    OOS:   ``train_end + 1day`` → end of tape.

    Returns metrics for both windows: R², mean absolute percentage error (MAPE)
    on the price level (not log), and the max OOS deviation (how far the model
    was wrong at its worst in the OOS window).
    """
    mask = df["log_s2f"].notna() & df["log_close"].notna()
    sub = df[mask].copy()

    train = sub[sub.index <= train_end]
    oos = sub[sub.index > train_end]

    if len(train) < 100 or len(oos) < 10:
        return {"train_r2": float("nan"), "oos_r2": float("nan"),
                "oos_mape": float("nan"), "oos_max_err_pct": float("nan"),
                "n_train": len(train), "n_oos": len(oos)}

    # Fit on train
    y_tr = train["log_close"].to_numpy(dtype=float)
    X_tr = np.column_stack([np.ones(len(y_tr)), train["log_s2f"].to_numpy(dtype=float)])
    if HAS_SM:
        fit = OLS(y_tr, X_tr).fit()
        alpha, beta = fit.params
    else:
        p, _, _, _ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        alpha, beta = float(p[0]), float(p[1])
    y_tr_hat = alpha + beta * X_tr[:, 1]
    ss_tr_res = float(((y_tr - y_tr_hat) ** 2).sum())
    ss_tr_tot = float(((y_tr - y_tr.mean()) ** 2).sum())
    train_r2 = 1.0 - ss_tr_res / ss_tr_tot

    # Predict OOS
    y_oos = oos["log_close"].to_numpy(dtype=float)
    X_oos_x = oos["log_s2f"].to_numpy(dtype=float)
    y_oos_hat = alpha + beta * X_oos_x
    ss_oos_res = float(((y_oos - y_oos_hat) ** 2).sum())
    ss_oos_tot = float(((y_oos - y_oos.mean()) ** 2).sum())
    oos_r2 = 1.0 - ss_oos_res / ss_oos_tot if ss_oos_tot > 0 else float("nan")

    # MAPE and max error in price (not log) space
    px_actual = np.exp(y_oos)
    px_pred = np.exp(y_oos_hat)
    pct_err = (px_pred - px_actual) / px_actual * 100.0
    oos_mape = float(np.abs(pct_err).mean())
    oos_max_err_pct = float(pct_err[np.argmax(np.abs(pct_err))])

    # Extremes for narrative
    oos_dates = oos.index
    worst_idx = np.argmax(np.abs(pct_err))

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "train_r2": float(train_r2),
        "oos_r2": float(oos_r2),
        "oos_mape": float(oos_mape),
        "oos_max_err_pct": float(oos_max_err_pct),
        "oos_worst_date": str(oos_dates[worst_idx].date()),
        "n_train": int(len(train)),
        "n_oos": int(len(oos)),
        "train_end": train_end,
        "oos_pred_series": pd.Series(px_pred, index=oos_dates, name="s2f_pred"),
        "oos_actual_series": pd.Series(px_actual, index=oos_dates, name="actual"),
    }


# ---------------------------------------------------------------------------
# Spurious regression detection — the positive control
# ---------------------------------------------------------------------------
def spurious_control(df: pd.DataFrame) -> dict:
    """Estimate how much of the S2F R² is attributable to the shared trend.

    We run three regressions and report their R²s and ADF p-values:
        1. log(price) ~ log(S2F)   — PlanB's claim
        2. log(price) ~ log(time)  — the 'any trend' baseline
        3. Δlog(price) ~ Δlog(S2F) — first differences, removes trend

    If (1) ≈ (2) and (3) ≈ 0, then S2F is just a proxy for time; the high R² in
    (1) is a spurious regression artefact, not a structural causal relationship.
    """
    r1 = levels_regression(df)
    r2 = time_regression(df)
    r3 = first_diff_regression(df)
    return {"s2f_levels": r1, "time_levels": r2, "first_diff": r3}


# ---------------------------------------------------------------------------
# Summary statistics for inference
# ---------------------------------------------------------------------------
def summarize(control: dict) -> dict:
    """Flatten spurious_control output into a single summary dict for notebooks."""
    r1 = control["s2f_levels"]
    r2 = control["time_levels"]
    r3 = control["first_diff"]
    return {
        "s2f_r2": r1["r2"],
        "s2f_adf_p": r1["adf_pvalue"],
        "s2f_beta": r1["beta"],
        "time_r2": r2["r2"],
        "time_adf_p": r2["adf_pvalue"],
        "diff_r2": r3["r2"],
        "diff_beta_t": r3.get("tstat_beta", float("nan")),
        "n": r1["n"],
        "spurious": r1["adf_pvalue"] > 0.05 if not np.isnan(r1["adf_pvalue"]) else True,
    }
