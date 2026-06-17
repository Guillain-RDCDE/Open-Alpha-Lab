"""The strategy and its honest controls — Study 245 (Oil-Equity-Correlation).

The folk claim: *crude oil prices lead the stock market* — when oil rises, stocks soon
follow (growth optimism) or fall (cost-push inflation). We test this as a predictive
regression:

    r_equity_{t+h} = alpha + beta * r_oil_t + epsilon_{t+h}

where ``r_oil_t`` is the lagged oil log-return and ``h`` is the forecast horizon
(weekly, monthly). The study asks three questions:

1. **Contemporaneous (C)** — do oil and equities move *together* in the same period?
   This is the visually compelling chart. It is real but untradable.
2. **In-sample (IS)** — does beta carry a statistically significant HAC t-stat on the
   full sample? Sign matters: positive (oil up → equities up) is the 'growth' channel;
   negative (oil up → equities down) is the 'cost-push' channel.
3. **Out-of-sample (OOS)** — does the regression beat a naive (zero-mean) forecast using
   expanding-window OOS R², à la Goyal & Welch (2008)?

The naive baseline is the historical mean return (the Goyal-Welch benchmark). A positive
OOS R² means the oil model beats the mean; a negative one means it does not.

No look-ahead: the oil return on week t is used to predict the equity return *starting*
at the close of week t (the next non-overlapping period).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature engineering — oil returns and forward equity returns
# ---------------------------------------------------------------------------
def compute_features(
    daily: pd.DataFrame,
    horizon: str = "W",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Resample daily data to ``horizon`` and compute lagged oil return + forward equity return.

    ``horizon`` follows pandas offset aliases: ``"W"`` = week-ending Friday,
    ``"ME"`` = month-end.  Returns a frame with:

    - ``ret_oil``       — log oil return of the current period (the predictor)
    - ``fwd_equity``    — log equity return for the *next* horizon period (the target)
    - ``ret_equity``    — log equity return of the current period (contemporaneous)

    All rows with NaN in predictor or target are dropped. Using log-returns ensures
    stationarity.
    """
    daily = daily.copy()
    agg = daily.resample(horizon, label="right", closed="right").last()
    agg = agg.dropna(subset=["oil", "equity"])

    # Log returns
    log_oil = np.log(agg["oil"])
    log_equity = np.log(agg["equity"])

    agg = agg.copy()
    agg["ret_oil"] = log_oil.diff()
    agg["ret_equity"] = log_equity.diff()

    # Forward returns (shift -1 so the predictor aligns with the *next* period's return)
    agg["fwd_equity"] = agg["ret_equity"].shift(-1)

    out = agg[["ret_oil", "ret_equity", "fwd_equity"]].dropna()
    return out


# ---------------------------------------------------------------------------
# In-sample regression
# ---------------------------------------------------------------------------
def regression_is(
    x: pd.Series,
    y: pd.Series,
) -> dict:
    """OLS regression of y on x (no intercept suppression), with HAC standard errors.

    Returns:
    - ``alpha``       — intercept
    - ``beta``        — slope on x
    - ``r2``          — in-sample R²
    - ``n``           — number of observations
    - ``tstat``       — Newey-West HAC t-stat on beta (the inference-bar number)
    - ``se_hac``      — HAC standard error on beta
    - ``tstat_naive`` — naive (non-HAC) t-stat for reference
    """
    r = np.asarray(y, dtype=float)
    x_ = np.asarray(x, dtype=float)
    mask = np.isfinite(r) & np.isfinite(x_)
    r, x_ = r[mask], x_[mask]
    n = r.size
    if n < 5:
        nan = float("nan")
        return {k: nan for k in ("alpha", "beta", "r2", "n", "tstat", "se_hac", "tstat_naive")}

    # OLS
    X = np.column_stack([np.ones(n), x_])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ (X.T @ r)
    alpha, beta = float(coef[0]), float(coef[1])

    r_hat = alpha + beta * x_
    resid = r - r_hat
    ss_res = float(resid @ resid)
    ss_tot = float(((r - r.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Naive SE
    s2 = ss_res / max(n - 2, 1)
    se_naive = float(np.sqrt(s2 * XtX_inv[1, 1]))
    tstat_naive = beta / se_naive if se_naive > 0 else float("nan")

    # Newey-West HAC SE on the slope
    e = resid * x_  # score for beta
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    # Sandwich estimator for the slope only
    xx = float(x_ @ x_) / n
    se_hac = float(np.sqrt(max(lrv, 0.0) / n) / xx) if xx > 0 else float("nan")
    tstat_hac = beta / se_hac if se_hac > 0 else float("nan")

    return {
        "alpha": alpha,
        "beta": beta,
        "r2": float(r2),
        "n": int(n),
        "tstat": float(tstat_hac),
        "se_hac": float(se_hac),
        "tstat_naive": float(tstat_naive),
    }


# ---------------------------------------------------------------------------
# Out-of-sample R² (Goyal-Welch 2008 benchmark)
# ---------------------------------------------------------------------------
def oos_r2(
    x: pd.Series,
    y: pd.Series,
    min_train: int = 52,
) -> dict:
    """Expanding-window OOS R² of the oil-return model vs the historical-mean benchmark.

    The Goyal-Welch (2008) OOS R² is defined as:

        R²_OOS = 1 - MSFE_model / MSFE_mean

    where MSFE is mean squared forecast error. A positive R²_OOS means the model beats
    the simple historical-mean forecast; negative means it does not.

    ``min_train`` is the minimum number of periods required before generating a forecast
    (here default 52 weeks = 1 year).

    Returns:
    - ``r2_oos``      — Goyal-Welch OOS R²
    - ``n_oos``       — number of OOS forecasts
    - ``msfe_model``  — model MSFE
    - ``msfe_mean``   — naive mean MSFE
    - ``dm_tstat``    — Diebold-Mariano t-stat
    """
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[mask], yv[mask]
    n = len(xv)

    errors_model = []
    errors_mean = []

    for t in range(min_train, n):
        # Expanding window [0..t-1] → forecast y[t]
        x_train = xv[:t]
        y_train = yv[:t]
        y_mean_fcst = y_train.mean()

        # OLS on expanding window
        X = np.column_stack([np.ones(t), x_train])
        try:
            coef = np.linalg.lstsq(X, y_train, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        y_model_fcst = coef[0] + coef[1] * xv[t]

        y_actual = yv[t]
        errors_model.append((y_actual - y_model_fcst) ** 2)
        errors_mean.append((y_actual - y_mean_fcst) ** 2)

    if not errors_model:
        nan = float("nan")
        return {k: nan for k in ("r2_oos", "n_oos", "msfe_model", "msfe_mean", "dm_tstat")}

    em = np.array(errors_model)
    en = np.array(errors_mean)
    msfe_m = float(em.mean())
    msfe_n = float(en.mean())
    r2_oos_val = float(1.0 - msfe_m / msfe_n) if msfe_n > 0 else float("nan")

    # DM test on error differences
    diff = en - em
    mu_d = diff.mean()
    lags = int(np.floor(4.0 * (len(diff) / 100.0) ** (2.0 / 9.0)))
    e = diff - mu_d
    lrv = float(e @ e) / len(e)
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / len(e)
    se_d = float(np.sqrt(max(lrv, 0.0) / len(e)))
    dm_tstat = float(mu_d / se_d) if se_d > 0 else float("nan")

    return {
        "r2_oos": r2_oos_val,
        "n_oos": len(em),
        "msfe_model": msfe_m,
        "msfe_mean": msfe_n,
        "dm_tstat": dm_tstat,
    }


# ---------------------------------------------------------------------------
# Contemporaneous vs predictive decomposition
# ---------------------------------------------------------------------------
def contemporaneous_vs_predictive(daily: pd.DataFrame, horizon: str = "W") -> dict:
    """Compare contemporaneous and lagged (predictive) regression R² and t-stats.

    The contemporaneous regression ``equity ~ ret_oil`` (same-period) answers
    whether oil and equities move *together*. The predictive regression (computed
    by ``regression_is``) asks whether oil *leads* equities.

    Returns a dict with both regressions' key stats.
    """
    feat = compute_features(daily, horizon=horizon)

    # Contemporaneous: same-period ret_equity ~ same-period ret_oil
    contemp = regression_is(feat["ret_oil"], feat["ret_equity"])
    # Predictive: next-period fwd_equity ~ ret_oil
    pred = regression_is(feat["ret_oil"], feat["fwd_equity"])

    return {
        "contemporaneous": contemp,
        "predictive_equity": pred,
        "n_periods": len(feat),
    }


# ---------------------------------------------------------------------------
# Summarize all regressions in one call
# ---------------------------------------------------------------------------
def summarize(daily: pd.DataFrame, horizon: str = "W") -> dict:
    """Full regression suite: IS, OOS, contemporaneous — for one horizon.

    Returns a nested dict:
    - ``is_equity``     — IS regression (equity target, predictive)
    - ``oos_equity``    — OOS R² result (equity target)
    - ``contemp``       — contemporaneous regression stats
    - ``n_periods``     — number of periods at this horizon
    - ``horizon``       — the horizon string
    """
    feat = compute_features(daily, horizon=horizon)

    is_eq = regression_is(feat["ret_oil"], feat["fwd_equity"])
    oos_eq = oos_r2(feat["ret_oil"], feat["fwd_equity"])
    contemp = regression_is(feat["ret_oil"], feat["ret_equity"])

    return {
        "is_equity": is_eq,
        "oos_equity": oos_eq,
        "contemp": contemp,
        "n_periods": len(feat),
        "horizon": horizon,
    }
