"""Signal and analysis for Study 120 (Excess-CAPE-Yield).

The study asks: does ECY (Shiller's 1/CAPE minus the real 10-year bond yield)
forecast the *next 10-year equity-minus-bond excess return* better than the plain
earnings yield (1/CAPE) alone — and if so, is that just in-sample or also
out-of-sample?

Three claims under test
-----------------------
H1 (long-horizon signal)  ECY strongly forecasts the 10-year excess return;
                          the in-sample R² and the OLS slope are large and
                          significant on non-overlapping windows.
H2 (ECY beats 1/CAPE)    Bond-yield adjustment (the "excess" part) makes ECY a
                          materially better predictor than the plain earnings
                          yield, because it controls for the varying risk-free
                          hurdle investors face.
H3 (out-of-sample)        The relationship survives a 50/50 in-sample / OOS
                          split on non-overlapping 10-year windows.

Control
-------
We pin the signal against the plainest benchmark: "always hold equities" — the
unconditional mean 10-year excess return.  An OOS R² > 0 means ECY beats the
historical-mean forecast.

Inference caveat on overlapping returns
---------------------------------------
Monthly observations overlap 120-fold for 10-year returns, so the effective
sample size is ~n_monthly / 120.  All headline t-statistics are computed on
*non-overlapping* 10-year windows to avoid the Valkanov (2003) size distortion.
Monthly R² is reported as context only; the non-overlapping OLS t drives the
Signal verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core regression helpers
# ---------------------------------------------------------------------------
def ols_regression(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS of y on x (with intercept) — returns slope, intercept, R², residuals.

    Standard errors are computed with OLS formula (assuming i.i.d. residuals).
    For overlapping windows call :func:`nonoverlay_regression` which sub-samples
    first so the i.i.d. assumption is reasonable.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "tstat": np.nan, "n": n}
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    resid = y - y_hat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    se = float(np.sqrt(ss_res / (n - 2) / np.sum((x - x.mean()) ** 2))) if n > 2 else np.nan
    tstat = float(slope / se) if se and se > 0 else np.nan
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "se": se,
        "tstat": tstat,
        "n": n,
        "resid": resid,
        "x": x,
        "y": y,
    }


def nonoverlay_regression(
    df: pd.DataFrame,
    signal_col: str,
    outcome_col: str,
    step: int = 120,
) -> dict:
    """Regression on non-overlapping sub-sampled observations.

    Takes every ``step``-th row of ``df`` to produce (approximately)
    non-overlapping outcome windows, then runs OLS.  The t-statistic is valid
    under mild heteroskedasticity; use this for the inference bar.

    Returns the same dict as :func:`ols_regression` plus ``step``.
    """
    sub = df[[signal_col, outcome_col]].dropna().iloc[::step].copy()
    result = ols_regression(sub[signal_col].values, sub[outcome_col].values)
    result["step"] = step
    return result


# ---------------------------------------------------------------------------
# Out-of-sample R² (Goyal & Welch 2008 convention)
# ---------------------------------------------------------------------------
def oos_r2(
    df: pd.DataFrame,
    signal_col: str,
    outcome_col: str,
    step: int = 120,
    train_frac: float = 0.5,
) -> dict:
    """Pseudo-OOS R² on the non-overlapping sub-sample.

    Fits on the first ``train_frac`` share of non-overlapping windows, predicts
    on the rest.  OOS R² = 1 − SS_OOS / SS_hist, where SS_hist is the squared
    error of the *historical-mean* benchmark (Goyal & Welch 2008).  Positive
    means ECY beats the constant-mean forecast; negative means it's beaten by
    a naïve "mean always" rule.

    Returns ``oos_r2``, ``train_n``, ``test_n``, and the arrays of actual and
    predicted OOS values for inspection.
    """
    sub = df[[signal_col, outcome_col]].dropna().iloc[::step].copy()
    n = len(sub)
    n_train = max(2, int(np.ceil(n * train_frac)))
    train = sub.iloc[:n_train]
    test = sub.iloc[n_train:]
    if len(test) < 2:
        return {"oos_r2": np.nan, "train_n": n_train, "test_n": len(test)}
    slope_tr, ic_tr = np.polyfit(train[signal_col].values, train[outcome_col].values, 1)
    y_pred = slope_tr * test[signal_col].values + ic_tr
    y_test = test[outcome_col].values
    # Benchmark: historical mean of the *training* target
    mu_hist = float(train[outcome_col].mean())
    y_hist = np.full_like(y_test, mu_hist)
    ss_model = float(np.sum((y_test - y_pred) ** 2))
    ss_hist = float(np.sum((y_test - y_hist) ** 2))
    r2 = 1.0 - ss_model / ss_hist if ss_hist > 0 else np.nan
    return {
        "oos_r2": float(r2),
        "train_n": int(n_train),
        "test_n": int(len(test)),
        "y_test": y_test,
        "y_pred_model": y_pred,
        "y_pred_hist": y_hist,
        "train_slope": float(slope_tr),
        "train_intercept": float(ic_tr),
    }


# ---------------------------------------------------------------------------
# Valuation buckets — the monotone ladder
# ---------------------------------------------------------------------------
def bucket_returns(
    df: pd.DataFrame,
    signal_col: str,
    outcome_col: str,
    n_buckets: int = 5,
) -> pd.Series:
    """Mean forward outcome by signal quintile (low ECY → high ECY).

    Returns a Series indexed Q1..Q5, values are annualised mean excess returns.
    Useful for showing the monotone relationship without assuming linearity.
    """
    sub = df[[signal_col, outcome_col]].dropna()
    labels = [f"Q{i+1}" for i in range(n_buckets)]
    sub = sub.copy()
    sub["bucket"] = pd.qcut(sub[signal_col], n_buckets, labels=labels)
    return sub.groupby("bucket", observed=True)[outcome_col].mean()


# ---------------------------------------------------------------------------
# Summary: headline statistics for results.md
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> dict:
    """Compute all headline metrics: in-sample (full monthly + non-overlap), OOS R².

    Returns a flat dict of floats ready to drop into the R dict and docs/results.md.
    """
    # -- Full monthly (context only; overlapping) --
    monthly = df[["ecy", "fwd10_excess", "earnings_yield", "fwd1_excess"]].dropna()

    full_ecy = ols_regression(monthly["ecy"].values, monthly["fwd10_excess"].values)
    full_ey = ols_regression(monthly["earnings_yield"].values, monthly["fwd10_excess"].values)
    ecy_1yr = ols_regression(monthly["ecy"].values, monthly["fwd1_excess"].values)

    # -- Non-overlapping 10-year windows (inference bar) --
    no_ecy = nonoverlay_regression(monthly, "ecy", "fwd10_excess", step=120)
    no_ey = nonoverlay_regression(monthly, "earnings_yield", "fwd10_excess", step=120)

    # -- OOS (non-overlapping, 50/50 split) --
    oos_ecy = oos_r2(monthly, "ecy", "fwd10_excess", step=120, train_frac=0.5)

    # -- ECY level stats --
    ecy_mean = float(monthly["ecy"].mean())
    ecy_std = float(monthly["ecy"].std())
    n_pos = int((monthly["ecy"] > 0).sum())
    mean_excess_pos = float(
        monthly.loc[monthly["ecy"] > 0, "fwd10_excess"].mean()
    )
    mean_excess_neg = float(
        monthly.loc[monthly["ecy"] <= 0, "fwd10_excess"].mean()
    )

    return {
        # sample dimensions
        "n_monthly": int(len(monthly)),
        "n_nonoverlap_10yr": int(no_ecy["n"]),
        "year_start": int(monthly.index[0].year),
        "year_end": int(monthly.index[-1].year),
        # full monthly R² (overlapping — context only)
        "r2_ecy_monthly": float(full_ecy["r2"]),
        "r2_ey_monthly": float(full_ey["r2"]),
        "r2_ecy_1yr_monthly": float(ecy_1yr["r2"]),
        "corr_ecy_monthly": float(np.sqrt(max(full_ecy["r2"], 0.0))) * np.sign(full_ecy["slope"]),
        # non-overlapping 10yr (inference bar)
        "r2_ecy_nonoverlap": float(no_ecy["r2"]),
        "r2_ey_nonoverlap": float(no_ey["r2"]),
        "slope_ecy_nonoverlap": float(no_ecy["slope"]),
        "slope_ey_nonoverlap": float(no_ey["slope"]),
        "tstat_ecy_nonoverlap": float(no_ecy["tstat"]),
        "tstat_ey_nonoverlap": float(no_ey["tstat"]),
        # OOS R²
        "oos_r2_ecy": float(oos_ecy["oos_r2"]),
        "oos_train_n": int(oos_ecy["train_n"]),
        "oos_test_n": int(oos_ecy["test_n"]),
        # ECY distributional stats
        "ecy_mean_pct": float(ecy_mean * 100),
        "ecy_std_pct": float(ecy_std * 100),
        "frac_pos_ecy": float(n_pos / len(monthly) * 100),
        "mean_excess_when_ecy_pos_pct": float(mean_excess_pos * 100),
        "mean_excess_when_ecy_neg_pct": float(mean_excess_neg * 100),
    }
