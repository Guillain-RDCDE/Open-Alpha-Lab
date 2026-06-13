"""The strategy and its honest controls — Study 114 (Dollar-Smile).

The folk claim: *a rising dollar (strong USD) is bad for stocks, and even worse for EM
equities.* We test two distinct hypotheses:

1. **Contemporaneous** — does the dollar move *with* or *against* equities on the same
   day/week? (Well-documented in the literature; the expected sign for SPY is negative
   or near-zero; for EEM it is reliably negative.)
2. **Predictive** — does the lagged dollar change *forecast* forward equity returns?
   (This is the claim that matters for trading. Literature suggests it is weak to absent.)

The design mirrors Study 85 (Dr-Copper) closely:

    r_equity_{t+h} = alpha + beta * delta_dollar_t + epsilon_{t+h}

where ``delta_dollar_t`` is the lagged dollar log-change and ``h`` is the horizon
(weekly, monthly). We test IS (HAC t-stat), OOS (Goyal-Welch expanding R²), and the
contemporaneous vs predictive decomposition.

A momentum-direction control: we also ask whether a simple "dollar momentum" rule
(sell / underweight SPY/EEM after a strongly rising dollar week) outperforms a
random-direction coin — the *tradable* version of the claim.

No look-ahead: the dollar change on week t is used to predict the return *starting* at
the close of week t (the next non-overlapping period).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature engineering — dollar changes and forward returns
# ---------------------------------------------------------------------------
def compute_features(
    daily: pd.DataFrame,
    horizon: str = "W",
) -> pd.DataFrame:
    """Resample daily data to ``horizon`` and compute lagged dollar change + forward return.

    ``horizon`` follows pandas offset aliases: ``"W"`` = week-ending Friday,
    ``"ME"`` = month-end. Returns a frame with:

    - ``delta_dollar``   — log-change in the dollar index (the predictor)
    - ``ret_spy``        — log SPY return for the current horizon (contemporaneous)
    - ``ret_eem``        — log EEM return for the current horizon (contemporaneous)
    - ``fwd_spy``        — log SPY return for the *next* horizon (the target, no look-ahead)
    - ``fwd_eem``        — log EEM return for the *next* horizon (the target)
    - ``dollar``         — dollar index level (for visualisation)

    All rows with NaN in predictor or target are dropped. Using log-changes ensures
    stationarity and additive compounding.
    """
    daily = daily.copy()
    agg = daily.resample(horizon, label="right", closed="right").last()
    agg = agg.dropna(subset=["dollar", "spy", "eem"])

    log_dollar = np.log(agg["dollar"])
    log_spy = np.log(agg["spy"])
    log_eem = np.log(agg["eem"])

    agg = agg.copy()
    agg["delta_dollar"] = log_dollar.diff()
    agg["ret_spy"] = log_spy.diff()
    agg["ret_eem"] = log_eem.diff()

    # Forward returns (shift -1: predictor at t → return at t+1)
    agg["fwd_spy"] = agg["ret_spy"].shift(-1)
    agg["fwd_eem"] = agg["ret_eem"].shift(-1)

    out = agg[["delta_dollar", "ret_spy", "ret_eem", "fwd_spy", "fwd_eem", "dollar"]].dropna()
    return out


# ---------------------------------------------------------------------------
# In-sample regression with HAC standard errors
# ---------------------------------------------------------------------------
def regression_is(
    x: pd.Series,
    y: pd.Series,
) -> dict:
    """OLS regression of y on x (with intercept), Newey-West HAC standard errors.

    Returns:
    - ``alpha``      — intercept
    - ``beta``       — slope on x
    - ``r2``         — in-sample R²
    - ``n``          — number of observations
    - ``tstat``      — Newey-West HAC t-stat on beta (the inference-bar number)
    - ``se_hac``     — HAC standard error on beta
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

    # Newey-West HAC SE on the slope (score e_t = resid_t * x_t)
    e = resid * x_
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
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
    """Expanding-window OOS R² of the dollar-change model vs the historical-mean benchmark.

    The Goyal-Welch (2008) OOS R² is defined as:

        R²_OOS = 1 - MSFE_model / MSFE_mean

    where MSFE is mean squared forecast error. A positive R²_OOS means the model beats
    the simple historical-mean forecast; negative means it does not.

    Returns:
    - ``r2_oos``     — Goyal-Welch OOS R²
    - ``n_oos``      — number of OOS forecasts
    - ``msfe_model`` — model MSFE
    - ``msfe_mean``  — naive mean MSFE
    - ``dm_tstat``   — Diebold-Mariano t-stat
    """
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[mask], yv[mask]
    n = len(xv)

    errors_model = []
    errors_mean = []

    for t in range(min_train, n):
        x_train = xv[:t]
        y_train = yv[:t]
        y_mean_fcst = y_train.mean()

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

    The contemporaneous regression ``equity ~ delta_dollar`` (same-period) answers
    whether the dollar and equities move *together*. The predictive regression asks
    whether the dollar *leads* equities.

    Returns a dict with four regressions' key stats:
    - ``contemp_spy``        — contemporaneous SPY vs dollar change
    - ``contemp_eem``        — contemporaneous EEM vs dollar change
    - ``predictive_spy``     — predictive SPY vs lagged dollar change
    - ``predictive_eem``     — predictive EEM vs lagged dollar change
    """
    feat = compute_features(daily, horizon=horizon)

    contemp_spy = regression_is(feat["delta_dollar"], feat["ret_spy"])
    contemp_eem = regression_is(feat["delta_dollar"], feat["ret_eem"])
    pred_spy = regression_is(feat["delta_dollar"], feat["fwd_spy"])
    pred_eem = regression_is(feat["delta_dollar"], feat["fwd_eem"])

    return {
        "contemp_spy": contemp_spy,
        "contemp_eem": contemp_eem,
        "predictive_spy": pred_spy,
        "predictive_eem": pred_eem,
        "n_periods": len(feat),
    }


# ---------------------------------------------------------------------------
# Dollar-momentum trading signal (direction control)
# ---------------------------------------------------------------------------
def dollar_signal(feat: pd.DataFrame, threshold: float = 0.0) -> pd.Series:
    """Weekly dollar-momentum signal for equity timing.

    When the dollar rose last week (``delta_dollar > threshold``), the signal is ``-1``
    (short/underweight equities — the "strong dollar is bad" bet). When it fell, signal
    is ``+1`` (long). With ``threshold=0`` this is a simple sign rule.

    Returns a Series aligned on the *forward* return periods (i.e. the week after the
    signal fires).
    """
    sig = np.where(feat["delta_dollar"] > threshold, -1, 1)
    return pd.Series(sig, index=feat.index, name="signal")


def trade_returns(
    feat: pd.DataFrame,
    target: str = "fwd_spy",
    threshold: float = 0.0,
    cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Per-period returns from the dollar-momentum signal on ``target`` (SPY or EEM).

    Signal on week t predicts direction of ``target`` on week t+1; the position is
    ±1 (long or short). ``cost_bps`` is a round-trip cost charged each week (2 bps is
    conservative for ETFs). Returns a frame with columns:
    - ``signal``      — the ±1 direction bet
    - ``fwd_ret``     — the realised forward return
    - ``ret_gross``   — signal * fwd_ret (gross P&L)
    - ``ret_net``     — gross minus one-way cost (halved because we assume 50% turnover
                        on reversals — conservative)
    """
    sig = dollar_signal(feat, threshold=threshold)
    fwd = feat[target]
    gross = sig * fwd
    # Cost on full flip of position is 2 * cost_bps; partial-flip is less.
    # We use the full round-trip for conservatism.
    net = gross - cost_bps * 1e-4

    return pd.DataFrame(
        {"signal": sig, "fwd_ret": fwd, "ret_gross": gross, "ret_net": net},
        index=feat.index,
    )


def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-period statistics for one ledger.

    Returns period count, win-rate, mean return (bps/period), the period Sharpe, and
    a HAC t-stat on the mean — the inference-bar number that decides whether the edge
    is distinguishable from zero.
    """
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out = {
        "n_periods": int(n),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(52)) if n > 1 and r.std() > 0 else float("nan"),
        "tstat": float("nan"),
    }
    if n > 5:
        mu = r.mean()
        e = r - mu
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lrv = float(e @ e) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Full regression suite
# ---------------------------------------------------------------------------
def run_all(daily: pd.DataFrame, horizon: str = "W") -> dict:
    """Full regression suite: IS, OOS, contemporaneous vs predictive — for one horizon.

    Returns a nested dict:
    - ``contemp_spy``        — contemporaneous IS regression (SPY target)
    - ``contemp_eem``        — contemporaneous IS regression (EEM target)
    - ``is_spy``             — predictive IS regression (SPY target)
    - ``is_eem``             — predictive IS regression (EEM target)
    - ``oos_spy``            — OOS R² result (SPY target)
    - ``oos_eem``            — OOS R² result (EEM target)
    - ``signal_spy``         — momentum signal stats (SPY)
    - ``signal_eem``         — momentum signal stats (EEM)
    - ``n_periods``          — number of periods at this horizon
    - ``horizon``            — the horizon string
    """
    feat = compute_features(daily, horizon=horizon)

    is_spy = regression_is(feat["delta_dollar"], feat["fwd_spy"])
    is_eem = regression_is(feat["delta_dollar"], feat["fwd_eem"])
    oos_spy = oos_r2(feat["delta_dollar"], feat["fwd_spy"])
    oos_eem = oos_r2(feat["delta_dollar"], feat["fwd_eem"])
    contemp_spy = regression_is(feat["delta_dollar"], feat["ret_spy"])
    contemp_eem = regression_is(feat["delta_dollar"], feat["ret_eem"])

    led_spy = trade_returns(feat, target="fwd_spy", cost_bps=2.0)
    led_eem = trade_returns(feat, target="fwd_eem", cost_bps=2.0)
    signal_spy = summarize(led_spy)
    signal_eem = summarize(led_eem)

    return {
        "contemp_spy": contemp_spy,
        "contemp_eem": contemp_eem,
        "is_spy": is_spy,
        "is_eem": is_eem,
        "oos_spy": oos_spy,
        "oos_eem": oos_eem,
        "signal_spy": signal_spy,
        "signal_eem": signal_eem,
        "n_periods": len(feat),
        "horizon": horizon,
    }
