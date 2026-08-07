"""Strategy + inference for Study 825 — Oil Predicts Equities.

The claim (Driesprong, Jacobsen & Maat 2008, *"Striking Oil: Another Puzzle?"*): a change
in the **oil price this month** predicts **next month's equity return** *negatively* —
oil is a slow-diffusing macro shock, so a rising oil price is a headwind for stocks one
month out. We test the self-contained monthly version:

    r_equity[t+1] = alpha + beta * r_oil[t] + eps ,   beta expected < 0 ,

a single-regressor predictive regression of the S&P 500 (SPY) *forward* one-month return
on the *trailing* one-month oil (USO) return.

This is distinct from:

* [245-oil-equity-correlation](../../245-oil-equity-correlation/) — the **contemporaneous**
  same-period oil↔equity co-movement (``corr(r_oil[t], r_equity[t])``), a real but
  untradable chart. Here the oil return and the equity return are one month **apart**
  (a genuine forecast, one documented lag).
* [226-crude-seasonality](../../226-crude-seasonality/) — crude's **calendar** seasonality
  (month-of-year effects in the oil price itself), not oil→equity cross-prediction.
* [85-dr-copper](../../85-dr-copper/) — **copper** as a growth barometer for equities, a
  different commodity and a positive (pro-cyclical) reading; oil here predicts equities
  *negatively*.

Method:

* **Month-end resample.** Daily adjusted close → month-end levels → monthly simple
  returns for oil and equity.
* **Predictive alignment, one documented lag.** The predictor is the trailing-month oil
  return known at the close of month ``t``; the target is the equity return realised over
  the *next* month ``t+1`` (``equity.shift(-1)``). Zero look-ahead.
* **Inference.** OLS slope with a **Newey-West (HAC)** *t* (the honest headline); the
  in-sample R²; a permutation placebo that shuffles the predictor→target link; a costed
  monthly timer that trades on the sign of the forecast; a seeded synthetic positive
  control. Newey-West / one-sample / Welch / Wilson primitives mirror the desk house
  style (study 803).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Returns + the predictive regression frame
# --------------------------------------------------------------------------- #
def monthly_returns(series: pd.DataFrame) -> pd.DataFrame:
    """Month-end simple returns for oil (USO) and equity (SPY).

    ``series`` is the daily adjusted-close frame (columns ``USO``, ``SPY``). We take the
    last close of each calendar month and its simple pct-change. Returns a frame with
    columns ``oil`` and ``equity`` indexed by month-end.
    """
    m = series.resample("ME").last().dropna()
    out = pd.DataFrame({
        "oil": m["USO"].pct_change(),
        "equity": m["SPY"].pct_change(),
    }).dropna()
    return out


def regression_frame(series: pd.DataFrame) -> pd.DataFrame:
    """Align the trailing-month oil return (predictor ``x``) with the forward-month equity
    return (target ``y``).

    ``x`` on month ``t`` = oil return over month ``t`` (known at the close of ``t``).
    ``y`` on month ``t`` = equity return over month ``t+1`` (``equity.shift(-1)``), the
    forecast target. One documented execution lag, zero look-ahead. Rows with a missing
    predictor or target (the last month has no forward return) are dropped.
    """
    m = monthly_returns(series)
    frame = pd.DataFrame({
        "x": m["oil"],
        "y": m["equity"].shift(-1),
        "equity_now": m["equity"],
    }).dropna()
    return frame


# --------------------------------------------------------------------------- #
# Inference primitives (mirror study 803's house style)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 6) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        var += 2.0 * w * cov
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


def newey_west_ols(x: np.ndarray, y: np.ndarray, lags: int = 6) -> dict:
    """Single-regressor OLS ``y = alpha + beta*x + eps`` with a Newey-West (HAC) *t* on
    the slope ``beta``.

    Closed-form OLS plus a HAC (Bartlett) sandwich variance of the slope, so we depend
    only on numpy. Returns ``alpha``, ``beta``, the OLS-*t* and the NW-*t* on the slope,
    the R², and ``n``. This is the study's honest headline: sign + magnitude + HAC *t*.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 4:
        return {"alpha": float("nan"), "beta": float("nan"), "t_ols": float("nan"),
                "t_nw": float("nan"), "r2": float("nan"), "n": n}
    xbar, ybar = x.mean(), y.mean()
    sxx = float(((x - xbar) ** 2).sum())
    if sxx <= 0:
        return {"alpha": float("nan"), "beta": float("nan"), "t_ols": float("nan"),
                "t_nw": float("nan"), "r2": float("nan"), "n": n}
    beta = float(((x - xbar) * (y - ybar)).sum() / sxx)
    alpha = float(ybar - beta * xbar)
    resid = y - (alpha + beta * x)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - ybar) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Plain OLS slope t.
    sigma2 = ss_res / (n - 2)
    se_ols = np.sqrt(sigma2 / sxx)
    t_ols = beta / se_ols if se_ols > 0 else float("nan")

    # Newey-West HAC slope variance: sandwich with score g_t = (x_t - xbar) * resid_t.
    g = (x - xbar) * resid
    S = float(g @ g) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        S += 2.0 * w * float(g[l:] @ g[:-l]) / n
    var_beta = n * S / (sxx ** 2)
    se_nw = np.sqrt(var_beta) if var_beta > 0 else float("nan")
    t_nw = beta / se_nw if se_nw and se_nw > 0 else float("nan")

    return {"alpha": alpha, "beta": beta, "t_ols": float(t_ols),
            "t_nw": float(t_nw), "r2": float(r2), "n": n}


# --------------------------------------------------------------------------- #
# Headline stats
# --------------------------------------------------------------------------- #
def regression_stats(series: pd.DataFrame, nw_lags: int = 6) -> dict:
    """Run the Driesprong predictive regression on a daily close frame.

    Returns the slope (``beta``, monthly return per unit oil return), its OLS- and
    NW-*t*, the R², the sample size, plus the monthly *equity* return in the top/bottom
    oil-return terciles (a model-free cross-check of the sign)."""
    fr = regression_frame(series)
    reg = newey_west_ols(fr["x"].to_numpy(), fr["y"].to_numpy(), lags=nw_lags)

    # Model-free tercile cross-check: forward equity return by trailing-oil tercile.
    x = fr["x"].to_numpy(); y = fr["y"].to_numpy()
    order = np.argsort(x, kind="stable")
    k = len(x) // 3
    lo_ret = float(np.mean(y[order[:k]])) if k else float("nan")   # oil fell most
    hi_ret = float(np.mean(y[order[-k:]])) if k else float("nan")  # oil rose most
    return {
        "n": reg["n"], "beta": reg["beta"], "alpha": reg["alpha"],
        "t_nw": reg["t_nw"], "t_ols": reg["t_ols"], "r2": reg["r2"],
        "r2_pct": reg["r2"] * 100.0,
        "fwd_after_oil_down_pct": lo_ret * 100.0,
        "fwd_after_oil_up_pct": hi_ret * 100.0,
        "welch_t": welch_t(y[order[-k:]], y[order[:k]]) if k else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the slope real, or a lucky alignment of predictor & target?
# --------------------------------------------------------------------------- #
def placebo_pvalue(series: pd.DataFrame, nw_lags: int = 6,
                   n_draws: int = 2000, base_seed: int = 825) -> dict:
    """Break the predictor→target link by permuting the target ``y`` while keeping ``x``.

    ``p`` = share of permuted worlds whose |slope| is >= the observed |slope| (two-sided
    on magnitude, since the claim fixes the sign a priori and we test whether *any*
    predictive slope this large arises by chance)."""
    fr = regression_frame(series)
    x = fr["x"].to_numpy(); y = fr["y"].to_numpy()
    obs = newey_west_ols(x, y, lags=nw_lags)["beta"]
    xbar = x.mean()
    sxx = float(((x - xbar) ** 2).sum())
    rng = np.random.default_rng(base_seed)
    betas = np.empty(n_draws)
    for i in range(n_draws):
        yp = rng.permutation(y)
        betas[i] = float(((x - xbar) * (yp - yp.mean())).sum() / sxx)
    p = float((np.abs(betas) >= abs(obs)).mean())
    return {
        "obs_beta": float(obs),
        "placebo_mean_beta": float(betas.mean()),
        "placebo_sd_beta": float(betas.std(ddof=1)),
        "p_value": p,
        "n_draws": n_draws,
    }


# --------------------------------------------------------------------------- #
# The costed monthly timer
# --------------------------------------------------------------------------- #
def timer_stats(series: pd.DataFrame, cost_bps: float = 5.0,
                borrow_bps_yr: float = 50.0, long_only: bool = False) -> dict:
    """Trade the Driesprong forecast: each month hold ``-sign(oil_ret[t])`` of SPY next
    month (short SPY after oil rises, long after oil falls). ``long_only`` caps the short
    at flat (a long/flat timer). Charge one-way ``cost_bps`` × NAV per rebalance leg and
    borrow on any short."""
    fr = regression_frame(series)
    x = fr["x"].to_numpy(); y = fr["y"].to_numpy()
    pos = -np.sign(x)
    if long_only:
        pos = np.clip(pos, 0.0, 1.0)
    gross = pos * y

    # Turnover: |Δposition| each month (position set from the fresh signal), plus the
    # entry leg. One-way cost charged on traded notional; borrow on the short leg.
    dpos = np.abs(np.diff(np.concatenate([[0.0], pos])))
    cost = dpos * (cost_bps / 1e4)
    borrow_monthly = (borrow_bps_yr / 1e4) / 12.0
    short_frac = np.clip(-pos, 0.0, 1.0)
    net = gross - cost - short_frac * borrow_monthly

    n = len(net)
    net_mean = float(net.mean())
    sd = float(net.std(ddof=1)) if n > 1 else float("nan")
    sharpe = net_mean / sd * np.sqrt(MONTHS_PER_YEAR) if sd and sd > 0 else float("nan")
    return {
        "n_months": n,
        "gross_pct_mo": float(gross.mean()) * 100.0,
        "net_pct_mo": net_mean * 100.0,
        "cost_pct_mo": float(cost.mean() + short_frac.mean() * borrow_monthly) * 100.0,
        "ann_net_pct": net_mean * MONTHS_PER_YEAR * 100.0,
        "sharpe_net": sharpe,
        "t_net": one_sample_t(net),
        "hit_rate": float((gross > 0).mean()),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(series: pd.DataFrame, nw_lags: int = 6) -> dict:
    """Run the headline predictive regression on a synthetic tape."""
    s = regression_stats(series, nw_lags=nw_lags)
    return {"beta": s["beta"], "t_nw": s["t_nw"], "r2_pct": s["r2_pct"], "n": s["n"]}
