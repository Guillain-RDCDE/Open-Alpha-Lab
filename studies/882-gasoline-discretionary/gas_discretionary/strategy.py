"""Strategy + inference for Study 882 — Gas-Price → Discretionary (the "pump tax" rotation).

The claim: a rise in the **gasoline price this month** is a tax on the consumer's wallet, so
it should forecast **discretionary (XLY) underperforming staples (XLP) next month** and a
**tailwind for energy (XLE)**. We test the self-contained monthly version:

    r_spread[t+1] = alpha + beta * r_gas[t] + eps ,   beta expected < 0 ,

a single-regressor predictive regression of the discretionary-minus-staples (XLY − XLP)
*forward* one-month return on the *trailing* one-month gasoline (RB=F) return. A parallel
regression on the energy tilt (XLE − SPY) expects ``beta > 0``.

This is distinct from:

* [825-oil-predicts-equities](../../825-oil-predicts-equities/) — the lagged **crude →
  aggregate-equity** forecast (Driesprong); here the predictor is **gasoline** (the retail
  pump price) and the target is a **within-equity sector rotation** (XLY vs XLP), not the
  whole market.
* [245-oil-equity-correlation](../../245-oil-equity-correlation/) — the **contemporaneous**
  oil↔equity co-movement, not a lagged forecast and not a sector spread.
* [226-crude-seasonality](../../226-crude-seasonality/) — crude's **calendar** seasonality
  in the oil price itself, not a cross-asset rotation.
* [639-gasoline-rvp-seasonality](../../639-gasoline-rvp-seasonality/) — gasoline's **own**
  RVP-driven (summer-blend) calendar seasonality, a property of the gas price, not gas as a
  cross-asset predictor of a consumer-sector spread.

Method:

* **Month-end resample.** Daily adjusted close → month-end levels → monthly simple returns
  for gas, XLY, XLP, XLE, SPY; the discretionary-minus-staples spread (``XLY − XLP``) and
  the energy tilt (``XLE − SPY``).
* **Predictive alignment, one documented lag.** The predictor is the trailing-month gas
  return known at the close of month ``t``; the target is the spread return realised over
  the *next* month ``t+1`` (``spread.shift(-1)``). Zero look-ahead.
* **Inference.** OLS slope with a **Newey-West (HAC)** *t* (the honest headline); the
  in-sample R²; a Welch tercile cross-check; a permutation placebo; a two-era robustness
  cut; a costed monthly timer; a seeded synthetic positive control. Newey-West /
  one-sample / Welch / Wilson primitives mirror the desk house style (study 803).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Returns + the predictive regression frame
# --------------------------------------------------------------------------- #
def monthly_returns(series: pd.DataFrame) -> pd.DataFrame:
    """Month-end simple returns for gas and the four ETFs, plus the two rotation spreads.

    ``series`` is the daily adjusted-close frame (columns ``GAS``, ``XLY``, ``XLP``,
    ``XLE``, ``SPY``). We take the last close of each calendar month and its simple
    pct-change. Returns a frame with columns ``gas``, ``xly``, ``xlp``, ``xle``, ``spy``,
    ``disc_stap`` (XLY − XLP, the pump-tax rotation) and ``enr_mkt`` (XLE − SPY, the energy
    tilt), indexed by month-end.
    """
    m = series.resample("ME").last().dropna()
    r = m.pct_change()
    out = pd.DataFrame({
        "gas": r["GAS"],
        "xly": r["XLY"], "xlp": r["XLP"], "xle": r["XLE"], "spy": r["SPY"],
        "disc_stap": r["XLY"] - r["XLP"],
        "enr_mkt": r["XLE"] - r["SPY"],
    }).dropna()
    return out


def regression_frame(series: pd.DataFrame, target: str = "disc_stap") -> pd.DataFrame:
    """Align the trailing-month gas return (predictor ``x``) with the forward-month
    ``target`` spread return (target ``y``).

    ``x`` on month ``t`` = gas return over month ``t`` (known at the close of ``t``).
    ``y`` on month ``t`` = ``target`` return over month ``t+1`` (``target.shift(-1)``), the
    forecast target. One documented execution lag, zero look-ahead. Rows with a missing
    predictor or target (the last month has no forward return) are dropped.
    """
    m = monthly_returns(series)
    frame = pd.DataFrame({
        "x": m["gas"],
        "y": m[target].shift(-1),
        "spread_now": m[target],
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
    """Single-regressor OLS ``y = alpha + beta*x + eps`` with a Newey-West (HAC) *t* on the
    slope ``beta``.

    Closed-form OLS plus a HAC (Bartlett) sandwich variance of the slope, so we depend only
    on numpy. Returns ``alpha``, ``beta``, the OLS-*t* and the NW-*t* on the slope, the R²,
    and ``n``. This is the study's honest headline: sign + magnitude + HAC *t*.
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
def regression_stats(series: pd.DataFrame, target: str = "disc_stap",
                     nw_lags: int = 6) -> dict:
    """Run the pump-tax predictive regression on a daily close frame.

    Returns the slope (``beta``, forward ``target`` spread return per unit gas return), its
    OLS- and NW-*t*, the R², the sample size, plus the forward spread return in the
    top/bottom gas-return terciles (a model-free cross-check of the sign)."""
    fr = regression_frame(series, target=target)
    reg = newey_west_ols(fr["x"].to_numpy(), fr["y"].to_numpy(), lags=nw_lags)

    # Model-free tercile cross-check: forward spread return by trailing-gas tercile.
    x = fr["x"].to_numpy(); y = fr["y"].to_numpy()
    order = np.argsort(x, kind="stable")
    k = len(x) // 3
    lo_ret = float(np.mean(y[order[:k]])) if k else float("nan")   # gas fell most
    hi_ret = float(np.mean(y[order[-k:]])) if k else float("nan")  # gas rose most
    return {
        "target": target,
        "n": reg["n"], "beta": reg["beta"], "alpha": reg["alpha"],
        "t_nw": reg["t_nw"], "t_ols": reg["t_ols"], "r2": reg["r2"],
        "r2_pct": reg["r2"] * 100.0,
        "fwd_after_gas_down_pct": lo_ret * 100.0,
        "fwd_after_gas_up_pct": hi_ret * 100.0,
        "welch_t": welch_t(y[order[-k:]], y[order[:k]]) if k else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the slope real, or a lucky alignment of predictor & target?
# --------------------------------------------------------------------------- #
def placebo_pvalue(series: pd.DataFrame, target: str = "disc_stap", nw_lags: int = 6,
                   n_draws: int = 2000, base_seed: int = 882) -> dict:
    """Break the predictor→target link by permuting the target ``y`` while keeping ``x``.

    ``p`` = share of permuted worlds whose |slope| is >= the observed |slope| (two-sided on
    magnitude, since the claim fixes the sign a priori and we test whether *any* predictive
    slope this large arises by chance)."""
    fr = regression_frame(series, target=target)
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
def timer_stats(series: pd.DataFrame, target: str = "disc_stap", cost_bps: float = 5.0,
                borrow_bps_yr: float = 50.0) -> dict:
    """Trade the pump-tax forecast on the ``target`` spread: each month hold
    ``-sign(gas_ret[t])`` of the spread next month (short the XLY−XLP spread after gas
    rises, long it after gas falls). The spread itself is a 2× NAV long/short book; charge
    one-way ``cost_bps`` × NAV per rebalance leg on BOTH legs plus borrow on the short
    leg."""
    fr = regression_frame(series, target=target)
    x = fr["x"].to_numpy(); y = fr["y"].to_numpy()
    pos = -np.sign(x)          # +1 long the spread, -1 short the spread
    gross = pos * y

    # Turnover: |Δposition| each month on a 2-leg (2× NAV) spread; charge both legs.
    dpos = np.abs(np.diff(np.concatenate([[0.0], pos])))
    cost = dpos * 2.0 * (cost_bps / 1e4)
    borrow_monthly = (borrow_bps_yr / 1e4) / 12.0
    # The spread is always half short (one leg short whichever way we lean).
    net = gross - cost - borrow_monthly

    n = len(net)
    net_mean = float(net.mean())
    sd = float(net.std(ddof=1)) if n > 1 else float("nan")
    sharpe = net_mean / sd * np.sqrt(MONTHS_PER_YEAR) if sd and sd > 0 else float("nan")
    return {
        "target": target,
        "n_months": n,
        "gross_pct_mo": float(gross.mean()) * 100.0,
        "net_pct_mo": net_mean * 100.0,
        "cost_pct_mo": float(cost.mean() + borrow_monthly) * 100.0,
        "ann_net_pct": net_mean * MONTHS_PER_YEAR * 100.0,
        "sharpe_net": sharpe,
        "t_net": one_sample_t(net),
        "hit_rate": float((gross > 0).mean()),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(series: pd.DataFrame, target: str = "disc_stap",
                     nw_lags: int = 6) -> dict:
    """Run the headline predictive regression on a synthetic tape."""
    s = regression_stats(series, target=target, nw_lags=nw_lags)
    return {"beta": s["beta"], "t_nw": s["t_nw"], "r2_pct": s["r2_pct"], "n": s["n"]}
