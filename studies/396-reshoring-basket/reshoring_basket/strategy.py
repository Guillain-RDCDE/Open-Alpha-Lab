"""Strategy + inference for Study 396 — Reshoring-Basket (a persistent thematic tilt).

The claim under test:

    "The reshoring / nearshoring trade — US industrials + Mexico (+ factory automation) —
     structurally out-earns a plain global index, because manufacturing is coming home."

Unlike an event study, this is a *persistent-tilt* test. The reshoring basket is held the
whole sample; the question is whether its return *in excess of* the benchmark is a real,
positive, autocorrelation-robust edge — and, crucially, whether anything survives once you
strip the part you were always paid for: **beta**. A high-beta industrials-plus-EM basket
will out-earn the market in a rising market for free; that is sector beta, not reshoring
alpha. So we test, in order of strength:

  * the **raw daily excess** (``basket - benchmark``) mean against zero, with a plain t and a
    **HAC / Newey-West** t (the desk's autocorrelation-robust bar for a REAL stamp);
  * the **CAPM alpha** — regress the basket's excess-of-cash return on the benchmark's
    excess-of-cash return; the *intercept* is the beta-adjusted edge, and its t is the honest
    "is there anything beyond beta?" test;
  * a **Sharpe race**, excess-of-cash basket vs excess-of-cash benchmark (a high-vol basket
    can out-*return* and under-*Sharpe* the index);
  * a **placebo / sign-flip** null on the monthly excess series — how often does a random
    relabelling produce a mean excess this large;
  * **one-way costs × turnover** on a yearly-rebalanced long-basket / short-benchmark book.

The decisive number is the **alpha t**, not the raw excess: raw excess is mostly beta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_YEAR = 252
RF_DAILY = 0.02 / TRADING_YEAR     # a flat 2%/yr cash rate for the excess-of-cash Sharpe race


# --------------------------------------------------------------------------- #
# Excess-return level
# --------------------------------------------------------------------------- #
def excess_series(frame: pd.DataFrame, bench: str = "mkt") -> pd.Series:
    """Daily reshoring-basket return in excess of ``bench`` (``mkt`` SPY or ``acwi``)."""
    x = (frame["basket"] - frame[bench]).dropna()
    return x


def plain_t(x: np.ndarray) -> float:
    """t-stat of mean(x) against zero (iid). NaN if fewer than 2 points."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(x.mean() / (sd / np.sqrt(len(x))))


def hac_t(x: np.ndarray, lags: int = 21) -> float:
    """Newey-West (HAC) t of mean(x) against zero (Bartlett kernel, ``lags`` lags).

    The autocorrelation-robust view: daily excess returns cluster, so the iid t over-states
    significance. This is the statistic the REAL stamp must clear (t >= 2) on the real tape.
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    if T < 3:
        return float("nan")
    m = x.mean()
    e = x - m
    gamma0 = (e @ e) / T
    s = gamma0
    L = max(1, lags)
    for k in range(1, L + 1):
        if k >= T:
            break
        w = 1.0 - k / (L + 1)
        cov = (e[k:] @ e[:-k]) / T
        s += 2.0 * w * cov
    se = np.sqrt(s / T)
    if se == 0:
        return float("nan")
    return float(m / se)


# --------------------------------------------------------------------------- #
# CAPM alpha — the beta-adjusted edge
# --------------------------------------------------------------------------- #
def capm_alpha(frame: pd.DataFrame, bench: str = "mkt",
               rf_daily: float = RF_DAILY, lags: int = 21) -> dict:
    """Regress (basket - rf) on (bench - rf): alpha = intercept, beta = slope.

    Returns daily and annualised alpha, beta, R-squared, and both the iid and HAC t-stats of
    the alpha intercept. The alpha is the part of the basket's return the benchmark does NOT
    explain — the honest "reshoring edge beyond sector beta". Its HAC t is the REAL bar.
    """
    sub = frame.dropna(subset=["basket", bench])
    y = sub["basket"].values - rf_daily
    xb = sub[bench].values - rf_daily
    T = len(y)
    if T < 10:
        return {"alpha_daily": float("nan"), "alpha_ann": float("nan"),
                "beta": float("nan"), "r2": float("nan"),
                "t_alpha": float("nan"), "t_alpha_hac": float("nan"), "n": T}
    X = np.column_stack([np.ones(T), xb])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = y - X @ coef
    # variance of the intercept under iid, then a HAC version of the alpha t
    sse = resid @ resid
    sigma2 = sse / (T - 2)
    XtX_inv = np.linalg.inv(X.T @ X)
    se_alpha_iid = np.sqrt(sigma2 * XtX_inv[0, 0])
    t_alpha_iid = alpha / se_alpha_iid if se_alpha_iid > 0 else float("nan")
    # HAC standard error of the intercept (Newey-West on the regression scores)
    L = max(1, lags)
    u = resid[:, None] * X                      # score contributions
    S = (u.T @ u) / T
    for k in range(1, L + 1):
        if k >= T:
            break
        w = 1.0 - k / (L + 1)
        G = (u[k:].T @ u[:-k]) / T
        S += w * (G + G.T)
    bread = XtX_inv * T
    cov_hac = bread @ S @ bread / T
    se_alpha_hac = np.sqrt(cov_hac[0, 0])
    t_alpha_hac = alpha / se_alpha_hac if se_alpha_hac > 0 else float("nan")
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
    return {
        "alpha_daily": alpha,
        "alpha_ann": alpha * TRADING_YEAR,
        "beta": beta,
        "r2": float(r2),
        "t_alpha": float(t_alpha_iid),
        "t_alpha_hac": float(t_alpha_hac),
        "n": T,
    }


# --------------------------------------------------------------------------- #
# Sharpe race (excess-of-cash, both legs)
# --------------------------------------------------------------------------- #
def sharpe(r: np.ndarray, rf_daily: float = RF_DAILY) -> float:
    """Annualised excess-of-cash Sharpe of a daily return series."""
    r = np.asarray(r, dtype=float)
    ex = r - rf_daily
    sd = ex.std(ddof=1)
    if sd == 0 or len(ex) < 2:
        return float("nan")
    return float(ex.mean() / sd * np.sqrt(TRADING_YEAR))


def sharpe_race(frame: pd.DataFrame, bench: str = "mkt",
                rf_daily: float = RF_DAILY) -> dict:
    """Annualised excess-of-cash Sharpe of the basket vs the benchmark, on the same dates."""
    sub = frame.dropna(subset=["basket", bench])
    return {
        "sharpe_basket": sharpe(sub["basket"].values, rf_daily),
        "sharpe_bench": sharpe(sub[bench].values, rf_daily),
        "n": int(len(sub)),
    }


# --------------------------------------------------------------------------- #
# Placebo / sign-flip null on the monthly excess
# --------------------------------------------------------------------------- #
def placebo_pvalue(x: pd.Series, n_draws: int = 20_000, seed: int = 396) -> dict:
    """Sign-flip placebo on the *monthly* excess series.

    Aggregates the daily excess to monthly, then asks: under a random Rademacher sign-flip of
    each month's excess (the sharp null of "no edge, symmetric noise"), how often is the mean
    >= the observed mean? The honest small-sample p-value for a persistent tilt.
    """
    monthly = (1.0 + x).resample("ME").prod() - 1.0
    m = monthly.values
    m = m[~np.isnan(m)]
    k = len(m)
    if k < 3:
        return {"k": k, "obs_mean": float("nan"), "p_value": float("nan")}
    obs = float(m.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_draws, k))
    means = (signs * m).mean(axis=1)
    p = float((means >= obs).mean())
    return {"k": k, "obs_mean": obs, "p_value": p}


# --------------------------------------------------------------------------- #
# Headline summary + costs
# --------------------------------------------------------------------------- #
def summarize(frame: pd.DataFrame, bench: str = "mkt",
              rf_daily: float = RF_DAILY) -> dict:
    """Headline stats vs one benchmark: annualised excess, plain + HAC t of daily excess,
    CAPM alpha (ann.) + its iid/HAC t + beta, Sharpe race, and the placebo p-value."""
    x = excess_series(frame, bench=bench)
    al = capm_alpha(frame, bench=bench, rf_daily=rf_daily)
    sr = sharpe_race(frame, bench=bench, rf_daily=rf_daily)
    pl = placebo_pvalue(x)
    return {
        "bench": bench,
        "n": int(len(x)),
        "excess_ann": float(x.mean() * TRADING_YEAR),
        "t_excess": plain_t(x.values),
        "t_excess_hac": hac_t(x.values),
        "alpha_ann": al["alpha_ann"],
        "beta": al["beta"],
        "t_alpha": al["t_alpha"],
        "t_alpha_hac": al["t_alpha_hac"],
        "sharpe_basket": sr["sharpe_basket"],
        "sharpe_bench": sr["sharpe_bench"],
        "p_placebo": pl["p_value"],
    }


def net_of_costs(frame: pd.DataFrame, bench: str = "mkt",
                 cost_bps: float = 10.0, rebalances_per_year: float = 1.0) -> dict:
    """Long-basket / short-benchmark book, annual rebalance, one-way costs × turnover.

    The reshoring tilt is a buy-and-hold thematic basket: turnover is low (we assume one
    rebalance a year, both legs). One-way cost ``cost_bps`` is charged on the traded notional
    per rebalance per leg. Returns the gross and net annualised excess — the point is whether
    the (beta-driven) raw excess survives costs, and, separately, whether the *alpha* does.
    """
    x = excess_series(frame, bench=bench)
    gross_ann = float(x.mean() * TRADING_YEAR)
    # two legs (long basket, short bench), traded `rebalances_per_year` times, one-way each
    annual_cost = 2.0 * rebalances_per_year * (cost_bps / 1e4)
    net_ann = gross_ann - annual_cost
    return {
        "gross_excess_ann": gross_ann,
        "net_excess_ann": net_ann,
        "cost_bps": cost_bps,
        "annual_cost": annual_cost,
        "rebalances_per_year": rebalances_per_year,
    }
