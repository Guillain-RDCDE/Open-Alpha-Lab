"""Strategy + inference for Study 801 — Employee Satisfaction ("100 Best" alpha).

The claim (Edmans 2011): a portfolio of the "100 Best Companies to Work For" earns
**positive risk-adjusted alpha** — employee satisfaction is an intangible the market
underprices. We test a modest, hand-coded, cited basket of *perennial listed members*,
equal-weighted and monthly-rebalanced, against the market (SPY total return).

Measurements (all on monthly total returns):

* **The headline — a market-model (CAPM) alpha.** Regress the equal-weight basket's monthly
  return on the market: ``r_basket = alpha + beta * r_mkt + eps``. The intercept ``alpha`` is
  the monthly risk-adjusted excess; we report it annualized and its **Newey-West (HAC) t**
  (the decisive number) alongside the plain OLS t. This is a CAPM alpha, **not** Edmans'
  4-factor alpha — it controls for the market only, so it is an *upper bound* on any true
  satisfaction alpha (a tech-tilted survivor basket also loads on size/value/momentum/quality
  we do not strip out).

* **The excess-over-market series.** ``d_t = r_basket,t - r_mkt,t``; its mean, one-sample
  Newey-West t, and the monthly **information ratio** (annualized). A direct "does the basket
  beat the market" read that needs no risk-free rate.

* **The survivorship placebo — the honest core.** Draw many random equal-weight baskets of
  the same size from the ``CONTROL`` universe of large-cap **survivors not selected for
  workplace prestige**, compute each one's CAPM alpha, and ask where the satisfaction basket
  falls. If a random survivor basket earns alpha just as easily, the basket's alpha is
  *survivorship*, not satisfaction. (An equal-weight basket of survivors also carries a small
  size / rebalancing tilt vs cap-weight SPY — the placebo prices that in too.)

* **Sub-period split** — first vs second half, to check the alpha is not one lucky stretch.

* **A costed timer** — hold the equal-weight basket, rebalance monthly; one documented
  execution convention (rebalance at month-end close, earn next month), costs = one-way ×
  NAV × realised turnover; long-only (no shorts, so no borrow). Net alpha after costs.

The decisive number is the basket's **market-model alpha NW t** on the real tape. Because the
basket is a hand-picked survivor set and we control only for the market, the honest ceiling on
the Signal stamp is WEAK unless that alpha is overwhelming AND clears the survivor placebo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


# --------------------------------------------------------------------------- #
# Basket construction
# --------------------------------------------------------------------------- #
def equal_weight_basket(monthly: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """Equal-weight monthly return of ``tickers`` (mean across names present each month)."""
    cols = [t for t in tickers if t in monthly.columns]
    return monthly[cols].mean(axis=1, skipna=True).rename("basket")


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def newey_west_alpha(y: np.ndarray, x: np.ndarray, lags: int = 3) -> dict:
    """Market model ``y = alpha + beta*x + u`` with Newey-West (Bartlett) HAC SEs.

    Returns alpha, beta, the HAC t of alpha, and the plain OLS t of alpha. ``y`` and ``x`` are
    monthly returns (basket and market). ``lags`` is the Bartlett bandwidth (3 for monthly).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    keep = np.isfinite(y) & np.isfinite(x)
    y, x = y[keep], x[keep]
    n = len(y)
    if n < 6:
        return {"alpha": float("nan"), "beta": float("nan"),
                "t_alpha_nw": float("nan"), "t_alpha_ols": float("nan"), "n": n}
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ (X.T @ y)
    u = y - X @ beta_hat
    # OLS (homoskedastic) SE for the plain t
    sigma2 = (u @ u) / (n - 2)
    se_ols = np.sqrt(sigma2 * XtX_inv[0, 0])
    # Newey-West HAC
    s = X * u[:, None]
    S = s.T @ s
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = s[l:].T @ s[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se_nw = np.sqrt(V[0, 0])
    return {
        "alpha": float(beta_hat[0]), "beta": float(beta_hat[1]),
        "t_alpha_nw": float(beta_hat[0] / se_nw) if se_nw > 0 else float("nan"),
        "t_alpha_ols": float(beta_hat[0] / se_ols) if se_ols > 0 else float("nan"),
        "n": n,
    }


def newey_west_mean_t(x: np.ndarray, lags: int = 3) -> float:
    """One-sample Newey-West (HAC) t of mean(x) vs 0 — a regression on a constant."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 6:
        return float("nan")
    u = x - x.mean()
    S = float(u @ u)
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        S += 2.0 * w * float(u[l:] @ u[:-l])
    se = np.sqrt(S) / n
    return float(x.mean() / se) if se > 0 else float("nan")


def one_sample_t(x: np.ndarray) -> float:
    """Plain one-sample t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n (monthly win-rate)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def ann_return(monthly_ret: np.ndarray) -> float:
    """Geometric annualized return of a monthly return series."""
    monthly_ret = np.asarray(monthly_ret, dtype=float)
    monthly_ret = monthly_ret[np.isfinite(monthly_ret)]
    if len(monthly_ret) == 0:
        return float("nan")
    return float(np.prod(1.0 + monthly_ret) ** (MONTHS / len(monthly_ret)) - 1.0)


def ann_vol(monthly_ret: np.ndarray) -> float:
    monthly_ret = np.asarray(monthly_ret, dtype=float)
    monthly_ret = monthly_ret[np.isfinite(monthly_ret)]
    return float(monthly_ret.std(ddof=1) * np.sqrt(MONTHS)) if len(monthly_ret) > 1 else float("nan")


# --------------------------------------------------------------------------- #
# The headline
# --------------------------------------------------------------------------- #
def headline_stats(basket: pd.Series, market: pd.Series, lags: int = 3) -> dict:
    """CAPM alpha of the basket vs market (NW + OLS t), the excess-over-market series,
    win-rate + Wilson, annualized return/vol/IR."""
    df = pd.concat([basket.rename("b"), market.rename("m")], axis=1).dropna()
    b, m = df["b"].to_numpy(), df["m"].to_numpy()
    mm = newey_west_alpha(b, m, lags=lags)
    d = b - m
    k_hit = int((d > 0).sum())
    lo, hi = wilson_interval(k_hit, len(d))
    ir = float(d.mean() / d.std(ddof=1) * np.sqrt(MONTHS)) if d.std(ddof=1) > 0 else float("nan")
    return {
        "n_months": len(df),
        "alpha_monthly_bps": mm["alpha"] * 1e4,
        "alpha_ann_pct": ((1.0 + mm["alpha"]) ** MONTHS - 1.0) * 100,
        "beta": mm["beta"],
        "t_alpha_nw": mm["t_alpha_nw"], "t_alpha_ols": mm["t_alpha_ols"],
        "excess_mean_bps": float(d.mean() * 1e4),
        "excess_t_nw": newey_west_mean_t(d, lags=lags),
        "excess_t_ols": one_sample_t(d),
        "ir_ann": ir,
        "hit": k_hit, "hit_rate": k_hit / len(d), "hit_lo": lo, "hit_hi": hi,
        "basket_ann_pct": ann_return(b) * 100, "market_ann_pct": ann_return(m) * 100,
        "basket_vol_pct": ann_vol(b) * 100, "market_vol_pct": ann_vol(m) * 100,
    }


# --------------------------------------------------------------------------- #
# The survivorship placebo — random large-cap-survivor baskets
# --------------------------------------------------------------------------- #
def survivor_placebo(monthly: pd.DataFrame, control: list[str], market: pd.Series,
                     basket_size: int, n_draws: int = 5000, seed: int = 801,
                     lags: int = 3) -> dict:
    """Distribution of the CAPM alpha of RANDOM equal-weight baskets drawn from the
    survivor ``control`` universe. The honest survivorship yardstick.

    Draws ``n_draws`` random size-``basket_size`` baskets (without replacement within a
    draw), computes each one's market-model alpha (annualized %), and returns the array plus
    summary. The real basket's alpha is compared against this cloud in the caller: a right-tail
    p = share of random survivor baskets whose alpha meets or beats the satisfaction basket.
    """
    cols = [t for t in control if t in monthly.columns]
    M = monthly[cols]
    mkt = market.reindex(M.index)
    rng = np.random.default_rng(seed)
    k = min(basket_size, len(cols))
    alphas = np.empty(n_draws)
    m_arr = mkt.to_numpy()
    for i in range(n_draws):
        pick = rng.choice(len(cols), size=k, replace=False)
        b = M.iloc[:, pick].mean(axis=1, skipna=True).to_numpy()
        mm = newey_west_alpha(b, m_arr, lags=lags)
        alphas[i] = ((1.0 + mm["alpha"]) ** MONTHS - 1.0) * 100
    alphas = alphas[np.isfinite(alphas)]
    return {"alphas_ann_pct": alphas, "mean": float(alphas.mean()),
            "sd": float(alphas.std(ddof=1)), "n": len(alphas), "k": k}


def placebo_pvalue(obs_alpha_ann_pct: float, null_alphas: np.ndarray) -> float:
    """Right-tail p: share of random survivor baskets whose alpha >= the observed basket alpha."""
    null_alphas = np.asarray(null_alphas, dtype=float)
    null_alphas = null_alphas[np.isfinite(null_alphas)]
    if null_alphas.size == 0:
        return float("nan")
    return float((null_alphas >= obs_alpha_ann_pct).mean())


# --------------------------------------------------------------------------- #
# Sub-period split
# --------------------------------------------------------------------------- #
def subperiod_split(basket: pd.Series, market: pd.Series, lags: int = 3) -> dict:
    """First-half vs second-half CAPM alpha (annualized %) + NW t of each half."""
    df = pd.concat([basket.rename("b"), market.rename("m")], axis=1).dropna()
    mid = len(df) // 2
    out = {}
    for tag, sl in (("early", slice(0, mid)), ("late", slice(mid, None))):
        sub = df.iloc[sl]
        mm = newey_west_alpha(sub["b"].to_numpy(), sub["m"].to_numpy(), lags=lags)
        out[tag] = {"n": mm["n"],
                    "alpha_ann_pct": ((1.0 + mm["alpha"]) ** MONTHS - 1.0) * 100,
                    "t_alpha_nw": mm["t_alpha_nw"]}
    return out


# --------------------------------------------------------------------------- #
# The costed timer — hold the basket, rebalance monthly
# --------------------------------------------------------------------------- #
def realised_turnover(monthly: pd.DataFrame, tickers: list[str]) -> float:
    """Average one-way turnover of an equal-weight monthly rebalance.

    Each month the weights drift with returns; rebalancing back to equal weight trades
    ``0.5 * sum|w_drifted - w_target|`` one-way. We average that over the sample. (Modest for a
    ~18-name equal-weight book — a few % a month.)
    """
    cols = [t for t in tickers if t in monthly.columns]
    R = monthly[cols].fillna(0.0)
    k = len(cols)
    w_target = np.full(k, 1.0 / k)
    tos = []
    for _, row in R.iterrows():
        grown = w_target * (1.0 + row.to_numpy())
        w_drift = grown / grown.sum()
        tos.append(0.5 * np.abs(w_drift - w_target).sum())
    return float(np.mean(tos)) if tos else float("nan")


def timer_stats(basket: pd.Series, market: pd.Series, monthly: pd.DataFrame,
                tickers: list[str], cost_bps: float = 5.0, lags: int = 3) -> dict:
    """Net-of-cost CAPM alpha of the long-only equal-weight basket, rebalanced monthly.

    One documented convention: weights are set at the month-end close and earn the next
    month's return (calendar rebalance, no look-ahead). Cost per month = one-way turnover ×
    cost_bps × NAV, charged against the basket return; long-only, so no borrow. Reports the
    gross and net annualized alpha and the net alpha NW t.
    """
    to = realised_turnover(monthly, tickers)
    monthly_cost = to * cost_bps / 1e4
    df = pd.concat([basket.rename("b"), market.rename("m")], axis=1).dropna()
    b_net = df["b"].to_numpy() - monthly_cost
    mm_g = newey_west_alpha(df["b"].to_numpy(), df["m"].to_numpy(), lags=lags)
    mm_n = newey_west_alpha(b_net, df["m"].to_numpy(), lags=lags)
    return {
        "turnover_1way_pct": to * 100, "cost_bps": cost_bps,
        "cost_drag_ann_pct": ((1.0 + monthly_cost) ** MONTHS - 1.0) * 100,
        "gross_alpha_ann_pct": ((1.0 + mm_g["alpha"]) ** MONTHS - 1.0) * 100,
        "net_alpha_ann_pct": ((1.0 + mm_n["alpha"]) ** MONTHS - 1.0) * 100,
        "gross_t_nw": mm_g["t_alpha_nw"], "net_t_nw": mm_n["t_alpha_nw"],
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(world: dict, lags: int = 3) -> dict:
    """Run the market-model alpha detector on a synthetic world (from data.synthetic_world)."""
    mm = newey_west_alpha(world["basket"].to_numpy(), world["market"].to_numpy(), lags=lags)
    return {"alpha_ann_pct": ((1.0 + mm["alpha"]) ** MONTHS - 1.0) * 100,
            "beta": mm["beta"], "t_alpha_nw": mm["t_alpha_nw"], "n": mm["n"]}
