"""The engine and its honest controls — Study 330 (Low-Volatility-Anomaly).

The claim, at full strength (Baker-Bradley-Wurgler 2011; Frazzini-Pedersen 2014): calm
stocks out-earn wild ones *risk-adjusted* — the security-market line is too flat. The
retail-tradable embodiment is the **SPLV** (S&P 500 Low Volatility) vs **SPHB** (S&P 500 High
Beta) ETF pair. This module measures three things, honestly:

1. **The race.** SPLV vs SPHB leg-by-leg: CAGR, *excess-of-cash* Sharpe (the house rule when
   one side is a different risk level), vol and drawdown. The decisive question is whether the
   calm leg beats the wild leg **per unit of risk**, not in raw return.

2. **The self-financing book.** A dollar-neutral long-SPLV / short-SPHB spread — the literal
   "bet against beta in ETF form". Its mean carries a Newey-West (HAC) *t* and a circular
   block-bootstrap CI; it is charged one-way turnover × NAV in costs, plus a short borrow on
   the SPHB leg (SPHB is liquid and cheap to borrow, but it is never free).

3. **Alpha vs beta.** The spread regressed on the market (SPY) excess return: how much of any
   "edge" is just the structural beta gap (SPLV ~0.7, SPHB ~1.4) you were always exposed to?

Execution lag: the rebalance is calendar-known (hold each fund, monthly weights), so the book
earns the *realised* return of the holding month — one shift is unnecessary and would
double-count (see METHODOLOGY → "one execution lag, documented exactly"). Costs are applied at
rebalance via one-way turnover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Leg statistics
# ---------------------------------------------------------------------------
def ann_return(r: pd.Series) -> float:
    """Geometric annualised (CAGR) of a monthly simple-return series."""
    r = r.dropna()
    if r.empty:
        return float("nan")
    growth = float((1.0 + r).prod())
    yrs = len(r) / MONTHS_PER_YEAR
    return growth ** (1.0 / yrs) - 1.0 if yrs > 0 and growth > 0 else float("nan")


def ann_vol(r: pd.Series) -> float:
    """Annualised volatility of a monthly simple-return series."""
    r = r.dropna()
    return float(r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)) if len(r) > 1 else float("nan")


def sharpe(r: pd.Series, rf_monthly: float = 0.0) -> float:
    """Annualised *excess-of-cash* Sharpe of a monthly simple-return series.

    Pass ``rf_monthly`` (a monthly risk-free rate) so a leg sitting at a different risk level
    is judged on excess-of-cash return — the house rule that stops a raw-vs-excess race from
    manufacturing a verdict.
    """
    ex = (r.dropna() - rf_monthly)
    if len(ex) < 2 or ex.std(ddof=1) == 0:
        return float("nan")
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))


def max_drawdown(r: pd.Series) -> float:
    """Maximum drawdown (a negative number) of a monthly simple-return series."""
    r = r.dropna()
    if r.empty:
        return float("nan")
    nav = (1.0 + r).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def leg_stats(r: pd.Series, rf_monthly: float = 0.0) -> dict:
    """The four-number leg card: CAGR, Sharpe (excess-of-cash), vol, max-DD."""
    return {
        "cagr": ann_return(r),
        "sharpe": sharpe(r, rf_monthly),
        "vol": ann_vol(r),
        "max_dd": max_drawdown(r),
    }


# ---------------------------------------------------------------------------
# The self-financing SPLV - SPHB book
# ---------------------------------------------------------------------------
def spread(
    ret: pd.DataFrame,
    low: str = "SPLV",
    high: str = "SPHB",
    cost_bps: float = 0.0,
    borrow_ann_bps: float = 0.0,
) -> pd.Series:
    """The dollar-neutral long-low / short-high monthly spread, net of costs and borrow.

    Gross spread is ``ret[low] - ret[high]`` (long the calm leg, short the wild leg, $1 each).
    Costs: a one-way turnover × NAV charge of ``cost_bps`` is applied each month the book is
    re-struck (both legs trade, so a full round-trip of the two-sided book is ``2 * cost_bps``
    per rebalance — counted here as a flat monthly drag, the conservative monthly-rebalance
    case). ``borrow_ann_bps`` charges short financing on the SPHB leg (annual, pro-rated
    monthly) exactly once.
    """
    s = (ret[low] - ret[high]).dropna()
    monthly_cost = 2.0 * cost_bps * 1e-4          # both legs trade each rebalance
    monthly_borrow = borrow_ann_bps * 1e-4 / MONTHS_PER_YEAR
    return s - monthly_cost - monthly_borrow


def beta_neutral_spread(
    ret: pd.DataFrame,
    low: str = "SPLV",
    high: str = "SPHB",
    mkt: str = "SPY",
    cost_bps: float = 0.0,
    borrow_ann_bps: float = 0.0,
    window: int | None = None,
) -> pd.Series:
    """The Frazzini-Pedersen-style *beta-neutral* low-minus-high book.

    The naive dollar-neutral ``spread`` (long $1 calm, short $1 wild) is dominated by the
    structural beta gap (SPLV ~0.7 vs SPHB ~1.4): in any up-market the high-beta leg simply
    earns more, so the raw spread is negative *by construction* regardless of any
    risk-adjusted edge. To isolate the **anomaly** (return per unit of risk), we hedge out the
    market: long the calm leg, short a beta-matched amount of the wild leg so the net market
    beta is ~0. What is left is the low-vol leg's risk-adjusted advantage — exactly the thing
    the anomaly claims and the thing a synthetic positive control must be able to plant.

    Betas are estimated on the *full sample* by default (``window=None``); pass ``window`` for a
    rolling-beta variant. Costs and borrow are charged as in ``spread``.
    """
    df = ret[[low, high, mkt]].dropna()
    if len(df) < 6:
        return pd.Series(dtype=float)
    m = df[mkt].to_numpy()

    def _beta(y):
        cov = np.cov(y, m, ddof=1)
        return cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0

    b_low = _beta(df[low].to_numpy())
    b_high = _beta(df[high].to_numpy())
    # Long calm at unit beta; short wild scaled so the two betas cancel.
    w_high = b_low / b_high if b_high != 0 else 1.0
    s = (df[low] - w_high * df[high]).rename("beta_neutral")
    monthly_cost = (1.0 + w_high) * cost_bps * 1e-4
    monthly_borrow = w_high * borrow_ann_bps * 1e-4 / MONTHS_PER_YEAR
    return s - monthly_cost - monthly_borrow


def hac_tstat(r: pd.Series) -> float:
    """Newey-West (HAC) t-stat on the mean of a monthly return series.

    No hard dependency on quantlab being importable — the same lag rule
    ``floor(4*(n/100)^(2/9))`` the desk uses elsewhere.
    """
    x = r.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def block_bootstrap_ci(
    r: pd.Series,
    block: int = 6,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 330,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the annualised mean of a monthly series.

    Circular blocks preserve the short-run autocorrelation that i.i.d. resampling destroys.
    Returns the (lo, hi) annualised-mean bounds at confidence ``1 - alpha``.
    """
    x = r.dropna().to_numpy(dtype=float)
    n = x.size
    if n < block + 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = x[idx[:n]].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo * MONTHS_PER_YEAR), float(hi * MONTHS_PER_YEAR))


def spread_stats(s: pd.Series) -> dict:
    """Headline statistics for a spread series: annualised mean, vol, Sharpe, HAC t, n."""
    s = s.dropna()
    return {
        "n": int(len(s)),
        "mean_ann": float(s.mean() * MONTHS_PER_YEAR),
        "vol_ann": ann_vol(s),
        "sharpe": sharpe(s),
        "tstat": hac_tstat(s),
    }


# ---------------------------------------------------------------------------
# Alpha vs beta — how much of the edge is the structural beta gap?
# ---------------------------------------------------------------------------
def market_regression(s: pd.Series, mkt: pd.Series, rf_monthly: float = 0.0) -> dict:
    """OLS of the spread on the market excess return: annualised alpha, beta, alpha t-stat.

    ``s`` is the (net) spread; ``mkt`` is the market total return (SPY). The intercept is the
    monthly alpha; the slope is the spread's net market beta (expected negative — long calm,
    short wild). Reported with a HAC t-stat on the residual mean is overkill for a single
    intercept, so we report the OLS alpha t-stat (homoskedastic) alongside the spread's own
    HAC t for honesty.
    """
    df = pd.concat([s.rename("s"), (mkt - rf_monthly).rename("m")], axis=1).dropna()
    if len(df) < 6:
        return {"alpha_ann": float("nan"), "beta": float("nan"), "alpha_t": float("nan")}
    x = df["m"].to_numpy()
    y = df["s"].to_numpy()
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = X.shape
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_alpha = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
    return {
        "alpha_ann": float(coef[0] * MONTHS_PER_YEAR),
        "beta": float(coef[1]),
        "alpha_t": float(coef[0] / se_alpha) if se_alpha > 0 else float("nan"),
    }
