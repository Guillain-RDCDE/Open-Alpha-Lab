"""The engine and its honest controls — Study 324 (Bitcoin-Treasury).

Three questions, one apparatus:

1. **How much of MSTR is just BTC?** — an OLS of MSTR daily returns on BTC daily returns.
   The slope is the *beta to Bitcoin* (the leverage), the intercept is the *daily alpha*
   (what's left after stripping the BTC exposure). The intercept's HAC (Newey-West)
   t-stat is the inference-bar number for the Signal axis: a real "MSTR is more than
   levered BTC" claim needs a robustly positive alpha (t ≥ 2). R² says how much of MSTR's
   variance is just BTC.

2. **Does holding MSTR beat holding BTC?** — the honest race. Comparing raw MSTR to raw
   BTC is the naive claim; the fair comparison is MSTR against a **synthetic levered-BTC
   proxy** that takes the *same* beta exposure with a documented daily financing cost on
   the borrowed notional. If MSTR can't beat its own levered-beta replica net of carry,
   the "treasury premium" paid nothing. We race on **excess-of-cash Sharpe** so a levered
   leg isn't flattered by the risk-free rate it implicitly earns.

3. **Does the NAV premium pay?** — the market-cap-to-coin-value premium swings wildly. We
   measure whether *buying when the premium is low* (a contrarian premium-timing overlay)
   beats *buying always*, with a HAC t on the difference. A premium that mean-reverts is
   risk, not return, unless you can time it.

Execution lag: any timing overlay uses a signal known at the close of *t* to earn the
return of *t+1* — one ``shift``, applied once, documented here. The beta regression and
the buy-and-hold race are calendar-contemporaneous (no signal, no lag).

Costs: one-way bps × NAV per rebalance; the levered-proxy control additionally pays a
daily financing rate on the borrowed ``(beta - 1)`` notional (shorts/borrow paid once).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 365


# ---------------------------------------------------------------------------
# Returns helpers
# ---------------------------------------------------------------------------
def log_returns(close: pd.Series) -> pd.Series:
    """Daily log returns of a close series (first value dropped)."""
    return np.log(close).diff().dropna()


def simple_returns(close: pd.Series) -> pd.Series:
    """Daily simple returns of a close series (first value dropped)."""
    return close.pct_change().dropna()


# ---------------------------------------------------------------------------
# HAC (Newey-West) t-stat on a mean — no hard quantlab dependency
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West HAC t-stat for the mean of ``x`` being zero."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def hac_tstat_resid(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float, float]:
    """OLS of ``y`` on ``x`` with intercept; HAC t-stat on the *intercept*.

    Returns ``(alpha, beta, r2, t_alpha)`` where ``alpha`` is the intercept (daily),
    ``beta`` the slope, ``r2`` the regression R², and ``t_alpha`` the Newey-West t on the
    intercept. This is the central inference-bar test: is there a return left in MSTR
    after stripping its Bitcoin exposure?
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    n = y.size
    if n < 6:
        return float("nan"), float("nan"), float("nan"), float("nan")
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ (X.T @ y)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = y - X @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    # Newey-West sandwich on the coefficient covariance.
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        Xe = X * resid[:, None]
        G = Xe[k:].T @ Xe[:-k]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se_alpha = np.sqrt(max(cov[0, 0], 0.0))
    t_alpha = alpha / se_alpha if se_alpha > 0 else float("nan")
    return alpha, beta, r2, float(t_alpha)


# ---------------------------------------------------------------------------
# Beta-to-BTC decomposition
# ---------------------------------------------------------------------------
def beta_decomposition(frame: pd.DataFrame) -> dict:
    """Decompose MSTR into its Bitcoin exposure: alpha, beta, R², HAC t on alpha.

    Uses simple daily returns. ``beta`` is the leverage to Bitcoin; ``alpha`` the daily
    residual drift; ``r2`` how much of MSTR's day-to-day variance is just BTC moving.
    """
    r_mstr = simple_returns(frame["mstr"])
    r_btc = simple_returns(frame["btc"])
    common = r_mstr.index.intersection(r_btc.index)
    a, b, r2, ta = hac_tstat_resid(r_mstr.loc[common].to_numpy(),
                                   r_btc.loc[common].to_numpy())
    return {
        "alpha_daily": a,
        "alpha_ann": a * TRADING_DAYS_PER_YEAR,
        "beta": b,
        "r2": r2,
        "t_alpha": ta,
        "n": int(len(common)),
    }


# ---------------------------------------------------------------------------
# Levered-BTC synthetic proxy — the fair control for "hold MSTR not BTC"
# ---------------------------------------------------------------------------
def levered_btc_proxy(
    frame: pd.DataFrame,
    leverage: float,
    financing_bps_per_day: float = 0.0,
) -> pd.Series:
    """A daily simple-return series for a constant-``leverage`` BTC position.

    The proxy takes ``leverage``× the BTC daily return and pays a daily financing cost on
    the *borrowed* notional ``(leverage - 1)`` (charged once; no double-count of the
    risk-free rate). ``financing_bps_per_day`` is the borrow cost in bps/day on that
    notional. This is the apples-to-apples replica of MSTR's leverage — if MSTR can't beat
    it, the "Bitcoin treasury" wrapper added nothing.
    """
    r_btc = simple_returns(frame["btc"])
    borrowed = max(leverage - 1.0, 0.0)
    carry = borrowed * financing_bps_per_day * 1e-4
    return (leverage * r_btc - carry).rename("lev_proxy")


def sharpe(returns: np.ndarray, rf_daily: float = 0.0,
           periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sharpe of a daily simple-return series, excess of ``rf_daily``."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)] - rf_daily
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))


def cagr(returns: np.ndarray, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound annual growth from a daily simple-return series."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    total = float(np.prod(1.0 + r))
    yrs = r.size / periods
    return total ** (1.0 / yrs) - 1.0 if total > 0 and yrs > 0 else float("nan")


def max_drawdown(returns: np.ndarray) -> float:
    """Worst peak-to-trough drawdown of the cumulative simple-return curve (<= 0)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    curve = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(curve)
    return float((curve / peak - 1.0).min())


def race_mstr_vs_proxy(
    frame: pd.DataFrame,
    rf_daily: float = 0.0,
    financing_bps_per_day: float = 0.0,
    leverage: float | None = None,
) -> dict:
    """The honest race: hold MSTR vs hold a levered-BTC replica at MSTR's own beta.

    ``leverage`` defaults to the empirical beta from :func:`beta_decomposition`. We report
    CAGR, max drawdown and **excess-of-cash Sharpe** for MSTR, raw BTC, and the levered
    proxy, plus the HAC t-stat on the daily **MSTR − proxy** return difference (the
    apples-to-apples edge of the wrapper over its own leverage).
    """
    if leverage is None:
        leverage = beta_decomposition(frame)["beta"]
    r_mstr = simple_returns(frame["mstr"])
    r_btc = simple_returns(frame["btc"])
    proxy = levered_btc_proxy(frame, leverage, financing_bps_per_day)
    idx = r_mstr.index.intersection(r_btc.index).intersection(proxy.index)
    rm, rb, rp = r_mstr.loc[idx], r_btc.loc[idx], proxy.loc[idx]
    diff = (rm - rp).to_numpy()
    return {
        "leverage": float(leverage),
        "mstr_cagr": cagr(rm.to_numpy()),
        "btc_cagr": cagr(rb.to_numpy()),
        "proxy_cagr": cagr(rp.to_numpy()),
        "mstr_sharpe": sharpe(rm.to_numpy(), rf_daily),
        "btc_sharpe": sharpe(rb.to_numpy(), rf_daily),
        "proxy_sharpe": sharpe(rp.to_numpy(), rf_daily),
        "mstr_mdd": max_drawdown(rm.to_numpy()),
        "btc_mdd": max_drawdown(rb.to_numpy()),
        "proxy_mdd": max_drawdown(rp.to_numpy()),
        "diff_mean_bps": float(np.nanmean(diff) * 1e4),
        "diff_t": hac_tstat(diff),
        "n": int(len(idx)),
    }


# ---------------------------------------------------------------------------
# NAV premium and a premium-timing overlay
# ---------------------------------------------------------------------------
def nav_premium(frame: pd.DataFrame, beta_anchor: float | None = None) -> pd.Series:
    """A proxy NAV-premium series: MSTR's price relative to a BTC-implied 'fair' value.

    With only price tapes (no share-count / coin-count data) we approximate the premium
    as the **log ratio of MSTR to a beta-scaled BTC reference**, demeaned. It is a *price*
    proxy for the true mNAV premium, not the accounting premium — it captures the same
    swing (MSTR rich/cheap vs the coins) up to a constant. Returned as a 0-mean Series.
    """
    if beta_anchor is None:
        beta_anchor = beta_decomposition(frame)["beta"]
    # Reference: BTC raised to the leverage, rebased to MSTR at the start.
    ref = (frame["btc"] / frame["btc"].iloc[0]) ** beta_anchor
    ratio = np.log(frame["mstr"] / frame["mstr"].iloc[0]) - np.log(ref)
    return (ratio - ratio.mean()).rename("nav_premium")


def premium_timing(
    frame: pd.DataFrame,
    premium: pd.Series | None = None,
    cost_bps: float = 0.0,
) -> dict:
    """Contrarian premium-timing: hold MSTR only when its premium is below its median.

    Signal known at the close of *t* (premium below trailing-free median) earns the MSTR
    return of *t+1* — **one** shift, applied once. Compared against always-hold MSTR. The
    HAC t on the daily (timed − always) difference says whether timing the premium adds
    anything. ``cost_bps`` is one-way bps × NAV charged on each position change.
    """
    if premium is None:
        premium = nav_premium(frame)
    r_mstr = simple_returns(frame["mstr"])
    # Position decided at close of t (cheap = premium below its full-sample median).
    cheap = (premium < premium.median()).astype(float)
    pos = cheap.shift(1).reindex(r_mstr.index).fillna(0.0)  # one execution lag
    turn = pos.diff().abs().fillna(pos.abs())
    timed = (pos * r_mstr - turn * cost_bps * 1e-4).to_numpy()
    always = r_mstr.to_numpy()
    diff = timed - always
    return {
        "timed_cagr": cagr(timed),
        "always_cagr": cagr(always),
        "timed_sharpe": sharpe(timed),
        "always_sharpe": sharpe(always),
        "diff_mean_bps": float(np.nanmean(diff) * 1e4),
        "diff_t": hac_tstat(diff),
        "pct_in_market": float(pos.mean()),
        "n": int(len(r_mstr)),
    }


# ---------------------------------------------------------------------------
# Block bootstrap CI for a mean (circular block bootstrap)
# ---------------------------------------------------------------------------
def block_bootstrap_ci(
    x: np.ndarray,
    block: int = 20,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 324,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean of ``x`` (preserves autocorrelation)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < block + 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = x[idx[:n]].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi
