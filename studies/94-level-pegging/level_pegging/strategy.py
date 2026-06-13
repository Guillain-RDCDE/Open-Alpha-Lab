"""The race and its honest controls — Study 94 (Level-Pegging).

The folk claim: *"equal-weighting **always** beats cap-weighting — RSP (equal-weight S&P)
outperforms SPY (cap-weight S&P) because it tilts to smaller names and harvests a rebalancing
premium. Just buy RSP."* We implement both arms as total-return buy-and-hold index curves and
pin the claim against the parts that decide whether it is a free lunch or a regime-dependent
bet:

- **Full-sample stats** — CAGR, vol, Sharpe, maxDD, terminal wealth for each arm, and the
  daily return *difference* (EW minus CW).
- **Sub-period contrast** — the same race on a justified split (the mega-cap leadership era
  started mid-decade). EW led early and lagged late; we test whether the two sub-periods'
  mean return differences are themselves *different* (block bootstrap on the difference-of-
  differences), not just eyeball two equity curves.
- **Alpha vs beta** — regress EW's excess-of-cash daily return on CW's. The slope is EW's
  beta to the cap-weight market; the intercept (annualised) is what's left after you strip
  the beta you were always paid for. "Always beats" needs a positive, robust alpha — not just
  a higher-beta ride that happens to win when small caps lead.
- **Costs** — RSP charges a higher expense ratio than SPY and rebalances quarterly, so it
  turns over more. We net the EW arm down by an annual fee/turnover drag and re-read the race.

Both arms are held continuously (no timing, no lag): the comparison is index-vs-index, so the
desk's one-day execution-lag convention doesn't apply — there is no signal to act on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Per-arm stats
# ---------------------------------------------------------------------------
def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def _stats(ret: pd.Series) -> dict:
    """Headline stats for a daily simple-return series (buy-and-hold, no costs)."""
    ret = ret.astype(float).fillna(0.0)
    equity = (1.0 + ret).cumprod()
    n = len(ret)
    years = n / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = (
        float(ret.mean() / ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if ret.std(ddof=1) > 0
        else float("nan")
    )
    return {
        "net": ret,
        "equity": equity,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": _max_drawdown(equity.to_numpy()),
        "final": float(equity.iloc[-1]),
    }


def arm_returns(frame: pd.DataFrame, col: str, fee_bps_yr: float = 0.0) -> dict:
    """Total-return stats for one arm (``'ew'`` or ``'cw'``), optionally net of an annual fee.

    ``fee_bps_yr`` is an annual cost in basis points charged as a flat daily drag
    (``fee/252`` per day) — the place RSP's higher expense ratio and rebalancing turnover go.
    Cap-weight (SPY) carries its own small fee; pass each arm its own number.
    """
    ret = frame[col].pct_change().fillna(0.0)
    if fee_bps_yr:
        ret = ret - (fee_bps_yr * 1e-4) / TRADING_DAYS
    return _stats(ret)


def race(frame: pd.DataFrame, ew_fee_bps_yr: float = 0.0, cw_fee_bps_yr: float = 0.0) -> dict:
    """Run both arms and return their stats plus the EW-minus-CW daily return difference.

    Returns ``{'ew': stats, 'cw': stats, 'diff': Series, 'ew_minus_cw_cagr': float}`` where
    ``diff`` is the net daily return difference (after each arm's fee) used by the inference
    helpers below.
    """
    ew = arm_returns(frame, "ew", fee_bps_yr=ew_fee_bps_yr)
    cw = arm_returns(frame, "cw", fee_bps_yr=cw_fee_bps_yr)
    diff = (ew["net"] - cw["net"]).rename("ew_minus_cw")
    return {
        "ew": ew,
        "cw": cw,
        "diff": diff,
        "ew_minus_cw_cagr": ew["cagr"] - cw["cagr"],
    }


# ---------------------------------------------------------------------------
# Alpha vs beta — strip the cap-weight market beta from EW
# ---------------------------------------------------------------------------
def alpha_beta(frame: pd.DataFrame, rf_annual: float = 0.0) -> dict:
    """Regress EW excess-of-cash daily return on CW excess-of-cash daily return.

    ``slope`` (beta) is EW's exposure to the cap-weight market; ``alpha_ann`` is the
    intercept annualised (×252). With ``rf_annual=0`` both sides are raw returns (excess of a
    0% cash rate) — a stated, conservative choice; the relative comparison is unchanged by a
    common risk-free subtraction since it nets out of the difference. Returns alpha, its HAC
    *t*-stat, beta and R².
    """
    rf_d = rf_annual / TRADING_DAYS
    y = (frame["ew"].pct_change() - rf_d).dropna().to_numpy()
    x = (frame["cw"].pct_change() - rf_d).dropna().to_numpy()
    n = min(len(x), len(y))
    x, y = x[-n:], y[-n:]
    X = np.column_stack([np.ones(n), x])
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_hat
    # HAC (Newey-West) t on the intercept (coefficient 0).
    t_alpha = _hac_t_coef(X, beta_hat, resid, 0)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "alpha_daily": float(beta_hat[0]),
        "alpha_ann": float(beta_hat[0] * TRADING_DAYS),
        "beta": float(beta_hat[1]),
        "t_alpha": t_alpha,
        "r2": float(r2),
    }


def _hac_t_coef(X: np.ndarray, beta_hat: np.ndarray, resid: np.ndarray, k: int,
                lags: int | None = None) -> float:
    """Newey-West HAC t-stat for coefficient ``k`` of an OLS fit (X, beta_hat, residuals)."""
    n = X.shape[0]
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    XtX_inv = np.linalg.inv(X.T @ X)
    u = X * resid[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return float(beta_hat[k] / se[k]) if se[k] > 0 else float("nan")


# ---------------------------------------------------------------------------
# Inference on a return-difference series: HAC t-stat
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat for the mean of ``x`` (local, no quantlab dep)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 5:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Sub-period contrast: block bootstrap on the difference-of-differences
# ---------------------------------------------------------------------------
def split_contrast(diff: pd.Series, split_date: str, n_boot: int = 5000, block: int = 21,
                   seed: int = 94) -> dict:
    """Test whether the EW-minus-CW mean return *differs* between two sub-periods.

    ``diff`` is the daily EW-minus-CW return series; ``split_date`` cuts it into an early and
    a late regime. The statistic is the difference-of-means (early minus late). Its sampling
    distribution under the null (same mean in both regimes) is approximated by a **circular
    block bootstrap** that resamples blocks of length ``block`` from the *pooled* series and
    re-splits at the same calendar fraction — preserving autocorrelation, which i.i.d.
    resampling would destroy. Returns the observed gap, a two-sided bootstrap p-value, and
    each sub-period's annualised mean difference.
    """
    diff = diff.dropna()
    early = diff.loc[:split_date]
    late = diff.loc[split_date:]
    # guard against the split row appearing in both
    late = late.iloc[1:] if len(late) and len(early) and late.index[0] == early.index[-1] else late
    obs = float(early.mean() - late.mean())

    pooled = diff.to_numpy()
    n = pooled.size
    n_early = len(early)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    n_blocks = int(np.ceil(n / block))
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idxs = (starts[:, None] + np.arange(block)[None, :]) % n
        sample = pooled[idxs.ravel()][:n]
        stats[b] = sample[:n_early].mean() - sample[n_early:].mean()
    # Two-sided p under the centered null (resamples have ~0 expected gap).
    centered = stats - stats.mean()
    p = float((np.abs(centered) >= abs(obs)).mean())
    return {
        "split_date": split_date,
        "obs_gap_daily": obs,
        "obs_gap_ann": obs * TRADING_DAYS,
        "early_ann": float(early.mean() * TRADING_DAYS),
        "late_ann": float(late.mean() * TRADING_DAYS),
        "early_t": hac_tstat(early.to_numpy()),
        "late_t": hac_tstat(late.to_numpy()),
        "p_boot": p,
        "n_boot": n_boot,
        "block": block,
    }


def block_bootstrap_mean_ci(x: np.ndarray, n_boot: int = 5000, block: int = 21,
                            seed: int = 94, alpha: float = 0.05) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean of an autocorrelated series ``x``."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idxs = (starts[:, None] + np.arange(block)[None, :]) % n
        means[b] = x[idxs.ravel()][:n].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
