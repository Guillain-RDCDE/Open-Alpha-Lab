"""Strategy + inference for Study 356 — the GLP-1 / Ozempic basket.

The theme: buy LLY + NVO (a 50/50 weight-loss-drug basket) to ride the Ozempic boom; does
it beat SPY and the healthcare sector (XLV)? The Signal axis is *alpha* — is the basket's
excess return over a benchmark statistically distinguishable from zero once we use
autocorrelation-robust (Newey-West / HAC) inference, not a naïve i.i.d. t?

Three tools settle the study:

* :func:`perf` / :func:`excess_test` — annualised return, vol, Sharpe, and the **HAC-t** of
  the mean excess return over a benchmark (the Signal test, t >= 2 to be ``REAL``).
* :func:`market_model` — a CAPM-style regression ``r = alpha + beta*mkt`` with a **HAC-t on
  alpha**; run per name it performs the decisive *decomposition* (is the basket's edge one
  name?) and the *recency* split (does it survive the GLP-1-mania sub-period?).
* :func:`cost_sweep` / :func:`max_drawdown` — the Tradability teardown: one-way costs on
  rebalancing turnover, and the concentration risk (drawdown) of a two-name bet.

Pure numpy / pandas. Deterministic. No network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Performance summary
# --------------------------------------------------------------------------- #
def perf(rets: pd.Series, ann: int = TRADING_DAYS) -> dict:
    """Annualised return, vol, Sharpe (rf=0), naïve t(mean>0), and total compounded return."""
    x = np.asarray(rets, dtype=float)
    n = len(x)
    mu_ann = x.mean() * ann
    vol_ann = x.std(ddof=1) * np.sqrt(ann)
    sharpe = mu_ann / vol_ann if vol_ann else float("nan")
    t_naive = x.mean() / x.std(ddof=1) * np.sqrt(n) if n > 1 else float("nan")
    total = float(np.prod(1.0 + x) - 1.0)
    return {"n": n, "cagr": float(mu_ann), "vol": float(vol_ann),
            "sharpe": float(sharpe), "t_naive": float(t_naive), "total": total}


def max_drawdown(rets: pd.Series) -> float:
    """Worst peak-to-trough fractional drawdown of the compounded equity curve."""
    eq = np.cumprod(1.0 + np.asarray(rets, dtype=float))
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


# --------------------------------------------------------------------------- #
# HAC (Newey-West) inference
# --------------------------------------------------------------------------- #
def _nw_lag(n: int) -> int:
    """Automatic Newey-West truncation lag, floor(4 (n/100)^(2/9))."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_t_mean(x: np.ndarray, lag: int | None = None) -> tuple[float, int]:
    """HAC (Newey-West) t-statistic for the null ``mean(x) = 0``. Returns ``(t, lag)``."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    e = x - m
    L = _nw_lag(n) if lag is None else lag
    s = float(np.dot(e, e) / n)               # gamma_0
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)               # Bartlett kernel
        s += 2.0 * w * float(np.dot(e[l:], e[:-l]) / n)
    se = np.sqrt(s / n)
    return (m / se if se else float("nan")), L


def excess_test(rets: pd.Series, bench: pd.Series, ann: int = TRADING_DAYS) -> dict:
    """Mean *excess* return of ``rets`` over ``bench`` with its HAC-t (the Signal test).

    Excess is compared like-for-like (both total-return daily series). HAC-t >= 2 is the bar
    for a ``REAL`` signal; below it the excess is not distinguishable from noise.
    """
    ex = np.asarray(rets, dtype=float) - np.asarray(bench, dtype=float)
    t, L = hac_t_mean(ex)
    return {"ann_excess": float(ex.mean() * ann), "t_hac": float(t),
            "lag": L, "n": len(ex)}


def market_model(y: pd.Series, mkt: pd.Series, ann: int = TRADING_DAYS) -> dict:
    """CAPM-style OLS ``y = alpha + beta*mkt`` with a **HAC-t on alpha** (and on beta).

    Run with ``mkt = SPY`` it asks "is there alpha beyond market beta?"; per single name it
    decomposes *where* any alpha lives. Returns annualised alpha, its HAC-t, beta, and R².
    """
    y = np.asarray(y, dtype=float)
    m = np.asarray(mkt, dtype=float)
    n = len(y)
    X = np.column_stack([np.ones(n), m])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    L = _nw_lag(n)
    Xu = X * resid[:, None]
    S = Xu.T @ Xu
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        Gl = Xu[l:].T @ Xu[:-l]
        S += w * (Gl + Gl.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot else float("nan")
    return {"alpha_daily": float(beta[0]), "alpha_ann": float(beta[0] * ann),
            "t_alpha": float(beta[0] / se[0]), "beta": float(beta[1]),
            "t_beta": float(beta[1] / se[1]), "r2": float(r2), "n": n}


def block_bootstrap_excess(rets: pd.Series, bench: pd.Series, block: int = 21,
                           n_boot: int = 5000, ann: int = TRADING_DAYS,
                           seed: int = 356) -> dict:
    """Circular-block bootstrap of the annualised mean excess return.

    Block resampling preserves the autocorrelation an i.i.d. bootstrap would destroy.
    Returns the point estimate, a 95% CI, and P(excess > 0).
    """
    ex = (np.asarray(rets, dtype=float) - np.asarray(bench, dtype=float))
    n = len(ex)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = []
        while len(idx) < n:
            s = int(rng.integers(0, n))
            idx.extend((np.arange(s, s + block) % n).tolist())
        means[b] = ex[np.asarray(idx[:n])].mean()
    return {"point": float(ex.mean() * ann),
            "lo": float(np.percentile(means, 2.5) * ann),
            "hi": float(np.percentile(means, 97.5) * ann),
            "p_pos": float((means > 0).mean())}


# --------------------------------------------------------------------------- #
# Tradability — costs on turnover
# --------------------------------------------------------------------------- #
def turnover_per_day(rets: pd.DataFrame, weights=None) -> float:
    """Mean one-way fraction of NAV traded to restore fixed weights each day.

    For a daily-rebalanced fixed-weight basket the drift to undo is tiny, so turnover (and
    thus cost) is small — the point being that costs are *not* what kills this idea; the
    concentration is.
    """
    if weights is None:
        weights = {"LLY": 0.5, "NVO": 0.5}
    w = pd.Series(weights)
    r = rets[w.index].to_numpy()
    tgt = w.to_numpy()
    drift = tgt * (1.0 + r)
    drift = drift / drift.sum(axis=1, keepdims=True)         # post-drift weights
    traded = np.abs(tgt - drift).sum(axis=1) / 2.0           # one-way turnover
    return float(traded.mean())


def cost_sweep(basket: pd.Series, bench: pd.Series, rets: pd.DataFrame,
               bps_grid=(0, 1, 2, 5, 10), weights=None,
               ann: int = TRADING_DAYS) -> list[dict]:
    """Annualised net excess over ``bench`` as one-way cost (bps of traded NAV) rises."""
    tpd = turnover_per_day(rets, weights)
    out = []
    for bps in bps_grid:
        cost_daily = tpd * (bps / 1e4)
        net = np.asarray(basket, dtype=float) - cost_daily
        ex = net - np.asarray(bench, dtype=float)
        t, _ = hac_t_mean(ex)
        out.append({"bps": bps, "ann_excess": float(ex.mean() * ann), "t_hac": float(t)})
    return out
