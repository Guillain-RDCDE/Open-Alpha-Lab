"""The convexity engine and its honest controls — Study 339 (Convertible-Bonds).

The folk claim: *'convertible bonds give you equity upside with bond downside — a convex,
asymmetric payoff that catches the rally but is cushioned in the crash by the bond floor.'*
We test the two halves separately:

1. **Is the payoff actually convex?** Regress the convertible's daily return on the equity
   index return *and* the squared upside of the index (a quadratic term). A real convexity
   claim needs a **positive, significant gamma** (the quadratic coefficient) under
   autocorrelation-robust (Newey-West) inference — not just a positive slope.

2. **Is it more than a re-priced linear blend?** Build a plain **stock/bond blend** whose
   equity beta is matched to the convertible's, race it on excess-of-cash Sharpe, and
   compare **up-capture vs down-capture**. Genuine convexity means the convertible keeps
   more of the up days than the down days *beyond* what the matched linear blend does.

Conventions: returns are simple daily total returns from the price frame; the matched blend
is a fixed-weight calendar rebalance (no signal, no execution lag needed); rebalance turnover
is charged one-way at ``cost_bps``; Sharpe races are excess-of-cash (a cash leg supplied as
``rf``) so a part-bond instrument and a part-bond blend compare like-for-like.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Return helpers
# ---------------------------------------------------------------------------
def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from a total-return price frame (first row dropped)."""
    return prices.pct_change().dropna()


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


# ---------------------------------------------------------------------------
# The matched stock/bond blend — fixed weights, calendar rebalance
# ---------------------------------------------------------------------------
def beta_to(returns: pd.DataFrame, asset: str, index: str) -> float:
    """OLS beta of ``asset`` daily returns on ``index`` daily returns."""
    x = returns[index].to_numpy(dtype=float)
    y = returns[asset].to_numpy(dtype=float)
    vx = x.var(ddof=1)
    return float(np.cov(x, y, ddof=1)[0, 1] / vx) if vx > 0 else float("nan")


def matched_blend(
    returns: pd.DataFrame,
    stock: str,
    bond: str,
    w_stock: float,
    rebalance: str = "annual",
    cost_bps: float = 0.0,
) -> pd.Series:
    """Daily net return of a fixed-weight ``w_stock`` stock / ``1-w_stock`` bond blend.

    Weights drift between rebalances; on each rebalance date the book resets to the target
    and the one-way turnover (sum of absolute weight changes) is charged at ``cost_bps``.
    ``rebalance='annual'`` resets on the first trading day of each calendar year; ``'monthly'``
    on the first of each month; ``'none'`` buys-and-holds the initial mix. Choose ``w_stock``
    to match the convertible's equity beta so the comparison isolates *convexity*, not
    *exposure*.
    """
    cols = [stock, bond]
    R = returns[cols].to_numpy(dtype=float)
    w_target = np.array([w_stock, 1.0 - w_stock], dtype=float)
    n = R.shape[0]
    idx = returns.index

    if rebalance == "annual":
        marks = idx.to_series().groupby(idx.year).head(1).index
    elif rebalance == "monthly":
        marks = idx.to_series().groupby([idx.year, idx.month]).head(1).index
    elif rebalance == "none":
        marks = idx[:1]
    else:
        raise ValueError(f"unknown rebalance schedule: {rebalance!r}")
    rebal = pd.Index(marks)

    w = w_target.copy()
    out = np.empty(n)
    cost = cost_bps * 1e-4
    for t in range(n):
        if idx[t] in rebal:
            turn = np.abs(w_target - w).sum()
            w = w_target.copy()
        else:
            turn = 0.0
        out[t] = float(w @ R[t]) - turn * cost
        w = w * (1.0 + R[t])
        s = w.sum()
        if s != 0:
            w = w / s
    return pd.Series(out, index=idx, name="blend")


def single_asset(returns: pd.DataFrame, ticker: str) -> pd.Series:
    """The daily return series of one asset (e.g. the convertible ETF)."""
    return returns[ticker].rename(ticker)


# ---------------------------------------------------------------------------
# Convexity measurement — the heart of the study
# ---------------------------------------------------------------------------
def convexity_regression(
    returns: pd.DataFrame, asset: str, index: str
) -> dict:
    """Quadratic regression of ``asset`` returns on the index: r_a = a + b·r + g·up² + e.

    ``up = max(r, 0)`` so ``up²`` isolates *upside* curvature; a positive ``g`` (gamma) means
    the asset captures big up days more than proportionally — option-like convexity. The
    quadratic term is the planted/claimed signal. Returns the coefficients, the gamma series
    of per-day contributions (for a HAC *t* on gamma), the R² and the matched fit.
    """
    r = returns[index].to_numpy(dtype=float)
    ya = returns[asset].to_numpy(dtype=float)
    up2 = np.maximum(r, 0.0) ** 2
    X = np.column_stack([np.ones_like(r), r, up2])
    coef, *_ = np.linalg.lstsq(X, ya, rcond=None)
    a, b, g = (float(c) for c in coef)
    fitted = X @ coef
    resid = ya - fitted
    ss_res = float(resid @ resid)
    ss_tot = float(((ya - ya.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # HAC t on gamma via Frisch-Waugh-Lovell: partial r and the constant out of up2, then
    # gamma = (up2_perp . ya) / (up2_perp . up2_perp). Its HAC-robust SE uses the per-obs
    # influence series (up2_perp * full_resid): SE(gamma) = sqrt(n*LRV(infl)) / denom.
    Z = np.column_stack([np.ones_like(r), r])
    bz, *_ = np.linalg.lstsq(Z, up2, rcond=None)
    up2_perp = up2 - Z @ bz
    denom = float(up2_perp @ up2_perp)
    g_check = float((up2_perp @ ya) / denom) if denom > 0 else float("nan")
    infl = up2_perp * resid  # influence (meat) series for the gamma sandwich SE
    t_gamma = _gamma_hac_tstat(g_check, infl, denom)
    return {
        "alpha": a, "beta": b, "gamma": g, "gamma_fwl": g_check,
        "r2": r2, "t_gamma": t_gamma,
        "infl": infl, "denom": denom, "n": len(r),
    }


def _gamma_hac_tstat(gamma: float, infl: np.ndarray, denom: float,
                     lags: int | None = None) -> float:
    """HAC (Newey-West) t-stat for an FWL coefficient ``gamma`` with influence series ``infl``.

    For ``gamma = (z . y) / (z . z)`` with the partialled-out regressor ``z`` (here ``up2_perp``),
    the sandwich variance is ``Var(gamma) = [sum_t LRV(z_t * resid_t)] / denom^2`` where ``denom
    = z . z`` and ``LRV`` is the Newey-West long-run variance summed over observations. So
    ``SE(gamma) = sqrt(n * LRV(infl)) / denom`` and ``t = gamma / SE(gamma)``.
    """
    s = np.asarray(infl, dtype=float)
    s = s[np.isfinite(s)]
    n = s.size
    if n <= 5 or denom <= 0:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    # long-run variance of the influence series (NOT de-meaned: it is a score, mean ~0)
    lrv = float(s @ s) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(s[k:] @ s[:-k]) / n
    se = np.sqrt(n * max(lrv, 0.0)) / denom
    return float(gamma / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Up / down capture — the asymmetry the brochure sells
# ---------------------------------------------------------------------------
def capture_ratios(returns: pd.DataFrame, asset: str, index: str) -> dict:
    """Up-capture and down-capture of ``asset`` vs ``index`` on the index's up/down days.

    Up-capture = mean(asset return on index-up days) / mean(index return on index-up days);
    down-capture analogously on index-down days. The convexity *brochure* claim is
    up-capture noticeably above down-capture (you keep more of the rally than the fall).
    """
    r = returns[index]
    a = returns[asset]
    up = r > 0
    dn = r < 0
    up_cap = float(a[up].mean() / r[up].mean()) if up.any() and r[up].mean() != 0 else float("nan")
    dn_cap = float(a[dn].mean() / r[dn].mean()) if dn.any() and r[dn].mean() != 0 else float("nan")
    return {"up_capture": up_cap, "down_capture": dn_cap,
            "asymmetry": up_cap - dn_cap, "n_up": int(up.sum()), "n_dn": int(dn.sum())}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def stats(net: pd.Series, rf: pd.Series | None = None) -> dict:
    """Headline stats for a daily return series.

    If ``rf`` (a daily cash return series) is given, the Sharpe is computed on the
    **excess-of-cash** return so two arms with different cash exposure are compared
    like-for-like; CAGR/vol/drawdown stay on the raw total-return series.
    """
    net = net.astype(float)
    equity = (1.0 + net).cumprod()
    n = len(net)
    years = n / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    if rf is not None:
        ex = (net - rf.reindex(net.index).fillna(0.0)).astype(float)
    else:
        ex = net
    sharpe = (float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS))
              if ex.std(ddof=1) > 0 else float("nan"))
    return {
        "net": net, "equity": equity, "cagr": cagr, "vol": vol, "sharpe": sharpe,
        "max_dd": _max_drawdown(equity.to_numpy()), "final": float(equity.iloc[-1]),
    }


# ---------------------------------------------------------------------------
# Inference: HAC t-stat and circular block bootstrap on a difference series
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


def _annualised_sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def bootstrap_sharpe_diff(
    a: pd.Series,
    b: pd.Series,
    rf: pd.Series | None = None,
    block: int = 21,
    n_boot: int = 2000,
    seed: int = 339,
) -> dict:
    """Circular block bootstrap CI for the Sharpe *difference* (arm ``a`` − arm ``b``).

    Resamples the two aligned excess-return series **jointly** in circular blocks (so the
    cross-correlation and the volatility clustering both survive), recomputes each arm's
    annualised Sharpe per resample, and returns the point difference, a 95% CI, and the
    bootstrap fraction of resamples in which ``a``'s Sharpe exceeds ``b``'s.
    """
    idx = a.index.intersection(b.index)
    ra = a.reindex(idx).to_numpy(dtype=float)
    rb = b.reindex(idx).to_numpy(dtype=float)
    if rf is not None:
        f = rf.reindex(idx).fillna(0.0).to_numpy(dtype=float)
        ra = ra - f
        rb = rb - f
    n = len(idx)
    point = _annualised_sharpe(ra) - _annualised_sharpe(rb)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    a_wins = 0
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        offsets = (starts[:, None] + np.arange(block)[None, :]) % n
        sel = offsets.reshape(-1)[:n]
        sa, sb = _annualised_sharpe(ra[sel]), _annualised_sharpe(rb[sel])
        diffs[i] = sa - sb
        if sa > sb:
            a_wins += 1
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "point": float(point), "ci95": (float(lo), float(hi)),
        "frac_a_wins": a_wins / n_boot, "n": n, "block": block, "n_boot": n_boot,
    }


def bootstrap_gamma(
    returns: pd.DataFrame, asset: str, index: str,
    block: int = 21, n_boot: int = 2000, seed: int = 339,
) -> dict:
    """Circular block bootstrap CI for the convexity coefficient gamma.

    Resamples the (asset, index) return pair jointly in circular blocks and re-runs the
    quadratic regression each time, returning the point gamma, a 95% CI and the fraction of
    resamples with gamma > 0. A genuine convexity claim wants the whole CI above zero.
    """
    r = returns[index].to_numpy(dtype=float)
    a = returns[asset].to_numpy(dtype=float)
    n = len(r)
    point = convexity_regression(returns, asset, index)["gamma"]
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    gammas = np.empty(n_boot)
    pos = 0
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        offsets = (starts[:, None] + np.arange(block)[None, :]) % n
        sel = offsets.reshape(-1)[:n]
        rr, aa = r[sel], a[sel]
        up2 = np.maximum(rr, 0.0) ** 2
        X = np.column_stack([np.ones_like(rr), rr, up2])
        coef, *_ = np.linalg.lstsq(X, aa, rcond=None)
        gammas[i] = coef[2]
        if coef[2] > 0:
            pos += 1
    lo, hi = np.percentile(gammas, [2.5, 97.5])
    return {"point": float(point), "ci95": (float(lo), float(hi)),
            "frac_pos": pos / n_boot, "n": n, "block": block, "n_boot": n_boot}
