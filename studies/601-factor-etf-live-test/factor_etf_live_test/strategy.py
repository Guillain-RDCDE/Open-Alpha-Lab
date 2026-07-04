"""Stats layer for Study 601 — Factor ETFs, Live Test.

The tested quantities, per fund vs SPY on monthly total returns:

* **CAPM regression** (excess-vs-excess) with Newey-West (HAC) errors — beta, the
  beta-below-one *t* (the low-vol delivery test), annualised alpha and its NW *t*
  (the alpha-delivery test).
* **Style-loading regression** — fund excess on [market excess, style-spread factor],
  NW *t* of the spread loading (the exposure-delivery test for MTUM/VLUE/QUAL).
* **Realized vol ratio** with a paired moving-block bootstrap CI (USMV's mechanical
  promise), **up/down capture**, annual stats and an excess-vs-excess Sharpe race.
* **Spread-sign month split** (Welch *t*) + a same-density random-calendar placebo —
  "in months when the style factor won, did the fund beat SPY?"

Everything is deterministic (fixed seeds); pure numpy + pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


# --------------------------------------------------------------------------- #
# HAC / Newey-West machinery
# --------------------------------------------------------------------------- #
def nw_tstat(x: pd.Series, lags: int = 6) -> float:
    """Newey-West (HAC) t of the mean of a monthly series."""
    v = np.asarray(x.dropna(), dtype=float)
    n = len(v)
    if n < 10:
        return np.nan
    e = v - v.mean()
    s0 = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        s0 += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(s0 / n)
    return float(v.mean() / se) if se > 0 else np.nan


def _nw_ols(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS with a Newey-West covariance; returns (beta, se, r2)."""
    n = len(y)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    G = X * resid[:, None]
    S = (G.T @ G) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        gk = (G[k:].T @ G[:-k]) / n
        S += w * (gk + gk.T)
    cov = XtX_inv @ (n * S) @ XtX_inv
    se = np.sqrt(np.diag(cov))
    denom = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / denom if denom > 0 else np.nan
    return beta, se, r2


def capm(fund_ex: pd.Series, mkt_ex: pd.Series, lags: int = 6) -> dict:
    """CAPM on monthly EXCESS returns: fund = a + b·mkt, Newey-West errors.

    Reports the alpha (annualised, with NW t — the alpha-delivery test) and the beta
    with BOTH the plain t and the beta-vs-ONE t ((b-1)/se — the risk-profile test:
    a low-vol product must show b significantly below 1).
    """
    df = pd.concat([fund_ex, mkt_ex], axis=1).dropna()
    y = df.iloc[:, 0].to_numpy(dtype=float)
    x = df.iloc[:, 1].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)), x])
    beta, se, r2 = _nw_ols(y, X, lags)
    return {"alpha_m": float(beta[0]), "alpha_ann": float(beta[0] * MONTHS),
            "t_alpha": float(beta[0] / se[0]) if se[0] > 0 else np.nan,
            "beta": float(beta[1]), "se_beta": float(se[1]),
            "t_beta_vs1": float((beta[1] - 1.0) / se[1]) if se[1] > 0 else np.nan,
            "r2": r2, "n": len(y)}


def style_loading(fund_ex: pd.Series, mkt_ex: pd.Series, spread: pd.Series,
                  lags: int = 6) -> dict:
    """Two-factor regression: fund excess = a + b·mkt excess + s·spread, NW errors.

    The spread loading ``s`` (with its NW t) is the exposure-delivery statistic: a fund
    that tracks its stated style must load positively and significantly on the style's
    realized long-short spread.
    """
    df = pd.concat([fund_ex, mkt_ex, spread], axis=1).dropna()
    y = df.iloc[:, 0].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)),
                         df.iloc[:, 1].to_numpy(dtype=float),
                         df.iloc[:, 2].to_numpy(dtype=float)])
    beta, se, r2 = _nw_ols(y, X, lags)
    return {"alpha_ann": float(beta[0] * MONTHS),
            "t_alpha": float(beta[0] / se[0]) if se[0] > 0 else np.nan,
            "beta": float(beta[1]),
            "loading": float(beta[2]),
            "t_loading": float(beta[2] / se[2]) if se[2] > 0 else np.nan,
            "r2": r2, "n": len(y)}


# --------------------------------------------------------------------------- #
# Risk profile / performance
# --------------------------------------------------------------------------- #
def ann_stats(total_ret: pd.Series, rf: pd.Series) -> dict:
    """CAGR, annualised vol, excess Sharpe, max drawdown on monthly TOTAL returns."""
    df = pd.concat([total_ret, rf], axis=1).dropna()
    r, f = df.iloc[:, 0], df.iloc[:, 1]
    n = len(r)
    wealth = float((1.0 + r).prod())
    cagr = wealth ** (MONTHS / n) - 1.0
    vol = float(r.std() * np.sqrt(MONTHS))
    ex = r - f
    sharpe = float(ex.mean() / ex.std() * np.sqrt(MONTHS)) if ex.std() > 0 else np.nan
    curve = (1.0 + r).cumprod()
    dd = float((curve / curve.cummax() - 1.0).min())
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "maxdd": dd,
            "wealth": wealth, "n_months": n}


def vol_ratio_ci(fund: pd.Series, bench: pd.Series, n_draws: int = 2000,
                 block: int = 6, seed: int = 601) -> dict:
    """Realized-vol ratio fund/bench with a PAIRED moving-block bootstrap 95% CI.

    Joint monthly rows are resampled in blocks so both the serial and the cross
    correlation survive; the ratio of annualised vols is recomputed per draw.
    """
    df = pd.concat([fund, bench], axis=1).dropna()
    a = df.iloc[:, 0].to_numpy(dtype=float)
    b = df.iloc[:, 1].to_numpy(dtype=float)
    n = len(a)
    obs = float(a.std() / b.std())
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    draws = np.empty(n_draws)
    for i in range(n_draws):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        draws[i] = a[idx].std() / b[idx].std()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"obs": obs, "lo": float(lo), "hi": float(hi), "n_months": n,
            "n_draws": n_draws, "block": block}


def up_down_capture(fund: pd.Series, bench: pd.Series) -> dict:
    """Up/down capture on monthly TOTAL returns (mean fund / mean bench, per regime)."""
    df = pd.concat([fund, bench], axis=1).dropna()
    f, b = df.iloc[:, 0], df.iloc[:, 1]
    up, dn = b > 0, b < 0
    up_cap = float(f[up].mean() / b[up].mean()) if up.any() else np.nan
    dn_cap = float(f[dn].mean() / b[dn].mean()) if dn.any() else np.nan
    return {"up": up_cap, "down": dn_cap, "n_up": int(up.sum()), "n_down": int(dn.sum())}


# --------------------------------------------------------------------------- #
# Spread-sign month split + placebo
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return float((a.mean() - b.mean()) / np.sqrt(va + vb)) if (va + vb) > 0 else np.nan


def spread_sign_split(active: pd.Series, spread: pd.Series,
                      n_draws: int = 10_000, seed: int = 601) -> dict:
    """Active return (fund - SPY) split by the SIGN of the style spread's month.

    A fund that carries its stated exposure must out-run SPY in months its style factor
    won and lag in months it lost. Welch t of the two-group difference + a same-density
    random-calendar placebo p (share of ``n_draws`` random month-taggings whose split
    beats the observed one). Deterministic (fixed seed).
    """
    df = pd.concat([active, spread], axis=1).dropna()
    act = df.iloc[:, 0].to_numpy(dtype=float)
    pos = df.iloc[:, 1].to_numpy(dtype=float) > 0
    a, b = act[pos], act[~pos]
    obs = float(a.mean() - b.mean())
    t = welch_t(a, b)
    rng = np.random.default_rng(seed)
    n, k = len(act), int(pos.sum())
    beats = 0
    for _ in range(n_draws):
        idx = rng.permutation(n)[:k]
        m = np.zeros(n, dtype=bool)
        m[idx] = True
        if act[m].mean() - act[~m].mean() >= obs:
            beats += 1
    return {"diff_m": obs, "diff_bps": obs * 1e4, "welch_t": t,
            "p_placebo": beats / n_draws, "n_pos": k, "n_neg": n - k,
            "mean_pos_bps": float(a.mean() * 1e4), "mean_neg_bps": float(b.mean() * 1e4)}
