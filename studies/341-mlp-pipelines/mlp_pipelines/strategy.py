"""Engine and honest controls — Study 341 (MLP-Pipelines).

The question: are pipeline MLPs (AMLP) a fat-yield free lunch, or a value trap that is really a
leveraged bet on the energy complex? The answer turns on three measurements, all done here:

1. **The income illusion.** A fund's total return splits into a **price** leg (what the NAV
   does) and a **distribution** leg (cash handed out). AMLP engineers a fat 7–8% distribution
   yield — but if the NAV erodes by roughly the same amount, the "income" is just your own NAV
   handed back. ``income_illusion`` quantifies the share of the distribution not matched by NAV
   growth (the return-of-capital share).

2. **The race, done honestly.** We compare an MLP fund to **SPY total return** (the market) and
   to **XLE total return** (the energy sector) on the **same months**, and report the mean
   monthly **return spread** with an autocorrelation-robust (Newey-West / HAC) *t*-stat and a
   **circular block-bootstrap** CI of the spread. Both legs are total return, so this is a fair
   excess-vs-excess comparison of two fully-invested books. A documented one-month **execution
   lag** is available for any timing variant.

3. **The energy beta.** The decisive identity: AMLP is sold as a yield play but trades like a
   leveraged energy bet. ``energy_beta`` regresses the fund's monthly return on an energy factor
   (XLE / crude) with a **HAC *t*** on the slope and a block-bootstrap CI — the inference-bar
   number for the *Signal* axis (is the energy exposure real?). ``capture`` measures up/down
   capture vs the energy factor — the crash signature.

Costs: ``cost_bps`` is a one-way fee × NAV, charged on the modeled rebalance turnover of the
*synthetic* MLP replicator (the real ETFs' fees are already inside their reported total return).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Performance primitives
# ---------------------------------------------------------------------------
def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate of a monthly simple-return series."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    growth = float((1.0 + r).prod())
    years = len(r) / MONTHS_PER_YEAR
    return growth ** (1.0 / years) - 1.0 if years > 0 and growth > 0 else float("nan")


def ann_vol(returns: pd.Series) -> float:
    """Annualised volatility of a monthly simple-return series."""
    r = returns.dropna()
    return float(r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)) if len(r) > 1 else float("nan")


def sharpe(returns: pd.Series, rf_monthly: float = 0.0) -> float:
    """Annualised Sharpe of excess monthly returns (rf in monthly units).

    Pass the *same* ``rf_monthly`` to both legs of a race so it is an honest excess-of-cash vs
    excess-of-cash comparison.
    """
    r = (returns.dropna() - rf_monthly)
    if len(r) < 2 or r.std(ddof=1) <= 1e-15:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough drawdown of the compounded series (a negative number)."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    curve = (1.0 + r).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


# ---------------------------------------------------------------------------
# The income illusion — distribution vs NAV
# ---------------------------------------------------------------------------
def income_illusion(price: pd.Series, dist: pd.Series) -> dict:
    """Decompose an MLP fund's return into price vs distribution and size the illusion.

    ``price`` and ``dist`` are aligned monthly simple-return series (the NAV's price move and the
    cash distribution yield). Returns annualised price CAGR, annualised distribution yield,
    total-return CAGR, and ``return_of_capital_share`` — the fraction of the distribution that is
    *not* backed by NAV growth, i.e. handed back out of capital.

    ``return_of_capital_share`` near 1 means "the income is almost entirely your own money" (the
    trap); near 0 means the distribution is genuine yield on top of a flat-or-rising NAV.
    """
    p = price.dropna()
    d = dist.reindex(p.index).fillna(0.0)
    price_cagr = cagr(p)
    dist_yield = float((1.0 + d).prod()) ** (MONTHS_PER_YEAR / len(d)) - 1.0 if len(d) else float("nan")
    total_cagr = cagr(p + d)
    roc = (dist_yield - max(price_cagr, 0.0)) / dist_yield if dist_yield > 0 else float("nan")
    return {
        "price_cagr": price_cagr,
        "dist_yield": dist_yield,
        "total_cagr": total_cagr,
        "return_of_capital_share": float(np.clip(roc, 0.0, 1.0)) if np.isfinite(roc) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Up/down capture — the crash signature vs the energy factor
# ---------------------------------------------------------------------------
def capture(fund: pd.Series, factor: pd.Series) -> dict:
    """Up- and down-capture of ``fund`` vs an energy ``factor`` on aligned monthly returns.

    Up-capture = mean fund return in months the factor was up, ÷ mean factor up-return.
    Down-capture is the mirror. A leveraged energy bet shows BOTH well above (or near) 1 — it
    crashes *with* energy, the opposite of the "bond-like income" pitch.
    """
    df = pd.concat([fund, factor], axis=1, keys=["f", "b"]).dropna()
    up, dn = df[df["b"] > 0], df[df["b"] < 0]
    up_cap = float(up["f"].mean() / up["b"].mean()) if len(up) and up["b"].mean() != 0 else float("nan")
    dn_cap = float(dn["f"].mean() / dn["b"].mean()) if len(dn) and dn["b"].mean() != 0 else float("nan")
    return {
        "up_capture": up_cap,
        "down_capture": dn_cap,
        "n_up": int(len(up)),
        "n_down": int(len(dn)),
        "asymmetry": up_cap - dn_cap if np.isfinite(up_cap) and np.isfinite(dn_cap) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Robust inference primitives
# ---------------------------------------------------------------------------
def newey_west_t(x: np.ndarray, lags: int | None = None) -> tuple[float, float]:
    """HAC (Newey-West) mean and t-stat of a 1-D series. No hard quantlab dependency."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float("nan"), float("nan")
    mu = x.mean()
    e = x - mu
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu), float(mu / se) if se > 0 else float("nan")


def block_bootstrap_ci(
    x: np.ndarray,
    block: int = 6,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 341,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean of ``x`` (preserves serial dependence)."""
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
        means[b] = x[idx][:n].mean()
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# The energy beta — the decisive identity (Signal axis)
# ---------------------------------------------------------------------------
def _nw_lags(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def energy_beta(
    fund: pd.Series,
    factor: pd.Series,
    block: int = 6,
    seed: int = 341,
) -> dict:
    """Regress ``fund`` monthly returns on an energy ``factor`` with a HAC *t* on the slope.

    OLS slope (beta) of fund on factor, with an autocorrelation-robust **Newey-West HAC standard
    error** on the slope (so the *t* survives the volatility clustering of energy returns), the
    R², and a **circular block-bootstrap 95% CI** for beta. A bond-like "income" sleeve would
    have beta ≈ 0; a leveraged energy bet has beta well above 1 with a HAC *t* clearing the bar.
    """
    df = pd.concat([fund, factor], axis=1, keys=["y", "x"]).dropna()
    n = len(df)
    if n < 5:
        return {"beta": float("nan"), "beta_t": float("nan"), "r2": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "n": int(n)}
    y = df["y"].to_numpy()
    x = df["x"].to_numpy()
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    # HAC (Newey-West) covariance of the OLS coefficients.
    lags = _nw_lags(n)
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        u = X * resid[:, None]
        G = u[k:].T @ u[:-k]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    beta = float(coef[1])
    se_beta = float(np.sqrt(max(cov[1, 1], 0.0)))
    beta_t = beta / se_beta if se_beta > 0 else float("nan")
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # Block-bootstrap CI for the slope (resample contiguous (y,x) blocks, refit).
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    betas = np.empty(2000)
    for b in range(2000):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        Xb, yb = X[idx], y[idx]
        cb, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        betas[b] = cb[1]
    lo, hi = np.quantile(betas, [0.025, 0.975])
    return {"beta": beta, "beta_t": float(beta_t), "r2": float(r2),
            "ci_lo": float(lo), "ci_hi": float(hi), "n": int(n)}


# ---------------------------------------------------------------------------
# The race — return spread with HAC t and block-bootstrap CI
# ---------------------------------------------------------------------------
def race(
    fund: pd.Series,
    bench: pd.Series,
    rf_monthly: float = 0.0,
    block: int = 6,
    seed: int = 341,
) -> dict:
    """Race an MLP fund against a benchmark on aligned monthly total returns.

    Reports each leg's CAGR / vol / Sharpe / max-drawdown, the mean monthly **return spread**
    (fund − bench), its **HAC t-stat** and **block-bootstrap 95% CI**, and the annualised Sharpe
    *difference*. A "fat-yield free lunch" must NOT show a robustly negative spread vs the market.
    """
    df = pd.concat([fund, bench], axis=1, keys=["f", "b"]).dropna()
    spread = (df["f"] - df["b"]).to_numpy()
    mu, t = newey_west_t(spread)
    lo, hi = block_bootstrap_ci(spread, block=block, seed=seed)
    return {
        "n": int(len(df)),
        "fund_cagr": cagr(df["f"]),
        "bench_cagr": cagr(df["b"]),
        "fund_vol": ann_vol(df["f"]),
        "bench_vol": ann_vol(df["b"]),
        "fund_sharpe": sharpe(df["f"], rf_monthly),
        "bench_sharpe": sharpe(df["b"], rf_monthly),
        "fund_maxdd": max_drawdown(df["f"]),
        "bench_maxdd": max_drawdown(df["b"]),
        "spread_mean_bps_mo": float(mu * 1e4),
        "spread_ann_pct": float(((1.0 + df["f"].mean()) ** 12 - (1.0 + df["b"].mean()) ** 12) * 100),
        "spread_t": t,
        "spread_ci_lo_bps": float(lo * 1e4) if np.isfinite(lo) else float("nan"),
        "spread_ci_hi_bps": float(hi * 1e4) if np.isfinite(hi) else float("nan"),
        "sharpe_diff": (sharpe(df["f"], rf_monthly) - sharpe(df["b"], rf_monthly)),
    }


# ---------------------------------------------------------------------------
# Synthetic MLP replicator with costs / one execution lag
# ---------------------------------------------------------------------------
def replicate_mlp(
    energy_tr: pd.Series,
    beta: float,
    dist: float,
    nav_drift: float = 0.0,
    cost_bps: float = 0.0,
    lag: int = 0,
) -> pd.DataFrame:
    """Mechanically replicate an MLP fund from an energy total-return series.

    Each month: price leg = ``beta * log(1+energy) + nav_drift`` (a levered multiple of the
    energy factor plus a steady NAV bleed), distribution leg = ``dist``. ``cost_bps`` is a
    **one-way fee × NAV** deducted for the modeled monthly rebalance. ``lag`` shifts the energy
    exposure by ``lag`` months as a documented execution-lag robustness knob (the convention: an
    exposure decided at the close of *t* binds the return of *t+lag*; ``lag=0`` is the
    contemporaneous replication, ``lag=1`` the conservative one-month-lagged variant).

    Columns: ``price, dist, total_gross, total_net``.
    """
    e = np.log1p(energy_tr.dropna())
    exposure = beta * e
    if lag:
        exposure = beta * e.shift(lag).fillna(0.0)
    price = exposure + nav_drift
    d = pd.Series(dist, index=e.index)
    total_gross = price + d
    total_net = total_gross - cost_bps * 1e-4
    return pd.DataFrame(
        {"price": price, "dist": d, "total_gross": total_gross, "total_net": total_net}
    )
