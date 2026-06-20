"""Engine and honest controls — Study 337 (Covered-Call-ETF).

The question: do buy-write "income" ETFs (JEPI/QYLD) give you yield *without* giving up
returns? The answer turns on three measurements, all done here:

1. **The income illusion.** A fund's total return splits into a **price** leg (what the NAV
   does) and a **distribution** leg (cash handed out). A buy-write engineers a fat
   distribution yield — but if the price line erodes by roughly the same amount, the
   "income" is just your own NAV paid back to you. ``income_illusion`` quantifies the share
   of the distribution that is *not* matched by total-return outperformance.

2. **The race, done honestly.** We compare a buy-write ETF to **SPY total return** on the
   **same months**, and report the mean monthly **return spread** with an autocorrelation-
   robust (Newey-West / HAC) *t*-stat and a **circular block-bootstrap** CI of the spread —
   the inference-bar number. Both legs are total return (distributions reinvested), so this is
   a fair excess-vs-excess comparison of two fully-invested equity books (no cash leg to mis-
   net). A documented one-month **execution lag** is available for any timing variant.

3. **Capped upside.** ``capture`` measures up-capture and down-capture vs the benchmark — the
   structural signature of a covered call: it keeps only part of the up months while taking
   most of the down months.

Costs: ``cost_bps`` is a one-way fee × NAV, charged on the modeled rebalance turnover of the
*synthetic* buy-write (the real ETFs' fees are already inside their reported total return).
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

    Pass the *same* ``rf_monthly`` to both legs of a race so it is an honest
    excess-of-cash vs excess-of-cash comparison.
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
    """Decompose a buy-write's return into price vs distribution and size the illusion.

    ``price`` and ``dist`` are aligned monthly simple-return series (the NAV's price move and
    the cash distribution yield). Returns annualised price CAGR, annualised distribution yield,
    total-return CAGR, and ``return_of_capital_share`` — the fraction of the distribution that
    is *not* backed by NAV growth, i.e. handed back out of capital.

    ``return_of_capital_share`` near 1 means "the income is almost entirely your own money"
    (the illusion); near 0 means the distribution is genuine yield on top of a flat-or-rising
    NAV.
    """
    p = price.dropna()
    d = dist.reindex(p.index).fillna(0.0)
    price_cagr = cagr(p)
    dist_yield = float((1.0 + d).prod()) ** (MONTHS_PER_YEAR / len(d)) - 1.0 if len(d) else float("nan")
    total_cagr = cagr(p + d)
    # Of the distribution paid, how much was NOT covered by NAV appreciation?
    roc = (dist_yield - max(price_cagr, 0.0)) / dist_yield if dist_yield > 0 else float("nan")
    return {
        "price_cagr": price_cagr,
        "dist_yield": dist_yield,
        "total_cagr": total_cagr,
        "return_of_capital_share": float(np.clip(roc, 0.0, 1.0)) if np.isfinite(roc) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Up/down capture — the capped-upside signature
# ---------------------------------------------------------------------------
def capture(fund: pd.Series, bench: pd.Series) -> dict:
    """Up- and down-capture of ``fund`` vs ``bench`` on aligned monthly returns.

    Up-capture = mean fund return in months the bench was up, ÷ mean bench up-return.
    Down-capture is the mirror. A covered call's tell is up-capture well below 1 (upside
    capped) while down-capture stays near 1 (downside uncapped) — keeping less of the good
    months than the bad ones.
    """
    df = pd.concat([fund, bench], axis=1, keys=["f", "b"]).dropna()
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
# The race — return spread with HAC t and block-bootstrap CI
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
    seed: int = 337,
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


def race(
    fund: pd.Series,
    bench: pd.Series,
    rf_monthly: float = 0.0,
    block: int = 6,
    seed: int = 337,
) -> dict:
    """Race a buy-write fund against a benchmark on aligned monthly total returns.

    Reports each leg's CAGR / vol / Sharpe / max-drawdown, the mean monthly **return spread**
    (fund − bench), its **HAC t-stat** and **block-bootstrap 95% CI**, and the annualised
    Sharpe *difference*. The spread's t-stat is the inference-bar number: a buy-write that
    "gives yield without giving up returns" must NOT show a robustly negative spread.
    """
    df = pd.concat([fund, bench], axis=1, keys=["f", "b"]).dropna()
    spread = (df["f"] - df["b"]).to_numpy()
    mu, t = newey_west_t(spread)
    lo, hi = block_bootstrap_ci(spread, block=block, seed=seed)
    return {
        "n": int(len(df)),
        "fund_cagr": cagr(df["f"]),
        "bench_cagr": cagr(df["b"]),
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
# Synthetic buy-write replicator with costs / one execution lag
# ---------------------------------------------------------------------------
def replicate_buywrite(
    index_tr: pd.Series,
    cap: float,
    premium: float,
    cost_bps: float = 0.0,
    lag: int = 0,
) -> pd.DataFrame:
    """Mechanically replicate a buy-write from an index total-return series.

    Each month: price leg = index capped at ``cap`` (``min(g, cap)`` in log space), distribution
    leg = ``premium``. ``cost_bps`` is a **one-way fee × NAV** deducted for rolling the option
    each month (one-way because you write a fresh call monthly). ``lag`` shifts the strike-cap
    application by ``lag`` months as a documented execution-lag robustness knob (the convention:
    a cap decided at the close of *t* binds the return of *t+lag*; ``lag=0`` is the contemporaneous
    mechanical replication, ``lag=1`` the conservative one-month-lagged variant).

    Columns: ``price, dist, total_gross, total_net``.
    """
    g = np.log1p(index_tr.dropna())
    capped = np.minimum(g, cap)
    if lag:
        # Apply the cap with a delay: months not yet "capped" pass the raw index through.
        capped = pd.Series(np.where(np.arange(len(g)) >= lag,
                                    np.minimum(g.shift(lag).fillna(g), cap), g),
                           index=g.index)
    price = np.expm1(capped)
    dist = pd.Series(premium, index=g.index)
    total_gross = price + dist
    total_net = total_gross - cost_bps * 1e-4
    return pd.DataFrame(
        {"price": price, "dist": dist, "total_gross": total_gross, "total_net": total_net}
    )
