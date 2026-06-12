"""The book — Global Equities Momentum (dual momentum), the way Antonacci defines it.

Three doors, decided monthly on trailing 12-month total return:

  1. **Relative momentum** — between US and foreign equity, prefer the stronger of the two.
  2. **Absolute momentum** — but only hold that equity if its 12-month return beats the **T-bill**
     over the same window; otherwise step aside into **bonds**. This is the crash filter: in a
     sustained bear market both equities fall below cash, and the book sits in bonds.

The decision is formed at each month-end and applied to the *next* month's return (no look-ahead).
A switch costs ``cost_bps`` once. Everything else in this module is benchmarks and metrics, so the
study can ask the only fair question: does GEM beat just holding a 60/40?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
DOORS = ("US", "INTL", "BOND")


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Monthly simple returns from a monthly total-return price frame."""
    return prices.pct_change()


def _mom(prices: pd.Series, lookback: int) -> pd.Series:
    return prices / prices.shift(lookback) - 1.0


def gem_hold(prices: pd.DataFrame, tbill_m: pd.Series, lookback: int = 12) -> pd.Series:
    """The door held **during** each month, decided at the prior month-end (so it's tradable).

    Returns a Series of {'US','INTL','BOND'} (NaN until the lookback is warm). Relative momentum
    picks the stronger equity; absolute momentum demotes it to BOND when even the winner trails the
    T-bill over the lookback window.
    """
    m_us, m_intl = _mom(prices["US"], lookback), _mom(prices["INTL"], lookback)
    tbill_lb = (1.0 + tbill_m).rolling(lookback).apply(np.prod, raw=True) - 1.0
    winner = np.where(m_us >= m_intl, "US", "INTL")
    eq_mom = np.maximum(m_us, m_intl)
    decision = pd.Series(np.where(eq_mom > tbill_lb, winner, "BOND"), index=prices.index)
    # warm-up months (any input NaN) are not investable decisions
    warm = m_us.notna() & m_intl.notna() & tbill_lb.notna()
    decision = decision.where(warm)
    return decision.shift(1).rename("hold")  # decided at t-1 close, held during t


def gem_returns(
    prices: pd.DataFrame, tbill_m: pd.Series, lookback: int = 12, cost_bps: float = 20.0
) -> pd.Series:
    """Net monthly return of the GEM book: held-door return minus a one-off cost on each switch."""
    rets = to_returns(prices)
    hold = gem_hold(prices, tbill_m, lookback)
    out, prev = {}, None
    for dt, door in hold.items():
        if not isinstance(door, str):
            continue
        r = rets.at[dt, door]
        if prev is not None and door != prev:
            r -= cost_bps / 1e4
        out[dt] = r
        prev = door
    return pd.Series(out, name="gem").astype(float).dropna()


def n_switches(prices: pd.DataFrame, tbill_m: pd.Series, lookback: int = 12) -> int:
    """How many times the book changes door over the sample (turnover ≈ this × full notional)."""
    hold = gem_hold(prices, tbill_m, lookback).dropna()
    return int((hold != hold.shift(1)).iloc[1:].sum())


def buy_and_hold(prices: pd.DataFrame, col: str = "US") -> pd.Series:
    """Always-invested benchmark: just own one door (default US equity)."""
    return to_returns(prices)[col].dropna().rename(f"hold_{col.lower()}")


def blend(prices: pd.DataFrame, weights=None) -> pd.Series:
    """A fixed-weight, monthly-rebalanced blend — the honest benchmark (default 60/40 US/BOND)."""
    weights = weights or {"US": 0.6, "BOND": 0.4}
    rets = to_returns(prices)
    parts = [w * rets[c] for c, w in weights.items()]
    return sum(parts).dropna().rename("blend")


def _excess(r: pd.Series, rf, periods_per_year: int) -> pd.Series:
    """Excess returns under the quantlab convention: scalar ``rf`` = annualised rate, Series = per-period."""
    if np.ndim(rf) == 0:
        return r - float(rf) / periods_per_year if float(rf) != 0.0 else r
    rf_s = pd.Series(rf).astype(float).reindex(r.index)
    return r - rf_s


def summary(
    returns: pd.Series, periods_per_year: int = MONTHS_PER_YEAR, rf: float | pd.Series = 0.0
) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown, Calmar, skew for a monthly return series.

    ``rf`` follows the quantlab convention (scalar annualised rate, or a per-period return series
    aligned on the index): the **Sharpe is computed on excess-of-cash returns** when given, while
    CAGR / vol / drawdown stay raw. Default 0 reproduces the historical raw-Sharpe behaviour —
    which, on an always-invested book, certifies the asset premium, not the strategy. Compare
    like with like: raw vs raw, or excess vs excess, never one of each.
    """
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "calmar", "skew", "n")}
    std = r.std(ddof=1)
    ex = _excess(r, rf, periods_per_year).dropna()
    ex_mean, ex_std = ex.mean(), ex.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {
        "sharpe": float(ex_mean / ex_std * np.sqrt(periods_per_year)) if ex_std > 0 else np.nan,
        "cagr": float(cagr),
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float(dd),
        "calmar": float(cagr / abs(dd)) if dd < 0 else np.nan,
        "skew": float(r.skew()),
        "n": int(len(r)),
    }


def spanning_alpha(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    lags: int | None = None,
    periods_per_year: int = MONTHS_PER_YEAR,
) -> dict:
    """The signal test that actually isolates the *timing* skill: a spanning regression.

    Regress the strategy's return on its own static ingredients,
        r_strat = alpha + sum_i beta_i * r_factor_i + eps,
    with a Newey-West (HAC, Bartlett kernel) covariance for the intercept. A long-only book that is
    always invested passes a t-test on its raw mean just by carrying the asset premium; the spanning
    intercept asks the only question dual momentum gets paid for — *does switching between the doors
    add return that a fixed mix of the same doors would not have earned?* Feed **excess-of-cash**
    returns on both sides so the cash leg nets out. ``lags=None`` uses the Newey-West rule of thumb
    ``floor(4*(n/100)^(2/9))``. Returns the monthly alpha (bps), its annualised value, the HAC t-stat,
    the betas, and R².
    """
    y = pd.Series(strategy_returns).astype(float)
    f = pd.DataFrame(factor_returns).astype(float)
    df = f.join(y.rename("_y"), how="inner").dropna()
    yv = df["_y"].to_numpy()
    names = [c for c in df.columns if c != "_y"]
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in names])
    n, k = X.shape

    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    u = yv - X @ beta
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))

    # HAC sandwich: V = (X'X)^-1 S (X'X)^-1, S the Bartlett-weighted long-run score covariance
    g = X * u[:, None]  # scores
    S = g.T @ g
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1.0)
        gamma = g[j:].T @ g[:-j]
        S += w * (gamma + gamma.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    V = XtX_inv @ S @ XtX_inv

    alpha, se = float(beta[0]), float(np.sqrt(max(V[0, 0], 0.0)))
    ss_res, ss_tot = float(u @ u), float(((yv - yv.mean()) ** 2).sum())
    return {
        "alpha_bps_m": alpha * 1e4,
        "alpha_ann_pct": ((1.0 + alpha) ** periods_per_year - 1.0) * 100.0,
        "tstat": alpha / se if se > 0 else np.nan,
        "betas": {nm: float(b) for nm, b in zip(names, beta[1:])},
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "n": int(n),
        "lags": int(lags),
    }
