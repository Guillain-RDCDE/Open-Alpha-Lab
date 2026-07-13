"""Strategy + inference for Study 769 — "parks attendance/pricing as a DIS tell".

The claim: theme-park attendance momentum (and Disney's pricing power) is a *consumer
tell* for Disney the stock — when the parks are booming, DIS should follow. We test the
strongest **strictly-lagged, no-look-ahead** version of that:

1. **Context — DIS vs SPY.** Does DIS even beat the market it's benchmarked against, over
   the window? CAGR / vol / Sharpe / max-drawdown + an annual-excess *t*. (If parks
   momentum is a DIS *tell*, the payoff is DIS-specific *excess* return, not broad beta.)
2. **Lead-lag.** Does the release-lagged parks-attendance growth **predict DIS forward
   returns** — and, the honest test, DIS's forward return *in excess of SPY*? A
   Newey-West (HAC) *t* on the slope. ``REAL`` needs |t| >= 2 on the excess tape.
3. **Regime split.** DIS forward return when parks momentum is positive vs the base rate
   (a Welch *t*) — both absolute and excess-of-SPY.
4. **A timing backtest, net of costs.** Hold DIS when the (release-lagged) parks momentum
   is positive, else sit in SPY; one execution lag, costs charged on each switch, shorts
   (if enabled) pay borrow. Raced against buy-and-hold DIS *and* SPY on a Sharpe basis.

Inference: a small-sample annual-excess *t*, a Newey-West (HAC) *t* of the lead-lag slope,
Welch *t* for the regime contrast, Sharpe ratios and a max-drawdown decomposition. Pure
numpy/pandas; scipy only for the *t*-distribution p-value (optional, guarded).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MONTHS = 12.0
ANN = 12


# --------------------------------------------------------------------------- #
# Return / risk primitives
# --------------------------------------------------------------------------- #
def cagr(level: pd.Series) -> float:
    """Compound annual growth rate of a level series (index carries the dates)."""
    yrs = (level.index[-1] - level.index[0]).days / 365.25
    if yrs <= 0 or level.iloc[0] <= 0:
        return float("nan")
    return (level.iloc[-1] / level.iloc[0]) ** (1.0 / yrs) - 1.0


def max_drawdown(level: pd.Series) -> float:
    """Worst peak-to-trough drawdown (a negative fraction)."""
    dd = level / level.cummax() - 1.0
    return float(dd.min())


def ann_vol(returns: pd.Series, periods_per_year: float = MONTHS) -> float:
    """Annualised volatility of periodic simple returns."""
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe(returns: pd.Series, rf_annual: float = 0.0,
           periods_per_year: float = MONTHS) -> float:
    """Annualised Sharpe of periodic simple returns (excess of a flat rf)."""
    rf_p = (1 + rf_annual) ** (1 / periods_per_year) - 1
    ex = returns - rf_p
    sd = ex.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(ex.mean() / sd * math.sqrt(periods_per_year))


def summarize(level: pd.Series, periods_per_year: float = MONTHS) -> dict:
    """CAGR / annualised vol / Sharpe / max-drawdown of a level series."""
    rets = level.pct_change().dropna()
    return {
        "cagr": cagr(level),
        "vol": ann_vol(rets, periods_per_year),
        "sharpe": sharpe(rets, 0.0, periods_per_year),
        "mdd": max_drawdown(level),
        "n": int(len(rets)),
    }


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def _t_p_value(t: float, df: int) -> float:
    """Two-sided p-value for a t-stat (scipy if available, else a normal approx)."""
    if not np.isfinite(t):
        return float("nan")
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), df))
    except Exception:  # pragma: no cover - scipy is in requirements but stay safe
        return float(2 * 0.5 * math.erfc(abs(t) / math.sqrt(2)))


def annual_excess_t(dis_level: pd.Series, bench_level: pd.Series) -> dict:
    """*t*-stat that DIS's **annual** return beats the benchmark's.

    Aligns both to year-end, takes annual simple returns, and tests the mean of the paired
    excess (DIS - bench) against 0. Small-sample by construction — the finding, not a flaw.
    """
    a = dis_level.resample("YE").last().pct_change().dropna()
    b = bench_level.resample("YE").last().pct_change().dropna()
    j = pd.concat([a, b], axis=1, keys=["d", "b"]).dropna()
    ex = j["d"] - j["b"]
    n = len(ex)
    if n < 2:
        return {"mean_excess": float("nan"), "t": float("nan"), "p": float("nan"), "n": n}
    se = ex.std(ddof=1) / math.sqrt(n)
    t = ex.mean() / se if se > 0 else float("nan")
    return {"mean_excess": float(ex.mean()), "t": float(t),
            "p": _t_p_value(t, n - 1), "n": n}


def forward_returns(level: pd.Series, horizon: int) -> pd.Series:
    """``horizon``-month forward simple return for every month-end (NaN near the tail)."""
    return level.shift(-horizon) / level - 1.0


def newey_west_slope_t(y: pd.Series, x: pd.Series, lags: int = 12) -> dict:
    """Newey-West (HAC) *t* of the slope in ``y = a + b*x + e`` (Bartlett kernel).

    The Signal-axis statistic for the lead-lag: does the release-lagged parks-momentum
    ``x`` predict the forward return ``y``? Returns slope, its HAC SE and *t*, the
    intercept, and n. Overlapping forward windows demand HAC — hence ``lags`` (default 12).
    """
    j = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    yv = j["y"].to_numpy()
    xv = j["x"].to_numpy()
    n = len(yv)
    X = np.column_stack([np.ones(n), xv])
    # A zero-variance (constant) regressor is collinear with the intercept -> no slope is
    # identified. Return an explicit NaN rather than let a singular solve blow up.
    if n < 3 or np.std(xv) == 0:
        return {"slope": float("nan"), "intercept": float("nan"), "se_slope": float("nan"),
                "t": float("nan"), "p": float("nan"), "n": n}
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ yv)
    resid = yv - X @ beta
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        Xe = X * resid[:, None]
        Gamma = Xe[L:].T @ Xe[:-L]
        S += w * (Gamma + Gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    se_b = math.sqrt(cov[1, 1])
    b = float(beta[1])
    t_b = b / se_b if se_b > 0 else float("nan")
    return {"slope": b, "intercept": float(beta[0]), "se_slope": float(se_b),
            "t": float(t_b), "p": _t_p_value(t_b, n - 2), "n": n}


def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of ``mean(sample) - mean(base)`` (unequal variance). NaN if sample < 2."""
    if len(sample) < 2:
        return float("nan")
    m1, m0 = sample.mean(), base.mean()
    se = np.sqrt(sample.var(ddof=1) / len(sample) + base.var(ddof=1) / len(base))
    if se == 0:
        return float("nan")
    return float((m1 - m0) / se)


def regime_split(frame: pd.DataFrame, horizon: int = 12, lag: int = 1,
                 excess: bool = False) -> dict:
    """DIS forward ``horizon``-m return when parks momentum > 0 vs the base rate.

    ``lag`` is the execution lag applied to the (already release-lagged) ``pg`` signal.
    ``excess=True`` measures DIS's forward return *in excess of SPY* — the DIS-specific
    'tell' test. Returns conditional mean/n, base mean, and a Welch *t* of the difference.
    """
    fwd = forward_returns(frame["dis"], horizon)
    if excess:
        fwd = fwd - forward_returns(frame["spy"], horizon)
    sig_lagged = frame["pg"].shift(lag)
    # Base rate is the unconditional forward return over the window the signal EXISTS
    # (apples-to-apples: the conditional set is a subset of exactly this window — including
    # pre-signal months in the base would contrast against a different regime).
    avail = sig_lagged.notna()
    cond = fwd[avail & (sig_lagged > 0)].dropna().to_numpy()
    base = fwd[avail].dropna().to_numpy()
    return {
        "horizon": horizon,
        "n_cond": int(len(cond)),
        "cond_mean": float(cond.mean()) if len(cond) else float("nan"),
        "base_mean": float(base.mean()) if len(base) else float("nan"),
        "t": welch_t(cond, base),
        "excess": excess,
    }


def lead_lag(frame: pd.DataFrame, signal: str = "pg", horizon: int = 12,
             lag: int = 1, excess: bool = False, lags: int = 12) -> dict:
    """Newey-West lead-lag: does a release-lagged parks signal predict DIS fwd return?

    ``signal`` is a column of ``frame`` (``pg`` attendance growth, ``ph`` price hike).
    ``excess=True`` regresses DIS's forward return *minus SPY's* on the signal — the
    honest 'DIS-specific tell' test. Returns :func:`newey_west_slope_t`'s dict.
    """
    fwd = forward_returns(frame["dis"], horizon)
    if excess:
        fwd = fwd - forward_returns(frame["spy"], horizon)
    return newey_west_slope_t(fwd, frame[signal].shift(lag), lags=lags)


# --------------------------------------------------------------------------- #
# Timing backtest, net of costs (the Tradability axis)
# --------------------------------------------------------------------------- #
def _stats(r: pd.Series) -> dict:
    mu = r.mean() * ANN
    sd = r.std(ddof=1) * math.sqrt(ANN)
    return {"ann_ret": float(mu), "ann_vol": float(sd),
            "sharpe": float(mu / sd) if sd > 0 else float("nan")}


def timing_backtest(frame: pd.DataFrame, lag: int = 1, cost_bps: float = 10.0,
                    hold_bench: bool = True, borrow_bps_annual: float = 50.0) -> dict:
    """Hold DIS when release-lagged parks momentum > 0, else SPY (or cash).

    Position for month ``m`` is decided by ``pg`` known ``lag`` months earlier: long DIS
    when ``pg > 0``; otherwise long SPY (``hold_bench=True``) or flat/cash. A one-way cost
    of ``cost_bps`` is charged on each leg of a switch (turnover counts both legs). A DIS↔
    SPY switch is a *rotation*, not a levered short — no borrow is due; ``borrow_bps_annual``
    is only charged if a flat/cash regime is replaced by a short (not used here, kept for
    the levered variant). Returns gross/net stats for the rule and both buy-and-holds
    (price-only Adj-Close tape — labelled price-only, no separate cash leg)."""
    dis_r = frame["dis"].pct_change().dropna()
    spy_r = frame["spy"].pct_change().dropna()
    in_dis = (frame["pg"].shift(lag) > 0).reindex(dis_r.index).fillna(False)
    base_r = spy_r if hold_bench else pd.Series(0.0, index=dis_r.index)
    gross = pd.Series(np.where(in_dis, dis_r, base_r), index=dis_r.index)
    # turnover: each switch moves the whole book from one asset to the other → 2 legs.
    switches = in_dis.astype(float).diff().abs().fillna(0.0)
    c = cost_bps / 1e4
    net = gross - switches * c * 2.0
    return {
        "n_months": int(len(dis_r)),
        "n_switches": float(switches.sum()),
        "exposure_dis": float(in_dis.mean()),
        "cost_bps": cost_bps,
        "hold_bench": hold_bench,
        "gross": _stats(gross),
        "net": _stats(net),
        "buy_hold_dis": _stats(dis_r),
        "buy_hold_spy": _stats(spy_r),
    }


def control_recovers(frame: pd.DataFrame, planted_edge: float, horizon: int = 12) -> dict:
    """Positive control: the engine lights up iff a forward edge was planted.

    Regresses the synthetic DIS forward return on the lagged synthetic momentum signal and
    returns the HAC *t* — near 0 when ``planted_edge == 0`` (a real null), large when the
    edge is planted (the harness can detect what it plants)."""
    ll = lead_lag(frame, signal="pg", horizon=horizon, lag=1, excess=False)
    return {"planted_edge": planted_edge, "t": ll["t"], "slope": ll["slope"], "n": ll["n"]}
