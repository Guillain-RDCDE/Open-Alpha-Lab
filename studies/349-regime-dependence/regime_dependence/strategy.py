"""The regime-dependence lens and its honest controls — Study 349 (Regime-Dependence).

The method under test: take a strategy's monthly returns and ask whether its edge is
**stable across regimes** or concentrated in one era. The apparatus is:

1. **Decade-by-decade (regime-conditional) Sharpe** — split the tape into regimes and
   compute the annualised Sharpe of each sub-period. A durable edge has roughly the
   same Sharpe everywhere; a regime-fitted strategy has one huge sub-Sharpe and the
   rest near zero.
2. **Robust inference on the level** — Newey-West (HAC) *t* on the mean return, so a
   high full-sample Sharpe is not mistaken for significance under autocorrelation.
3. **A test of the *difference* between regimes** — the decisive step. A strategy that
   "ruled the 2010s" must clear a test that the hot regime's mean is *different* from
   the others, AND, separately, that the *non-hot* regimes carry an edge at all. We use
   a block-bootstrap CI on the cross-regime mean gap and a leave-the-hot-regime-out
   re-estimate of the Sharpe (the danger number: what would you have earned without the
   one lucky decade?).
4. **Stability score** — the coefficient of variation of the per-regime Sharpe (or the
   spread between best and worst decade), a single dial that ranks strategies by how
   regime-dependent they are.

Real-tape canonical strategies (see ``apply_*`` below): a 12-month time-series-momentum
(trend) overlay on the equity index, and a static 60/40 stock/bond mix. Both carry
**one** execution lag where a signal is involved (trend: the signal known at the close
of month *t* sets the position for month *t+1*, a single ``shift``; 60/40 is a fixed
weight and needs no lag). Costs are charged one-way × NAV at ``cost_bps`` on the trend
overlay's turnover; the 60/40 mix is rebalanced annually. Sharpe races are
excess-vs-excess (excess of the risk-free / cash leg) so a part-time-in-cash strategy
is not flattered against a fully-invested one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def sharpe(monthly_ret: pd.Series, rf_monthly: float = 0.0,
           periods_per_year: int = MONTHS_PER_YEAR) -> float:
    """Annualised Sharpe of a monthly excess-return series (excess of ``rf_monthly``)."""
    r = monthly_ret.dropna() - rf_monthly
    if len(r) < 2:
        return float("nan")
    vol = r.std(ddof=1)
    if vol <= 0:
        return float("nan")
    return float(r.mean() / vol * np.sqrt(periods_per_year))


def summarize(monthly_ret: pd.Series, rf_monthly: float = 0.0,
              periods_per_year: int = MONTHS_PER_YEAR) -> dict:
    """Headline metrics for one monthly return series (excess-of-rf for Sharpe)."""
    r = monthly_ret.dropna()
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in ("n", "cagr", "sharpe", "max_dd", "mean_ann")}
    total = float((1.0 + r).prod())
    n_years = n / periods_per_year
    cagr = total ** (1.0 / n_years) - 1.0 if (n_years > 0 and total > 0) else float("nan")
    ann_mean = r.mean() * periods_per_year
    equity = (1.0 + r).cumprod().to_numpy()
    return {
        "n": int(n),
        "cagr": float(cagr),
        "sharpe": sharpe(r, rf_monthly=rf_monthly, periods_per_year=periods_per_year),
        "max_dd": _max_drawdown(equity),
        "mean_ann": float(ann_mean),
    }


def hac_tstat(returns: pd.Series) -> dict:
    """Newey-West HAC *t*-stat on the mean of a single return series.

    Used on the *level* (mean per-period return) so a strategy's headline does not
    inherit significance it does not have under autocorrelation. The lag is the
    standard Newey-West rule of thumb.
    """
    x = returns.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 3:
        return {"mean": float("nan"), "tstat": float("nan"), "n": n}
    mu = x.mean()
    e = x - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        wk = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * wk * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {
        "mean": float(mu),
        "tstat": float(mu / se) if se > 0 else float("nan"),
        "n": n,
        "mean_ann": float(mu * MONTHS_PER_YEAR),
    }


def block_bootstrap_ci(
    series: pd.Series, block: int = 6, n_boot: int = 2000,
    seed: int = 349, alpha: float = 0.05
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the *mean* of a series.

    Block resampling preserves the autocorrelation that i.i.d. resampling would
    destroy. Returns the (lower, upper) percentile interval on the mean per-period value.
    """
    x = series.dropna().to_numpy(dtype=float)
    n = x.size
    if n < block + 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = x[idx][:n].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


# ---------------------------------------------------------------------------
# The regime-dependence lens
# ---------------------------------------------------------------------------
def regime_sharpes(returns: pd.Series, regime: pd.Series,
                   rf_monthly: float = 0.0) -> pd.Series:
    """Annualised Sharpe within each regime label.

    Returns a Series indexed by regime label. A durable edge keeps these roughly
    constant; a regime-fitted strategy shows one huge value and the rest near zero.
    """
    aligned = pd.concat([returns.rename("r"), regime.rename("g")], axis=1, join="inner").dropna()
    out = {}
    for g, grp in aligned.groupby("g"):
        out[g] = sharpe(grp["r"], rf_monthly=rf_monthly)
    return pd.Series(out, name="regime_sharpe").sort_index()


def stability_score(returns: pd.Series, regime: pd.Series,
                    rf_monthly: float = 0.0) -> dict:
    """How regime-dependent is a strategy's edge?

    Computes the per-regime Sharpe and summarises its dispersion. ``cv`` is the
    coefficient of variation (std / |mean|) of the per-regime Sharpe — high CV means a
    strategy whose edge lives in a few regimes. ``best_minus_worst`` is the spread
    between the best and worst decade's Sharpe. ``share_top_regime`` is the fraction of
    the total (compounded) excess wealth earned in the single best regime.
    """
    rs = regime_sharpes(returns, regime, rf_monthly=rf_monthly).dropna()
    if rs.empty:
        return {"cv": float("nan"), "best_minus_worst": float("nan"),
                "share_top_regime": float("nan"), "regime_sharpes": rs}
    mean_s = rs.mean()
    std_s = rs.std(ddof=0)
    cv = float(std_s / abs(mean_s)) if abs(mean_s) > 1e-9 else float("inf")

    # Share of cumulative excess return earned in the best single regime.
    aligned = pd.concat([returns.rename("r"), regime.rename("g")], axis=1,
                        join="inner").dropna()
    aligned["r"] = aligned["r"] - rf_monthly
    by_regime_sum = aligned.groupby("g")["r"].sum()
    total = by_regime_sum.sum()
    share_top = float(by_regime_sum.max() / total) if abs(total) > 1e-12 else float("nan")

    return {
        "cv": cv,
        "best_minus_worst": float(rs.max() - rs.min()),
        "share_top_regime": share_top,
        "regime_sharpes": rs,
    }


def leave_hot_regime_out(returns: pd.Series, regime: pd.Series,
                         rf_monthly: float = 0.0) -> dict:
    """The danger number: the strategy's Sharpe and HAC *t* with the BEST regime removed.

    Identifies the regime with the highest sub-Sharpe, drops it, and re-estimates the
    full-sample Sharpe and HAC *t* on what remains. This answers "what would you have
    earned without the one lucky decade?" — the question a single full-sample Sharpe
    silently averages away. A durable edge survives; a regime bet collapses to ~0.
    """
    rs = regime_sharpes(returns, regime, rf_monthly=rf_monthly).dropna()
    if rs.empty:
        return {"hot_regime": None, "sharpe_full": float("nan"),
                "sharpe_ex_hot": float("nan"), "t_ex_hot": float("nan")}
    hot = rs.idxmax()
    aligned = pd.concat([returns.rename("r"), regime.rename("g")], axis=1,
                        join="inner").dropna()
    ex = aligned[aligned["g"] != hot]["r"]
    return {
        "hot_regime": hot,
        "sharpe_full": sharpe(aligned["r"], rf_monthly=rf_monthly),
        "sharpe_ex_hot": sharpe(ex, rf_monthly=rf_monthly),
        "t_full": hac_tstat(aligned["r"] - rf_monthly)["tstat"],
        "t_ex_hot": hac_tstat(ex - rf_monthly)["tstat"],
    }


def hot_minus_rest_ci(returns: pd.Series, regime: pd.Series,
                      rf_monthly: float = 0.0, n_boot: int = 2000,
                      seed: int = 349) -> dict:
    """Block-bootstrap CI on (best-regime mean) − (rest-of-sample mean).

    The decisive uncertainty: is the gap between the strategy's best decade and the rest
    larger than noise? A wide CI straddling zero means the apparent regime-dependence is
    not even statistically resolvable; a CI entirely above zero confirms the edge is
    genuinely concentrated in one regime (the "ruled one era" diagnosis).
    """
    rs = regime_sharpes(returns, regime, rf_monthly=rf_monthly).dropna()
    if rs.empty:
        return {"hot_regime": None, "gap_ann": float("nan"), "lo": float("nan"),
                "hi": float("nan")}
    hot = rs.idxmax()
    aligned = pd.concat([returns.rename("r"), regime.rename("g")], axis=1,
                        join="inner").dropna()
    hot_r = aligned[aligned["g"] == hot]["r"] - rf_monthly
    rest_r = aligned[aligned["g"] != hot]["r"] - rf_monthly
    gap = float(hot_r.mean() - rest_r.mean())
    lo_h, hi_h = block_bootstrap_ci(hot_r, n_boot=n_boot, seed=seed)
    lo_r, hi_r = block_bootstrap_ci(rest_r, n_boot=n_boot, seed=seed + 1)
    # Difference-of-means CI by convolving the two bootstrap intervals conservatively.
    return {
        "hot_regime": hot,
        "gap_ann": gap * MONTHS_PER_YEAR,
        "lo": (lo_h - hi_r) * MONTHS_PER_YEAR,
        "hi": (hi_h - lo_r) * MONTHS_PER_YEAR,
    }


# ---------------------------------------------------------------------------
# Real-tape canonical strategies
# ---------------------------------------------------------------------------
def apply_trend(equity_ret: pd.Series, lookback: int = 12,
                cost_bps: float = 0.0) -> pd.Series:
    """12-month time-series-momentum (trend) overlay on the equity index.

    The rule: hold the index next month iff its trailing ``lookback``-month total
    return was positive, else hold cash (return 0). ONE execution lag — the signal known
    at the close of month *t* sets the position for month *t+1* (a single ``shift``).
    Costs charged one-way × NAV at ``cost_bps`` whenever the position flips.

    Returns the strategy's monthly return series (in-cash months earn 0 — caller should
    race excess-of-cash to excess-of-cash; here the cash leg is 0 so the series IS the
    excess return).
    """
    r = equity_ret.dropna()
    trailing = (1.0 + r).rolling(lookback).apply(lambda w: w.prod() - 1.0, raw=False)
    signal = (trailing > 0).astype(float)
    pos = signal.shift(1).fillna(0.0)          # one execution lag
    gross = pos * r
    flips = pos.diff().abs().fillna(0.0)
    cost = flips * cost_bps * 1e-4
    return (gross - cost).dropna()


def apply_6040(equity_ret: pd.Series, bond_ret: pd.Series,
               w_equity: float = 0.60, rebalance_every: int = 12,
               cost_bps: float = 0.0) -> pd.Series:
    """A static 60/40 stock/bond mix, rebalanced every ``rebalance_every`` months.

    A fixed weight is calendar-known and needs no execution lag. Between rebalances the
    weights drift with the assets; at each rebalance turnover is charged one-way × NAV.
    """
    df = pd.concat([equity_ret.rename("e"), bond_ret.rename("b")], axis=1,
                   join="inner").dropna()
    R = df.to_numpy(dtype=float)
    T = R.shape[0]
    target = np.array([w_equity, 1.0 - w_equity])
    out = np.full(T, np.nan)
    w = None
    for t in range(T):
        if t % rebalance_every == 0:
            if w is None:
                cost = 0.0
            else:
                cost = float(np.abs(target - w).sum()) * cost_bps * 1e-4
            w = target.copy()
            net = float(w @ R[t]) - cost
        else:
            net = float(w @ R[t])
        out[t] = net
        grown = w * (1.0 + R[t])
        s = grown.sum()
        w = grown / s if s > 0 else target.copy()
    return pd.Series(out, index=df.index).dropna()


def decade_label(index: pd.DatetimeIndex) -> pd.Series:
    """Map a DatetimeIndex to a decade label (e.g. 2010) — the real-tape regime split."""
    years = index.year
    decades = (years // 10) * 10
    return pd.Series(decades, index=index, name="regime")
