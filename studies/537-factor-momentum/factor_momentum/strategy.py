"""Strategy + inference for Study 537 — Factor-Momentum (Ehsani-Linnainmaa 2022).

The idea: don't ask whether *stocks* trend — ask whether the **factors** trend. Build a panel
of monthly factor returns (``data.build_factor_panel``), then run **time-series momentum** on
each factor: if a factor's trailing return is positive, hold it long next month; if negative,
hold it short (or flat). The **factor-momentum premium** is the average of these timed-factor
returns. Ehsani & Linnainmaa (2022) argue this meta-premium is large, spans much of individual
stock momentum, and is more robust than the underlying factors traded statically.

What we measure on the desk's terms:

  * **The meta-premium.** For each factor *f*, the time-series-momentum-timed return is
    ``sign(trailing_f) * f_next``. Average across factors = the factor-momentum strategy's
    monthly return. We test its mean against zero with a one-sample (and HAC) *t*.
  * **A placebo / label-shuffle null.** Destroy the time ordering of each factor series before
    timing it (sign of a *shuffled* history); if the premium is real persistence, the shuffled
    timing should collapse to ~0. ``p`` = P[shuffled mean >= observed].
  * **One execution lag.** The timing signal at month-end *t* (trailing return through *t*)
    sets next month's position; we **enter the close of month t and earn month t+1** — the
    factor panel is already lagged one month, so there is no look-ahead.
  * **Costs.** One-way bps × turnover (a factor flips sign -> round-trip on that sleeve) +
    borrow on any sleeve held short. Gross vs net.
  * **A deterministic synthetic control.** On planted AR(1) factors the timing must light up;
    on white-noise factors it must NOT manufacture a premium.

The honest expectation on a 5-factor, large-cap **survivor** panel with real costs: the
meta-premium is plausible but thin, and clearing |t| >= 2 *and* a placebo is the high bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACKS = (1, 3, 6, 12)          # trailing-return windows (months) for the timing signal
DEFAULT_LB = 12


def _rolling_compound(p: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Trailing ``lookback``-month compounded return (vectorised, no Python apply)."""
    logr = np.log1p(p.clip(lower=-0.999))
    trail_log = logr.rolling(lookback).sum()
    return np.expm1(trail_log)


# --------------------------------------------------------------------------- #
# Time-series momentum timing of the factor panel
# --------------------------------------------------------------------------- #
def timed_factor_returns(panel: pd.DataFrame, lookback: int = DEFAULT_LB,
                         long_only_on_negative: bool = False) -> pd.DataFrame:
    """Time-series-momentum-timed factor returns.

    For each factor column and each month *t*, the position is the **sign of the trailing
    ``lookback``-month return through t-1** (strictly past; the panel is already one-month
    lagged vs its signal). Timed return at *t* = position * factor_return[t]. If
    ``long_only_on_negative`` is True, a negative trailing factor is held **flat** (0) rather
    than short.

    Returns a (month × factor) frame of timed returns aligned to ``panel``'s later rows.
    """
    p = panel.copy()
    trail = _rolling_compound(p, lookback)
    sig = trail.shift(1)                      # use info through t-1 to trade month t
    pos = np.sign(sig)
    if long_only_on_negative:
        pos = pos.clip(lower=0.0)
    timed = pos * p
    return timed.dropna(how="all")


def factor_momentum_series(panel: pd.DataFrame, lookback: int = DEFAULT_LB,
                           long_only_on_negative: bool = False) -> pd.Series:
    """The factor-momentum strategy's monthly return = mean of timed-factor returns."""
    timed = timed_factor_returns(panel, lookback, long_only_on_negative)
    return timed.mean(axis=1).dropna()


def static_factor_series(panel: pd.DataFrame) -> pd.Series:
    """Benchmark: the equal-weight **static** (untimed) factor return — the average factor.

    This is the comparison Ehsani-Linnainmaa draw: does *timing* the factors beat just
    holding them long? The factor-momentum premium is the spread of timed over static.
    """
    return panel.mean(axis=1).dropna()


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def ttest_vs_zero(sample: np.ndarray | pd.Series) -> float:
    """One-sample t of ``sample`` against 0."""
    x = np.asarray(pd.Series(sample).dropna(), dtype=float)
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def hac_t(sample: np.ndarray | pd.Series) -> float:
    """Newey-West HAC t-stat of the mean against 0 (autocorrelation-robust)."""
    x = np.asarray(pd.Series(sample).dropna(), dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def placebo_pvalue(panel: pd.DataFrame, lookback: int = DEFAULT_LB,
                   n_draws: int = 5000, seed: int = 537,
                   long_only_on_negative: bool = False) -> dict:
    """Label/time-shuffle placebo null for the factor-momentum mean.

    Each draw **shuffles the time order of every factor column independently**, then runs the
    same time-series-momentum timing on the shuffled history. If the premium is genuine
    persistence, shuffling (which destroys autocorrelation) should kill it. ``p`` =
    P[shuffled mean >= observed].
    """
    obs = float(factor_momentum_series(panel, lookback, long_only_on_negative).mean())
    rng = np.random.default_rng(seed)
    vals = panel.to_numpy()
    n, k = vals.shape
    means = np.empty(n_draws)
    cols = panel.columns
    for d in range(n_draws):
        shuffled = np.empty_like(vals)
        for j in range(k):
            col = vals[:, j].copy()
            rng.shuffle(col)
            shuffled[:, j] = col
        sp = pd.DataFrame(shuffled, index=panel.index, columns=cols)
        means[d] = float(factor_momentum_series(sp, lookback, long_only_on_negative).mean())
    p = float((means >= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()), "p_value": p, "draws": means}


def summary(series: pd.Series, periods_per_year: int = 12) -> dict:
    """Annualised stats for a monthly return series: mean, vol, Sharpe, t, HAC t, hit, MDD."""
    r = pd.Series(series).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "t", "hac_t",
                                    "hit_rate", "max_drawdown", "n")}
    mean_m, vol_m = float(r.mean()), float(r.std(ddof=1))
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "mean": mean_m * periods_per_year,
        "vol": vol_m * np.sqrt(periods_per_year),
        "sharpe": (mean_m * periods_per_year) / (vol_m * np.sqrt(periods_per_year))
        if vol_m > 0 else np.nan,
        "t": ttest_vs_zero(r),
        "hac_t": hac_t(r),
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": dd,
        "n": n,
    }


# --------------------------------------------------------------------------- #
# Costs × turnover
# --------------------------------------------------------------------------- #
def net_of_costs(panel: pd.DataFrame, lookback: int = DEFAULT_LB,
                 cost_bps: float = 10.0, borrow_bps_ann: float = 50.0,
                 long_only_on_negative: bool = False) -> dict:
    """Factor-momentum mean net of one-way costs × position turnover + short borrow.

    Turnover per factor sleeve = |position_t - position_{t-1}| / 2 (a sign flip = a full
    round-trip on that sleeve). Cost charged = ``cost_bps`` one-way × turnover, averaged
    across factors. Borrow = ``borrow_bps_ann`` annualised on the average fraction of sleeves
    held short. Returns gross/net annualised means and the realised turnover.
    """
    p = panel.copy()
    trail = _rolling_compound(p, lookback)
    sig = trail.shift(1)
    pos = np.sign(sig)
    if long_only_on_negative:
        pos = pos.clip(lower=0.0)
    pos = pos.dropna(how="all").fillna(0.0)
    timed = (pos * p).dropna(how="all")

    # turnover: change in position magnitude per sleeve, per month
    dpos = pos.diff().abs().fillna(pos.abs())   # first month: establishing the position
    turn = dpos.mean(axis=1)                     # avg across factors, per month
    short_frac = (pos < 0).mean(axis=1)          # fraction of sleeves short, per month

    c = cost_bps / 1e4
    borrow_m = (borrow_bps_ann / 1e4) / 12.0
    gross_m = timed.mean(axis=1)
    cost_m = c * turn.reindex(gross_m.index).fillna(0.0)
    borrow_cost_m = borrow_m * short_frac.reindex(gross_m.index).fillna(0.0)
    net_m = gross_m - cost_m - borrow_cost_m

    return {
        "gross": float(gross_m.mean() * 12.0),
        "net": float(net_m.mean() * 12.0),
        "avg_turnover": float(turn.mean()),
        "avg_short_frac": float(short_frac.mean()),
        "gross_t": ttest_vs_zero(gross_m),
        "net_t": ttest_vs_zero(net_m),
        "net_series": net_m,
    }
