"""The books — cross-sectional carry, the dollar-carry tilt, an FX-momentum sleeve, and their combo.

Three sleeves and a combiner:

  (a) :func:`carry_returns` — the **cross-sectional carry** book: long the high-rate currencies, short the
      low-rate ones, dollar-neutral, vol-scaled to a risk target. This is the Steamroller premium itself —
      a real, paid, but negative-skew (crash-prone) stream.
  (b) :func:`dollar_carry_returns` — the **dollar-carry** tilt: be *long the whole basket vs USD* when the
      average foreign rate exceeds the USD rate, short it when it doesn't (Lustig–Roussanov–Verdelhan
      2011's dollar factor — a *time-series* tilt on the basket, not a cross-sectional sort).
  (c) :func:`momentum_returns` — the **momentum** sleeve: rank currencies by their trailing FX trend, long
      the up-trenders, short the down-trenders (Asness–Moskowitz–Pedersen 2013; Menkhoff–Sarno–Schmeling–
      Schrimpf 2012b on FX momentum).

  :func:`combine` — the **carry⊕momentum combo**: an equal-risk blend of the carry and momentum books.
      The thesis (Koijen et al "Carry" 2018; AMP 2013): the two legs pay at *different times*, so the
      combo's Sharpe beats either standalone and momentum dulls the carry crash.

Everything is monthly; we annualise by ``×12`` (mean) and ``×√12`` (vol).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


def summary(returns: pd.Series, periods_per_year: int = MONTHS_PER_YEAR) -> dict:
    """Annualised Sharpe, return, vol, CAGR, max-drawdown, skew for a (monthly) return series."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "ann_return", "vol_ann", "cagr", "max_drawdown", "skew", "n_months")}
    mean, std = r.mean(), r.std(ddof=1)
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    years = len(r) / periods_per_year
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and equity.iloc[-1] > 0 else np.nan
    return {"sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
            "ann_return": float(mean * periods_per_year), "vol_ann": float(std * np.sqrt(periods_per_year)),
            "cagr": float(cagr), "max_drawdown": float(dd.min()), "skew": float(r.skew()),
            "n_months": int(len(r))}


def _vol_scale(gross: pd.Series, target_vol_ann: float, window: int = 24, max_lev: float = 3.0) -> pd.Series:
    """Scale a gross stream to a constant risk target by its own *trailing* (past-only) volatility."""
    target_m = target_vol_ann / np.sqrt(MONTHS_PER_YEAR)
    lev = (target_m / gross.rolling(window, min_periods=6).std(ddof=1).shift(1)).clip(upper=max_lev)
    return (lev * gross).rename(gross.name)


# --------------------------------------------------------------------------- #
# (a) cross-sectional carry
# --------------------------------------------------------------------------- #

def carry_weights(rate_diffs: pd.DataFrame, frac: float = 0.34) -> pd.DataFrame:
    """Dollar-neutral monthly weights: long the highest rate-differential, short the lowest.

    Each month the top and bottom ``frac`` of currencies by rate differential are equal-weighted long and
    short. Weights are lagged one month (set on the differential known at month-end, earning next month).
    """
    cols = rate_diffs.columns
    W = np.zeros((len(rate_diffs), len(cols)))
    rv = rate_diffs.to_numpy()
    for t in range(len(rate_diffs)):
        row = rv[t]
        valid = np.where(~np.isnan(row))[0]
        if len(valid) < 3:
            continue
        k = max(1, int(len(valid) * frac))
        order = valid[np.argsort(row[valid])]
        W[t, order[-k:]] = 0.5 / k
        W[t, order[:k]] = -0.5 / k
    return pd.DataFrame(W, index=rate_diffs.index, columns=cols).shift(1)


def carry_returns(xret: pd.DataFrame, rate_diffs: pd.DataFrame, frac: float = 0.34, cost_bps: float = 10.0,
                  target_vol_ann: float | None = 0.08) -> pd.Series:
    """The cross-sectional carry book's monthly return, net of ``cost_bps`` per unit traded.

    Optionally vol-scaled to ``target_vol_ann`` so it sits at a comparable risk to the momentum sleeve
    (required for an honest equal-risk combo). Pass ``target_vol_ann=None`` for the raw book.
    """
    w = carry_weights(rate_diffs, frac).reindex(xret.index)
    gross = (w * xret).sum(axis=1)
    cost = (cost_bps * 1e-4) * w.diff().abs().sum(axis=1)
    net = (gross - cost).dropna().rename("carry")
    if target_vol_ann is not None:
        net = _vol_scale(net, target_vol_ann).dropna().rename("carry")
    return net


# --------------------------------------------------------------------------- #
# (b) dollar carry — a time-series tilt on the whole basket vs USD
# --------------------------------------------------------------------------- #

def dollar_carry_returns(xret: pd.DataFrame, rate_diffs: pd.DataFrame, base: str = "C0",
                         cost_bps: float = 10.0, target_vol_ann: float | None = 0.08) -> pd.Series:
    """The **dollar-carry** tilt: long the equal-weight basket vs USD when the average rate gap is positive.

    Lustig–Roussanov–Verdelhan (2011): the *level* of the average forward discount (here the mean
    rate-differential vs the USD/base leg) is a tradable signal — when foreign rates are high vs the dollar,
    be long the basket (short USD); when low, short it. A time-series sign on the basket, lagged one month.
    """
    avg_gap = (rate_diffs.mean(axis=1) - rate_diffs.get(base, 0.0))
    sign = np.sign(avg_gap).shift(1).fillna(0.0)
    basket = xret.mean(axis=1)                                   # equal-weight basket excess return
    gross = sign * basket
    cost = (cost_bps * 1e-4) * sign.diff().abs() / len(xret.columns)
    net = (gross - cost).dropna().rename("dollar_carry")
    if target_vol_ann is not None:
        net = _vol_scale(net, target_vol_ann).dropna().rename("dollar_carry")
    return net


# --------------------------------------------------------------------------- #
# (c) momentum — trailing FX trend
# --------------------------------------------------------------------------- #

def momentum_weights(xret: pd.DataFrame, lookback: int = 12, frac: float = 0.34) -> pd.DataFrame:
    """Dollar-neutral weights from the trailing ``lookback``-month return: long up-trenders, short down."""
    trail = (1.0 + xret).rolling(lookback).apply(np.prod, raw=True) - 1.0
    cols = xret.columns
    W = np.zeros((len(xret), len(cols)))
    tv = trail.to_numpy()
    for t in range(len(xret)):
        row = tv[t]
        valid = np.where(~np.isnan(row))[0]
        if len(valid) < 3:
            continue
        k = max(1, int(len(valid) * frac))
        order = valid[np.argsort(row[valid])]
        W[t, order[-k:]] = 0.5 / k
        W[t, order[:k]] = -0.5 / k
    return pd.DataFrame(W, index=xret.index, columns=cols).shift(1)


def momentum_returns(xret: pd.DataFrame, lookback: int = 12, frac: float = 0.34, cost_bps: float = 10.0,
                     target_vol_ann: float | None = 0.08) -> pd.Series:
    """The FX-momentum sleeve's monthly return, net of ``cost_bps`` per unit traded, optionally vol-scaled."""
    w = momentum_weights(xret, lookback, frac)
    gross = (w * xret).sum(axis=1)
    cost = (cost_bps * 1e-4) * w.diff().abs().sum(axis=1)
    net = (gross - cost).dropna().rename("momentum")
    if target_vol_ann is not None:
        net = _vol_scale(net, target_vol_ann).dropna().rename("momentum")
    return net


# --------------------------------------------------------------------------- #
# the carry⊕momentum combo
# --------------------------------------------------------------------------- #

def combine(carry: pd.Series, momentum: pd.Series, w_carry: float = 0.5) -> pd.Series:
    """Equal-(ish) risk blend of the carry and momentum streams: ``w·carry + (1-w)·momentum``.

    Both legs are already vol-scaled to the same target, so a 50/50 blend is an equal-risk combo. The
    thesis: because carry and momentum pay at different times, the blend's Sharpe beats either leg.
    """
    idx = carry.index.intersection(momentum.index)
    return (w_carry * carry.reindex(idx) + (1.0 - w_carry) * momentum.reindex(idx)).rename("combo")


def carry_premium_by_bucket(xret: pd.DataFrame, rate_diffs: pd.DataFrame, n_buckets: int = 3) -> dict:
    """Sort currencies by rate differential into ``n_buckets`` each month; average excess return per bucket.

    A **rising** profile (high-rate buckets out-earn low-rate ones) is the carry premium; the high-minus-low
    spread (annualised) is its size. Under full UIRP (the null) the profile is flat.
    """
    common = xret.index.intersection(rate_diffs.index)
    xr, rt = xret.loc[common], rate_diffs.loc[common]
    bucket_ret = {b: [] for b in range(n_buckets)}
    for dt in common:
        row_r = rt.loc[dt].dropna()
        row_x = xr.loc[dt].reindex(row_r.index).dropna()
        row_r = row_r.reindex(row_x.index)
        if len(row_x) < n_buckets:
            continue
        b = pd.qcut(row_r.rank(method="first"), n_buckets, labels=False)
        for k in range(n_buckets):
            v = row_x[b == k]
            if len(v):
                bucket_ret[k].append(v.mean())
    table = pd.Series({b: np.mean(v) if v else np.nan for b, v in bucket_ret.items()})
    ann = table * MONTHS_PER_YEAR * 100.0
    lo, hi = float(table.iloc[0]), float(table.iloc[-1])
    return {"bucket_ann_pct": ann.rename("ann_pct"),
            "hml_ann_pct": float((hi - lo) * MONTHS_PER_YEAR * 100.0),
            "premium_present": bool(hi > lo), "n_months": int(len(common))}


def turnover_ann(weights: pd.DataFrame) -> float:
    return float(weights.diff().abs().sum(axis=1).mean() * MONTHS_PER_YEAR)
