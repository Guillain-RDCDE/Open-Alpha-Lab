"""Natural-gas winter-seasonality: the seasonal long, roll-drag accounting, and the controls.

The thesis: buy UNG (United States Natural Gas Fund) in October (heating-season start) and exit in March
(end of winter). Natural-gas demand peaks in winter; if the futures curve prices this imperfectly, a
seasonal long should profit.

The widow-maker: UNG is a pure front-month futures roll. When the curve is in contango (as it almost
always is in natural gas), rolling from the expiring front contract to the next one costs a toll each
month — the 'roll yield drag'. Over a year the drag can exceed 30-50%. A seasonal trade that holds for
five winter months inherits five months of contango drag on top of any spot-price move.

Protocol:
  * :func:`seasonal_signal` — returns 1.0 in Oct-Mar ('winter'), 0.0 otherwise (the buy/hold signal).
  * :func:`seasonal_returns` — winter-only monthly UNG returns (the strategy leg).
  * :func:`buy_hold_returns` — full-period UNG monthly returns (the benchmark).
  * :func:`winter_premium` — OLS t-stat: is the mean winter return > mean non-winter return?
  * :func:`summary` — annualised Sharpe, CAGR, vol, max-drawdown.

Two conventions, stated up front:
  * Months are defined by calendar month number (Oct=10, Nov=11, Dec=12, Jan=1, Feb=2, Mar=3) — no
    ambiguity from positional shifts.
  * 'Winter' strategy is in cash at 0% outside Oct-Mar (no T-bill credited, conservative baseline).
    This gives the seasonal strategy the benefit of the doubt — even with zero cash yield it fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
WINTER_MONTHS = {10, 11, 12, 1, 2, 3}   # Oct-Mar inclusive


def seasonal_signal(index: pd.DatetimeIndex) -> pd.Series:
    """Return 1.0 in Oct-Mar ('winter'), 0.0 otherwise."""
    return pd.Series(
        [1.0 if m in WINTER_MONTHS else 0.0 for m in index.month],
        index=index,
        name="winter_signal",
    )


def seasonal_returns(ung: pd.Series) -> pd.Series:
    """Monthly UNG returns for the seasonal strategy (in during Oct-Mar, cash=0 otherwise)."""
    sig = seasonal_signal(pd.DatetimeIndex(ung.index))
    return (ung * sig).rename("seasonal")


def buy_hold_returns(ung: pd.Series) -> pd.Series:
    """Full-period monthly UNG returns (the benchmark)."""
    return pd.Series(ung).astype(float).dropna().rename("buy_hold")


def winter_premium(ung: pd.Series) -> dict:
    """OLS t-stat on the winter dummy: is the mean winter return above the rest?

    Regresses monthly UNG return on an intercept + a 'winter' dummy (1 = Oct-Mar, 0 = Apr-Sep).
    A positive, significant slope (|t| >= 2) would support the seasonal claim.
    Returns slope, t-stat, mean_winter, mean_nonwinter, and n.
    """
    r = pd.Series(ung).astype(float).dropna()
    sig = seasonal_signal(pd.DatetimeIndex(r.index))
    sig = sig.reindex(r.index).fillna(0.0)
    n = len(r)
    if n < 4:
        return {"slope": np.nan, "tstat": np.nan, "mean_winter": np.nan, "mean_nonwinter": np.nan, "n": n}
    w = sig.to_numpy()
    y = r.to_numpy()
    # OLS: y = a + b*w  => b = Cov(w,y)/Var(w)
    w_bar, y_bar = w.mean(), y.mean()
    cov_wy = ((w - w_bar) * (y - y_bar)).mean()
    var_w = ((w - w_bar) ** 2).mean()
    if var_w < 1e-12:
        return {"slope": np.nan, "tstat": np.nan, "mean_winter": np.nan, "mean_nonwinter": np.nan, "n": n}
    b = cov_wy / var_w
    a = y_bar - b * w_bar
    resid = y - (a + b * w)
    s2 = resid.var(ddof=2)
    # var(b) = s2 / sum((w-w_bar)^2)
    se_b = np.sqrt(s2 / ((w - w_bar) ** 2).sum()) if s2 > 0 else np.nan
    tstat = b / se_b if se_b and np.isfinite(se_b) else np.nan
    mean_winter = float(y[w == 1].mean()) if (w == 1).any() else np.nan
    mean_nonwinter = float(y[w == 0].mean()) if (w == 0).any() else np.nan
    return {
        "slope": float(b),
        "tstat": float(tstat),
        "mean_winter": mean_winter,
        "mean_nonwinter": mean_nonwinter,
        "n": n,
    }


def summary(returns: pd.Series, periods_per_year: int = MONTHS_PER_YEAR) -> dict:
    """Annualised Sharpe (raw), CAGR, vol, max-drawdown for a monthly return series."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "n")}
    mean_r, std_r = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if eq.iloc[-1] > 0 else np.nan
    sharpe = float(mean_r / std_r * np.sqrt(periods_per_year)) if std_r > 0 else np.nan
    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "vol_ann": float(std_r * np.sqrt(periods_per_year)),
        "max_drawdown": float(dd),
        "n": int(len(r)),
    }


def roll_drag_estimate(ung: pd.Series, spot_proxy: pd.Series | None = None) -> dict:
    """Estimate the implied roll drag from UNG vs spot returns (if spot available).

    When a spot proxy (e.g. front-month spot price series) is provided, roll drag ≈ UNG return minus
    spot return, annualised. When unavailable, returns NaN with a note. Used for disclosure only.
    """
    r_ung = pd.Series(ung).astype(float).dropna()
    if spot_proxy is None:
        return {"roll_drag_ann": np.nan, "note": "no spot proxy available"}
    r_spot = pd.Series(spot_proxy).astype(float).reindex(r_ung.index).dropna()
    both = pd.concat([r_ung.rename("ung"), r_spot.rename("spot")], axis=1).dropna()
    diff = both["ung"] - both["spot"]
    drag_ann = float(diff.mean() * MONTHS_PER_YEAR)
    return {"roll_drag_ann": drag_ann, "note": "UNG minus spot proxy, annualised"}
