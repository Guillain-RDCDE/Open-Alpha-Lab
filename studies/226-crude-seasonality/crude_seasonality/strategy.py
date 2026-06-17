"""Crude-oil seasonality: monthly pattern tests and calendar timer.

The textbook claim: crude oil reliably rallies in spring (April–June) as refiners switch to summer
blends and driving demand picks up, then weakens in autumn (August–November) after the driving season
ends and refinery maintenance peaks. We test: (1) are spring months significantly stronger than
autumn months? (2) does a long-spring / short-autumn calendar timer beat buy-and-hold?

Two conventions, stated up front:

  * **Each month is tested against its own sample mean using a one-sample t-stat.**  Comparing a
    group of months (spring vs autumn) uses Welch's two-sample t-stat — never a positional trick.
  * **The cash leg earns the T-bill.** :func:`seasonal_timer` credits a monthly cash return when
    the rule is flat (neither long nor short), so the timing race is done on *excess-of-cash*
    Sharpe, like-for-like.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12
SPRING_MONTHS = [4, 5, 6]   # Apr–Jun: driving-season ramp-up
AUTUMN_MONTHS = [8, 9, 10, 11]  # Aug–Nov: post-summer, refinery maintenance


def month_stats(series: pd.Series) -> pd.DataFrame:
    """Per-calendar-month mean, std, count and one-sample t-stat for a monthly return series.

    Returns a DataFrame indexed 1..12 with columns ``mean``, ``std``, ``n``, ``tstat``.
    The t-stat is the classic ``mean / (std / sqrt(n))`` — testing whether each month's average
    return differs from zero. A robust seasonality claim needs |t| ≥ 2 after multiple-testing
    adjustment (Bonferroni: 0.05/12 ≈ 0.004, so effectively |t| ≥ 3 at n ≈ 26).
    """
    s = pd.Series(series).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    rows = {}
    for m in range(1, 13):
        vals = s[s.index.month == m].dropna()
        n = len(vals)
        if n < 2:
            rows[m] = {"mean": np.nan, "std": np.nan, "n": n, "tstat": np.nan}
            continue
        mu, sigma = vals.mean(), vals.std(ddof=1)
        t = mu / (sigma / np.sqrt(n))
        rows[m] = {"mean": float(mu), "std": float(sigma), "n": int(n), "tstat": float(t)}
    return pd.DataFrame(rows).T.rename_axis("month")


def spring_autumn_tstat(series: pd.Series) -> dict:
    """Welch two-sample t-stat comparing spring (Apr–Jun) vs autumn (Aug–Nov) monthly returns.

    Returns ``spring_mean``, ``autumn_mean``, ``tstat``, ``n_spring``, ``n_autumn``.
    The hypothesis: spring months earn more than autumn months. A robust result needs |t| ≥ 2.
    """
    s = pd.Series(series).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    sp = s[s.index.month.isin(SPRING_MONTHS)].dropna().to_numpy()
    au = s[s.index.month.isin(AUTUMN_MONTHS)].dropna().to_numpy()
    if len(sp) < 2 or len(au) < 2:
        return {k: np.nan for k in ("spring_mean", "autumn_mean", "tstat", "n_spring", "n_autumn")}
    # Welch t-stat
    mu_sp, mu_au = sp.mean(), au.mean()
    var_sp, var_au = sp.var(ddof=1), au.var(ddof=1)
    n_sp, n_au = len(sp), len(au)
    se = np.sqrt(var_sp / n_sp + var_au / n_au)
    t = (mu_sp - mu_au) / se if se > 0 else np.nan
    return {
        "spring_mean": float(mu_sp),
        "autumn_mean": float(mu_au),
        "tstat": float(t),
        "n_spring": int(n_sp),
        "n_autumn": int(n_au),
    }


def seasonal_timer(
    crude: pd.Series,
    tbill: pd.Series | None = None,
    spring: list[int] = SPRING_MONTHS,
    autumn: list[int] = AUTUMN_MONTHS,
) -> pd.Series:
    """Long crude (or energy) in spring, short in autumn, T-bill otherwise.

    ``tbill`` (a monthly cash-return series) is credited when the position is flat (months outside
    spring and autumn). ``tbill=None`` leaves cash at 0. Months inside ``spring`` go long (return =
    crude return); months inside ``autumn`` go short (return = −crude return); all other months earn
    T-bill. Returns a monthly return series aligned to ``crude``.
    """
    r = pd.Series(crude).astype(float)
    r.index = pd.DatetimeIndex(r.index)
    if tbill is None:
        cash = pd.Series(0.0, index=r.index)
    else:
        cash = pd.Series(tbill).astype(float).reindex(r.index).fillna(0.0)

    position = pd.Series(0.0, index=r.index)
    position[r.index.month.isin(spring)] = 1.0
    position[r.index.month.isin(autumn)] = -1.0

    result = position * r + (position == 0).astype(float) * cash
    return result.rename("seasonal_timer")


def buy_hold(series: pd.Series) -> pd.Series:
    return pd.Series(series).astype(float).dropna().rename("buy_hold")


def summary(returns: pd.Series, periods_per_year: int = MONTHS, rf: pd.Series | None = None) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown for a monthly return series.

    **Sharpe convention**: raw (``mean/std``) when ``rf`` is None; excess-of-cash when ``rf`` is
    given (``mean(r−rf)/std(r−rf)``). Pass the *same* ``rf`` to both legs of a race so the
    comparison is like-for-like. CAGR / vol / max-drawdown always describe the raw series.
    """
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "n")}
    if rf is None:
        ex = r
    else:
        ex = (r - pd.Series(rf).astype(float).reindex(r.index).fillna(0.0)).dropna()
    ex_mean, ex_std = ex.mean(), ex.std(ddof=1)
    std = r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {
        "sharpe": float(ex_mean / ex_std * np.sqrt(periods_per_year)) if ex_std > 0 else np.nan,
        "cagr": float(cagr),
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float(dd),
        "n": int(len(r)),
    }
