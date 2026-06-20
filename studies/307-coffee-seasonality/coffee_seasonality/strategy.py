"""Coffee-seasonality: monthly pattern tests and a frost/harvest calendar timer.

The folklore: Arabica coffee (KC=F) should rally into the Brazilian frost-risk window (Jun–Aug, when
a single overnight frost in the southern-hemisphere winter can destroy a crop and spike prices) and
weaken under harvest pressure (May/Sep, when new crop floods the market). We test: (1) per-month
one-sample HAC t-stats; (2) is the frost group significantly stronger than the harvest group? (3) a
block-bootstrap CI on the frost-minus-harvest spread; (4) does a long-frost / short-harvest calendar
timer beat buy-and-hold?

Conventions, stated up front:

  * **HAC (Newey-West) t-stats**, not naive ones — coffee returns cluster (weather scares, supply
    shocks), and the naive t overstates significance. We expose both for honesty.
  * **The cash leg earns the T-bill.** :func:`seasonal_timer` credits a monthly cash return when the
    rule is flat, so the timing race is done on *excess-of-cash* Sharpe, like-for-like.
  * **Calendar-known rule, no execution lag.** The month is known in advance, so the position is set
    at the start of each month with no ``shift`` — a frost-season rule needs no signal-to-trade lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12
FROST_MONTHS = [6, 7, 8]      # Jun–Aug: Brazilian winter, frost risk → bullish leg
HARVEST_MONTHS = [5, 9]       # May/Sep: harvest-pressure shoulders → bearish leg


def _hac_se(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (Bartlett-kernel) standard error of the sample mean of ``x``."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.nan
    e = x - x.mean()
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        if k >= n:
            break
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * (float(e[k:] @ e[:-k]) / n)
    return np.sqrt(max(lrv, 0.0) / n)


def month_stats(series: pd.Series) -> pd.DataFrame:
    """Per-calendar-month mean, std, count, naive t-stat and HAC t-stat for a monthly return series.

    Returns a DataFrame indexed 1..12 with columns ``mean``, ``std``, ``n``, ``tstat`` (naive),
    ``tstat_hac`` (Newey-West). A robust seasonality claim needs |t_HAC| ≥ 2 *after* multiple-testing
    adjustment (Bonferroni for 12 months: 0.05/12 ≈ 0.004, so effectively |t| ≈ 3 at n ≈ 30).
    """
    s = pd.Series(series).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    rows = {}
    for m in range(1, 13):
        vals = s[s.index.month == m].dropna()
        n = len(vals)
        if n < 2:
            rows[m] = {"mean": np.nan, "std": np.nan, "n": n, "tstat": np.nan, "tstat_hac": np.nan}
            continue
        v = vals.to_numpy()
        mu, sigma = v.mean(), v.std(ddof=1)
        se_hac = _hac_se(v)
        rows[m] = {
            "mean": float(mu),
            "std": float(sigma),
            "n": int(n),
            "tstat": float(mu / (sigma / np.sqrt(n))) if sigma > 0 else np.nan,
            "tstat_hac": float(mu / se_hac) if se_hac and se_hac > 0 else np.nan,
        }
    return pd.DataFrame(rows).T.rename_axis("month")


def frost_harvest_tstat(series: pd.Series) -> dict:
    """Welch two-sample t-stat comparing frost (Jun–Aug) vs harvest (May/Sep) monthly returns.

    Returns ``frost_mean``, ``harvest_mean``, ``spread``, ``tstat``, ``n_frost``, ``n_harvest``.
    Hypothesis: frost-risk months earn more than harvest-pressure months. Robust result needs |t| ≥ 2.
    """
    s = pd.Series(series).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    fr = s[s.index.month.isin(FROST_MONTHS)].dropna().to_numpy()
    ha = s[s.index.month.isin(HARVEST_MONTHS)].dropna().to_numpy()
    if len(fr) < 2 or len(ha) < 2:
        return {k: np.nan for k in ("frost_mean", "harvest_mean", "spread", "tstat", "n_frost", "n_harvest")}
    mu_f, mu_h = fr.mean(), ha.mean()
    var_f, var_h = fr.var(ddof=1), ha.var(ddof=1)
    n_f, n_h = len(fr), len(ha)
    se = np.sqrt(var_f / n_f + var_h / n_h)
    return {
        "frost_mean": float(mu_f),
        "harvest_mean": float(mu_h),
        "spread": float(mu_f - mu_h),
        "tstat": float((mu_f - mu_h) / se) if se > 0 else np.nan,
        "n_frost": int(n_f),
        "n_harvest": int(n_h),
    }


def spread_bootstrap_ci(
    series: pd.Series, n_boot: int = 2000, block: int = 12, seed: int = 307, alpha: float = 0.05
) -> dict:
    """Circular block-bootstrap CI for the frost-minus-harvest monthly-mean spread.

    Resamples 12-month blocks (one calendar year) to respect the annual seasonal structure, recomputes
    the frost-minus-harvest spread on each resample, and returns the percentile CI. ``[lo, hi]``
    straddling 0 means the spread is indistinguishable from noise.
    """
    s = pd.Series(series).astype(float).dropna()
    s.index = pd.DatetimeIndex(s.index)
    months = s.index.month.to_numpy()
    vals = s.to_numpy()
    n = len(vals)
    if n < block * 2:
        return {"point": np.nan, "lo": np.nan, "hi": np.nan, "n_boot": 0}
    rng = np.random.default_rng(seed)

    def spread_of(idx):
        mv, vv = months[idx], vals[idx]
        fr = vv[np.isin(mv, FROST_MONTHS)]
        ha = vv[np.isin(mv, HARVEST_MONTHS)]
        if len(fr) < 1 or len(ha) < 1:
            return np.nan
        return fr.mean() - ha.mean()

    point = spread_of(np.arange(n))
    n_blocks = int(np.ceil(n / block))
    draws = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(st, st + block) % n) for st in starts])[:n]
        d = spread_of(idx)
        if np.isfinite(d):
            draws.append(d)
    draws = np.array(draws)
    return {
        "point": float(point),
        "lo": float(np.quantile(draws, alpha / 2)),
        "hi": float(np.quantile(draws, 1 - alpha / 2)),
        "n_boot": int(len(draws)),
    }


def seasonal_timer(
    coffee: pd.Series,
    tbill: pd.Series | None = None,
    frost: list[int] = FROST_MONTHS,
    harvest: list[int] = HARVEST_MONTHS,
) -> pd.Series:
    """Long coffee in the frost window, short in the harvest window, T-bill otherwise.

    Calendar-known rule → no execution lag. ``tbill`` (a monthly cash-return series) is credited when
    the position is flat (months outside both windows). ``tbill=None`` leaves cash at 0. Returns a
    monthly return series aligned to ``coffee``.
    """
    r = pd.Series(coffee).astype(float)
    r.index = pd.DatetimeIndex(r.index)
    cash = pd.Series(0.0, index=r.index) if tbill is None else pd.Series(tbill).astype(float).reindex(r.index).fillna(0.0)

    position = pd.Series(0.0, index=r.index)
    position[r.index.month.isin(frost)] = 1.0
    position[r.index.month.isin(harvest)] = -1.0
    return (position * r + (position == 0).astype(float) * cash).rename("seasonal_timer")


def buy_hold(series: pd.Series) -> pd.Series:
    return pd.Series(series).astype(float).dropna().rename("buy_hold")


def summary(returns: pd.Series, periods_per_year: int = MONTHS, rf: pd.Series | None = None) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown for a monthly return series.

    **Sharpe convention**: raw (``mean/std``) when ``rf`` is None; excess-of-cash when ``rf`` is given
    (``mean(r−rf)/std(r−rf)``). Pass the *same* ``rf`` to both legs of a race so it is like-for-like.
    CAGR / vol / max-drawdown always describe the raw series.
    """
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "n")}
    ex = r if rf is None else (r - pd.Series(rf).astype(float).reindex(r.index).fillna(0.0)).dropna()
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


def apply_costs(returns: pd.Series, n_trades_per_year: float, cost_bps_one_way: float) -> pd.Series:
    """Subtract round-trip transaction cost spread evenly across the months that trade.

    ``cost_bps_one_way`` is one-way cost in basis points × NAV; a frost/harvest timer enters and exits
    each leg, so a full round trip is charged per entry+exit. We deduct the annual cost budget
    (``n_trades_per_year`` one-way legs × ``cost_bps_one_way``) spread across the 12 months.
    """
    r = pd.Series(returns).astype(float).copy()
    monthly_cost = (n_trades_per_year * cost_bps_one_way / 1e4) / MONTHS
    return (r - monthly_cost).rename("net")
