"""Strategy + inference for Study 648 — Grain Seasonality.

The claim: grains have a calendar. Every spring, weather risk during planting/pollination puts a
premium into corn, wheat and soybean prices ("the weather market"); every autumn, the new crop
hits the elevator and prices give it back ("harvest pressure"). This is the "old-crop/new-crop"
pattern grain desks have talked about for a century (Working 1949's theory of storage gives it a
textbook mechanism: near-harvest inventories are scarce and rebuild after the new crop arrives).

Measurements, run per grain (corn/CORN, wheat/WEAT, soy/SOYB) and pooled across all three:

* **Month-of-year mean returns** — one-sample naive + Newey-West (HAC) *t* for each of the 12
  calendar months, on the ETF's own monthly return (36 cells across 3 grains). A **Bonferroni**
  bar (`0.05 / 36`) is the honesty rail against cherry-picking one lucky month out of 36 draws.
* **Best/worst month vs rest** — per grain, the single highest- and lowest-mean month, Welch *t*
  against every other month pooled (this is the number a chart-watcher would actually trade).
* **Spring vs harvest, per grain and pooled** — each grain's own USDA-calendar weather-scare
  window vs its own harvest window (hardcoded facts in ``data.py``), Welch *t* plus a
  circular block-bootstrap CI on the spread.
* **Seasonal long/short timer** — long the spring window, short the harvest window, cash
  otherwise, on the *ETF* (the tradable, already-roll-costed instrument), gross and net of
  costs, vs buy-and-hold.
* **Roll caveat** — the ETF's own monthly return vs the roll-naive front-month futures splice
  over the identical window: the gap is what the ETF's contango/roll mechanics already took
  before any of the above numbers were computed.

The decisive number is the pooled spring-vs-harvest Welch/HAC *t* on the REAL ETF tape; the
honest question is whether anything survives 36-way multiple testing and the ETF's own roll.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

MONTHS = 12
TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Monthly returns from a daily close series
# --------------------------------------------------------------------------- #
def monthly_log_returns(close: pd.Series) -> pd.Series:
    """Month-end log returns from a daily close series (last print of each calendar month)."""
    s = pd.Series(close).astype(float).sort_index()
    me = s.resample("ME").last().dropna()
    return np.log(me).diff().dropna().rename("ret")


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def _hac_se(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (Bartlett-kernel) standard error of the sample mean of ``x``."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return float("nan")
    e = x - x.mean()
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        if k >= n:
            break
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * (float(e[k:] @ e[:-k]) / n)
    return float(np.sqrt(max(lrv, 0.0) / n))


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either has < 2 obs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def bonferroni_crit_t(n_tests: int, df: int, alpha: float = 0.05) -> float:
    """Two-sided Bonferroni-corrected critical |t| for ``n_tests`` simultaneous tests."""
    return float(sps.t.ppf(1.0 - alpha / (2.0 * n_tests), df))


# --------------------------------------------------------------------------- #
# Month-of-year table
# --------------------------------------------------------------------------- #
def month_stats(series: pd.Series) -> pd.DataFrame:
    """Per-calendar-month mean, std, n, naive t and HAC t (one-sample, vs 0) for one grain."""
    s = pd.Series(series).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    rows = {}
    for m in range(1, 13):
        vals = s[s.index.month == m].dropna().to_numpy()
        n = len(vals)
        if n < 2:
            rows[m] = {"mean": np.nan, "std": np.nan, "n": n, "tstat": np.nan, "tstat_hac": np.nan}
            continue
        mu, sigma = vals.mean(), vals.std(ddof=1)
        se_hac = _hac_se(vals)
        rows[m] = {
            "mean": float(mu), "std": float(sigma), "n": int(n),
            "tstat": float(mu / (sigma / np.sqrt(n))) if sigma > 0 else np.nan,
            "tstat_hac": float(mu / se_hac) if se_hac and se_hac > 0 else np.nan,
        }
    return pd.DataFrame(rows).T.rename_axis("month")


def best_worst_vs_rest(series: pd.Series) -> dict:
    """Best- and worst-mean calendar month for one grain, Welch t vs every other month pooled."""
    s = pd.Series(series).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    means = {m: s[s.index.month == m].mean() for m in range(1, 13) if (s.index.month == m).sum() >= 2}
    best_m = max(means, key=means.get)
    worst_m = min(means, key=means.get)
    best_x = s[s.index.month == best_m].dropna().to_numpy()
    worst_x = s[s.index.month == worst_m].dropna().to_numpy()
    rest_of_best = s[s.index.month != best_m].dropna().to_numpy()
    rest_of_worst = s[s.index.month != worst_m].dropna().to_numpy()
    return {
        "best_month": int(best_m), "best_mean": float(best_x.mean()),
        "best_t": welch_t(best_x, rest_of_best), "best_n": int(len(best_x)),
        "worst_month": int(worst_m), "worst_mean": float(worst_x.mean()),
        "worst_t": welch_t(worst_x, rest_of_worst), "worst_n": int(len(worst_x)),
    }


# --------------------------------------------------------------------------- #
# Spring vs harvest — the "old-crop/new-crop" headline
# --------------------------------------------------------------------------- #
def spring_harvest_tstat(series: pd.Series, spring: tuple[int, ...], harvest: tuple[int, ...]) -> dict:
    """Welch two-sample t of spring (weather-scare) vs harvest monthly returns, one grain."""
    s = pd.Series(series).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    sp = s[s.index.month.isin(spring)].dropna().to_numpy()
    hv = s[s.index.month.isin(harvest)].dropna().to_numpy()
    if len(sp) < 2 or len(hv) < 2:
        return {k: np.nan for k in ("spring_mean", "harvest_mean", "spread", "t", "n_spring", "n_harvest")}
    return {
        "spring_mean": float(sp.mean()), "harvest_mean": float(hv.mean()),
        "spread": float(sp.mean() - hv.mean()), "t": welch_t(sp, hv),
        "n_spring": int(len(sp)), "n_harvest": int(len(hv)),
    }


def pooled_spring_harvest(monthly: dict[str, pd.Series], spring: dict[str, tuple[int, ...]],
                          harvest: dict[str, tuple[int, ...]]) -> dict:
    """Pool spring- and harvest-tagged monthly returns across all grains, Welch t of the spread."""
    sp_all, hv_all = [], []
    for g, s in monthly.items():
        s = pd.Series(s).astype(float)
        s.index = pd.DatetimeIndex(s.index)
        sp_all.append(s[s.index.month.isin(spring[g])].dropna().to_numpy())
        hv_all.append(s[s.index.month.isin(harvest[g])].dropna().to_numpy())
    sp = np.concatenate(sp_all)
    hv = np.concatenate(hv_all)
    return {
        "spring_mean": float(sp.mean()), "harvest_mean": float(hv.mean()),
        "spread": float(sp.mean() - hv.mean()), "t": welch_t(sp, hv),
        "n_spring": int(len(sp)), "n_harvest": int(len(hv)),
    }


def pooled_spread_bootstrap_ci(monthly: dict[str, pd.Series], spring: dict[str, tuple[int, ...]],
                               harvest: dict[str, tuple[int, ...]], n_boot: int = 5000,
                               block: int = 12, seed: int = 648, alpha: float = 0.05) -> dict:
    """Circular block-bootstrap CI for the pooled spring-minus-harvest spread.

    Resamples 12-month blocks *within each grain* (respecting the annual seasonal structure and
    each grain's own calendar), recombines across grains, and returns the percentile CI on the
    pooled spread. A CI straddling 0 means the pooled spread is indistinguishable from noise.
    """
    rng = np.random.default_rng(seed)
    grains = list(monthly.keys())
    series = {}
    for g in grains:
        s = pd.Series(monthly[g]).astype(float)
        s.index = pd.DatetimeIndex(s.index)
        series[g] = s

    def spread_of(idx_map):
        sp_all, hv_all = [], []
        for g in grains:
            v = series[g].to_numpy()[idx_map[g]]
            m = series[g].index.month.to_numpy()[idx_map[g]]
            sp_all.append(v[np.isin(m, spring[g])])
            hv_all.append(v[np.isin(m, harvest[g])])
        sp, hv = np.concatenate(sp_all), np.concatenate(hv_all)
        if len(sp) < 1 or len(hv) < 1:
            return np.nan
        return float(sp.mean() - hv.mean())

    point = spread_of({g: np.arange(len(series[g])) for g in grains})
    draws = []
    for _ in range(n_boot):
        idx_map = {}
        for g in grains:
            n = len(series[g])
            n_blocks = int(np.ceil(n / block))
            starts = rng.integers(0, n, size=n_blocks)
            idx = np.concatenate([(np.arange(st, st + block) % n) for st in starts])[:n]
            idx_map[g] = idx
        d = spread_of(idx_map)
        if np.isfinite(d):
            draws.append(d)
    draws = np.asarray(draws)
    return {"point": point, "lo": float(np.quantile(draws, alpha / 2)),
            "hi": float(np.quantile(draws, 1 - alpha / 2)), "n_boot": int(len(draws))}


# --------------------------------------------------------------------------- #
# Seasonal long/short timer + costs
# --------------------------------------------------------------------------- #
def seasonal_timer(returns: pd.Series, spring: tuple[int, ...], harvest: tuple[int, ...],
                   tbill: pd.Series | None = None) -> pd.Series:
    """Long the spring window, short the harvest window, T-bill (or 0) otherwise.

    Calendar-known rule -> no execution lag: USDA's typical crop-progress windows are the same
    every year, so the position for month *m* is set from the calendar alone, not from a signal
    observed the prior close.
    """
    r = pd.Series(returns).astype(float)
    r.index = pd.DatetimeIndex(r.index)
    cash = pd.Series(0.0, index=r.index) if tbill is None else \
        pd.Series(tbill).astype(float).reindex(r.index).fillna(0.0)
    position = pd.Series(0.0, index=r.index)
    position[r.index.month.isin(spring)] = 1.0
    position[r.index.month.isin(harvest)] = -1.0
    return (position * r + (position == 0).astype(float) * cash).rename("seasonal_timer")


def apply_costs(returns: pd.Series, n_trades_per_year: float, cost_bps_one_way: float) -> pd.Series:
    """Subtract the annual one-way-cost budget spread evenly across the 12 months."""
    r = pd.Series(returns).astype(float).copy()
    monthly_cost = (n_trades_per_year * cost_bps_one_way / 1e4) / MONTHS
    return (r - monthly_cost).rename("net")


def summary(returns: pd.Series, periods_per_year: int = MONTHS, rf: pd.Series | None = None) -> dict:
    """Annualised Sharpe (excess-of-cash if ``rf`` given), CAGR, vol, max-drawdown."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "n")}
    ex = r if rf is None else (r - pd.Series(rf).astype(float).reindex(r.index).fillna(0.0)).dropna()
    ex_mean, ex_std = ex.mean(), ex.std(ddof=1)
    std = r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {
        "sharpe": float(ex_mean / ex_std * np.sqrt(periods_per_year)) if ex_std > 0 else np.nan,
        "cagr": float(cagr), "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": dd, "n": int(len(r)),
    }


# --------------------------------------------------------------------------- #
# Roll caveat — the ETF vs the roll-naive futures splice
# --------------------------------------------------------------------------- #
def roll_drag(etf_monthly: pd.Series, fut_monthly: pd.Series) -> dict:
    """Mean monthly return gap: ETF (roll-costed, tradable) minus futures splice (roll-naive proxy).

    A negative gap is the ETF quietly paying away return to its own roll mechanics relative to the
    (untradable) front-month proxy -- i.e. the "ETF contango" caveat, sized in basis points/month.
    """
    e = pd.Series(etf_monthly).astype(float)
    f = pd.Series(fut_monthly).astype(float)
    idx = e.index.intersection(f.index)
    e, f = e.loc[idx], f.loc[idx]
    gap = e - f
    return {
        "etf_mean_bps": float(e.mean() * 1e4), "fut_mean_bps": float(f.mean() * 1e4),
        "drag_bps": float(gap.mean() * 1e4), "t": _one_sample_t(gap.to_numpy()),
        "n": int(len(idx)),
    }


def _one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(df: pd.DataFrame, spring: tuple[int, ...], harvest: tuple[int, ...]) -> dict:
    """Run the spring-vs-harvest Welch split on a synthetic monthly-return world."""
    return spring_harvest_tstat(df["ret"], spring, harvest)
