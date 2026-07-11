"""Strategy + inference for Study 651 — Sugar-Seasonality.

The claim: raw sugar (ICE No.11) has a harvest calendar driven by the world's two largest cane
suppliers. Brazil's Center-South crush runs roughly **April → November** and India's cane-crushing
season runs roughly **October → April**; between them, old-crop stocks are supposed to be scarcest
in the Northern-Hemisphere winter, right before Brazil's new crush gets into full swing — a
"pre-harvest tight" premium — and give it back every spring as the Brazilian crush floods the market
with new supply ("crush glut"). This is the same "old-crop/new-crop" shape as study 648's grain
calendar, applied to a different crop and a different crush cycle.

Measurements, on CANE (the tradable ETF) with SB=F (roll-naive futures) as the cross-check:

* **Month-of-year mean returns** — one-sample naive + Newey-West (HAC) *t* for each of the 12
  calendar months (12 cells). A **Bonferroni** bar (`0.05 / 12`) is the honesty rail against
  cherry-picking one lucky month out of 12 draws.
* **Best/worst month vs rest** — the single highest- and lowest-mean month, Welch *t* against every
  other month pooled (the number a chart-watcher would actually trade).
* **Tight vs crush** — the claimed pre-harvest-tight window (Jan-Mar) vs the claimed crush-glut
  window (Apr-Jul), Welch *t* plus a circular block-bootstrap CI on the spread.
* **Seasonal long/short timer** — long the tight window, short the crush window, cash otherwise, on
  the *ETF* (the tradable, already-roll-costed instrument), gross and net of costs, vs buy-and-hold.
  Active-leg hit rate vs a fair coin.
* **Roll caveat** — CANE's own monthly return vs the roll-naive SB=F front-month splice over the
  identical window: the gap is what the ETF's contango/roll mechanics already took before any of the
  above numbers were computed.

The decisive number is the tight-vs-crush Welch/HAC *t* on the REAL CANE tape; the honest question is
whether anything survives 12-way multiple testing and the ETF's own roll.
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


def _one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Month-of-year table
# --------------------------------------------------------------------------- #
def month_stats(series: pd.Series) -> pd.DataFrame:
    """Per-calendar-month mean, std, n, naive t and HAC t (one-sample, vs 0)."""
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
    """Best- and worst-mean calendar month, Welch t vs every other month pooled."""
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
# Tight vs crush — the "pre-harvest tight / crush glut" headline
# --------------------------------------------------------------------------- #
def tight_crush_tstat(series: pd.Series, tight: tuple[int, ...], crush: tuple[int, ...]) -> dict:
    """Welch two-sample t of pre-harvest-tight vs crush-glut monthly returns."""
    s = pd.Series(series).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    tg = s[s.index.month.isin(tight)].dropna().to_numpy()
    cr = s[s.index.month.isin(crush)].dropna().to_numpy()
    if len(tg) < 2 or len(cr) < 2:
        return {k: np.nan for k in ("tight_mean", "crush_mean", "spread", "t", "n_tight", "n_crush")}
    return {
        "tight_mean": float(tg.mean()), "crush_mean": float(cr.mean()),
        "spread": float(tg.mean() - cr.mean()), "t": welch_t(tg, cr),
        "n_tight": int(len(tg)), "n_crush": int(len(cr)),
    }


def spread_bootstrap_ci(series: pd.Series, tight: tuple[int, ...], crush: tuple[int, ...],
                        n_boot: int = 5000, block: int = 12, seed: int = 651,
                        alpha: float = 0.05) -> dict:
    """Circular block-bootstrap CI for the tight-minus-crush monthly-mean spread.

    Resamples 12-month blocks (one calendar year) to respect the annual seasonal structure,
    recomputes the tight-minus-crush spread on each resample, and returns the percentile CI. A CI
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
        tg = vv[np.isin(mv, tight)]
        cr = vv[np.isin(mv, crush)]
        if len(tg) < 1 or len(cr) < 1:
            return np.nan
        return float(tg.mean() - cr.mean())

    point = spread_of(np.arange(n))
    n_blocks = int(np.ceil(n / block))
    draws = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(st, st + block) % n) for st in starts])[:n]
        d = spread_of(idx)
        if np.isfinite(d):
            draws.append(d)
    draws = np.asarray(draws)
    return {"point": point, "lo": float(np.quantile(draws, alpha / 2)),
            "hi": float(np.quantile(draws, 1 - alpha / 2)), "n_boot": int(len(draws))}


# --------------------------------------------------------------------------- #
# Seasonal long/short timer + costs
# --------------------------------------------------------------------------- #
def seasonal_timer(returns: pd.Series, tight: tuple[int, ...], crush: tuple[int, ...],
                   tbill: pd.Series | None = None) -> pd.Series:
    """Long the tight window, short the crush window, T-bill (or 0) otherwise.

    Calendar-known rule -> no execution lag: Brazil's and India's crush windows are the same every
    year, so the position for month *m* is set from the calendar alone, not from a signal observed
    the prior close.
    """
    r = pd.Series(returns).astype(float)
    r.index = pd.DatetimeIndex(r.index)
    cash = pd.Series(0.0, index=r.index) if tbill is None else \
        pd.Series(tbill).astype(float).reindex(r.index).fillna(0.0)
    position = pd.Series(0.0, index=r.index)
    position[r.index.month.isin(tight)] = 1.0
    position[r.index.month.isin(crush)] = -1.0
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


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


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


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(df: pd.DataFrame, tight: tuple[int, ...], crush: tuple[int, ...]) -> dict:
    """Run the tight-vs-crush Welch split on a synthetic monthly-return world."""
    return tight_crush_tstat(df["ret"], tight, crush)
