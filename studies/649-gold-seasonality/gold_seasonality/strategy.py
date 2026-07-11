"""Strategy + inference for Study 649 — Gold Seasonality.

The claim: gold has a calendar. **September** is "gold's best month" — Indian wedding-season and
pre-Diwali physical jewellery demand, plus Northern-hemisphere restocking ahead of year-end —
while the Northern-hemisphere **summer** (roughly May-August) is a quiet "lull" between the
spring Akshaya Tritiya buying and the autumn wedding season.

Measurements, all on GLD's own monthly return series (the tradable spot-gold proxy):

* **Month-of-year mean returns** — one-sample naive + Newey-West (HAC) *t* for each of the 12
  calendar months (12 cells). A **Bonferroni** bar (`0.05 / 12`) is the honesty rail against
  reporting one lucky month out of 12 draws as "the" seasonal.
* **September vs the rest** — the headline: Welch *t* of September's monthly return against the
  other 11 months pooled.
* **Summer vs the rest** — the second half of the claim: Welch *t* of the pooled May-Aug months
  against the other 8 months pooled.
* **Era contrast** — September's effect before vs after the 2013 gold crash (2013-04-12/15, an
  externally-dated, justified split — the end of the 2001-2012 bull "supercycle"), within-era
  Welch *t*'s plus a Welch *t* **of the difference** between the two eras.
* **The "own gold only in strong months" timer** — long GLD only in September (the claimed
  strong month), cash the other 11 months, gross and net of costs, vs simple buy-and-hold — both
  raced excess-of-cash (^IRX) so a partly-invested timer isn't flattered for merely holding less
  risk.

The decisive number is the September-vs-rest Welch/HAC *t*, checked against the Bonferroni bar
for 12 simultaneous month-of-year tests; the honest question is whether the "best month" survives
multiple testing and whether owning gold only in September beats simply owning it all year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

MONTHS = 12


# --------------------------------------------------------------------------- #
# Monthly returns from a daily close series
# --------------------------------------------------------------------------- #
def monthly_log_returns(close: pd.Series) -> pd.Series:
    """Month-end log returns from a daily close series (last print of each calendar month)."""
    s = pd.Series(close).astype(float).sort_index()
    me = s.resample("ME").last().dropna()
    return np.log(me).diff().dropna().rename("ret")


def monthly_cash_return(irx_close: pd.Series) -> pd.Series:
    """Monthly simple T-bill return from the ^IRX annualised discount yield (month-end print)."""
    s = pd.Series(irx_close).astype(float).sort_index()
    me = s.resample("ME").last().dropna()
    return ((me / 100.0) / 12.0).rename("cash")


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


def month_vs_rest(series: pd.Series, months: tuple[int, ...]) -> dict:
    """Welch t of the pooled ``months`` group vs every other month pooled."""
    s = pd.Series(series).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    a = s[s.index.month.isin(months)].dropna().to_numpy()
    b = s[~s.index.month.isin(months)].dropna().to_numpy()
    return {
        "mean": float(a.mean()), "rest_mean": float(b.mean()), "spread": float(a.mean() - b.mean()),
        "t": welch_t(a, b), "n": int(len(a)), "n_rest": int(len(b)),
    }


def spread_bootstrap_ci(series: pd.Series, months: tuple[int, ...], n_boot: int = 5000,
                        block: int = 12, seed: int = 649, alpha: float = 0.05) -> dict:
    """Circular block-bootstrap CI for the ``months``-vs-rest spread.

    Resamples 12-month blocks (respecting the annual seasonal structure) and returns the
    percentile CI on the spread. A CI straddling 0 means the spread is indistinguishable from
    noise.
    """
    rng = np.random.default_rng(seed)
    s = pd.Series(series).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    x = s.to_numpy()
    m = s.index.month.to_numpy()
    n = len(x)

    def spread_of(idx: np.ndarray) -> float:
        v, mm = x[idx], m[idx]
        a = v[np.isin(mm, months)]
        b = v[~np.isin(mm, months)]
        if len(a) < 1 or len(b) < 1:
            return float("nan")
        return float(a.mean() - b.mean())

    point = spread_of(np.arange(n))
    draws = []
    n_blocks = int(np.ceil(n / block))
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
# Era contrast — pre vs post the 2013 gold crash (justified split)
# --------------------------------------------------------------------------- #
def era_contrast(series: pd.Series, months: tuple[int, ...], split: str) -> dict:
    """``months``-group mean before vs since ``split``: within-era Welch t's + Welch t OF THE
    DIFFERENCE between the two eras' group means."""
    s = pd.Series(series).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    grp = s.index.month.isin(months)
    early = s[grp & (s.index < split)].dropna().to_numpy()
    late = s[grp & (s.index >= split)].dropna().to_numpy()
    rest_early = s[~grp & (s.index < split)].dropna().to_numpy()
    rest_late = s[~grp & (s.index >= split)].dropna().to_numpy()
    return {
        "n_early": len(early), "n_late": len(late),
        "early_mean": float(np.nanmean(early)) if len(early) else float("nan"),
        "late_mean": float(np.nanmean(late)) if len(late) else float("nan"),
        "welch_t_early": welch_t(early, rest_early),
        "welch_t_late": welch_t(late, rest_late),
        "welch_t_diff": welch_t(late, early),
    }


# --------------------------------------------------------------------------- #
# The "own gold only in strong months" timer
# --------------------------------------------------------------------------- #
def strong_month_timer(returns: pd.Series, strong: tuple[int, ...],
                       cash: pd.Series | None = None) -> pd.Series:
    """Long GLD only in the ``strong`` calendar months, cash (or 0) otherwise.

    Calendar-known rule -> no execution lag: September is the same calendar slot every year, so
    the position for a given month is set from the calendar alone, not from a signal observed the
    prior close. The monthly return itself already runs from the *prior* month-end close to the
    current month-end close, so being "in" September means entering at the August close and
    exiting at the September close — exactly the folklore's own trade, zero look-ahead.
    """
    r = pd.Series(returns).astype(float)
    r.index = pd.DatetimeIndex(r.index)
    c = pd.Series(0.0, index=r.index) if cash is None else \
        pd.Series(cash).astype(float).reindex(r.index).fillna(0.0)
    position = r.index.month.isin(strong)
    return pd.Series(np.where(position, r.to_numpy(), c.to_numpy()), index=r.index, name="timer")


def apply_timer_costs(timer: pd.Series, strong: tuple[int, ...],
                      cost_bps_one_way: float) -> pd.Series:
    """Subtract 2 one-way legs (enter + exit) x ``cost_bps_one_way`` on every active month."""
    r = pd.Series(timer).astype(float).copy()
    r.index = pd.DatetimeIndex(r.index)
    mask = r.index.month.isin(strong)
    r.loc[mask] = r.loc[mask] - 2.0 * cost_bps_one_way / 1e4
    return r.rename("timer_net")


def summary(returns: pd.Series, periods_per_year: int = MONTHS,
           rf: pd.Series | None = None) -> dict:
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


def hit_rate(series: pd.Series, months: tuple[int, ...]) -> dict:
    """Share of ``months``-group observations that are positive, with a Wilson interval."""
    s = pd.Series(series).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    x = s[s.index.month.isin(months)].dropna().to_numpy()
    k = int((x > 0).sum())
    n = len(x)
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(df: pd.DataFrame, strong: tuple[int, ...]) -> dict:
    """Run the strong-months-vs-rest Welch split on a synthetic monthly-return world."""
    return month_vs_rest(df["ret"], strong)
