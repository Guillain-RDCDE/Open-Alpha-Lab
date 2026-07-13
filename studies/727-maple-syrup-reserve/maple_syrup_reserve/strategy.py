"""Strategy + inference for Study 727 — "the maple-syrup reserve as a trade".

The claim, steelmanned: Quebec runs a strategic maple-syrup reserve like a central bank
runs gold; maple is a real soft commodity; so (a) the maple price should reward a holder,
(b) *something* listed should give you that exposure, and (c) there should be a tradable
sugaring-season seasonal. We test the strongest tradable version of each:

1. **The administered maple price itself** (hardcoded, cited, approximate) vs the S&P/TSX
   — CAGR, vol, drawdown, and a *t*-stat of the annual excess. The "does maple reward a
   holder better than stocks?" question, on the PPAQ-negotiated series believers quote.
2. **The tradable proxy** (Rogers Sugar, `RSI.TO`) vs the TSX — the only listed thing
   with real maple exposure. Monthly returns, Newey-West (HAC) *t* of the alpha, Sharpe,
   drawdown. Plus a sugar-futures placebo (`SB=F`) to show the nearest sweetener isn't it.
3. **The sugaring-season seasonal** — per-month HAC *t*-stats, a Feb–Apr vs rest-of-year
   Welch test with a circular-block-bootstrap CI, and a long-in-spring calendar timer vs
   buy-and-hold, costs net, cash earning the benchmark when flat.

Conventions, stated up front:
  * **HAC (Newey-West) t-stats**, not naive ones — equity returns cluster; the naive t
    overstates significance. We expose both.
  * **Calendar-known rule, no execution lag.** The sugaring months are known in advance,
    so the seasonal position is set at the start of each month with no ``shift``.
  * **Costs one-way × NAV; the flat leg earns the benchmark**, so the timing race is on
    excess-of-benchmark Sharpe, like-for-like.

Pure numpy/pandas; scipy only for the t-distribution p-value (optional, guarded).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MONTHS = 12.0
SUGARING_MONTHS = [2, 3, 4]


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
    roll_max = level.cummax()
    dd = level / roll_max - 1.0
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
# Inference
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


def annual_excess_t(price_level: pd.Series, bench_level: pd.Series) -> dict:
    """*t*-stat that the maple price's **annual** return beats the benchmark's.

    Aligns both to year-end, takes annual simple returns, and tests the mean of the
    paired excess (maple - bench) against 0. Small-sample, so this is a weak test by
    construction — which is itself part of the finding.
    """
    a = price_level.resample("YE").last().pct_change().dropna()
    b = bench_level.resample("YE").last().pct_change().dropna()
    j = pd.concat([a, b], axis=1, keys=["maple", "bench"]).dropna()
    ex = j["maple"] - j["bench"]
    n = len(ex)
    if n < 2:
        return {"mean_excess": float("nan"), "t": float("nan"), "p": float("nan"), "n": n}
    se = ex.std(ddof=1) / math.sqrt(n)
    t = ex.mean() / se if se > 0 else float("nan")
    return {"mean_excess": float(ex.mean()), "t": float(t),
            "p": _t_p_value(t, n - 1), "n": n}


def newey_west_alpha_t(proxy_ret: pd.Series, bench_ret: pd.Series,
                       lags: int = 6) -> dict:
    """Newey-West (HAC) *t* of the monthly alpha from r_proxy = a + b*r_bench + e.

    Returns the OLS alpha (monthly + annualised), beta, the HAC standard error of alpha
    and its *t*. ``lags`` is the Bartlett-kernel truncation. The Signal-axis statistic for
    a proxy: is there alpha vs the market the proxy is exposed to? `REAL` needs |t| >= 2.
    """
    j = pd.concat([proxy_ret, bench_ret], axis=1, keys=["y", "x"]).dropna()
    y = j["y"].to_numpy()
    x = j["x"].to_numpy()
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        Xe = X * resid[:, None]
        Gamma = Xe[L:].T @ Xe[:-L]
        S += w * (Gamma + Gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    se_alpha = math.sqrt(cov[0, 0])
    a_m = float(beta[0])
    t_a = a_m / se_alpha if se_alpha > 0 else float("nan")
    return {
        "alpha_m": a_m,
        "alpha_ann": (1 + a_m) ** 12 - 1,
        "beta": float(beta[1]),
        "se_alpha": float(se_alpha),
        "t_alpha": float(t_a),
        "p_alpha": _t_p_value(t_a, n - 2),
        "n": n,
    }


# --------------------------------------------------------------------------- #
# Seasonality machinery (the sugaring-season test)
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
    return math.sqrt(max(lrv, 0.0) / n)


def month_stats(returns: pd.Series) -> pd.DataFrame:
    """Per-calendar-month mean, std, count, naive t and HAC t for a monthly return series.

    Returns a DataFrame indexed 1..12. A robust seasonality claim needs |t_HAC| >= 2
    *after* multiple-testing adjustment (Bonferroni for 12 months ~ |t| >= 3).
    """
    s = pd.Series(returns).astype(float)
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


def season_tstat(returns: pd.Series, season: list[int] = SUGARING_MONTHS) -> dict:
    """Welch two-sample t comparing sugaring-season months vs the rest of the year.

    Returns ``season_mean``, ``rest_mean``, ``spread``, ``tstat``, ``n_season``,
    ``n_rest``. Hypothesis: sugaring months (Feb–Apr) earn more. Robust needs |t| >= 2.
    """
    s = pd.Series(returns).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    se = s[s.index.month.isin(season)].dropna().to_numpy()
    re_ = s[~s.index.month.isin(season)].dropna().to_numpy()
    if len(se) < 2 or len(re_) < 2:
        return {k: np.nan for k in ("season_mean", "rest_mean", "spread", "tstat", "n_season", "n_rest")}
    mu_s, mu_r = se.mean(), re_.mean()
    var_s, var_r = se.var(ddof=1), re_.var(ddof=1)
    n_s, n_r = len(se), len(re_)
    sd = math.sqrt(var_s / n_s + var_r / n_r)
    return {
        "season_mean": float(mu_s),
        "rest_mean": float(mu_r),
        "spread": float(mu_s - mu_r),
        "tstat": float((mu_s - mu_r) / sd) if sd > 0 else np.nan,
        "n_season": int(n_s),
        "n_rest": int(n_r),
    }


def season_bootstrap_ci(returns: pd.Series, season: list[int] = SUGARING_MONTHS,
                        n_boot: int = 2000, block: int = 12, seed: int = 727,
                        alpha: float = 0.05) -> dict:
    """Circular block-bootstrap CI for the season-minus-rest monthly-mean spread.

    Resamples 12-month blocks (one calendar year) to respect the annual seasonal
    structure. ``[lo, hi]`` straddling 0 means the spread is indistinguishable from noise.
    """
    s = pd.Series(returns).astype(float).dropna()
    s.index = pd.DatetimeIndex(s.index)
    months = s.index.month.to_numpy()
    vals = s.to_numpy()
    n = len(vals)
    if n < block * 2:
        return {"point": np.nan, "lo": np.nan, "hi": np.nan, "n_boot": 0}
    rng = np.random.default_rng(seed)
    seasarr = np.array(season)

    def spread_of(idx):
        mv, vv = months[idx], vals[idx]
        se = vv[np.isin(mv, seasarr)]
        re_ = vv[~np.isin(mv, seasarr)]
        if len(se) < 1 or len(re_) < 1:
            return np.nan
        return se.mean() - re_.mean()

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


# --------------------------------------------------------------------------- #
# The sugaring-season timer (the only tradable expression) vs buy-and-hold
# --------------------------------------------------------------------------- #
def seasonal_timer(proxy_ret: pd.Series, bench_ret: pd.Series | None = None,
                   season: list[int] = SUGARING_MONTHS) -> pd.Series:
    """Long the proxy during the sugaring season, else hold the benchmark (cash-of-market).

    Calendar-known rule → no execution lag. When flat (outside Feb–Apr) the book earns the
    benchmark return, so the race is excess-of-benchmark on both legs. Returns a monthly
    return series aligned to ``proxy_ret``.
    """
    r = pd.Series(proxy_ret).astype(float)
    r.index = pd.DatetimeIndex(r.index)
    if bench_ret is None:
        flat = pd.Series(0.0, index=r.index)
    else:
        flat = pd.Series(bench_ret).astype(float).reindex(r.index).fillna(0.0)
    in_season = r.index.month.isin(season)
    return (pd.Series(np.where(in_season, r.values, flat.values), index=r.index)
            .rename("seasonal_timer"))


def apply_costs(returns: pd.Series, n_legs_per_year: float = 2.0,
                cost_bps_one_way: float = 15.0) -> pd.Series:
    """Subtract one-way cost × NAV, spread across the 12 months.

    A long-in-spring timer enters once and exits once a year → ``n_legs_per_year`` one-way
    legs. ``cost_bps_one_way`` is one-way cost in bps × NAV. We deduct the annual budget
    (legs × cost) spread evenly across the months.
    """
    r = pd.Series(returns).astype(float).copy()
    monthly_cost = (n_legs_per_year * cost_bps_one_way / 1e4) / MONTHS
    return (r - monthly_cost).rename("net")


def summary_ret(returns: pd.Series, periods_per_year: float = MONTHS,
                rf: pd.Series | None = None) -> dict:
    """Annualised Sharpe / CAGR / vol / max-drawdown for a monthly *return* series.

    Sharpe is raw when ``rf`` is None, else excess-of-``rf`` (pass the same rf to both
    legs of a race for like-for-like). CAGR/vol/MDD describe the raw series.
    """
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol", "mdd", "n")}
    ex = r if rf is None else (r - pd.Series(rf).astype(float).reindex(r.index).fillna(0.0)).dropna()
    ex_mean, ex_std = ex.mean(), ex.std(ddof=1)
    std = r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cg = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {
        "sharpe": float(ex_mean / ex_std * math.sqrt(periods_per_year)) if ex_std > 0 else np.nan,
        "cagr": float(cg),
        "vol": float(std * math.sqrt(periods_per_year)),
        "mdd": float(dd),
        "n": int(len(r)),
    }


def control_recovers(returns: pd.Series, planted_sign: int,
                     season: list[int] = SUGARING_MONTHS) -> dict:
    """Positive control: the seasonality engine recovers the planted season's sign."""
    st = season_tstat(returns, season)
    return {"spread": st["spread"], "tstat": st["tstat"],
            "sign_ok": int(np.sign(st["spread"]) == planted_sign)}
