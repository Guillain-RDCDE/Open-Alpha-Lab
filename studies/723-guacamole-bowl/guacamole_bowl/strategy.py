"""Guacamole-bowl: monthly seasonality tests, a placebo across all month-pairs, and a Jan-Feb timer.

The folklore: America's Super-Bowl guacamole binge (early February) should print a January–February
seasonal in the avocado / produce trade — buy ahead of the game, ride the surge. We test the strongest
tradable version on ``PEP`` (PepsiCo/Frito-Lay: Tostitos + the branded dips, the Super-Bowl chip-and-dip
complex; a *labelled proxy* because the pure-play avocado name CVGW is unavailable on the current feed):

  1. **Per-month one-sample HAC t-stats** — is any calendar month reliably positive?
  2. **The guac-window spread** — is Jan–Feb significantly stronger than the rest of the year?
  3. **A placebo across all C(12,2) = 66 month-pairs** — where does the Super-Bowl window rank? A real
     seasonal should sit in the extreme tail; folklore sits in the crowd.
  4. **A block-bootstrap CI** on the window spread.
  5. **A Jan–Feb timer** (long the proxy in the window, T-bill otherwise) raced against buy-and-hold SPY.
  6. **Newey-West alpha** of the proxy vs SPY — is there any harvestable edge, or just beta?

Conventions, stated up front:

  * **HAC (Newey-West) t-stats**, not naive ones — we expose both for honesty.
  * **The cash leg earns the T-bill.** :func:`seasonal_timer` credits a monthly cash return when the
    rule is flat, so the timing race is done on *excess-of-cash* Sharpe, like-for-like.
  * **Calendar-known rule, no execution lag.** The month is known in advance, so the position is set at
    the start of each month with no ``shift`` — a Super-Bowl-calendar rule needs no signal-to-trade lag.
  * **Costs one-way × NAV**, both legs charged; the proxy is a liquid large-cap so shorts (none here)
    would pay borrow, but the timer is long-or-cash only.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

MONTHS = 12
GUAC_MONTHS = [1, 2]  # Jan build-up + Feb game month — the "guacamole surge" window


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
    adjustment (Bonferroni for 12 months: 0.05/12 ≈ 0.004, so effectively |t| ≈ 3 at n ≈ 33).
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


def window_spread_tstat(series: pd.Series, window: list[int] = GUAC_MONTHS) -> dict:
    """Welch two-sample t-stat: window months (Jan–Feb) vs every other month.

    Returns ``window_mean``, ``rest_mean``, ``spread``, ``tstat``, ``n_window``, ``n_rest``.
    Hypothesis: the Super-Bowl window earns more than the rest of the year. A robust seasonal needs
    a spread that is positive with |t| ≥ 2. (Here it lands negative — the surge window under-performs.)
    """
    s = pd.Series(series).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    win = s[s.index.month.isin(window)].dropna().to_numpy()
    rest = s[~s.index.month.isin(window)].dropna().to_numpy()
    if len(win) < 2 or len(rest) < 2:
        return {k: np.nan for k in ("window_mean", "rest_mean", "spread", "tstat", "n_window", "n_rest")}
    mu_w, mu_r = win.mean(), rest.mean()
    var_w, var_r = win.var(ddof=1), rest.var(ddof=1)
    n_w, n_r = len(win), len(rest)
    se = np.sqrt(var_w / n_w + var_r / n_r)
    return {
        "window_mean": float(mu_w),
        "rest_mean": float(mu_r),
        "spread": float(mu_w - mu_r),
        "tstat": float((mu_w - mu_r) / se) if se > 0 else np.nan,
        "n_window": int(n_w),
        "n_rest": int(n_r),
    }


def placebo_pairs(series: pd.Series, thesis: tuple[int, int] = (1, 2)) -> dict:
    """Compute the (month-pair vs rest) spread for all C(12,2) = 66 pairs; rank the thesis window.

    The placebo: if the Super-Bowl pair (Jan, Feb) carried a real seasonal it would sit in the extreme
    tail of the 66 spreads. Folklore sits in the crowd. Returns the thesis spread/t, its rank (1 =
    lowest spread), its percentile, the distribution mean/std of the 66 spreads, and the thesis z-score.
    """
    s = pd.Series(series).astype(float)
    s.index = pd.DatetimeIndex(s.index)

    def spread_of(pair):
        win = s[s.index.month.isin(pair)].dropna().to_numpy()
        rest = s[~s.index.month.isin(pair)].dropna().to_numpy()
        if len(win) < 2 or len(rest) < 2:
            return np.nan, np.nan
        se = np.sqrt(win.var(ddof=1) / len(win) + rest.var(ddof=1) / len(rest))
        sp = win.mean() - rest.mean()
        return float(sp), float(sp / se) if se > 0 else np.nan

    pairs = list(itertools.combinations(range(1, 13), 2))
    spreads = {p: spread_of(p) for p in pairs}
    sp_vals = np.array([v[0] for v in spreads.values()])
    th_sp, th_t = spreads[thesis]
    rank = int(np.sum(sp_vals <= th_sp))  # 1..66, 1 = lowest
    mu, sd = float(np.nanmean(sp_vals)), float(np.nanstd(sp_vals, ddof=1))
    most_pos = sorted(spreads.items(), key=lambda kv: -kv[1][0])[:3]
    return {
        "thesis": thesis,
        "thesis_spread": th_sp,
        "thesis_t": th_t,
        "rank": rank,
        "n_pairs": len(pairs),
        "pct": rank / len(pairs),
        "dist_mean": mu,
        "dist_std": sd,
        "z": (th_sp - mu) / sd if sd > 0 else np.nan,
        "most_positive": [(p, round(v[0] * 100, 2)) for p, v in most_pos],
    }


def spread_bootstrap_ci(
    series: pd.Series, window: list[int] = GUAC_MONTHS,
    n_boot: int = 2000, block: int = 12, seed: int = 723, alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the (window − rest) monthly-mean spread.

    Resamples 12-month blocks (one calendar year) to respect the annual seasonal structure, recomputes
    the window-minus-rest spread on each resample, and returns the percentile CI. ``[lo, hi]``
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
        win = vv[np.isin(mv, window)]
        rest = vv[~np.isin(mv, window)]
        if len(win) < 1 or len(rest) < 1:
            return np.nan
        return win.mean() - rest.mean()

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
    asset: pd.Series,
    tbill: pd.Series | None = None,
    window: list[int] = GUAC_MONTHS,
) -> pd.Series:
    """Long the asset in the guac window (Jan–Feb), T-bill otherwise.

    Calendar-known rule → no execution lag. ``tbill`` (a monthly cash-return series) is credited when
    the position is flat (months outside the window). ``tbill=None`` leaves cash at 0. Returns a
    monthly return series aligned to ``asset``.
    """
    r = pd.Series(asset).astype(float)
    r.index = pd.DatetimeIndex(r.index)
    cash = pd.Series(0.0, index=r.index) if tbill is None else pd.Series(tbill).astype(float).reindex(r.index).fillna(0.0)
    position = pd.Series(0.0, index=r.index)
    position[r.index.month.isin(window)] = 1.0
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
    """Subtract transaction cost spread evenly across the months, one-way × NAV.

    ``cost_bps_one_way`` is one-way cost in basis points × NAV; the Jan–Feb timer enters (Jan) and
    exits (end of Feb) once a year, so ``n_trades_per_year`` counts the one-way legs. We deduct the
    annual cost budget spread across the 12 months.
    """
    r = pd.Series(returns).astype(float).copy()
    monthly_cost = (n_trades_per_year * cost_bps_one_way / 1e4) / MONTHS
    return (r - monthly_cost).rename("net")


def newey_west_alpha_t(proxy_ret: pd.Series, bench_ret: pd.Series, lags: int = 6) -> dict:
    """Newey-West (HAC) *t* of the monthly alpha from r_proxy = a + b*r_bench + e.

    Returns the OLS alpha (monthly + annualised), beta, the HAC standard error of alpha and its *t*.
    The Signal-axis statistic for a proxy: is there alpha vs the market the proxy is exposed to?
    ``REAL`` needs a HAC *t* ≥ 2 in the proxy's favour.
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
        "n": n,
    }
