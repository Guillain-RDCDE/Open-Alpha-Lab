"""Chicken-wing-index: Super-Bowl seasonality on Wingstop (WING) and a calendar timer.

The claim: Americans eat ~1.4 billion wings on Super Bowl Sunday, so wing demand spikes into the
early-February game — and Wingstop (WING), the pure-play wing chain, should rally into it. We test the
strongest tradable version: (1) per-month one-sample HAC t-stats on WING monthly returns; (2) is the
run-up window (January) significantly stronger than every other month? (3) a block-bootstrap CI on the
window-minus-rest spread; (4) a 12-month **placebo** — where does January rank among all single-month
bets? (5) is January's pop just the generic turn-of-year effect, or WING-specific alpha over SPY?
(6) does a long-January calendar timer beat buy-and-hold?

Conventions, stated up front:

  * **HAC (Newey-West) t-stats** are reported alongside naive ones — but with only ~11 Januaries the HAC
    long-run variance is estimated on a handful of points and the HAC t is **not trustworthy** at this
    sample size; we lean on the naive t, the Welch window test, the bootstrap CI and the placebo instead.
  * **Data-snooping is the whole risk.** A single month picked from twelve is a 12-way search; the
    Bonferroni threshold for α = 0.05/12 ≈ 0.004 is roughly |t| ≈ 3. One month clearing a naive |t| = 2
    is exactly what selection manufactures.
  * **The cash leg earns the T-bill.** :func:`superbowl_timer` credits a monthly cash return when the rule
    is flat, so the timing race is done on *excess-of-cash* Sharpe, like-for-like.
  * **Calendar-known rule, no execution lag.** The Super Bowl date is known years in advance, so the
    position is set at the start of the window with no ``shift`` — a calendar rule needs no signal lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12
SUPERBOWL_MONTHS = [1]   # January: run-up to the early-Feb game → the bullish leg
GAME_MONTH = [2]         # February: the Super Bowl itself → the sell-the-news check


def _hac_se(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (Bartlett-kernel) standard error of the sample mean of ``x``.

    NB: with the tiny per-month samples here (~11 Januaries) the HAC long-run variance is unreliable;
    treat the HAC t as indicative only and defer to the naive t and the resampling tests.
    """
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
    ``tstat_hac`` (Newey-West). A robust seasonality claim needs |t| ≥ 2 *after* multiple-testing
    adjustment (Bonferroni for 12 months: 0.05/12 ≈ 0.004, so effectively |t| ≈ 3).
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


def superbowl_window_test(series: pd.Series, window: list[int] = SUPERBOWL_MONTHS) -> dict:
    """Welch two-sample t-stat comparing the Super-Bowl window vs every other month.

    Returns ``window_mean``, ``rest_mean``, ``spread``, ``tstat``, ``n_window``, ``n_rest``. This is the
    statistic that actually encodes the thesis — "the wing window beats the rest of the year" — as opposed
    to cherry-picking a single significant month out of twelve. Robust result needs |t| ≥ 2.
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


def spread_bootstrap_ci(
    series: pd.Series,
    window: list[int] = SUPERBOWL_MONTHS,
    n_boot: int = 2000,
    block: int = 12,
    seed: int = 726,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the (window-minus-rest) monthly-mean spread.

    Resamples 12-month blocks (one calendar year) to respect the annual seasonal structure, recomputes the
    window-minus-rest spread on each resample, and returns the percentile CI. ``[lo, hi]`` straddling 0
    means the spread is indistinguishable from noise.
    """
    s = pd.Series(series).astype(float).dropna()
    s.index = pd.DatetimeIndex(s.index)
    months = s.index.month.to_numpy()
    vals = s.to_numpy()
    n = len(vals)
    if n < block * 2:
        return {"point": np.nan, "lo": np.nan, "hi": np.nan, "n_boot": 0}
    rng = np.random.default_rng(seed)
    win_set = set(window)

    def spread_of(idx):
        mv, vv = months[idx], vals[idx]
        in_win = np.isin(mv, list(win_set))
        w, r = vv[in_win], vv[~in_win]
        if len(w) < 1 or len(r) < 1:
            return np.nan
        return w.mean() - r.mean()

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


def placebo_months(
    wing: pd.Series, tbill: pd.Series | None = None
) -> pd.DataFrame:
    """The 12-month placebo: rank every single-month long-timer by excess Sharpe.

    For each calendar month, build the "long WING that month, cash otherwise" timer and score it (mean,
    naive t of that month's returns, and excess-of-cash Sharpe of the timer). If January's rank is not #1
    — or if off-thesis months score as high — the January "signal" is a data-snooping artefact, not a
    Super-Bowl effect. Returns a DataFrame indexed 1..12, sorted by Sharpe descending.
    """
    r = pd.Series(wing).astype(float)
    r.index = pd.DatetimeIndex(r.index)
    cash = (
        pd.Series(0.0, index=r.index)
        if tbill is None
        else pd.Series(tbill).astype(float).reindex(r.index).fillna(0.0)
    )
    rows = {}
    for m in range(1, 13):
        mask = r.index.month == m
        vals = r[mask].to_numpy()
        n = len(vals)
        mu = vals.mean() if n else np.nan
        t = mu / (vals.std(ddof=1) / np.sqrt(n)) if n > 1 and vals.std(ddof=1) > 0 else np.nan
        pos = pd.Series(0.0, index=r.index)
        pos[mask] = 1.0
        timer = pos * r + (pos == 0).astype(float) * cash
        ex = timer - cash
        sh = ex.mean() / ex.std(ddof=1) * np.sqrt(MONTHS) if ex.std(ddof=1) > 0 else np.nan
        rows[m] = {"mean": float(mu), "tstat": float(t), "sharpe": float(sh), "n": int(n)}
    df = pd.DataFrame(rows).T.rename_axis("month")
    return df.sort_values("sharpe", ascending=False)


def window_alpha_vs_market(
    wing: pd.Series, spy: pd.Series, window: list[int] = SUPERBOWL_MONTHS
) -> dict:
    """Is the window's WING pop just the market's own January, or WING-specific alpha?

    OLS of the window-month WING returns on the *same-month* SPY returns: ``r_wing = a + b·r_spy + e``.
    Returns the alpha (monthly), beta, its naive t and the mean window excess of SPY. A positive,
    significant alpha would argue the window is WING-specific; an alpha that leans on ~11 points is fragile
    by construction and we say so. (Small n → naive t only; HAC is meaningless at this size.)
    """
    w = pd.Series(wing).astype(float)
    m = pd.Series(spy).astype(float)
    w.index, m.index = pd.DatetimeIndex(w.index), pd.DatetimeIndex(m.index)
    j = pd.concat([w, m], axis=1, keys=["w", "s"]).dropna()
    j = j[j.index.month.isin(window)]
    n = len(j)
    if n < 3:
        return {k: np.nan for k in ("alpha_m", "beta", "t_alpha", "mean_excess", "n")}
    X = np.column_stack([np.ones(n), j["s"].to_numpy()])
    y = j["w"].to_numpy()
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    dof = n - 2
    sigma2 = float(resid @ resid) / dof if dof > 0 else np.nan
    se_alpha = np.sqrt(sigma2 * XtX_inv[0, 0]) if np.isfinite(sigma2) else np.nan
    excess = (j["w"] - j["s"]).mean()
    return {
        "alpha_m": float(beta[0]),
        "beta": float(beta[1]),
        "t_alpha": float(beta[0] / se_alpha) if se_alpha and se_alpha > 0 else np.nan,
        "mean_excess": float(excess),
        "n": int(n),
    }


def superbowl_timer(
    wing: pd.Series,
    tbill: pd.Series | None = None,
    window: list[int] = SUPERBOWL_MONTHS,
) -> pd.Series:
    """Long WING in the Super-Bowl run-up window (January), T-bill otherwise.

    Calendar-known rule → no execution lag. ``tbill`` (a monthly cash-return series) is credited when the
    position is flat (all months outside the window). ``tbill=None`` leaves cash at 0. Returns a monthly
    return series aligned to ``wing``.
    """
    r = pd.Series(wing).astype(float)
    r.index = pd.DatetimeIndex(r.index)
    cash = (
        pd.Series(0.0, index=r.index)
        if tbill is None
        else pd.Series(tbill).astype(float).reindex(r.index).fillna(0.0)
    )
    position = pd.Series(0.0, index=r.index)
    position[r.index.month.isin(window)] = 1.0
    return (position * r + (position == 0).astype(float) * cash).rename("superbowl_timer")


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

    ``cost_bps_one_way`` is one-way cost in basis points × NAV; the long-January timer enters at the start
    of January and exits at the end (a round trip = two one-way legs per year). We deduct the annual cost
    budget (``n_trades_per_year`` one-way legs × ``cost_bps_one_way``) spread across the 12 months.
    """
    r = pd.Series(returns).astype(float).copy()
    monthly_cost = (n_trades_per_year * cost_bps_one_way / 1e4) / MONTHS
    return (r - monthly_cost).rename("net")
