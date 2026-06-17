"""Strategy and analysis layer for Study 288 (Ramadan-Effect).

The Ramadan effect (Bialkowski, Etebari & Wisniewski 2012, JBF): equity returns
in Muslim-majority markets are claimed to be *higher* and *less volatile* during
the holy month of Ramadan, attributed to investor optimism, social cohesion, and
reduced trading activity.

We implement this as a calendar split -- each month is either a "Ramadan month"
(>= a chosen overlap with the hardcoded Ramadan window) or not -- and pin the
Ramadan-vs-rest return difference against:

  - A **HAC (Newey-West) t-stat** on the Ramadan-month excess return (the desk's
    >= 2 inference bar). Monthly returns have little serial correlation, but we
    use the HAC standard error to be conservative.
  - A **Welch t-test** comparing the mean return in Ramadan vs non-Ramadan months
    (unequal variance -- the whole *premise* is that Ramadan months are calmer).
  - A **permutation test**: shuffle the Ramadan labels many times and locate the
    real mean-difference inside the null distribution.
  - A **placebo tape** (the S&P 500): a non-Muslim-majority index where, under any
    real behavioural mechanism, no Ramadan effect should exist. A "significant"
    Ramadan effect on the placebo would expose the whole thing as data-snooping.

The tiny-n reckoning: with ~18 years of a single MENA-proxy ETF there are only
~18 Ramadan months. Monthly equity vol of ~5% means the minimum detectable
mean difference at |t| = 2 is large -- we report the power calculation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# HAC (Newey-West) t-stat on a single return series
# ---------------------------------------------------------------------------
def nw_tstat(x: np.ndarray, lags: int = 3) -> tuple[float, float]:
    """Newey-West HAC t-stat for the mean of ``x`` (H0: mean = 0).

    Returns ``(mean, tstat)``. ``lags`` is the Bartlett-kernel truncation.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float("nan"), float("nan")
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1)
        cov = (e[k:] @ e[:-k]) / n
        var += 2.0 * w * cov
    se = np.sqrt(max(var, 0.0) / n)
    if se == 0:
        return float(mu), float("nan")
    return float(mu), float(mu / se)


def summarize(returns: pd.Series, periods_per_year: int = 12, lags: int = 3) -> dict:
    """Compact stats for a monthly return series: mean, ann, vol, Sharpe, HAC t, hit-rate."""
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    n = r.size
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ann": float("nan"),
                "vol_ann": float("nan"), "sharpe": float("nan"),
                "tstat": float("nan"), "hit_rate": float("nan")}
    mean = float(r.mean())
    sd = float(r.std(ddof=1)) if n > 1 else float("nan")
    _, t = nw_tstat(r, lags=lags)
    return {
        "n": int(n),
        "mean": mean,
        "ann": float((1 + mean) ** periods_per_year - 1),
        "vol_ann": float(sd * np.sqrt(periods_per_year)) if np.isfinite(sd) else float("nan"),
        "sharpe": float(mean / sd) if (np.isfinite(sd) and sd > 0) else float("nan"),
        "tstat": float(t),
        "hit_rate": float((r > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Ramadan-vs-rest comparison on one tape
# ---------------------------------------------------------------------------
def ramadan_comparison(
    df: pd.DataFrame,
    ret_col: str,
    n_permutations: int = 10_000,
    lags: int = 3,
    seed: int = 288,
) -> dict:
    """Compare Ramadan-month vs non-Ramadan-month returns for one return column.

    Expects ``df`` with a boolean ``ramadan`` column and the named ``ret_col``.

    Returns a dict with:
      - ``n``, ``n_ram``, ``n_rest``
      - ``mean_ram``, ``mean_rest`` (monthly), ``diff`` (= mean_ram - mean_rest)
      - ``ann_ram``, ``ann_rest`` (annualised)
      - ``vol_ram``, ``vol_rest`` (annualised, the 'calmer in Ramadan' claim)
      - ``welch_t``, ``welch_p`` (Ramadan vs rest, unequal var)
      - ``nw_t_ram``, ``nw_t_diff`` (HAC t on Ramadan months; HAC t on the
        de-meaned 'Ramadan dummy' regression -- here the mean-difference t)
      - ``perm_p`` (two-sided permutation p on |diff|)
      - ``hit_ram``, ``hit_rest`` (fraction of up months)
    """
    sub = df[[ret_col, "ramadan"]].dropna()
    ram = sub.loc[sub["ramadan"], ret_col].to_numpy(dtype=float)
    rest = sub.loc[~sub["ramadan"], ret_col].to_numpy(dtype=float)
    n_ram, n_rest = ram.size, rest.size

    mean_ram = float(ram.mean()) if n_ram else float("nan")
    mean_rest = float(rest.mean()) if n_rest else float("nan")
    diff = mean_ram - mean_rest

    def _ann(m):
        return float((1 + m) ** 12 - 1) if np.isfinite(m) else float("nan")

    def _volann(a):
        return float(a.std(ddof=1) * np.sqrt(12)) if a.size > 1 else float("nan")

    if n_ram >= 2 and n_rest >= 2:
        welch_t, welch_p = scipy_stats.ttest_ind(ram, rest, equal_var=False)
    else:
        welch_t, welch_p = float("nan"), float("nan")

    _, nw_t_ram = nw_tstat(ram, lags=lags)

    # Mean-difference HAC t via a Ramadan-dummy regression (OLS slope = diff).
    all_r = sub[ret_col].to_numpy(dtype=float)
    dummy = sub["ramadan"].to_numpy(dtype=float)
    nw_t_diff = _dummy_nw_t(all_r, dummy, lags=lags)

    # Permutation test on |diff| (shuffle the Ramadan labels).
    rng = np.random.default_rng(seed)
    labels = sub["ramadan"].to_numpy()
    vals = sub[ret_col].to_numpy(dtype=float)
    obs = abs(diff)
    perm_diffs = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        sh = rng.permutation(labels)
        m_r = vals[sh].mean() if sh.any() else 0.0
        m_o = vals[~sh].mean() if (~sh).any() else 0.0
        perm_diffs[i] = abs(m_r - m_o)
    perm_p = float((perm_diffs >= obs).mean())

    return {
        "n": int(n_ram + n_rest),
        "n_ram": int(n_ram),
        "n_rest": int(n_rest),
        "mean_ram": mean_ram,
        "mean_rest": mean_rest,
        "diff": float(diff),
        "ann_ram": _ann(mean_ram),
        "ann_rest": _ann(mean_rest),
        "vol_ram": _volann(ram),
        "vol_rest": _volann(rest),
        "welch_t": float(welch_t),
        "welch_p": float(welch_p),
        "nw_t_ram": float(nw_t_ram),
        "nw_t_diff": float(nw_t_diff),
        "perm_p": perm_p,
        "perm_diffs": perm_diffs,
        "hit_ram": float((ram > 0).mean()) if n_ram else float("nan"),
        "hit_rest": float((rest > 0).mean()) if n_rest else float("nan"),
    }


def _dummy_nw_t(y: np.ndarray, x: np.ndarray, lags: int = 3) -> float:
    """HAC (Newey-West) t-stat on the slope of y = a + b*x + e (x = Ramadan dummy)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = y.size
    if n < 3 or x.std() == 0:
        return float("nan")
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    # Newey-West meat matrix
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1)
        u = X * resid[:, None]
        gamma = u[k:].T @ u[:-k]
        S += w * (gamma + gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    se_b = np.sqrt(max(cov[1, 1], 0.0))
    if se_b == 0:
        return float("nan")
    return float(beta[1] / se_b)


# ---------------------------------------------------------------------------
# Tradable strategy: long the MENA proxy only during Ramadan months
# ---------------------------------------------------------------------------
def ramadan_timing_backtest(
    df: pd.DataFrame,
    ret_col: str = "mena_ret",
    one_way_bps: float = 30.0,
) -> dict:
    """Long the proxy in Ramadan months, flat otherwise -- vs buy-and-hold.

    Applies a one-month execution lag is unnecessary (the calendar is known
    ahead), but entering/exiting the position around each Ramadan month incurs
    a round-trip cost. We charge ``one_way_bps`` x NAV on each position change.

    Returns annualised gross and net strategy returns, buy-and-hold, turnover,
    and the cost drag.
    """
    sub = df[[ret_col, "ramadan"]].dropna().copy()
    r = sub[ret_col].to_numpy(dtype=float)
    pos = sub["ramadan"].to_numpy(dtype=float)  # 1 in Ramadan months, else 0
    # Position changes (entering/leaving) -> round-trip cost on each switch.
    prev = np.concatenate([[0.0], pos[:-1]])
    switches = np.abs(pos - prev)
    cost = switches * (one_way_bps * 1e-4)
    gross = pos * r
    net = gross - cost

    n = len(r)
    n_yrs = n / 12.0
    bah_ann = float((1 + r).prod() ** (1 / n_yrs) - 1) if n else float("nan")
    gross_ann = float((1 + gross).prod() ** (1 / n_yrs) - 1) if n else float("nan")
    net_ann = float((1 + net).prod() ** (1 / n_yrs) - 1) if n else float("nan")
    turnover = float(switches.mean())  # avg fraction of months with a switch
    drag = float(cost.sum() / n_yrs)   # annual cost drag

    return {
        "n_months": int(n),
        "bah_ann": bah_ann,
        "strat_gross_ann": gross_ann,
        "strat_net_ann": net_ann,
        "turnover": turnover,
        "cost_drag_ann": drag,
        "months_invested": int(pos.sum()),
    }


# ---------------------------------------------------------------------------
# Power calculation
# ---------------------------------------------------------------------------
def min_detectable_diff(
    vol_mo: float, n_ram: int, n_rest: int, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Minimum detectable monthly mean-difference at given power (two-sample t)."""
    from scipy.stats import t as t_dist
    dof = n_ram + n_rest - 2
    t_crit = t_dist.ppf(1 - alpha / 2, dof)
    t_pow = t_dist.ppf(power, dof)
    return float((t_crit + t_pow) * vol_mo * np.sqrt(1 / n_ram + 1 / n_rest))


# ---------------------------------------------------------------------------
# One-stop summary dict for notebooks / results.md
# ---------------------------------------------------------------------------
def headline(
    df: pd.DataFrame,
    n_permutations: int = 10_000,
    seed: int = 288,
) -> dict:
    """Flat dict mirroring the R = dict(...) in build_notebooks.py and results.md.

    Runs the MENA-proxy (``mena_ret``) comparison and the S&P placebo
    (``spx_ret``) comparison plus the tradable backtest. Percentages rounded.
    """
    g = ramadan_comparison(df, "mena_ret", n_permutations=n_permutations, seed=seed)
    p = ramadan_comparison(df, "spx_ret", n_permutations=n_permutations, seed=seed)
    bt = ramadan_timing_backtest(df, "mena_ret", one_way_bps=30.0)
    return {
        "n": g["n"], "n_ram": g["n_ram"], "n_rest": g["n_rest"],
        # MENA proxy
        "mena_ann_ram": round(g["ann_ram"] * 100, 1),
        "mena_ann_rest": round(g["ann_rest"] * 100, 1),
        "mena_mean_ram_bps": round(g["mean_ram"] * 1e4, 1),
        "mena_mean_rest_bps": round(g["mean_rest"] * 1e4, 1),
        "mena_diff_bps": round(g["diff"] * 1e4, 1),
        "mena_vol_ram": round(g["vol_ram"] * 100, 1),
        "mena_vol_rest": round(g["vol_rest"] * 100, 1),
        "mena_welch_t": round(g["welch_t"], 3),
        "mena_welch_p": round(g["welch_p"], 4),
        "mena_nw_t_ram": round(g["nw_t_ram"], 3),
        "mena_nw_t_diff": round(g["nw_t_diff"], 3),
        "mena_perm_p": round(g["perm_p"], 4),
        "mena_hit_ram": round(g["hit_ram"] * 100, 1),
        "mena_hit_rest": round(g["hit_rest"] * 100, 1),
        # placebo (S&P)
        "spx_diff_bps": round(p["diff"] * 1e4, 1),
        "spx_welch_t": round(p["welch_t"], 3),
        "spx_nw_t_diff": round(p["nw_t_diff"], 3),
        "spx_perm_p": round(p["perm_p"], 4),
        # tradable
        "bah_ann": round(bt["bah_ann"] * 100, 1),
        "strat_gross_ann": round(bt["strat_gross_ann"] * 100, 1),
        "strat_net_ann": round(bt["strat_net_ann"] * 100, 1),
        "turnover": round(bt["turnover"] * 100, 1),
        "cost_drag_ann_bps": round(bt["cost_drag_ann"] * 1e4, 1),
    }
