"""Strategy and honest controls — Study 279 (Geomagnetic-Storms).

The Krivelyova-Robotti (2003) claim: stock returns following geomagnetically
*stormy* periods are significantly *lower* than returns following *calm*
periods, because geomagnetic storms trigger mild depression / risk-aversion in
a fraction of the population (a "bad-mood" misattribution channel, à la the
weather-and-mood literature).

We implement this as a monthly regime test and pin it against:

  - The **storm-minus-calm mean-return gap**, with a **Newey-West HAC** t-stat
    (monthly returns have mild serial correlation; the HAC SE is the honest one).
  - A **two-sample Welch t-test** as a cross-check.
  - A **permutation test**: shuffle the storm/calm labels and see where the real
    gap lands in the null distribution.
  - A **tradable long/short** overlay (long after calm months, short after stormy
    months) with an explicit **one-month execution lag** and **one-way trading
    costs × NAV** (shorts pay borrow), reported gross and net.

The honest reckoning: monthly equity vol is ~4.3%/month, and with ~250 stormy
and ~250 calm months over nine decades, the storm-calm gap must clear a
Newey-West |t| ≥ 2 *on the real tape* to earn a REAL signal stamp — literature
support alone is only WEAK.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Newey-West HAC machinery
# ---------------------------------------------------------------------------
def nw_tstat(x: np.ndarray, lags: int | None = None) -> tuple[float, float, float]:
    """Newey-West HAC t-stat for the mean of a 1-D series.

    Returns ``(mean, se, tstat)``.  ``lags=None`` uses the standard
    ``floor(4 (n/100)^(2/9))`` automatic bandwidth (Newey-West 1994).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    mu = float(x.mean())
    e = x - mu
    if lags is None:
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    t = float(mu / se) if se > 0 else float("nan")
    return mu, float(se), t


# ---------------------------------------------------------------------------
# Storm-vs-calm regime statistics
# ---------------------------------------------------------------------------
def storm_calm_stats(
    df: pd.DataFrame,
    ret_col: str = "ret",
    regime_col: str = "regime",
    n_permutations: int = 10_000,
    seed: int = 279,
) -> dict:
    """Compare returns in stormy vs calm months, with HAC, Welch, and permutation.

    The 'gap' is ``mean(calm) - mean(stormy)`` — positive if storms depress
    returns (the Krivelyova-Robotti direction).  The HAC t-stat is computed on a
    *signed pooled* series: calm-month returns enter with +1, stormy-month returns
    with -1 (de-meaned within each leg is NOT done — we test the raw difference in
    means via a pooled regression-style construction below).

    Returns a dict with means, the gap, a Newey-West HAC t-stat on the gap, a
    Welch t-test, and a permutation p-value.
    """
    df = df.copy()
    r = df[ret_col].astype(float)
    reg = df[regime_col].astype(str)

    calm = r[reg == "calm"].to_numpy()
    stormy = r[reg == "stormy"].to_numpy()
    normal = r[reg == "normal"].to_numpy()

    n_calm, n_stormy = calm.size, stormy.size
    mean_calm = float(calm.mean()) if n_calm else float("nan")
    mean_stormy = float(stormy.mean()) if n_stormy else float("nan")
    mean_normal = float(normal.mean()) if normal.size else float("nan")
    gap = mean_calm - mean_stormy  # >0 if storms depress returns

    # HAC t-stat on the gap: build a calm/stormy-only ordered series and regress
    # ret on a +1(calm)/-1(stormy) dummy.  The OLS slope = (mean_calm-mean_stormy)/2
    # times 2 ... we instead compute the HAC SE of the difference directly via a
    # pooled, time-ordered "contribution" series so serial correlation is captured.
    sub = df[reg.isin(["calm", "stormy"])].sort_index()
    y = sub[ret_col].astype(float).to_numpy()
    d = np.where(sub[regime_col].to_numpy() == "calm", 1.0, -1.0)
    # Centre the dummy so the slope on d estimates half the gap; OLS with HAC.
    dc = d - d.mean()
    denom = float(dc @ dc)
    if denom > 0:
        beta = float(dc @ (y - y.mean())) / denom  # slope per unit dummy
        resid = (y - y.mean()) - beta * dc
        n = y.size
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
        # HAC variance of beta: (sum (dc_t resid_t))-based sandwich
        u = dc * resid
        lrv = float(u @ u) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(u[k:] @ u[:-k]) / n
        var_beta = n * lrv / (denom ** 2)
        se_beta = np.sqrt(max(var_beta, 0.0))
        # gap = mean_calm - mean_stormy = 2*beta (dummy spans +1..-1, range 2)
        hac_t = float(beta / se_beta) if se_beta > 0 else float("nan")
    else:
        hac_t = float("nan")

    # Welch two-sample t-test (cross-check; assumes independence).
    if n_calm >= 2 and n_stormy >= 2:
        from scipy import stats as scipy_stats
        welch_t, welch_p = scipy_stats.ttest_ind(calm, stormy, equal_var=False)
        welch_t, welch_p = float(welch_t), float(welch_p)
    else:
        welch_t, welch_p = float("nan"), float("nan")

    # Permutation test: shuffle calm/stormy labels, recompute the gap.
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([calm, stormy])
    labels = np.array([1] * n_calm + [0] * n_stormy)
    perm_gaps = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        pl = rng.permutation(labels)
        perm_gaps[i] = pooled[pl == 1].mean() - pooled[pl == 0].mean()
    perm_p = float((perm_gaps >= gap).mean())

    return {
        "n_calm": int(n_calm),
        "n_stormy": int(n_stormy),
        "n_normal": int(normal.size),
        "mean_calm": round(mean_calm, 6),
        "mean_stormy": round(mean_stormy, 6),
        "mean_normal": round(mean_normal, 6),
        "gap": round(float(gap), 6),
        "hac_t": round(float(hac_t), 3),
        "welch_t": round(float(welch_t), 3),
        "welch_p": round(float(welch_p), 4),
        "perm_p": round(float(perm_p), 4),
        "perm_gaps": perm_gaps,  # full permutation distribution, for plots
    }


# ---------------------------------------------------------------------------
# Tradable long/short overlay — execution lag + costs
# ---------------------------------------------------------------------------
def long_short_storm(
    df: pd.DataFrame,
    ret_col: str = "ret",
    regime_col: str = "regime",
    one_way_bps: float = 10.0,
    borrow_bps_month: float = 8.0,
) -> pd.DataFrame:
    """Tradable overlay: position on month t from month t-1's storm regime (lagged).

    Rule (Krivelyova-Robotti, tradable form):
      * prior month 'calm'   → +1 (long the market this month)
      * prior month 'stormy' → -1 (short the market this month)
      * prior month 'normal' → 0  (flat)

    The **one-month execution lag** (``regime.shift(1)``) makes the rule
    implementable: you only know last month's geomagnetic activity when you set
    this month's position.  Costs: ``one_way_bps`` per side on every position
    change (turnover × one-way × NAV), plus ``borrow_bps_month`` while short.

    Returns a DataFrame with columns ``pos`` (-1/0/+1), ``gross`` (pre-cost
    strategy return), ``cost`` (per-month cost drag), ``net`` (post-cost),
    ``mkt`` (the buy-and-hold benchmark return).
    """
    df = df.sort_index().copy()
    reg = df[regime_col].astype(str)
    pos_map = {"calm": 1.0, "stormy": -1.0, "normal": 0.0}
    pos = reg.map(pos_map).shift(1).fillna(0.0)  # one-month execution lag

    mkt = df[ret_col].astype(float)
    gross = pos * mkt

    # Turnover cost: |Δposition| × one-way bps (round-trip captured over time).
    turn = pos.diff().abs().fillna(pos.abs())
    cost = turn * one_way_bps * 1e-4
    # Borrow cost while short.
    cost = cost + (pos < 0).astype(float) * borrow_bps_month * 1e-4

    net = gross - cost

    out = pd.DataFrame(
        {"pos": pos, "gross": gross, "cost": cost, "net": net, "mkt": mkt}
    )
    return out


# ---------------------------------------------------------------------------
# Summary statistics for a monthly return series
# ---------------------------------------------------------------------------
def summarize(series: pd.Series, periods: int = 12) -> dict:
    """Headline statistics for a monthly return series, with Newey-West HAC t-stat."""
    r = pd.Series(series).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: float("nan") for k in
                ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}
    mu, se, t = nw_tstat(r.to_numpy())
    std = float(r.std(ddof=1))
    sr = mu / std if std > 0 else float("nan")
    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "mean": float(mu),
        "vol": float(std),
        "sharpe": float(sr),
        "sharpe_ann": float(sr * np.sqrt(periods)) if np.isfinite(sr) else float("nan"),
        "mean_ann": float(mu * periods),
        "vol_ann": float(std * np.sqrt(periods)),
        "tstat": float(t),
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": max_dd,
        "n": n,
    }


# ---------------------------------------------------------------------------
# One-stop summary dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize_study(
    df: pd.DataFrame,
    n_permutations: int = 10_000,
    seed: int = 279,
) -> dict:
    """Flat dict mirroring the R = dict(...) in build_notebooks.py and results.md."""
    s = storm_calm_stats(df, n_permutations=n_permutations, seed=seed)
    ls = long_short_storm(df)
    gross_stats = summarize(ls["gross"])
    net_stats = summarize(ls["net"])
    mkt_stats = summarize(df["ret"])

    return {
        "n": int(len(df)),
        "n_calm": s["n_calm"],
        "n_stormy": s["n_stormy"],
        "n_normal": s["n_normal"],
        "mean_calm_pct": round(s["mean_calm"] * 100, 3),
        "mean_stormy_pct": round(s["mean_stormy"] * 100, 3),
        "mean_normal_pct": round(s["mean_normal"] * 100, 3),
        "gap_pct": round(s["gap"] * 100, 3),
        "gap_ann_pct": round(s["gap"] * 12 * 100, 2),
        "hac_t": s["hac_t"],
        "welch_t": s["welch_t"],
        "welch_p": s["welch_p"],
        "perm_p": s["perm_p"],
        "ls_gross_ann_pct": round(gross_stats["mean_ann"] * 100, 2),
        "ls_gross_t": round(gross_stats["tstat"], 3),
        "ls_net_ann_pct": round(net_stats["mean_ann"] * 100, 2),
        "ls_net_t": round(net_stats["tstat"], 3),
        "mkt_ann_pct": round(mkt_stats["mean_ann"] * 100, 2),
    }
