"""Strategy and honest controls -- Study 290 (September-Effect).

The September effect (folklore): September is the worst calendar month for the
S&P 500 -- the only month with a negative long-run average return.  We test
this three ways:

  - **The monthly profile.** Mean return per calendar month, with September
    isolated against the pooled "all other months" baseline.
  - **A Newey-West (HAC) t-test** on the September dummy, plus a Welch t-test
    of September returns vs the rest.  Monthly equity returns have mild serial
    correlation, so we report the HAC t as the headline.  The REAL-tape bar for
    a "Signal = REAL" verdict is |HAC t| >= 2.
  - **A permutation test.** Shuffle the September label across all months
    10,000 times and see where the real September mean lands in the null
    distribution.

The honest trading counterpart is a **seasonal-avoidance rule**: hold the S&P
every month *except* September, when you sit in cash (0% return).  We compare
its annualised growth to passive buy-and-hold, charge a one-way transaction
cost on the two trades a year it requires (out at end-August, in at end-
September), and note that on a taxable book the realised gain triggers tax --
so the tiny gross edge is dwarfed by frictions.

**Survivorship**: none in the usual sense -- this is the S&P *index*, not a
stock basket, so there is no delisting bias.  But the index itself is
reconstituted (winners replace losers), so the September drag measured on the
modern S&P is already an *upper bound* on the drag a fixed-membership basket
would show.  This is named on the Signal axis.

**Price-only vs total return**: the Shiller SP500 column is price-only; the
September gap is a price-return statistic.  Adding back a roughly month-uniform
dividend yield makes the total-return September gap slightly *less* negative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

SEPTEMBER = 9
_MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# ---------------------------------------------------------------------------
# Newey-West (HAC) helper
# ---------------------------------------------------------------------------
def _hac_se_mean(x: np.ndarray) -> tuple[float, float, int]:
    """Newey-West HAC standard error of the sample mean of ``x``.

    Returns ``(mean, se, lags)``.  The bandwidth follows the standard
    ``floor(4 * (n/100)**(2/9))`` rule.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    mu = float(x.mean())
    e = x - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = float(np.sqrt(max(lrv, 0.0) / n))
    return mu, se, lags


def hac_tstat(x) -> float:
    """HAC t-stat that the mean of ``x`` differs from zero."""
    mu, se, _ = _hac_se_mean(np.asarray(x, dtype=float))
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Per-month profile
# ---------------------------------------------------------------------------
def monthly_profile(df: pd.DataFrame, ret_col: str = "ret") -> pd.DataFrame:
    """Mean / std / up-rate / HAC-t per calendar month.

    Returns a DataFrame indexed by month name (Jan..Dec) with columns
    ``mean_pct``, ``std_pct``, ``up_rate``, ``n``, ``hac_t``.
    """
    rows = []
    for m in range(1, 13):
        r = df.loc[df["month"] == m, ret_col].to_numpy(dtype=float)
        if r.size == 0:
            continue
        rows.append(
            {
                "month_num": m,
                "month": _MONTH_NAMES[m - 1],
                "mean_pct": round(float(r.mean()) * 100, 3),
                "std_pct": round(float(r.std(ddof=1)) * 100, 2),
                "up_rate": round(float((r > 0).mean()), 3),
                "n": int(r.size),
                "hac_t": round(hac_tstat(r), 3),
            }
        )
    out = pd.DataFrame(rows).set_index("month")
    return out


# ---------------------------------------------------------------------------
# September vs the rest -- the core inference
# ---------------------------------------------------------------------------
def september_stats(
    df: pd.DataFrame,
    ret_col: str = "ret",
    n_permutations: int = 10_000,
    seed: int = 290,
) -> dict:
    """September-vs-rest inference: means, gap, HAC t, Welch t, permutation p.

    The 'gap' is ``mean(September) - mean(all other months)``.  The folklore
    predicts a *negative* gap.  Reported tests:

      - ``hac_t_sep``    : HAC t-stat that September's own mean differs from 0.
      - ``hac_t_gap``    : HAC t-stat on the September dummy in the regression
                           ``ret ~ const + 1[month == Sep]`` (the headline).
      - ``welch_t``/``welch_p`` : Welch two-sample test, Sep vs the rest.
      - ``perm_p``       : permutation p (fraction of label-shuffles whose
                           September mean is <= the observed September mean;
                           a one-sided test for the folklore's negative drag).

    Up-rates are reported too: ``sep_up_rate`` vs ``all_up_rate`` (the honest
    base rate -- the S&P rises in most months regardless).
    """
    sep = df.loc[df["month"] == SEPTEMBER, ret_col].to_numpy(dtype=float)
    oth = df.loc[df["month"] != SEPTEMBER, ret_col].to_numpy(dtype=float)
    allr = df[ret_col].to_numpy(dtype=float)
    n = allr.size

    sep_mean = float(sep.mean())
    oth_mean = float(oth.mean())
    gap = sep_mean - oth_mean

    hac_t_sep = hac_tstat(sep)

    # HAC t on the September dummy via OLS with Newey-West covariance.
    is_sep = (df["month"].to_numpy() == SEPTEMBER).astype(float)
    X = np.column_stack([np.ones(n), is_sep])
    y = allr
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    XtX_inv = np.linalg.inv(X.T @ X)
    u = X * resid[:, None]
    S = u.T @ u
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        G = u[k:].T @ u[:-k]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se_beta = float(np.sqrt(cov[1, 1]))
    hac_t_gap = float(beta[1] / se_beta) if se_beta > 0 else float("nan")

    if sep.size >= 2 and oth.size >= 2:
        welch_t, welch_p = scipy_stats.ttest_ind(sep, oth, equal_var=False)
    else:
        welch_t, welch_p = float("nan"), float("nan")

    # Permutation test: shuffle which months carry the 'September' label.
    rng = np.random.default_rng(seed)
    n_sep = sep.size
    perm_means = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        idx = rng.choice(n, size=n_sep, replace=False)
        perm_means[i] = allr[idx].mean()
    perm_p = float((perm_means <= sep_mean).mean())  # one-sided: drag

    return {
        "n_total": int(n),
        "n_sep": int(n_sep),
        "n_other": int(oth.size),
        "sep_mean_pct": round(sep_mean * 100, 3),
        "other_mean_pct": round(oth_mean * 100, 3),
        "gap_pct": round(gap * 100, 3),
        "sep_up_rate": round(float((sep > 0).mean()), 3),
        "all_up_rate": round(float((allr > 0).mean()), 3),
        "hac_t_sep": round(float(hac_t_sep), 3),
        "hac_t_gap": round(float(hac_t_gap), 3),
        "welch_t": round(float(welch_t), 3),
        "welch_p": round(float(welch_p), 4),
        "perm_p": round(float(perm_p), 4),
        "perm_means": perm_means,  # full distribution, for plots
    }


# ---------------------------------------------------------------------------
# Sub-period persistence
# ---------------------------------------------------------------------------
def subperiod_table(df: pd.DataFrame, bounds=((1950, 1985), (1986, 2005), (2006, 2025))) -> pd.DataFrame:
    """September mean / HAC-t over coarse sub-periods (persistence check)."""
    rows = []
    for lo, hi in bounds:
        sub = df[(df["year"] >= lo) & (df["year"] <= hi)]
        sep = sub.loc[sub["month"] == SEPTEMBER, "ret"].to_numpy(dtype=float)
        if sep.size < 2:
            continue
        rows.append(
            {
                "period": f"{lo}-{hi}",
                "sep_mean_pct": round(float(sep.mean()) * 100, 3),
                "sep_hac_t": round(hac_tstat(sep), 3),
                "n_sep": int(sep.size),
            }
        )
    return pd.DataFrame(rows).set_index("period")


# ---------------------------------------------------------------------------
# The honest trading rule: avoid September
# ---------------------------------------------------------------------------
def avoid_september_backtest(
    df: pd.DataFrame,
    one_way_bps: float = 5.0,
    ret_col: str = "ret",
) -> dict:
    """Hold the S&P every month except September (sit in cash); compare to buy-hold.

    Two trades a year (exit at end-August, re-enter at end-September) cost
    ``2 * one_way_bps`` per year applied to NAV.  Returns annualised growth
    rates (gross and net of costs) and the cost-charged net edge.

    Note: cash earns 0% here (a conservative, low-rate assumption); in a
    high-rate regime cash would earn the risk-free rate and the rule would look
    *better*, but it still loses on a taxable book where dodging one month a
    year realises a taxable gain.
    """
    work = df.copy()
    n_years = int(work["year"].nunique())

    bh = work[ret_col].to_numpy(dtype=float)
    bh_ann = float((1.0 + bh).prod() ** (1.0 / n_years) - 1.0)

    flat = np.where(work["month"].to_numpy() == SEPTEMBER, 0.0, bh)
    strat_gross_ann = float((1.0 + flat).prod() ** (1.0 / n_years) - 1.0)

    # Costs: 2 one-way trades per year on the dodge.
    cost_per_year = 2.0 * one_way_bps * 1e-4
    strat_net_ann = strat_gross_ann - cost_per_year

    return {
        "n_years": n_years,
        "buyhold_ann_pct": round(bh_ann * 100, 3),
        "strat_gross_ann_pct": round(strat_gross_ann * 100, 3),
        "strat_net_ann_pct": round(strat_net_ann * 100, 3),
        "gross_edge_pp": round((strat_gross_ann - bh_ann) * 100, 3),
        "net_edge_pp": round((strat_net_ann - bh_ann) * 100, 3),
        "one_way_bps": one_way_bps,
    }


# ---------------------------------------------------------------------------
# Compact summary dict (mirror of docs/results.md and the notebook R dict)
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, n_permutations: int = 10_000, seed: int = 290) -> dict:
    """One-stop summary for the real monthly tape."""
    s = september_stats(df, n_permutations=n_permutations, seed=seed)
    bt = avoid_september_backtest(df)
    s.pop("perm_means", None)
    return {**s, **{f"bt_{k}": v for k, v in bt.items()}}
