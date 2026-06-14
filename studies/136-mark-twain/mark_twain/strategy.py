"""The analysis and its honest controls — Study 136 (Mark-Twain).

The claim: October is the most dangerous month for stocks (quoting Mark Twain's 1894 joke
in *Pudd'nhead Wilson*). Three concrete sub-claims underlie the reputation:

  (A) October mean return is *negative* or the worst of the twelve months.
  (B) October volatility is measurably higher than other months.
  (C) The effect persists even when you strip the three famous October crashes
      (1929, 1987, 2008) from the record.

This module tests all three with clean inference:

  - :func:`monthly_stats` — per-month mean, vol, worst-day, and HAC t-stat.
  - :func:`crash_strip_test` — repeats the test with the three crash months removed.
  - :func:`compare_october_vs_rest` — a Welch-t comparing October against the other
    eleven months — the honest "is October special?" test.
  - :func:`null_baseline` — the random-permutation (coin) control: shuffle month labels
    and ask how often a randomly-chosen month is as bad as October claims to be.

No look-ahead: calendar labels are assigned to the observation's own calendar month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The three famous October crashes — we strip these to test robustness.
CRASH_MONTHS = {
    "1929-10",  # Black Tuesday / the 1929 crash
    "1987-10",  # Black Monday (Oct 19 1987)
    "2008-10",  # GFC (Oct 2008 peak drawdown month)
}

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


# ---------------------------------------------------------------------------
# Core: per-month statistics
# ---------------------------------------------------------------------------
def monthly_stats(ret: pd.Series, exclude_ym: set[str] | None = None) -> pd.DataFrame:
    """Per-calendar-month statistics over the daily return series.

    Parameters
    ----------
    ret :
        Daily close-to-close simple returns indexed by date.
    exclude_ym :
        Optional set of ``"YYYY-MM"`` strings to drop before computing stats.
        Pass ``CRASH_MONTHS`` for the crash-stripped robustness test.

    Returns a DataFrame indexed 1..12 with columns:

    - ``mean_bps``  — average daily return in basis points.
    - ``vol_ann``   — annualised daily vol (std * sqrt(252)).
    - ``worst_day_pct`` — worst single-day return in percent.
    - ``n``         — number of trading days in the sample.
    - ``hac_t``     — Newey-West HAC t-stat on the daily mean (within-month).
    - ``month``     — 3-letter abbreviation.
    """
    r = pd.Series(ret).astype(float).dropna()
    if exclude_ym:
        ym = r.index.to_period("M").astype(str)
        r = r[~pd.Series(ym, index=r.index).isin(exclude_ym)]

    rows = []
    for m in range(1, 13):
        rm = r[r.index.month == m]
        n = len(rm)
        if n < 5:
            rows.append(
                {"month_num": m, "month": MONTH_NAMES[m], "mean_bps": np.nan,
                 "vol_ann": np.nan, "worst_day_pct": np.nan, "n": n, "hac_t": np.nan}
            )
            continue
        arr = rm.to_numpy()
        mean_bp = float(arr.mean()) * 1e4
        vol = float(arr.std(ddof=1)) * np.sqrt(252)
        worst = float(arr.min()) * 100.0
        hac_t = _hac_tstat(arr)
        rows.append(
            {"month_num": m, "month": MONTH_NAMES[m], "mean_bps": mean_bp,
             "vol_ann": vol, "worst_day_pct": worst, "n": n, "hac_t": hac_t}
        )
    df = pd.DataFrame(rows).set_index("month_num")
    return df


def _hac_tstat(arr: np.ndarray) -> float:
    """Newey-West HAC t-statistic for the sample mean."""
    n = arr.size
    if n < 6:
        return float("nan")
    mu = arr.mean()
    e = arr - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# The honest pairwise test: October vs the rest
# ---------------------------------------------------------------------------
def compare_october_vs_rest(
    ret: pd.Series, exclude_ym: set[str] | None = None
) -> dict:
    """Welch t-test: October daily returns vs the other eleven months combined.

    This is the primary falsifiable test: if October is genuinely bad, its mean
    return should be significantly below the rest, even on a robust test.

    Returns a dict with ``oct_mean_bps``, ``rest_mean_bps``, ``diff_bps``,
    ``welch_t``, ``n_oct``, ``n_rest``.
    """
    r = pd.Series(ret).astype(float).dropna()
    if exclude_ym:
        ym = r.index.to_period("M").astype(str)
        r = r[~pd.Series(ym, index=r.index).isin(exclude_ym)]

    oct_r = r[r.index.month == 10].to_numpy()
    rest_r = r[r.index.month != 10].to_numpy()
    n_o, n_r = len(oct_r), len(rest_r)
    m_o, m_r = oct_r.mean(), rest_r.mean()
    se = np.sqrt(oct_r.var(ddof=1) / n_o + rest_r.var(ddof=1) / n_r)
    return {
        "oct_mean_bps": float(m_o * 1e4),
        "rest_mean_bps": float(m_r * 1e4),
        "diff_bps": float((m_o - m_r) * 1e4),
        "welch_t": float((m_o - m_r) / se) if se > 0 else float("nan"),
        "n_oct": int(n_o),
        "n_rest": int(n_r),
    }


# ---------------------------------------------------------------------------
# Crash-stripped robustness: repeat both with the three events removed
# ---------------------------------------------------------------------------
def crash_strip_test(ret: pd.Series) -> dict:
    """The key robustness check: does the October story survive stripping the three famous crashes?

    Returns dicts for the full-sample and crash-stripped comparisons so callers
    can display both side by side.
    """
    full = compare_october_vs_rest(ret, exclude_ym=None)
    stripped = compare_october_vs_rest(ret, exclude_ym=CRASH_MONTHS)
    return {"full": full, "stripped": stripped}


# ---------------------------------------------------------------------------
# Null baseline — permutation control
# ---------------------------------------------------------------------------
def permutation_null(
    ret: pd.Series,
    n_perm: int = 5000,
    seed: int = 136,
) -> dict:
    """How often does a *randomly labelled* month look as bad as October?

    We shuffle the month labels (while keeping the return series intact) and ask
    what fraction of randomly-chosen months produce a diff <= October's observed diff.
    This is the honest sanity check: if October were truly anomalous, even randomly
    chosen months should not routinely match it.

    Returns ``pct_worse_than_oct`` (fraction of permutations where the random month's
    mean is at least as low as October's), and the October observed mean.
    """
    r = pd.Series(ret).astype(float).dropna()
    rng = np.random.default_rng(seed)
    # Observed October mean
    oct_mean = float(r[r.index.month == 10].mean())
    # Permutation: shuffle month assignment
    months = r.index.month.to_numpy()
    vals = r.to_numpy()
    count = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(months)
        perm_mean = float(vals[shuffled == 10].mean())
        if perm_mean <= oct_mean:
            count += 1
    return {
        "oct_mean_bps": float(oct_mean * 1e4),
        "pct_worse_than_oct": float(count / n_perm),
        "n_perm": n_perm,
    }


# ---------------------------------------------------------------------------
# Annualised summary helper
# ---------------------------------------------------------------------------
def period_summary(ret: pd.Series) -> dict:
    """Headline stats for the full series: CAGR, annualised vol, Sharpe."""
    r = pd.Series(ret).astype(float).dropna()
    mean = r.mean()
    std = r.std(ddof=1)
    cagr = (1.0 + r).prod() ** (252.0 / len(r)) - 1.0
    return {
        "n": int(len(r)),
        "cagr": float(cagr),
        "vol_ann": float(std * np.sqrt(252)),
        "sharpe": float(mean / std * np.sqrt(252)) if std > 0 else float("nan"),
        "start": str(r.index.min().date()),
        "end": str(r.index.max().date()),
    }
