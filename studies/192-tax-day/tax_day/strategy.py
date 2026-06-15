"""The strategy and its honest controls — Study 192 (Tax-Day).

The folk claim: the US equity market shows systematically elevated returns in the
~10 trading days *before* April 15 (IRA contribution flows, refund money entering
the market) and/or distinctive returns in the days *after* the deadline.  Two
sub-hypotheses are tested:

H1 — pre-tax window (10 trading days before April 15) outperforms the rest.
H2 — post-tax window (5 trading days on/after April 15) is distinct from the rest.

Controls:
- **Unconditional baseline** — all non-window trading days, as the natural comparator.
- **Month-of-April** — the entire April to separate any broad April seasonal from the
  specific deadline effect.
- **Placebo window** — same-sized windows around October 15 (no IRS deadline),
  to confirm the windows are not just picking up any 2-week random draw.
- **Multiple-comparisons reckoning** — two hypotheses (pre and post); we apply a
  Bonferroni correction (k=2) and report both raw and corrected p-values.

No look-ahead: the date label (pre_tax / post_tax) is known before the open; the
return is the close-to-close log-return of that day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal: Newey-West HAC t-statistic
# ---------------------------------------------------------------------------
def _hac_tstat(r: np.ndarray) -> dict:
    """Newey-West (HAC) t-statistic on the sample mean of ``r``."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 4:
        return {
            "tstat": float("nan"),
            "mean": float(r.mean()) if n else float("nan"),
            "se": float("nan"),
            "n": n,
        }
    mu = r.mean()
    e = r - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {
        "tstat": float(mu / se) if se > 0 else float("nan"),
        "mean": float(mu),
        "se": float(se),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Primary tests: pre-tax and post-tax windows vs unconditional baseline
# ---------------------------------------------------------------------------
def window_comparison(df: pd.DataFrame, window_col: str) -> dict:
    """Compare one tax window to all non-window trading days.

    Parameters
    ----------
    df : DataFrame with columns ``ret`` and *window_col* (boolean).
    window_col : str
        ``"pre_tax"`` or ``"post_tax"``.

    Returns
    -------
    dict with keys: n_window, mean_window_bps, tstat_window_hac,
    n_rest, mean_rest_bps, tstat_rest_hac, contrast_bps,
    tstat_welch, pval_welch.
    """
    from scipy.stats import ttest_ind

    win = df.loc[df[window_col], "ret"].to_numpy(dtype=float)
    rest = df.loc[~df[window_col], "ret"].to_numpy(dtype=float)
    win = win[np.isfinite(win)]
    rest = rest[np.isfinite(rest)]

    hw = _hac_tstat(win)
    hr = _hac_tstat(rest)

    if len(win) > 1 and len(rest) > 1:
        t_welch, pval = ttest_ind(win, rest, equal_var=False)
    else:
        t_welch, pval = float("nan"), float("nan")

    return {
        "n_window": int(len(win)),
        "mean_window_bps": float(win.mean() * 1e4) if len(win) else float("nan"),
        "tstat_window_hac": hw["tstat"],
        "n_rest": int(len(rest)),
        "mean_rest_bps": float(rest.mean() * 1e4) if len(rest) else float("nan"),
        "tstat_rest_hac": hr["tstat"],
        "contrast_bps": float((win.mean() - rest.mean()) * 1e4)
        if len(win) and len(rest) else float("nan"),
        "tstat_welch": float(t_welch),
        "pval_welch": float(pval),
    }


# ---------------------------------------------------------------------------
# April-month comparison (broad seasonal control)
# ---------------------------------------------------------------------------
def april_comparison(df: pd.DataFrame) -> dict:
    """Whole-month April vs all other months — the broad seasonal baseline."""
    from scipy.stats import ttest_ind

    apr = df.loc[df["tax_month"], "ret"].to_numpy(dtype=float)
    rest = df.loc[~df["tax_month"], "ret"].to_numpy(dtype=float)
    apr = apr[np.isfinite(apr)]
    rest = rest[np.isfinite(rest)]

    if len(apr) > 1 and len(rest) > 1:
        t_welch, pval = ttest_ind(apr, rest, equal_var=False)
    else:
        t_welch, pval = float("nan"), float("nan")

    return {
        "n_apr": int(len(apr)),
        "mean_apr_bps": float(apr.mean() * 1e4) if len(apr) else float("nan"),
        "n_rest": int(len(rest)),
        "mean_rest_bps": float(rest.mean() * 1e4) if len(rest) else float("nan"),
        "contrast_bps": float((apr.mean() - rest.mean()) * 1e4)
        if len(apr) and len(rest) else float("nan"),
        "tstat_welch": float(t_welch),
        "pval_welch": float(pval),
    }


# ---------------------------------------------------------------------------
# Placebo: same-sized windows around October 15 (no IRS deadline)
# ---------------------------------------------------------------------------
def _october_windows(
    dates: pd.DatetimeIndex, pre_window: int = 10, post_window: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Build boolean arrays for Oct-15 placebo windows on *dates*."""
    tz_aware = dates.tz is not None
    dates_naive = dates.tz_localize(None) if tz_aware else dates
    years = sorted(set(dates_naive.year))
    pre = np.zeros(len(dates), dtype=bool)
    post = np.zeros(len(dates), dtype=bool)

    for yr in years:
        # Oct 15, shifted for weekends.
        d = pd.Timestamp(year=yr, month=10, day=15)
        if d.dayofweek == 5:
            d += pd.Timedelta(days=2)
        elif d.dayofweek == 6:
            d += pd.Timedelta(days=1)

        yr_mask = dates_naive.year == yr
        yr_idx = np.where(yr_mask)[0]
        if len(yr_idx) == 0:
            continue
        yr_dates = dates_naive[yr_idx]
        before = yr_dates[yr_dates < d]
        after_or_eq = yr_dates[yr_dates >= d]

        if len(before) > 0:
            pre_start = max(0, len(before) - pre_window)
            global_pre = yr_idx[: len(before)][pre_start:]
            pre[global_pre] = True
        if len(after_or_eq) > 0:
            global_post = yr_idx[len(before) : len(before) + post_window]
            post[global_post] = True

    return pre, post


def placebo_comparison(df: pd.DataFrame, pre_window: int = 10, post_window: int = 5) -> dict:
    """Oct-15 placebo windows (same size, no IRS event) vs unconditional baseline.

    Returns a dict with the same key shape as ``window_comparison`` but for the
    *pre* and *post* placebo windows combined (we test whether the placebo is as
    'anomalous' as the tax windows — if yes, the result is noise, not signal).
    """
    from scipy.stats import ttest_ind

    pre, post = _october_windows(df.index, pre_window=pre_window, post_window=post_window)
    oct_mask = pre | post
    win = df.loc[oct_mask, "ret"].to_numpy(dtype=float)
    rest = df.loc[~oct_mask, "ret"].to_numpy(dtype=float)
    win = win[np.isfinite(win)]
    rest = rest[np.isfinite(rest)]

    if len(win) > 1 and len(rest) > 1:
        t_welch, pval = ttest_ind(win, rest, equal_var=False)
    else:
        t_welch, pval = float("nan"), float("nan")

    return {
        "n_oct_window": int(len(win)),
        "mean_oct_bps": float(win.mean() * 1e4) if len(win) else float("nan"),
        "n_rest": int(len(rest)),
        "mean_rest_bps": float(rest.mean() * 1e4) if len(rest) else float("nan"),
        "contrast_bps": float((win.mean() - rest.mean()) * 1e4)
        if len(win) and len(rest) else float("nan"),
        "tstat_welch": float(t_welch),
        "pval_welch": float(pval),
    }


# ---------------------------------------------------------------------------
# Multiple-comparisons sweep: Bonferroni k=2 (pre + post)
# ---------------------------------------------------------------------------
def bonferroni_summary(pre_result: dict, post_result: dict) -> dict:
    """Apply Bonferroni correction (k=2) to the two primary p-values.

    Returns a dict with corrected p-values for both hypotheses and whether either
    survives the 0.05/2 = 0.025 threshold.
    """
    n_tests = 2
    p_pre = pre_result["pval_welch"]
    p_post = post_result["pval_welch"]
    return {
        "n_tests": n_tests,
        "pval_pre_bonferroni": float(min(p_pre * n_tests, 1.0)) if np.isfinite(p_pre) else float("nan"),
        "pval_post_bonferroni": float(min(p_post * n_tests, 1.0)) if np.isfinite(p_post) else float("nan"),
        "any_survives_bonferroni": bool(
            (np.isfinite(p_pre) and p_pre * n_tests < 0.05)
            or (np.isfinite(p_post) and p_post * n_tests < 0.05)
        ),
    }


# ---------------------------------------------------------------------------
# Summarize — all tests in one call
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, pre_window: int = 10, post_window: int = 5) -> dict:
    """Headline numbers for the tax-day study: all tests combined.

    Returns a flat dict (plus nested sub-dicts and tables) mirroring
    docs/results.md for use in notebooks and verify.py.
    """
    pre = window_comparison(df, "pre_tax")
    post = window_comparison(df, "post_tax")
    apr = april_comparison(df)
    plac = placebo_comparison(df, pre_window=pre_window, post_window=post_window)
    bonf = bonferroni_summary(pre, post)

    return {
        # Pre-tax window
        "n_pre": pre["n_window"],
        "mean_pre_bps": pre["mean_window_bps"],
        "tstat_pre_hac": pre["tstat_window_hac"],
        "contrast_pre_bps": pre["contrast_bps"],
        "tstat_pre_welch": pre["tstat_welch"],
        "pval_pre": pre["pval_welch"],
        # Post-tax window
        "n_post": post["n_window"],
        "mean_post_bps": post["mean_window_bps"],
        "tstat_post_hac": post["tstat_window_hac"],
        "contrast_post_bps": post["contrast_bps"],
        "tstat_post_welch": post["tstat_welch"],
        "pval_post": post["pval_welch"],
        # Broad April baseline
        "n_apr": apr["n_apr"],
        "mean_apr_bps": apr["mean_apr_bps"],
        "contrast_apr_bps": apr["contrast_bps"],
        "pval_apr": apr["pval_welch"],
        # Unconditional baseline
        "mean_all_bps": float(df["ret"].dropna().mean() * 1e4),
        # Placebo
        "n_oct_window": plac["n_oct_window"],
        "mean_oct_bps": plac["mean_oct_bps"],
        "contrast_oct_bps": plac["contrast_bps"],
        "pval_oct": plac["pval_welch"],
        # Multiple comparisons
        "pval_pre_bonferroni": bonf["pval_pre_bonferroni"],
        "pval_post_bonferroni": bonf["pval_post_bonferroni"],
        "any_survives_bonferroni": bonf["any_survives_bonferroni"],
    }
