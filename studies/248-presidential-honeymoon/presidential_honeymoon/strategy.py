"""Strategy and analysis layer for Study 248 (Presidential-Honeymoon).

The folk recipe: a new president's first 100 days carry a market 'honeymoon' — relief at
the transition, optimism about new policies, a brief suspension of partisan fear.  We test it
the only honest way: not 'are first-100-day returns positive?' (trivially true for any equity
subperiod) but 'are they *distinguishably better* than the rest of the year's returns on the
same tape, using a paired within-term HAC t-test?'

The small-n problem is structural: we have at most 25 presidential inaugurations since
Hoover (1929), one 100-day window each.  With daily index vol of ~1% the SE of a
100-day cumulative return is ~10 pp.  To clear |t| ≥ 2 we need the gap vs the baseline to
exceed ~20 pp — a very high bar.

Three analysis functions:
- :func:`period_returns` — 100-day cumulative return per inauguration (honeymoon) and the
  next 265 calendar days (the 'rest-of-year' control), both measured from the same tape.
- :func:`honeymoon_stats` — mean, std, t-stat (HAC) for honeymoon returns, rest-of-year
  returns, and the difference (paired).
- :func:`summarize` — a compact dict suitable for the R-dict in notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import INAUGURATION_DATES, HONEYMOON_DAYS

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Per-inauguration returns
# ---------------------------------------------------------------------------
def period_returns(
    df: pd.DataFrame,
    window_days: int = HONEYMOON_DAYS,
    rest_days: int = 265,
) -> pd.DataFrame:
    """Cumulative return for each inauguration's honeymoon window and control window.

    For each inauguration date in ``INAUGURATION_DATES``:
    - ``hm_ret``:   cumulative simple return over the next ``window_days`` calendar days
      (i.e., trading days that fall in [inau, inau + window_days)).
    - ``rest_ret``: cumulative simple return over the subsequent ``rest_days`` calendar
      days (i.e., trading days that fall in [inau + window_days, inau + window_days + rest_days)).
    - ``hm_n``:     number of trading days in the honeymoon window.
    - ``rest_n``:   number of trading days in the rest window.

    Only inaugurations where *both* windows have at least 20 trading days are included.

    Returns a DataFrame indexed by inauguration date with columns
    ``president``, ``hm_ret``, ``rest_ret``, ``hm_n``, ``rest_n``.
    """
    from .data import INAUGURATIONS

    rows = []
    ret_series = df["ret"].dropna().astype(float)
    idx = pd.DatetimeIndex(ret_series.index)

    for i, (pres, _) in enumerate(INAUGURATIONS):
        inau = INAUGURATION_DATES[i]
        hm_end = inau + pd.Timedelta(days=window_days)
        rest_end = hm_end + pd.Timedelta(days=rest_days)

        hm_mask = (idx >= inau) & (idx < hm_end)
        rest_mask = (idx >= hm_end) & (idx < rest_end)

        hm_td = ret_series.iloc[hm_mask]
        rest_td = ret_series.iloc[rest_mask]

        if len(hm_td) < 20 or len(rest_td) < 20:
            continue  # skip if insufficient data

        hm_ret = float((1.0 + hm_td).prod() - 1.0)
        rest_ret = float((1.0 + rest_td).prod() - 1.0)
        rows.append(
            {
                "president": pres,
                "inau_date": inau,
                "hm_ret": hm_ret,
                "rest_ret": rest_ret,
                "hm_n": int(len(hm_td)),
                "rest_n": int(len(rest_td)),
            }
        )
    return pd.DataFrame(rows).set_index("inau_date")


# ---------------------------------------------------------------------------
# HAC t-stat helper
# ---------------------------------------------------------------------------
def _hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC t-stat for the mean of a small sample."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    mu = x.mean()
    e = x - mu
    if n < 6:
        se = x.std(ddof=1) / np.sqrt(n)
        return float(mu / se) if se > 0 else float("nan")
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def honeymoon_stats(periods: pd.DataFrame) -> dict:
    """Mean, std and HAC t-stat for honeymoon and rest-of-year returns, plus the paired diff.

    ``periods`` is the output of :func:`period_returns`.  Returns a dict with:
    - ``hm_mean``, ``hm_std``, ``hm_tstat``  — statistics on raw honeymoon returns
    - ``rest_mean``, ``rest_std``, ``rest_tstat`` — statistics on rest-of-year returns
    - ``diff_mean``, ``diff_std``, ``diff_tstat`` — paired diff (honeymoon minus rest)
    - ``n``  — number of inaugurations included
    - ``pos_rate``  — fraction of terms where honeymoon > rest
    """
    hm = periods["hm_ret"].values
    rest = periods["rest_ret"].values
    diff = hm - rest
    n = len(hm)

    def _stats(arr: np.ndarray) -> tuple[float, float, float]:
        return (
            float(arr.mean() * 100),
            float(arr.std(ddof=1) * 100) if len(arr) > 1 else float("nan"),
            _hac_tstat(arr),
        )

    hm_m, hm_s, hm_t = _stats(hm)
    rest_m, rest_s, rest_t = _stats(rest)
    diff_m, diff_s, diff_t = _stats(diff)
    pos_rate = float((diff > 0).mean())

    return {
        "n": n,
        "hm_mean": round(hm_m, 2),
        "hm_std": round(hm_s, 2),
        "hm_tstat": round(hm_t, 2),
        "rest_mean": round(rest_m, 2),
        "rest_std": round(rest_s, 2),
        "rest_tstat": round(rest_t, 2),
        "diff_mean": round(diff_m, 2),
        "diff_std": round(diff_s, 2),
        "diff_tstat": round(diff_t, 2),
        "pos_rate": round(pos_rate, 3),
    }


def first_year_stats(df: pd.DataFrame) -> dict:
    """Compare first-365-day return vs next-365-day return for each term.

    A broader test: does the full first calendar year after inauguration
    differ from the next year?  Returns a dict analogous to :func:`honeymoon_stats`.
    """
    from .data import INAUGURATIONS

    rows = []
    ret_series = df["ret"].dropna().astype(float)
    idx = pd.DatetimeIndex(ret_series.index)

    for i, (pres, _) in enumerate(INAUGURATIONS):
        inau = INAUGURATION_DATES[i]
        yr1_end = inau + pd.Timedelta(days=365)
        yr2_end = yr1_end + pd.Timedelta(days=365)

        yr1_td = ret_series.iloc[(idx >= inau) & (idx < yr1_end)]
        yr2_td = ret_series.iloc[(idx >= yr1_end) & (idx < yr2_end)]

        if len(yr1_td) < 50 or len(yr2_td) < 50:
            continue

        rows.append(
            {
                "yr1_ret": float((1.0 + yr1_td).prod() - 1.0),
                "yr2_ret": float((1.0 + yr2_td).prod() - 1.0),
            }
        )

    if not rows:
        return {}
    yr1 = np.array([r["yr1_ret"] for r in rows])
    yr2 = np.array([r["yr2_ret"] for r in rows])
    diff = yr1 - yr2
    return {
        "n_terms": len(rows),
        "yr1_mean": round(float(yr1.mean() * 100), 2),
        "yr2_mean": round(float(yr2.mean() * 100), 2),
        "diff_mean": round(float(diff.mean() * 100), 2),
        "diff_tstat": round(_hac_tstat(diff), 2),
    }


def summarize(df: pd.DataFrame) -> dict:
    """One-stop summary for a daily returns DataFrame (as from :func:`fetch_prices`).

    Computes per-inauguration honeymoon stats and the first-year extended test.
    Returns a flat dict suitable for the R-dict in notebooks and for docs/results.md.
    """
    periods = period_returns(df)
    stats = honeymoon_stats(periods)
    fy = first_year_stats(df)
    out: dict = {**stats}
    out["fy_n"] = fy.get("n_terms", 0)
    out["fy_yr1_mean"] = fy.get("yr1_mean", float("nan"))
    out["fy_yr2_mean"] = fy.get("yr2_mean", float("nan"))
    out["fy_diff_mean"] = fy.get("diff_mean", float("nan"))
    out["fy_diff_tstat"] = fy.get("diff_tstat", float("nan"))
    return out
