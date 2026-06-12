"""Strategy and analysis layer for Study 81 (Four-Year-Itch).

The folk recipe: year 3 of a U.S. presidential term is the equity bonanza (the incumbent
pumps the economy before the vote) and years 1–2 are the lean years (political capital spent
on unpopular reforms). We test it the only honest way: with a **baseline / control** that
asks not "are year-3 returns positive?" but "are they distinguishably different from a
random-year draw and from years 1, 2, and 4?" — and we quantify uncertainty via a HAC t-stat
on the ~24 non-overlapping annual return observations.

The small-n problem is structural: a four-year Presidential term began with Hoover in 1929,
so by 2024 we have only ~24 complete cycles — a training set a quant would laugh at for any
other effect. The honest analysis says so loudly.

Three analysis functions:
- :func:`annual_returns` — calendar-year (or term-year) returns from a daily return series.
- :func:`cycle_stats` — mean, std, t-stat (Newey-West) per year-of-term, plus the
  baseline (all-years pooled). HAC is used even for the per-year series: the ~24
  annual observations for each slot are not exchangeable (crisis years cluster), so
  plain SE would overstate precision.
- :func:`year3_vs_rest` — the headline scalar: is year-3 distinguishably better than
  the pooled alternative (years 1+2+4)? Welch t on the difference in means.
- :func:`summarize` — a compact dict suitable for the R-dict in notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Annual compounded return series
# ---------------------------------------------------------------------------
def annual_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compound daily returns to calendar-year totals, keeping ``term_year``.

    For each calendar year in ``df`` (which has columns ``ret`` and ``term_year``), we
    compound the daily simple returns to get the annual total return and record the
    majority term-year label (a year occasionally straddles an inauguration, but in
    practice the label is constant within a calendar year because we assign by calendar
    year).

    Returns a DataFrame indexed by calendar year with columns ``annual_ret`` and
    ``term_year``.
    """
    r = df["ret"].dropna().astype(float)
    ty = df["term_year"].reindex(r.index)

    rows = []
    for yr, grp in r.groupby(r.index.year):
        annual_ret = (1.0 + grp).prod() - 1.0
        t_labels = ty.reindex(grp.index).dropna()
        dominant_ty = int(t_labels.mode().iloc[0]) if len(t_labels) else np.nan
        rows.append({"year": yr, "annual_ret": annual_ret, "term_year": dominant_ty})
    return pd.DataFrame(rows).set_index("year")


# ---------------------------------------------------------------------------
# Per-term-year statistics
# ---------------------------------------------------------------------------
def _hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC t-stat for the mean of a small sample.

    Uses the same Bartlett-kernel estimator as the desk standard (see quantlab.analytics).
    For very small n (< 6) falls back to plain OLS SE.
    """
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


def cycle_stats(ann: pd.DataFrame) -> pd.DataFrame:
    """Per-year-of-term summary: n, mean return (%), std (%), HAC t-stat.

    ``ann`` is the output of :func:`annual_returns`. The returned DataFrame is indexed by
    term_year (1–4) with columns ``n``, ``mean_pct``, ``std_pct``, ``tstat``.
    """
    rows = []
    for ty in (1, 2, 3, 4):
        vals = ann.loc[ann["term_year"] == ty, "annual_ret"].dropna().values
        n = len(vals)
        mean_pct = float(vals.mean() * 100) if n > 0 else float("nan")
        std_pct = float(vals.std(ddof=1) * 100) if n > 1 else float("nan")
        tstat = _hac_tstat(vals)
        rows.append({"term_year": ty, "n": n, "mean_pct": mean_pct, "std_pct": std_pct, "tstat": tstat})
    return pd.DataFrame(rows).set_index("term_year")


def year3_vs_rest(ann: pd.DataFrame) -> dict:
    """Welch t-test: is year-3 mean annual return different from the pooled rest (years 1+2+4)?

    This is the headline test: the Presidential-cycle claim in its strongest form says year-3
    (pre-election) is meaningfully *better*, not just different. We return a Welch t (not
    paired — the years are different draws) and a plain-difference summary.
    """
    y3 = ann.loc[ann["term_year"] == 3, "annual_ret"].dropna().values
    rest = ann.loc[ann["term_year"].isin([1, 2, 4]), "annual_ret"].dropna().values
    if len(y3) < 2 or len(rest) < 2:
        return {"diff_pct": float("nan"), "welch_t": float("nan"), "n_y3": 0, "n_rest": 0}
    diff = y3.mean() - rest.mean()
    se = np.sqrt(y3.var(ddof=1) / len(y3) + rest.var(ddof=1) / len(rest))
    t = float(diff / se) if se > 0 else float("nan")
    return {
        "diff_pct": float(diff * 100),
        "welch_t": float(t),
        "n_y3": int(len(y3)),
        "n_rest": int(len(rest)),
    }


def summarize(df: pd.DataFrame) -> dict:
    """One-stop summary for a daily returns DataFrame (as from :func:`fetch_prices`).

    Computes annual returns, per-term-year stats, and the year-3-vs-rest headline test.
    Returns a flat dict suitable for the R-dict in notebooks and for docs/results.md.
    """
    ann = annual_returns(df)
    stats = cycle_stats(ann)
    y3r = year3_vs_rest(ann)
    out: dict = {}
    for ty in (1, 2, 3, 4):
        row = stats.loc[ty]
        out[f"y{ty}_n"] = int(row["n"])
        out[f"y{ty}_mean"] = round(float(row["mean_pct"]), 2)
        out[f"y{ty}_std"] = round(float(row["std_pct"]), 2)
        out[f"y{ty}_tstat"] = round(float(row["tstat"]), 2)
    out["y3_minus_rest_pct"] = round(y3r["diff_pct"], 2)
    out["y3_vs_rest_t"] = round(y3r["welch_t"], 2)
    out["n_y3"] = y3r["n_y3"]
    out["n_rest"] = y3r["n_rest"]
    return out
