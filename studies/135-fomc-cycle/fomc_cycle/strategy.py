"""The FOMC-cycle strategy and its honest controls — Study 135 (FOMC-Cycle).

The claim (Cieslak, Morse & Vuolteenaho 2019): US equity returns since 1994 are
concentrated in the *even* weeks (0, 2, 4) of the ~6-week FOMC cycle and are
flat-to-negative in the *odd* weeks (1, 3, 5). Week 0 starts on the FOMC statement
day; each week is 5 trading days.

We implement three complementary tests:

1. ``even_odd_table`` — headline split: mean daily return and HAC t-stat for even
   vs odd weeks, plus a Welch t on the *difference*. The honest control is the
   buy-and-hold return on all days: the gap only counts if it beats the full-market
   baseline (days-in-market = all, so no timing advantage is assumed away).

2. ``random_week_split`` — a placebo: randomly shuffle cycle-week labels (preserving
   the even/odd structure across n seeds) and measure the mean even-vs-odd gap. If
   the true gap barely exceeds the shuffle distribution, the effect is noise.

3. ``split_by_era`` — pre- vs post-publication decay (paper submitted ~2014,
   published 2019). McLean & Pontiff (2016) document that anomaly returns shrink
   after publication. We split at 2019-01-01 and measure the gap in each era.

No look-ahead: cycle-week assignment uses only the most recent *past* FOMC date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metric: even-vs-odd mean return gap, with HAC t-stat
# ---------------------------------------------------------------------------
def _hac_tstat(r: np.ndarray) -> float:
    """Newey-West HAC t-stat for the mean of ``r``."""
    r = r[np.isfinite(r)]
    n = r.size
    if n < 6:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def _arm_stats(rets: np.ndarray, ann: int = 252) -> dict:
    """Mean, vol, annualised Sharpe, and HAC t-stat for a daily-return array."""
    r = rets[np.isfinite(rets)]
    if r.size < 2:
        return {"n": 0, "mean_bps": float("nan"), "vol_ann": float("nan"),
                "sharpe_ann": float("nan"), "tstat": float("nan")}
    mu = r.mean()
    sd = r.std(ddof=1)
    return {
        "n": int(r.size),
        "mean_bps": mu * 1e4,
        "vol_ann": sd * np.sqrt(ann) * 1e4,
        "sharpe_ann": mu / sd * np.sqrt(ann) if sd > 0 else float("nan"),
        "tstat": _hac_tstat(r),
    }


# ---------------------------------------------------------------------------
# 1. Even vs odd week split
# ---------------------------------------------------------------------------
def even_odd_table(frame: pd.DataFrame, ann: int = 252) -> dict:
    """Headline even-week vs odd-week return comparison.

    Parameters
    ----------
    frame : DataFrame with columns ``ret`` (daily return) and ``cycle_week``
            (0-5 float, NaN for pre-sample days).
    ann   : Annualisation factor (default 252 trading days).

    Returns a dict with stats for the even arm, odd arm, all days, and the
    Welch t-stat on the even-minus-odd difference.
    """
    valid = frame.dropna(subset=["cycle_week"])
    even = valid[valid["cycle_week"].isin([0, 2, 4])]["ret"].to_numpy()
    odd = valid[valid["cycle_week"].isin([1, 3, 5])]["ret"].to_numpy()
    all_r = valid["ret"].to_numpy()

    # Welch t on even − odd (the headline test, Cieslak et al. eq. (2))
    def _welch_t(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size < 2 or b.size < 2:
            return float("nan")
        se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
        return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")

    return {
        "even": _arm_stats(even, ann),
        "odd":  _arm_stats(odd, ann),
        "all":  _arm_stats(all_r, ann),
        "diff_bps": float((even[np.isfinite(even)].mean() -
                           odd[np.isfinite(odd)].mean()) * 1e4),
        "welch_t": _welch_t(even, odd),
    }


# ---------------------------------------------------------------------------
# 2. Placebo — random cycle-week permutation
# ---------------------------------------------------------------------------
def random_week_split(
    frame: pd.DataFrame,
    n_seeds: int = 200,
    ann: int = 252,
) -> dict:
    """Placebo: randomly shuffle which trading days are labelled 'even' vs 'odd'.

    Within each seed we draw a random binary mask of the same size and density as
    the true even-week mask, compute the mean difference, and compare with the
    observed gap. If the real gap exceeds the 95th percentile of this null
    distribution, the effect is unlikely to be noise.

    Returns a dict with the real gap, the placebo mean, std, and p-value.
    """
    valid = frame.dropna(subset=["cycle_week"])
    real_even = valid["cycle_week"].isin([0, 2, 4])
    even_frac = real_even.mean()
    r = valid["ret"].to_numpy()
    real_gap = (r[real_even.values].mean() - r[~real_even.values].mean()) * 1e4

    rng = np.random.default_rng(135)
    gaps = np.empty(n_seeds)
    n = len(r)
    n_even = int(round(even_frac * n))
    for s in range(n_seeds):
        perm = np.zeros(n, dtype=bool)
        perm[:n_even] = True
        rng.shuffle(perm)
        gaps[s] = (r[perm].mean() - r[~perm].mean()) * 1e4

    p_value = float((gaps >= real_gap).mean())
    return {
        "real_gap_bps": real_gap,
        "placebo_mean_bps": float(gaps.mean()),
        "placebo_std_bps": float(gaps.std(ddof=1)),
        "p_value": p_value,
        "n_seeds": n_seeds,
    }


# ---------------------------------------------------------------------------
# 3. Pre- vs post-publication split
# ---------------------------------------------------------------------------
def split_by_era(
    frame: pd.DataFrame,
    cut: str = "2019-01-01",
    ann: int = 252,
) -> dict:
    """Even-vs-odd gap before and after the publication year (McLean-Pontiff test).

    The paper was published in 2019 (Journal of Finance, 74:5). We split at
    ``cut`` and measure whether the even-week premium survived. Decay is expected
    — but how much decay matters for the Tradability verdict.
    """
    cut_ts = pd.Timestamp(cut)
    pre = frame[frame.index < cut_ts]
    post = frame[frame.index >= cut_ts]

    def _gap(df: pd.DataFrame) -> dict:
        valid = df.dropna(subset=["cycle_week"])
        even = valid[valid["cycle_week"].isin([0, 2, 4])]["ret"].to_numpy()
        odd = valid[valid["cycle_week"].isin([1, 3, 5])]["ret"].to_numpy()
        if even.size < 10 or odd.size < 10:
            return {"diff_bps": float("nan"), "welch_t": float("nan"),
                    "even_mean_bps": float("nan"), "odd_mean_bps": float("nan"),
                    "n_even": 0, "n_odd": 0}
        e, o = even[np.isfinite(even)], odd[np.isfinite(odd)]
        se = np.sqrt(e.var(ddof=1) / e.size + o.var(ddof=1) / o.size)
        welch = float((e.mean() - o.mean()) / se) if se > 0 else float("nan")
        return {
            "diff_bps": float((e.mean() - o.mean()) * 1e4),
            "welch_t": welch,
            "even_mean_bps": float(e.mean() * 1e4),
            "odd_mean_bps": float(o.mean() * 1e4),
            "n_even": int(e.size),
            "n_odd": int(o.size),
        }

    return {
        "pre_publication": _gap(pre),
        "post_publication": _gap(post),
        "cut": cut,
    }


# ---------------------------------------------------------------------------
# 4. Week-by-week breakdown (weeks 0–5)
# ---------------------------------------------------------------------------
def week_by_week(frame: pd.DataFrame, ann: int = 252) -> pd.DataFrame:
    """Mean daily return for each FOMC-cycle week (0-5) and HAC t-stat.

    Returns a DataFrame indexed by week (0-5) with columns mean_bps, tstat, n,
    and is_even (True for weeks 0, 2, 4).
    """
    valid = frame.dropna(subset=["cycle_week"])
    rows = []
    for w in range(6):
        r = valid[valid["cycle_week"] == w]["ret"].to_numpy()
        r = r[np.isfinite(r)]
        mean_bps = float(r.mean() * 1e4) if r.size > 0 else float("nan")
        t = _hac_tstat(r)
        rows.append({"week": int(w), "n": int(r.size),
                     "mean_bps": mean_bps, "tstat": t,
                     "is_even": w % 2 == 0})
    return pd.DataFrame(rows).set_index("week")


# ---------------------------------------------------------------------------
# 5. Tradability cost sweep — biweekly even-week long
# ---------------------------------------------------------------------------
def cost_sweep(
    frame: pd.DataFrame,
    costs_bps: list[float] | None = None,
    ann: int = 252,
) -> pd.DataFrame:
    """Net annual return of the 'long on even weeks only' strategy vs cost.

    The strategy is long SPY only on even-week days, cash otherwise. The
    annualised gross return is the even-week mean × number of even-week days per
    year. We charge a one-way cost each time the strategy enters or exits
    (approx. 8 trades per FOMC cycle × 8 meetings/year = ~64 one-way trips/yr).
    """
    if costs_bps is None:
        costs_bps = [0.0, 0.5, 1.0, 2.0, 5.0]

    valid = frame.dropna(subset=["cycle_week"])
    even_mask = valid["cycle_week"].isin([0, 2, 4])
    even_r = valid.loc[even_mask, "ret"].to_numpy()
    even_r = even_r[np.isfinite(even_r)]

    # Approximate number of cycle transitions per year
    # 8 meetings/year × 2 transitions per cycle (in at start of even week, out at end)
    transitions_per_year = 8 * 2 * 3  # ~48 one-way trips for 3 even-week blocks
    gross_bps = float(even_r.mean() * 1e4) if even_r.size > 0 else float("nan")
    gross_ann = float(even_r.mean() * ann) if even_r.size > 0 else float("nan")

    rows = []
    for c in costs_bps:
        # Cost per year = transitions * one-way cost
        annual_cost = transitions_per_year * c * 1e-4
        net_ann = gross_ann - annual_cost
        rows.append({
            "cost_bps": c,
            "gross_ann_pct": gross_ann * 100,
            "net_ann_pct": net_ann * 100,
        })
    return pd.DataFrame(rows).set_index("cost_bps")
