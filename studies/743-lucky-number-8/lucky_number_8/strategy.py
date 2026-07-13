"""Strategy + inference for Study 743 — Lucky-Number-8.

Two independent tests of one superstition, each with the desk's standard honesty rails.

**A · The 8/8 superstition premium (an event study).** The 8th of August is the most
auspicious date in Chinese numerology. If the folklore is right, a China large-cap ETF
should get a feel-good/buying bump around it, over and above whatever emerging markets
as a whole are doing. Because 8/8 is *calendar-known* — a date on everyone's wall, not
a private announcement — there is **no look-ahead problem at all**: you could always
have placed the trade. The one documented execution convention is the snap:

* **day(-1)** = the last trading session strictly *before* Aug 8 (typically Aug 7, or
  the Friday before an Aug-8-weekend).
* **day(0)**  = the first trading session on/after Aug 8 (Aug 8 itself when it is a
  weekday; the Monday after when it falls on a weekend).

Two measurements per event, both **abnormal** (`FXI` minus the `EEM` emerging-markets
benchmark, both total-return), so a common EM move is differenced out and only the
China-specific piece remains:

* **Signal.** Cumulative abnormal return day(-1)→day(-1)+k for k ∈ {1 (the lucky day
  itself), 5 (the lucky week)}. Because the date is calendar-known this is *already*
  a tradable window — there is no un-tradable weekend jump to strip out, unlike an
  announcement study.
* **Tradability.** The same window net of one round-trip's costs (2× one-way × NAV).

The unit of inference is the **one-sample t across the ~21 independent, non-overlapping
years** — not a daily panel. A random-window placebo (same-length windows drawn at
random non-8/8 anchors from `FXI`'s own history vs `EEM`) checks whether the observed
mean sits outside `FXI`'s ordinary week-to-week tracking noise.

**B · Trailing-digit clustering.** Pool the raw closing prices of a China-ADR basket
and, separately, a US-control basket; extract the trailing cent digit (``round(px*100)
% 10``); count digit frequencies. Both baskets trade with a uniform $0.01 tick, so under
the null each digit is 1/10 and the universal round-number (0/5) preference is common to
both. The superstition predicts a **China-specific** 8-excess and 4-deficit — so the
decisive statistic is the two-proportion z of (China − control) on digit 8 and on digit
4, which cancels the shared round-number structure. A pooled chi-square-vs-uniform is
reported per basket for context (with its serial-dependence caveat stated out loud).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import data as dt

DAY_K = 1      # the lucky day itself (day(-1) -> day(0))
WEEK_K = 5     # the lucky week (~1 trading week)
COST_BPS = 5.0  # one-way, per leg


# =========================================================================== #
# PART A — the 8/8 event study
# =========================================================================== #
def build_event_table(prices: dict[str, pd.DataFrame], cost_bps: float = COST_BPS
                      ) -> pd.DataFrame:
    """One row per lucky-date (8/8) event: abnormal returns + inclusion reason.

    A row is INCLUDED only if both `FXI` and `EEM` cover [day(-1) .. day(-1)+WEEK_K]
    around that year's Aug 8. Excluded rows are kept (with a reason) so the funnel is
    auditable.
    """
    fxi = dt.adj_close(prices, dt.CHINA_PROXY)
    eem = dt.adj_close(prices, dt.EM_BENCHMARK)
    common = fxi.index.intersection(eem.index).sort_values()
    rows = []
    for year, date_str, note in dt.LUCKY_DATES:
        d8 = pd.Timestamp(date_str)
        before = common[common < d8]
        onafter = common[common >= d8]
        row = dict(year=year, date=date_str, note=note)
        if len(before) == 0 or len(onafter) == 0:
            row.update(included=False, reason="FXI/EEM predate or postdate the lucky day")
            rows.append(row)
            continue
        p = common.get_loc(before[-1])           # day(-1)
        if p + WEEK_K >= len(common):
            row.update(included=False, reason="insufficient trailing history")
            rows.append(row)
            continue
        d_m1 = common[p]
        d_0 = common[p + DAY_K]
        d_wk = common[p + WEEK_K]

        def ar(t_end):
            r_fxi = float(fxi.loc[t_end] / fxi.loc[d_m1] - 1.0)
            r_eem = float(eem.loc[t_end] / eem.loc[d_m1] - 1.0)
            return r_fxi - r_eem

        ar_day, ar_week = ar(d_0), ar(d_wk)
        row.update(
            included=True, reason="",
            anchor_date=str(d_m1.date()), day0_date=str(d_0.date()),
            ar_day=ar_day, ar_week=ar_week,
            # calendar-known: the same window IS the tradable trade, net of one round-trip
            cap_day_gross=ar_day, cap_day_net=ar_day - 2.0 * cost_bps / 1e4,
            cap_week_gross=ar_week, cap_week_net=ar_week - 2.0 * cost_bps / 1e4,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def one_sample_t(x: np.ndarray) -> dict:
    """One-sample t of mean(x) vs 0 — the right unit for independent yearly events."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else float("nan"), "sd": float("nan"),
                "t": float("nan")}
    se = x.std(ddof=1) / np.sqrt(n)
    return {"n": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "t": float(x.mean() / se) if se > 0 else float("nan")}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def hit_rate(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    k = int((x > 0).sum())
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


def placebo_pvalue(events: pd.DataFrame, prices: dict[str, pd.DataFrame], col: str,
                   k: int, cost_bps: float = 0.0, n_seeds: int = 20,
                   n_draws_per_seed: int = 200, base_seed: int = 743,
                   tail: str = "right") -> dict:
    """Random-window placebo: for each INCLUDED event redraw a random (non-8/8) k-session
    window on `FXI`-minus-`EEM` and recompute the abnormal return; average across the
    same number of events; repeat n_seeds x n_draws_per_seed times. ``tail`` = "right"
    (claim predicts a POSITIVE bump) or "left"."""
    fxi = dt.adj_close(prices, dt.CHINA_PROXY)
    eem = dt.adj_close(prices, dt.EM_BENCHMARK)
    common = fxi.index.intersection(eem.index).sort_values()
    inc = events[events["included"]]
    n_events = len(inc)
    obs = float(inc[col].mean())

    fxi_a = fxi.reindex(common).to_numpy()
    eem_a = eem.reindex(common).to_numpy()
    lo, hi = 1, len(common) - k - 1
    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            p = rng.integers(lo, hi, size=n_events)
            r_fxi = fxi_a[p + k] / fxi_a[p] - 1.0
            r_eem = eem_a[p + k] / eem_a[p] - 1.0
            vals = (r_fxi - r_eem) - 2.0 * cost_bps / 1e4
            means.append(float(np.mean(vals)))
    means = np.asarray(means)
    pv = float((means >= obs).mean()) if tail == "right" else float((means <= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)), "p_value": pv, "n_draws": len(means)}


def car_path(events: pd.DataFrame, prices: dict[str, pd.DataFrame],
             pre: int = 5, post: int = 5) -> pd.Series:
    """Mean cumulative abnormal return (FXI - EEM) at each offset -pre..+post around
    day(0) = the first session on/after 8/8, re-anchored so CAR at day(-1) = 0.

    Anchoring at day(-1) (offset -1, the study's event anchor) makes the offset(-1)->
    offset(0) step equal the ``ar_day`` headline: everything before -1 is the run-IN to
    the lucky day, everything after 0 is the hangover.
    """
    fxi = dt.adj_close(prices, dt.CHINA_PROXY)
    eem = dt.adj_close(prices, dt.EM_BENCHMARK)
    common = fxi.index.intersection(eem.index).sort_values()
    fxi_c = fxi.reindex(common).to_numpy()
    eem_c = eem.reindex(common).to_numpy()
    inc = events[events["included"]]
    anchor = pre - 1  # segment position of day(-1)
    paths = []
    for _, row in inc.iterrows():
        d0 = pd.Timestamp(row["day0_date"])
        pos = common.get_loc(d0)
        if pos - pre < 0 or pos + post >= len(common):
            continue
        seg_f = fxi_c[pos - pre: pos + post + 1]
        seg_e = eem_c[pos - pre: pos + post + 1]
        base_f, base_e = seg_f[anchor], seg_e[anchor]
        rf = seg_f / base_f - 1.0
        re = seg_e / base_e - 1.0
        paths.append(rf - re)
    arr = np.asarray(paths)
    return pd.Series(arr.mean(axis=0), index=range(-pre, post + 1))


def synthetic_detect(bump: float, seed: int, k: int = 1) -> dict:
    """Run the one-sample-t detector on a synthetic paired world with a planted bump."""
    a, b, events = dt.synthetic_world(bump=bump, seed=seed)
    ar = []
    for p in events:
        if p + k >= len(a):
            continue
        ra = a.iloc[p:p + k + 1].sum()
        rb = b.iloc[p:p + k + 1].sum()
        ar.append(float(ra - rb))
    return one_sample_t(np.asarray(ar))


# =========================================================================== #
# PART B — trailing-digit clustering
# =========================================================================== #
def trailing_digit(px: np.ndarray) -> np.ndarray:
    """The trailing cent digit of a raw price: round(px*100) % 10.

    A $27.38 close -> 8; a $27.34 -> 4. The digit the superstition should tilt.
    """
    px = np.asarray(px, dtype=float)
    px = px[np.isfinite(px) & (px > 0)]
    cents = np.rint(px * 100.0).astype(np.int64)
    return cents % 10


def basket_digit_counts(prices: dict[str, pd.DataFrame], tickers: list[str]) -> np.ndarray:
    """Pooled counts of the trailing cent digit (length-10 array, digits 0..9) across
    every raw daily close of every ticker in the basket."""
    counts = np.zeros(10, dtype=np.int64)
    for t in tickers:
        if t not in prices:
            continue
        d = trailing_digit(dt.raw_close(prices, t).to_numpy())
        counts += np.bincount(d, minlength=10)
    return counts


def chi_square_uniform(counts: np.ndarray) -> dict:
    """Pearson chi-square of a length-10 digit-count vector against Uniform(1/10).

    NOTE the honest caveat: pooled daily closes are serially dependent, so the nominal
    p-value OVERSTATES significance; read the effect SIZE (the per-digit frequency
    deviation), not the raw p. df = 9; the 0.05 critical value is 16.92.
    """
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    exp = np.full(10, n / 10.0)
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    p = float(stats.chi2.sf(chi2, df=9))
    return {"n": int(n), "chi2": chi2, "p_value": p, "crit_05": 16.919,
            "freq": (counts / n).tolist()}


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Two-proportion z-test of p1 (group 1 frequency of a digit) vs p2 (group 2).

    Used for the decisive China-minus-control contrast on digit 8 (and 4): the shared
    round-number structure cancels, so a non-zero z is China-SPECIFIC.
    """
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = float((p1 - p2) / se) if se > 0 else float("nan")
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")
    return {"p1": p1, "p2": p2, "diff": p1 - p2, "z": z, "p_value": p}


def digit_report(prices: dict[str, pd.DataFrame]) -> dict:
    """The full clustering read: per-basket digit frequencies, chi-square, and the
    decisive China-minus-control two-proportion z on digit 8 and digit 4."""
    china = basket_digit_counts(prices, dt.CHINA_ADRS)
    control = basket_digit_counts(prices, dt.CONTROL_US)
    nc, nu = int(china.sum()), int(control.sum())
    out = {
        "china_counts": china.tolist(), "control_counts": control.tolist(),
        "china_n": nc, "control_n": nu,
        "china_chi2": chi_square_uniform(china),
        "control_chi2": chi_square_uniform(control),
        "z8": two_proportion_z(int(china[8]), nc, int(control[8]), nu),
        "z4": two_proportion_z(int(china[4]), nc, int(control[4]), nu),
    }
    return out


def synthetic_digit_detect(excess: float, seed: int, n: int = 40000) -> dict:
    """Positive control for the clustering detector: generate a digit sample with a
    planted 8-excess and report the one-proportion z on digit 8 vs the 1/10 null."""
    d = dt.synthetic_digits(excess=excess, seed=seed, n=n)
    counts = np.bincount(d, minlength=10)
    k8 = int(counts[8])
    p8 = k8 / n
    se = np.sqrt(0.1 * 0.9 / n)
    z = float((p8 - 0.1) / se)
    chi = chi_square_uniform(counts)
    return {"n": n, "p8": p8, "z8": z, "chi2": chi["chi2"], "chi2_p": chi["p_value"]}
