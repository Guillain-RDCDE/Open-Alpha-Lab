"""Estimator + break-even arithmetic for Study 920 — Total Cost of Ownership.

The question. Two wrappers hold the same index. One is the mega-liquid original with the
higher fee and the tightest quote; the other is the cheap clone with a wider quote. The
cheap one wins on the *carry* (fee + realised tracking difference) and loses on the
*entry* (round-trip spread). So there is a **break-even holding period**:

    H* (years) = (round-trip spread of cheap - round-trip spread of liquid)
                 / (annual tracking-difference advantage of cheap over liquid)

Above ``H*`` the cheap wrapper is the cheaper thing to own; below it, the liquid one is.
Everything on the numerator is a **quote-level assumption** (a daily-close tape carries no
spreads), so it is swept end to end. Everything on the denominator comes off the tape.

Sign convention, used everywhere: ``td_bp_yr`` is the **cheap minus liquid** annualised
total-return difference in basis points per year. Positive = the cheap wrapper really did
deliver more, which is what a lower fee is supposed to buy.

Three estimators of the same quantity, reported side by side. **On the real tape they do
not agree**, and the disagreement is itself a finding rather than a rounding detail:

- **cumulative** — the full-window drift of ``log(cheap / liquid)``, annualised. Maximum
  power, minimum robustness: it is two closing prints and a slope, so it carries the level
  error of *both* endpoints at full weight.
- **annual** — the mean of the complete-calendar-year tracking differences, with a
  one-sample *t* across years. This is the study's headline: it lets the year-to-year
  variation of a real wrapper (lending revenue, sampling, cash drag, print noise) speak,
  and — because it keeps only complete calendar years — it discards the partial opening
  and closing stubs where a single mis-timed closing print does the most damage.
- **monthly HAC** — the mean of the complete-calendar-month tracking differences with a
  Newey-West *t*, plus a circular block bootstrap CI on the annualised mean.

They agree on synthetic tape (see the calibration panel) and they do **not** agree on the
public one. On the common window the QQQ/QQQM pair prints +7.19 bp/yr annual, +4.48
cumulative and +2.23 monthly-annualised; the same-fee placebo pair prints +0.82 annual but
+4.69 cumulative. :func:`stub_decomposition` shows why: the partial periods that the annual
estimator drops (here 2020 Q4 and 2026 H1) carry level artefacts of −15 to +24 bp, i.e.
three to five times the whole annual signal. The complete-calendar-period rule is therefore
load-bearing, it was fixed ex ante by the as-of convention, and the *reason* to believe it
rather than the cumulative estimator is the placebo: two funds charging the same 3 bp
cannot really drift +23.6 bp apart in six months. **Every headline in this study is the
annual estimator's, and it is the largest of the three on the pair that carries the
verdict — a reader who prefers the cumulative one gets a materially weaker result.**

One execution lag, documented: in :func:`holding_period_edge` the wrapper choice is made
with information through the close of day ``t`` and both candidate positions are entered
at the close of day ``t+1``. Costs are charged one-way x NAV per leg (a round trip is two
of them). The only short leg in the study is :func:`long_short_harvest`, which pays an
explicit borrow rate on the shorted liquid wrapper, swept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTHS = 12


# --------------------------------------------------------------------------- #
# Inference primitives (shared desk mirror)
# --------------------------------------------------------------------------- #
def one_sample_t(x) -> float:
    """Plain one-sample *t* of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def newey_west_t(x, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) *t* of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(int(lags), n - 2))
    u = x - x.mean()
    var = float(u @ u) / n
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(var / n))


# Two-sided 95% Student-t critical values, df = 1..30. Hard-coded so the study needs no
# scipy: the annual estimator runs on five to twenty-five observations, exactly the regime
# where using 1.96 (or a percentile bootstrap) understates the interval badly.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
    30: 2.042,
}


def t95(df: int) -> float:
    """Two-sided 95% Student-*t* critical value for ``df`` degrees of freedom."""
    if df < 1:
        return float("nan")
    return float(_T95.get(int(df), 1.96))


def t_interval(x) -> tuple[float, float]:
    """Small-sample Student-*t* 95% interval for the mean of ``x``.

    This is the interval the study quotes whenever it makes a *pessimistic* claim, and it
    exists because the circular block bootstrap in :func:`bootstrap_td_ci` is not to be
    trusted at the sample sizes this study actually has. With five annual observations the
    percentile bootstrap resamples a five-point empirical distribution: its spread is
    ``sd/sqrt(n)`` with no allowance for the fact that ``sd`` is itself estimated from five
    numbers, so it comes out roughly ``t95(4)/1.96 ~ 1.4x`` too narrow before the block
    structure narrows it further. On the common window the two disagree badly — IVV/SPY
    bootstraps to [+3.55, +7.90] and *t*-intervals to [+0.28, +11.28] — and where they
    disagree the *t* interval is the honest one.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    half = t95(n - 1) * x.std(ddof=1) / np.sqrt(n)
    return (float(x.mean() - half), float(x.mean() + half))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a win rate — used on the switch-wins-or-not counts."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (float(mid - half), float(mid + half))


# --------------------------------------------------------------------------- #
# The tracking-difference estimator
# --------------------------------------------------------------------------- #
def log_ratio(liquid: pd.Series, cheap: pd.Series) -> pd.Series:
    """``log(cheap / liquid)`` on the common index — the cumulative TD path (in logs)."""
    common = liquid.dropna().index.intersection(cheap.dropna().index)
    return np.log(cheap.reindex(common) / liquid.reindex(common)).sort_index().rename("log_ratio")


def _period_td_bp(lr: pd.Series, by: str, complete_only: bool = True) -> pd.Series:
    """Per-period change of the log ratio, in bp. ``by`` is 'year' or 'month'.

    The periods are **chained**, not measured inside each period: each entry is the change
    from the *last* observation of the previous period to the last observation of this one.
    That matters more than it sounds. A wrapper's tracking difference does not arrive
    smoothly — it lands in discrete steps on distribution dates, which cluster at quarter
    ends. Measuring first-to-last *within* each period would silently drop every
    period-boundary gap, i.e. exactly the days that carry the signal, and the monthly and
    annual estimators would then disagree with the cumulative one. Chaining makes the three
    arithmetically consistent: the mean of the chained periods, annualised, *is* the
    cumulative drift.

    The first period is consumed as the anchor by the differencing (so a partial opening
    period cannot bias the average), and ``complete_only`` drops the final period when it is
    a partial calendar year or month — the as-of rule.
    """
    idx = lr.index
    if by == "year":
        keys = idx.year
    elif by == "month":
        keys = idx.year * 100 + idx.month
    else:  # pragma: no cover - guarded by the callers
        raise ValueError("by must be 'year' or 'month'")

    last = lr.groupby(keys).last()
    out = (last.diff().dropna() * 1e4).rename("td_bp")
    if complete_only and len(out):
        tail = idx[keys == last.index[-1]]
        if by == "year":
            incomplete = tail[-1].month < 12
        else:
            incomplete = (tail[-1] + pd.Timedelta(days=7)).month == tail[-1].month
        if incomplete:
            out = out.iloc[:-1]
    return out.sort_index()


def annual_td_table(liquid: pd.Series, cheap: pd.Series) -> pd.Series:
    """Complete-calendar-year tracking difference (cheap - liquid) in bp/yr."""
    return _period_td_bp(log_ratio(liquid, cheap), by="year")


def monthly_td_table(liquid: pd.Series, cheap: pd.Series) -> pd.Series:
    """Complete-calendar-month tracking difference (cheap - liquid) in bp/month."""
    return _period_td_bp(log_ratio(liquid, cheap), by="month")


def stub_decomposition(liquid: pd.Series, cheap: pd.Series) -> dict:
    """Split the window's total log drift into opening stub + complete years + closing stub.

    The annual estimator keeps only the middle term. This function shows what it threw
    away, which is the study's main data-quality exhibit: on the real common window the two
    stubs carry −15 to +24 bp against an annual signal of 5 to 7 bp/yr, and the *placebo*
    pair — two funds that charge the same 3 bp — has the biggest closing stub of the four.
    That is a level artefact in a single closing print, not a fee.
    """
    lr = log_ratio(liquid, cheap)
    ann = annual_td_table(liquid, cheap)
    if not len(ann) or not len(lr):
        return {"total_bp": float("nan"), "head_bp": float("nan"),
                "years_bp": float("nan"), "tail_bp": float("nan"), "n_years": 0}
    first_full, last_full = int(ann.index[0]), int(ann.index[-1])
    anchor = lr[lr.index.year == first_full - 1]
    close = lr[lr.index.year == last_full]
    head = float(anchor.iloc[-1] - lr.iloc[0]) * 1e4 if len(anchor) else 0.0
    tail = float(lr.iloc[-1] - close.iloc[-1]) * 1e4 if len(close) else 0.0
    return {
        "total_bp": float(lr.iloc[-1] - lr.iloc[0]) * 1e4,
        "head_bp": head, "years_bp": float(ann.sum()), "tail_bp": tail,
        "n_years": int(len(ann)), "first_year": first_full, "last_year": last_full,
    }


def sign_count(annual_bp: pd.Series) -> tuple[int, int]:
    """``(years positive, years total)`` — the study's one distribution-free statement.

    Five complete years all positive is a one-sided sign-test *p* of 1/32 = 0.031 and owes
    nothing to the annual dispersion, the bootstrap or the *t* approximation.
    """
    a = pd.Series(annual_bp).dropna()
    return int((a > 0).sum()), int(len(a))


def bootstrap_td_ci(
    annual_bp: pd.Series,
    n_boot: int = 2000,
    block: int = 2,
    seed: int = 920,
    alpha: float = 0.05,
) -> dict:
    """Circular block bootstrap CI for the mean of the **annual** TD series (bp/yr).

    The bootstrap runs on complete years, not months, and the reason is arithmetic rather
    than taste. Chained monthly tracking differences are dominated by *level* noise in the
    closing prints, which telescopes: month-to-month TDs are strongly negatively
    autocorrelated, so twelve of them sum to far less than sqrt(12) times one of them. On
    QQQ/QQQM the monthly series has a standard deviation of ~10 bp while the annual series
    — the sum of twelve of them — has one of ~3 bp. Any monthly-frequency interval
    therefore over-states the uncertainty by roughly an order of magnitude. Blocks of
    ``block`` consecutive years keep whatever slow drift survives at the annual scale.

    **Read this interval with care, and never as the pessimistic case.** A percentile block
    bootstrap resamples the empirical distribution of ``n`` annual numbers; at the ``n = 5``
    of the common window that distribution is five points, and the resulting interval is
    materially too narrow — IVV/SPY bootstraps to [+3.55, +7.90] where the Student-*t*
    interval on the same five numbers is [+0.28, +11.28]. The study reports both, quotes
    the *t* interval whenever it states how bad things could be, and uses the bootstrap
    only as a same-shape comparison across pairs. See :func:`t_interval`.
    """
    r = np.asarray(pd.Series(annual_bp).dropna(), dtype=float)
    n = r.size
    if n < 3:
        return {"td_bp_yr": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_le_zero": float("nan"), "n_years": int(n), "n_boot": 0, "block": int(block)}
    block = max(1, min(int(block), n - 1))
    point = float(r.mean())
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = r[idx].mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "td_bp_yr": point, "ci_low": float(lo), "ci_high": float(hi),
        "frac_le_zero": float((boots <= 0).mean()), "n_years": int(n),
        "n_boot": int(n_boot), "block": int(block),
    }


def td_estimate(
    liquid: pd.Series,
    cheap: pd.Series,
    n_boot: int = 2000,
    seed: int = 920,
) -> dict:
    """Estimate the annual tracking-difference advantage of ``cheap`` over ``liquid``.

    Returns the cumulative, annual-mean and monthly-HAC estimators of the same number
    (bp/yr, cheap minus liquid), the annual dispersion, the block-bootstrap CI, and the
    **power** figure that matters more than any of them: how many complete years of tape a
    gap of this size would need before a one-sample *t* on annual TDs reached 2.
    """
    lr = log_ratio(liquid, cheap)
    n_days = int(len(lr))
    years = n_days / TRADING_DAYS
    cum = float(lr.iloc[-1] - lr.iloc[0]) * 1e4 / years if years > 0 else float("nan")

    ann = annual_td_table(liquid, cheap)
    mon = monthly_td_table(liquid, cheap)

    ann_mean = float(ann.mean()) if len(ann) else float("nan")
    ann_sd = float(ann.std(ddof=1)) if len(ann) > 1 else float("nan")
    t_ann = one_sample_t(ann.to_numpy()) if len(ann) > 1 else float("nan")
    t_mon = newey_west_t(mon.to_numpy(), lags=12) if len(mon) > 3 else float("nan")

    # Robust sensitivities. Yahoo's adjusted-close tape occasionally mis-times a
    # distribution, which lands as a 40-60 bp step in a single year of a series whose
    # signal is 5 bp — an order of magnitude bigger than the thing being measured. The
    # median and the min/max-trimmed mean say what the estimate is worth without the worst
    # of those artefacts; where they disagree with the mean, the mean is the suspect one.
    ann_median = float(ann.median()) if len(ann) else float("nan")
    if len(ann) >= 5:
        trimmed = ann.drop(index=[ann.idxmin(), ann.idxmax()])
        ann_trim = float(trimmed.mean())
        t_trim = one_sample_t(trimmed.to_numpy())
    else:
        ann_trim, t_trim = float("nan"), float("nan")

    ci = bootstrap_td_ci(ann, n_boot=n_boot, seed=seed)
    t_lo, t_hi = t_interval(ann.to_numpy()) if len(ann) > 1 else (float("nan"), float("nan"))
    pos, tot = sign_count(ann)

    return {
        "start": lr.index[0], "end": lr.index[-1], "n_days": n_days, "years": years,
        "td_cum_bp_yr": cum,
        "t_ci_low": t_lo, "t_ci_high": t_hi,
        "years_positive": pos, "years_counted": tot,
        "td_ann_bp_yr": ann_mean, "td_ann_sd_bp": ann_sd, "n_years": int(len(ann)),
        "t_annual": t_ann,
        "td_ann_median_bp": ann_median, "td_ann_trimmed_bp": ann_trim, "t_trimmed": t_trim,
        "worst_year": (int(ann.abs().idxmax()), float(ann.loc[ann.abs().idxmax()]))
        if len(ann) else None,
        "td_mon_bp_yr": float(mon.mean() * MONTHS) if len(mon) else float("nan"),
        "t_monthly_hac": t_mon, "n_months": int(len(mon)),
        "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "frac_le_zero": ci["frac_le_zero"],
        "years_to_detect": years_to_detect(ann_mean, ann_sd),
        "annual": ann, "monthly": mon,
    }


def years_to_detect(gap_bp_yr: float, sd_bp_yr: float, target_t: float = 2.0) -> float:
    """Complete years of tape needed for a one-sample *t* on annual TDs to reach ``target_t``.

    ``n = (target_t * sd / gap) ** 2``. This is the study's sharpest instrument: it turns
    "we could not measure the fee gap" into "you would have needed N years of this fund's
    life to measure it", which is a statement about the tape, not about the fund.
    """
    if not np.isfinite(gap_bp_yr) or not np.isfinite(sd_bp_yr) or abs(gap_bp_yr) < 1e-12:
        return float("inf")
    return float((target_t * sd_bp_yr / abs(gap_bp_yr)) ** 2)


# --------------------------------------------------------------------------- #
# The break-even curve
# --------------------------------------------------------------------------- #
def breakeven_days(td_bp_yr: float, delta_rt_spread_bps: float) -> float:
    """Trading days you must hold before the cheap wrapper repays its wider round trip.

    ``H* = delta_spread / td * 252``. Returns ``inf`` when the tracking-difference
    advantage is non-positive (the cheap wrapper never repays the spread) and 0 when the
    cheap wrapper is *also* the tighter quote.
    """
    if delta_rt_spread_bps <= 0:
        return 0.0
    if not np.isfinite(td_bp_yr) or td_bp_yr <= 0:
        return float("inf")
    return float(delta_rt_spread_bps / td_bp_yr * TRADING_DAYS)


def breakeven_curve(
    est: dict,
    spreads_bps=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0),
) -> pd.DataFrame:
    """Break-even holding period against a sweep of round-trip spread *differentials*.

    Days from the point estimate and from each end of *two* intervals: the block bootstrap
    and the small-sample Student-*t*. An interval's *low* end gives the pessimistic (long,
    possibly infinite) break-even, its high end the optimistic one. When the low end is
    <= 0 the break-even is unbounded and the honest answer is "the tape cannot promise the
    cheap wrapper ever repays".

    ``days_t_low_td`` is the column to quote for a pessimistic case. On the common window
    the bootstrap says a 1 bp differential is repaid within 53-73 days even at its bad end,
    while the *t* interval says 74 days (QQQM/QQQ), 208 (VOO/SPY) and 900 (IVV/SPY) — the
    difference between "two and a half months" and "three and a half years", and the *t*
    interval is the one that respects five degrees of freedom.
    """
    rows = []
    for s in spreads_bps:
        rows.append({
            "delta_rt_spread_bps": float(s),
            "days_point": breakeven_days(est["td_ann_bp_yr"], s),
            "days_ci_low_td": breakeven_days(est["ci_low"], s),
            "days_ci_high_td": breakeven_days(est["ci_high"], s),
            "days_t_low_td": breakeven_days(est.get("t_ci_low", float("nan")), s),
            "days_t_high_td": breakeven_days(est.get("t_ci_high", float("nan")), s),
        })
    df = pd.DataFrame(rows).set_index("delta_rt_spread_bps")
    df["months_point"] = df["days_point"] / (TRADING_DAYS / 12.0)
    return df


def stated_fee_gap_bps(liquid: str, cheap: str, stated_er_bps: dict) -> float:
    """The PROSPECTUS fee gap (liquid - cheap) in bp/yr — an ASSUMPTION, never a measurement.

    Reported next to the realised tracking difference so the reader can see how much of
    the stated fee actually showed up on the tape.
    """
    return float(stated_er_bps[liquid] - stated_er_bps[cheap])


# --------------------------------------------------------------------------- #
# The lived test — overlapping holding-period races on the tape
# --------------------------------------------------------------------------- #
def holding_period_edge(
    liquid: pd.Series,
    cheap: pd.Series,
    hold_days: int,
    delta_rt_spread_bps: float = 1.0,
    lag: int = 1,
) -> dict:
    """Race the two wrappers over every overlapping ``hold_days`` window, with one lag.

    The wrapper choice is made with information through the close of day ``t``; both
    candidate positions are entered at the close of day ``t + lag`` and exited
    ``hold_days`` later. The cheap wrapper is charged the extra round-trip spread
    ``delta_rt_spread_bps`` once, at entry — this is the *whole* difference in execution
    cost between the two choices, since you were going to pay one round trip either way.

    Returns the mean edge in bp per window, the win rate with a Wilson interval, and a
    Newey-West *t* on the overlapping edges (lags = ``hold_days``, the overlap length).

    **The HAC *t* here is only trustworthy at short horizons.** Newey-West needs the
    bandwidth to be small relative to the sample; with ``hold_days = 63`` on ~1,400 daily
    windows the ratio is 0.05 and the *t* means what it says, but at ``hold_days = 504`` on
    929 windows it is 0.54 — those 929 windows contain barely two independent draws, and
    the resulting *t* (+4.3, +6.7) is an artefact of the bandwidth, not evidence. The study
    quotes the 63-day row as its significance claim and treats every row past 252 days as
    descriptive only. The win rate and its Wilson interval have the same problem for the
    same reason: overlapping windows are not independent trials.
    """
    common = liquid.dropna().index.intersection(cheap.dropna().index)
    l = liquid.reindex(common).sort_index()
    c = cheap.reindex(common).sort_index()
    n = len(l)
    h = int(hold_days)
    if n < lag + h + 2:
        return {"hold_days": h, "n_windows": 0, "mean_edge_bp": float("nan"),
                "win_rate": float("nan"), "wilson_low": float("nan"),
                "wilson_high": float("nan"), "t_hac": float("nan")}

    lv = l.to_numpy()
    cv = c.to_numpy()
    entry = np.arange(lag, n - h)
    exit_ = entry + h
    r_liquid = lv[exit_] / lv[entry] - 1.0
    r_cheap = cv[exit_] / cv[entry] - 1.0
    edge = (r_cheap - r_liquid) * 1e4 - float(delta_rt_spread_bps)

    wins = int((edge > 0).sum())
    lo, hi = wilson_interval(wins, len(edge))
    return {
        "hold_days": h,
        "n_windows": int(len(edge)),
        "mean_edge_bp": float(edge.mean()),
        "median_edge_bp": float(np.median(edge)),
        "win_rate": float(wins / len(edge)),
        "wilson_low": lo, "wilson_high": hi,
        "t_hac": newey_west_t(edge, lags=h),
    }


def holding_period_table(
    liquid: pd.Series,
    cheap: pd.Series,
    horizons=(5, 21, 63, 126, 252, 504, 1260),
    delta_rt_spread_bps: float = 1.0,
    lag: int = 1,
) -> pd.DataFrame:
    """:func:`holding_period_edge` across a ladder of holding periods — the empirical curve."""
    rows = [holding_period_edge(liquid, cheap, h, delta_rt_spread_bps, lag) for h in horizons]
    return pd.DataFrame(rows).set_index("hold_days")


def empirical_breakeven(
    liquid: pd.Series,
    cheap: pd.Series,
    delta_rt_spread_bps: float = 1.0,
    horizons=(5, 10, 21, 42, 63, 126, 189, 252, 378, 504, 756, 1008, 1260),
    lag: int = 1,
) -> dict:
    """First holding period at which the mean edge turns positive, and at which it clears |t| = 2.

    The *analytic* break-even divides one number by another; this one walks the tape and
    asks when the average window actually stopped losing. They should agree; where they do
    not, the tape wins.
    """
    tbl = holding_period_table(liquid, cheap, horizons, delta_rt_spread_bps, lag)
    pos = tbl.index[tbl["mean_edge_bp"] > 0]
    sig = tbl.index[(tbl["mean_edge_bp"] > 0) & (tbl["t_hac"] >= 2.0)]
    maj = tbl.index[tbl["win_rate"] > 0.5]
    return {
        "delta_rt_spread_bps": float(delta_rt_spread_bps),
        "first_positive_days": int(pos[0]) if len(pos) else None,
        "first_majority_days": int(maj[0]) if len(maj) else None,
        "first_significant_days": int(sig[0]) if len(sig) else None,
        "table": tbl,
    }


# --------------------------------------------------------------------------- #
# The long/short version — where the borrow rate lives
# --------------------------------------------------------------------------- #
def long_short_harvest(
    liquid: pd.Series,
    cheap: pd.Series,
    borrow_bps_yr: float = 30.0,
    cost_bps_one_way: float = 1.0,
    lag: int = 1,
) -> dict:
    """Long the cheap wrapper, short the liquid one, hold to the end: the pure TD harvest.

    This is the only construction in the study with a short leg, so it is the only one that
    pays **borrow**: ``borrow_bps_yr`` basis points a year on the short notional, accrued
    daily, plus one-way costs on all four legs (in and out, two legs). One execution lag:
    the pair is put on at the close of day ``t + lag``.

    ``borrow_bps_yr`` is an ASSUMPTION — a daily-close tape carries no borrow curve. It is
    swept by :func:`borrow_sweep`, and the sweep is the point: a fee gap of a few basis
    points a year cannot survive a borrow cost an order of magnitude larger.
    """
    common = liquid.dropna().index.intersection(cheap.dropna().index)
    l = liquid.reindex(common).sort_index().iloc[lag:]
    c = cheap.reindex(common).sort_index().iloc[lag:]
    r_l = l.pct_change().dropna()
    r_c = c.pct_change().dropna()
    idx = r_l.index.intersection(r_c.index)
    gross = (r_c.reindex(idx) - r_l.reindex(idx))
    borrow_daily = borrow_bps_yr * 1e-4 / TRADING_DAYS
    net = gross - borrow_daily
    n = len(net)
    years = n / TRADING_DAYS
    round_trip = 4.0 * cost_bps_one_way * 1e-4  # two legs, in and out
    total_net_bp = float(net.sum() * 1e4 - round_trip * 1e4)
    return {
        "n_days": int(n), "years": float(years),
        "gross_bp_yr": float(gross.mean() * TRADING_DAYS * 1e4),
        "net_bp_yr": float(total_net_bp / years) if years > 0 else float("nan"),
        "borrow_bps_yr": float(borrow_bps_yr),
        "vol_bp_yr": float(gross.std(ddof=1) * np.sqrt(TRADING_DAYS) * 1e4),
        "sharpe": float(net.mean() / net.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if net.std(ddof=1) > 0 else float("nan"),
        "t_hac": newey_west_t(net.to_numpy(), lags=21),
    }


def borrow_sweep(liquid: pd.Series, cheap: pd.Series,
                 borrow_grid=(0.0, 10.0, 25.0, 50.0, 100.0),
                 cost_bps_one_way: float = 1.0) -> pd.DataFrame:
    """Net TD harvest of the long-cheap / short-liquid pair across a borrow-rate sweep."""
    rows = [long_short_harvest(liquid, cheap, b, cost_bps_one_way) for b in borrow_grid]
    return pd.DataFrame(rows).set_index("borrow_bps_yr")


# --------------------------------------------------------------------------- #
# The Sharpe race (excess-of-cash vs excess-of-cash) — and why it is the wrong tool
# --------------------------------------------------------------------------- #
def excess_sharpe_race(liquid: pd.Series, cheap: pd.Series, cash: pd.Series) -> dict:
    """Excess-of-cash annualised Sharpe of each wrapper, and the difference.

    Both arms are measured against the *same* cash leg (BIL's total return), so the cash
    leg cancels in the difference and the Sharpe gap is a pure wrapper-vs-wrapper number.
    Included for completeness and to make a point: a 6 bp/yr fee gap sits roughly three
    orders of magnitude below the annual vol of an equity index, so the Sharpe difference
    is numerically zero and the Sharpe race is the *wrong instrument* for this question.
    """
    idx = liquid.dropna().index.intersection(cheap.dropna().index).intersection(cash.dropna().index)
    r_l = liquid.reindex(idx).pct_change()
    r_c = cheap.reindex(idx).pct_change()
    r_cash = cash.reindex(idx).pct_change()
    e_l = (r_l - r_cash).dropna()
    e_c = (r_c - r_cash).dropna()
    common = e_l.index.intersection(e_c.index)
    e_l, e_c = e_l.reindex(common), e_c.reindex(common)

    def _sh(x):
        sd = x.std(ddof=1)
        return float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")

    diff = (e_c - e_l).dropna()
    return {
        "n_days": int(len(common)),
        "sharpe_liquid": _sh(e_l), "sharpe_cheap": _sh(e_c),
        "sharpe_diff": _sh(e_c) - _sh(e_l),
        "t_diff_hac": newey_west_t(diff.to_numpy(), lags=21),
        "vol_liquid_pct": float(e_l.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100),
    }


# --------------------------------------------------------------------------- #
# Era cut
# --------------------------------------------------------------------------- #
def era_cut(liquid: pd.Series, cheap: pd.Series, split: str, n_boot: int = 500) -> dict:
    """Re-estimate the tracking difference on each side of ``split``.

    A structural fee gap should print with the same sign in both halves; a measurement
    artefact should not.
    """
    out: dict[str, dict | None] = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        l = liquid.loc[sl].dropna()
        c = cheap.loc[sl].dropna()
        if len(l.index.intersection(c.index)) < 2 * TRADING_DAYS:
            out[tag] = None
            continue
        est = td_estimate(l, c, n_boot=n_boot)
        out[tag] = {k: est[k] for k in
                    ("start", "end", "n_days", "n_years", "td_cum_bp_yr", "td_ann_bp_yr",
                     "td_ann_sd_bp", "t_annual", "t_monthly_hac", "ci_low", "ci_high")}
    return out


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, n_boot: int = 500, seed: int = 920) -> dict:
    """Run the TD estimator on a synthetic (liquid, cheap, cash) tape.

    On a planted gap the estimator must recover it (right sign, right magnitude); on the
    zero-gap null it must sit at zero with a CI that straddles it.
    """
    est = td_estimate(prices["liquid"], prices["cheap"], n_boot=n_boot, seed=seed)
    return {
        "td_ann_bp_yr": est["td_ann_bp_yr"],
        "td_cum_bp_yr": est["td_cum_bp_yr"],
        "td_ann_sd_bp": est["td_ann_sd_bp"],
        "t_annual": est["t_annual"],
        "t_monthly_hac": est["t_monthly_hac"],
        "ci_low": est["ci_low"], "ci_high": est["ci_high"],
        "n_years": est["n_years"], "n_days": est["n_days"],
    }


def panel_calibration(panel: dict, truths: dict, n_boot: int = 300, seed: int = 920) -> pd.DataFrame:
    """Recovered vs planted gap across a synthetic panel — a calibration check, not a stamp.

    The estimator should sit on the 45-degree line: recovering 6 bp when 6 bp were planted
    and 0 bp when none were.
    """
    rows = []
    for key, px in panel.items():
        d = synthetic_detect(px, n_boot=n_boot, seed=seed)
        rows.append({
            "pair": key,
            "planted_bp_yr": truths["pairs"][key]["planted_gap_bp_yr"],
            "recovered_bp_yr": d["td_ann_bp_yr"],
            "t_annual": d["t_annual"],
            "ci_low": d["ci_low"], "ci_high": d["ci_high"],
        })
    return pd.DataFrame(rows).set_index("pair")
