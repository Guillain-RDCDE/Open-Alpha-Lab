"""Strategy + inference for Study 639 — Gasoline RVP Seasonality.

The claim: every spring, US gasoline specs switch to the expensive summer blend on a **dated
regulatory calendar** (40 CFR 1090.215 — summer-RVP fuel at refineries/terminals by **May 1**,
winter blend legal again after **September 15**), so the gasoline-minus-crude spread should
run **up** in Feb–Apr and **down** in Sep — a seasonal with an actual law behind it.

Measurement:

  * **The crack seasonal (Signal).** Monthly log-excess of gasoline over crude,
    ``log(1+r_RB) − log(1+r_CL)`` — a dollar-neutral, monthly-rebalanced long-gasoline /
    short-crude spread that nets out the oil-price level. Per-year panel: each year contributes
    its Feb–Apr mean monthly excess and its rest-of-year mean monthly excess; the primary test
    is a **Welch t across ~20 yearly pairs** (window years vs rest years), plus a one-sample t
    of the per-year cumulative window excess and a pooled-months Welch t. September gets its own
    one-sample t (expected negative).

  * **The curve test (third axis).** Per-year, same window: cumulative log return of **UGA**
    (a real front-month holder who pays every roll) minus cumulative log return of the
    **spliced** RB=F chain (the spot-proxy that banks the roll markup for free). A paired
    one-sample t on that gap asks: does the futures curve already price the seasonal — does the
    roll eat it for anyone actually holding?

  * **Tradability.** The retail vehicle is UGA: hold it **only** Feb–Apr (one round trip per
    year, one-way costs × NAV), per-year window return panel, one-sample t of the net window
    return. The futures crack's investable estimate = spliced window excess minus the roll gap.

Inference: Welch t for the group split across years, one-sample t for per-year panels (annual
observations of a monthly-rebalanced spread are essentially serially uncorrelated, so the plain
t is the honest statistic; no HAC needed on a 20-point annual panel). Execution lag: the signal
is a **statute** — the calendar is known years in advance; entries use the prior month-end close
(strictly earlier information), which more than satisfies the desk's one-lag rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import RUNUP_MONTHS, SWITCHBACK_MONTH


# --------------------------------------------------------------------------- #
# Basic inference
# --------------------------------------------------------------------------- #
def ttest_vs_zero(sample) -> float:
    """One-sample t of ``sample`` against 0."""
    sample = np.asarray(sample, dtype=float)
    sample = sample[~np.isnan(sample)]
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float(sample.mean() / se)


def welch_t(a, b) -> float:
    """Welch t of mean(a) − mean(b), unequal variances."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / se)


# --------------------------------------------------------------------------- #
# The crack seasonal
# --------------------------------------------------------------------------- #
def excess_series(mret: pd.DataFrame, long: str = "RB=F", short: str = "CL=F") -> pd.Series:
    """Monthly log-excess of ``long`` over ``short`` (both legs must be present)."""
    sub = mret[[long, short]].dropna()
    return (np.log1p(sub[long]) - np.log1p(sub[short])).rename("excess")


def per_year_panel(excess: pd.Series, months: tuple = RUNUP_MONTHS) -> pd.DataFrame:
    """One row per year with a COMPLETE window: window mean/sum + rest-of-year mean.

    ``win_mean``/``rest_mean`` are mean monthly log-excess (window vs the year's other months);
    ``win_sum`` is the cumulative window log-excess. Years missing any window month are dropped
    (no partial windows); the rest-mean needs >= 6 non-window months.
    """
    df = excess.to_frame()
    df["year"] = df.index.year
    df["month"] = df.index.month
    rows = []
    for y, g in df.groupby("year"):
        w = g[g["month"].isin(months)]["excess"]
        r = g[~g["month"].isin(months)]["excess"]
        if len(w) < len(months) or len(r) < 6:
            continue
        rows.append({"year": int(y), "win_mean": float(w.mean()),
                     "win_sum": float(w.sum()), "rest_mean": float(r.mean())})
    return pd.DataFrame(rows).set_index("year")


def window_stats(excess: pd.Series, months: tuple = RUNUP_MONTHS) -> dict:
    """Headline stats for a calendar window on the excess series.

    Primary: Welch t across the per-year panel (window-mean years vs rest-mean years). Also a
    one-sample t of the per-year cumulative window excess, and a pooled-months Welch t.
    """
    pan = per_year_panel(excess, months)
    m = excess.index.month
    in_w = np.isin(m, months)
    pooled_w, pooled_r = excess[in_w].values, excess[~in_w].values
    return {
        "n_years": int(len(pan)),
        "win_mean_pct": float(pan["win_mean"].mean() * 100),
        "rest_mean_pct": float(pan["rest_mean"].mean() * 100),
        "gap_pct": float((pan["win_mean"] - pan["rest_mean"]).mean() * 100),
        "welch_t_years": welch_t(pan["win_mean"], pan["rest_mean"]),
        "win_sum_pct": float(pan["win_sum"].mean() * 100),
        "t_win_sum": ttest_vs_zero(pan["win_sum"]),
        "hit_rate": float((pan["win_sum"] > 0).mean()),
        "welch_t_pooled": welch_t(pooled_w, pooled_r),
        "n_pooled_w": int(in_w.sum()), "n_pooled_r": int((~in_w).sum()),
        "panel": pan,
    }


def single_month_stats(excess: pd.Series, month: int = SWITCHBACK_MONTH) -> dict:
    """Per-year panel for one calendar month (the September switch-back): mean + t."""
    vals = excess[excess.index.month == month]
    return {"n_years": int(len(vals)), "mean_pct": float(vals.mean() * 100),
            "t": ttest_vs_zero(vals.values), "hit_rate": float((vals < 0).mean())}


def month_table(excess: pd.Series) -> pd.DataFrame:
    """Per-calendar-month mean monthly log-excess (%) + one-sample t + n (12 rows)."""
    rows = []
    for mo in range(1, 13):
        vals = excess[excess.index.month == mo].values
        rows.append({"month": mo, "mean_pct": float(np.mean(vals) * 100),
                     "t": ttest_vs_zero(vals), "n": int(len(vals))})
    return pd.DataFrame(rows).set_index("month")


# --------------------------------------------------------------------------- #
# The curve test — holder (UGA) vs spliced chain (RB=F)
# --------------------------------------------------------------------------- #
def roll_gap_panel(mret: pd.DataFrame, months: tuple = RUNUP_MONTHS,
                   holder: str = "UGA", chain: str = "RB=F") -> pd.DataFrame:
    """Per-year window gap: cumulative log return of the HOLDER minus the SPLICED chain.

    The spliced chain banks each roll's price jump for free; the holder pays it (and earns
    collateral interest, and pays the expense ratio). If the futures curve already prices the
    RVP transition, the gap is systematically NEGATIVE inside the run-up window.
    """
    sub = mret[[holder, chain]].dropna()
    lr = np.log1p(sub)
    df = pd.DataFrame({"gap": lr[holder] - lr[chain],
                       "holder": lr[holder], "chain": lr[chain]})
    df["year"] = df.index.year
    df["month"] = df.index.month
    rows = []
    for y, g in df.groupby("year"):
        w = g[g["month"].isin(months)]
        if len(w) < len(months):
            continue
        rows.append({"year": int(y), "gap_sum": float(w["gap"].sum()),
                     "holder_sum": float(w["holder"].sum()),
                     "chain_sum": float(w["chain"].sum())})
    return pd.DataFrame(rows).set_index("year")


def roll_gap_stats(mret: pd.DataFrame, months: tuple = RUNUP_MONTHS,
                   holder: str = "UGA", chain: str = "RB=F") -> dict:
    """Paired one-sample t of the per-year window holder-minus-chain gap."""
    pan = roll_gap_panel(mret, months, holder=holder, chain=chain)
    return {"n_years": int(len(pan)),
            "gap_pct": float(pan["gap_sum"].mean() * 100),
            "t": ttest_vs_zero(pan["gap_sum"]),
            "holder_pct": float(pan["holder_sum"].mean() * 100),
            "chain_pct": float(pan["chain_sum"].mean() * 100),
            "share_negative": float((pan["gap_sum"] < 0).mean()),
            "panel": pan}


# --------------------------------------------------------------------------- #
# Tradability — the UGA seasonal overlay
# --------------------------------------------------------------------------- #
def overlay_stats(mret: pd.DataFrame, months: tuple = RUNUP_MONTHS,
                  ticker: str = "UGA", cost_bps: float = 5.0) -> dict:
    """Hold ``ticker`` ONLY in the window months; one round trip per year.

    Entry at the prior month-end close (the calendar is a statute, known years ahead — the
    desk's one-lag rule is satisfied with room to spare). Gross per-year window simple return;
    net = gross − 2 × one-way cost (long-only, no borrow). One-sample t of the net panel.
    """
    r = mret[ticker].dropna()
    df = r.to_frame("r")
    df["year"] = df.index.year
    df["month"] = df.index.month
    rows = []
    for y, g in df.groupby("year"):
        w = g[g["month"].isin(months)]["r"]
        if len(w) < len(months):
            continue
        rows.append({"year": int(y), "gross": float((1 + w).prod() - 1)})
    pan = pd.DataFrame(rows).set_index("year")
    c = 2.0 * cost_bps / 1e4
    net = pan["gross"] - c
    return {"n_years": int(len(pan)), "cost_bps": cost_bps,
            "gross_pct": float(pan["gross"].mean() * 100),
            "net_pct": float(net.mean() * 100),
            "t_net": ttest_vs_zero(net.values),
            "hit_rate": float((net > 0).mean()),
            "panel": pan}


def overlay_excess_cash(mret: pd.DataFrame, tbill: pd.Series,
                        months: tuple = RUNUP_MONTHS, ticker: str = "UGA",
                        cost_bps: float = 10.0) -> dict:
    """The overlay's net window return MINUS the same window's T-bill return (excess-vs-cash).

    Sitting in cash outside (and instead of) the window earns the bill, so the honest race
    subtracts the per-year window T-bill return from the net overlay return.
    """
    ov = overlay_stats(mret, months, ticker=ticker, cost_bps=cost_bps)
    pan = ov["panel"]
    tb = tbill.to_frame("tb")
    tb["year"] = tb.index.year
    tb["month"] = tb.index.month
    wtb = tb[tb["month"].isin(months)].groupby("year")["tb"].sum()
    j = pan.join(wtb.rename("tb"), how="inner").dropna()
    xs = j["gross"] - 2.0 * cost_bps / 1e4 - j["tb"]
    return {"n_years": int(len(xs)), "cost_bps": cost_bps,
            "excess_pct": float(xs.mean() * 100),
            "t": ttest_vs_zero(xs.values),
            "hit_rate": float((xs > 0).mean())}
