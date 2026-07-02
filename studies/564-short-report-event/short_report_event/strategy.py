"""Strategy + inference for Study 564 — Short-Report-Event (the short-side cousin of 390).

The folklore, made precise. Around each activist *short-seller* report we measure two distinct
objects, because they have completely different tradability:

  * **The report-day crater** — the target's return on the report day (day 0) and the
    day-0->+1 move. This is the allegation being priced in; you cannot capture it unless you
    were already short before the report dropped (you weren't), so it is *not* a strategy — it
    is the efficient-market re-pricing. We report it to show the news is real (and, for a short
    report, it should be *negative*).

  * **The post-report drift** — what you get if you **short** at the close *one day after* the
    report (a realistic, no-look-ahead entry) and hold 1 / 3 / 6 months, measured **in excess
    of SPY** over the same window (so we race the target against the market, not against cash).
    This is the only leg you can actually trade, and it is the one the "the stock keeps
    falling" claim lives or dies on.

We report everything from the perspective of the **excess return of the target vs SPY**
(``excess = target_ret - spy_ret``). For a short report the claim is ``excess < 0`` (the target
*underperforms* after the report). The **short book's P&L** is then ``-excess`` minus a borrow
cost (shorting a freshly-crashed, hard-to-borrow name is expensive — the borrow is punitive).

Inference on the drift leg (the Signal axis):

  * a **Welch t** of the mean excess against zero (no-drift null) — a *negative* t at |t| >= 2
    would be a real post-report underperformance;
  * a **placebo / bootstrap** null — for each event, draw the *same* horizon excess from a
    *random* date on the same target, repeat thousands of times, and ask how often a
    random-date basket is at least as *negative* as the real report basket (the honest
    small-sample test, controlling for the targets' own drift);
  * a **hit-rate** (P[excess < 0], i.e. the short worked) vs the 50% coin-flip base rate;
  * **one-day execution lag** baked into the entry, and **one-way costs + a short borrow**
    applied to the short-the-drift trade.

The decisive number is not the report-day crater (it is real and negative — the allegation
matters) but the *post-report excess drift you can short*: across a basket selected because it
is famous, is the average post-report underperformance bigger (more negative) than what random
dates on the same names deliver?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = {1: 21, 3: 63, 6: 126}        # months -> trading days


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pos_on_or_after(index: pd.DatetimeIndex, date: pd.Timestamp) -> int | None:
    """Integer position of the first index date >= ``date`` (None if past the tape)."""
    i = index.searchsorted(date, side="left")
    if i >= len(index):
        return None
    return int(i)


# --------------------------------------------------------------------------- #
# Report-day crater (day-0 effect — reported, not tradable)
# --------------------------------------------------------------------------- #
def report_day_moves(prices: pd.DataFrame, table: pd.DataFrame) -> np.ndarray:
    """Day-0 report return of each target (the allegation being priced in).

    Day 0 = the first trading day on/after the report date. The move is that day's simple
    return (close vs prior close). For a short report this should be *negative*. Events whose
    target has no usable window dropped.
    """
    out = []
    for _, r in table.iterrows():
        tkr = r["ticker"]
        if tkr not in prices.columns:
            continue
        s = prices[tkr].dropna()
        i = _pos_on_or_after(s.index, r["report"])
        if i is None or i < 1 or i >= len(s):
            continue
        out.append(s.iloc[i] / s.iloc[i - 1] - 1.0)
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Post-report drift (excess of SPY) — the tradable (short) leg
# --------------------------------------------------------------------------- #
def event_excess_drift(prices: pd.DataFrame, table: pd.DataFrame, months: int,
                       lag: int = 1) -> np.ndarray:
    """Excess-of-SPY return of the target if you enter ``lag`` days after the report and hold
    ``months`` months (no look-ahead).

    For each event: entry = the close ``lag`` trading days after day 0; exit = ``horizon``
    days later. Excess = target return - SPY return over the identical calendar window. A
    *negative* excess is the short working (the target underperformed the market). Events whose
    horizon overruns the target's or SPY's tape are dropped.
    """
    h = TRADING_DAYS[months]
    spy = prices["SPY"].dropna()
    out = []
    for _, r in table.iterrows():
        tkr = r["ticker"]
        if tkr not in prices.columns:
            continue
        s = prices[tkr].dropna()
        i0 = _pos_on_or_after(s.index, r["report"])
        if i0 is None:
            continue
        entry = i0 + lag
        exit_ = entry + h
        if exit_ >= len(s):
            continue
        d_entry, d_exit = s.index[entry], s.index[exit_]
        je = _pos_on_or_after(spy.index, d_entry)
        jx = _pos_on_or_after(spy.index, d_exit)
        if je is None or jx is None or jx >= len(spy):
            continue
        tgt_ret = s.iloc[exit_] / s.iloc[entry] - 1.0
        spy_ret = spy.iloc[jx] / spy.iloc[je] - 1.0
        out.append(tgt_ret - spy_ret)
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t_vs_zero(sample: np.ndarray) -> float:
    """One-sample t of ``mean(sample)`` against 0 (the no-drift null). NaN if n < 2.

    On the excess-drift sample a *negative* t means the target underperformed after the report
    (the short worked); the claim needs t <= -2.
    """
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float(sample.mean() / se)


def placebo_pvalue(prices: pd.DataFrame, table: pd.DataFrame, months: int,
                   n_draws: int = 20_000, lag: int = 1, seed: int = 564) -> dict:
    """Small-sample placebo null controlling for the targets' own drift.

    For each event, the alternative-history return is the same ``months``-horizon excess
    (target - SPY) measured from a *random* valid entry date on the **same target**. We draw a
    full basket of such random-date excess returns ``n_draws`` times and ask how often the
    random basket's mean is at least as **negative** as the real report basket. This nulls out
    the fact that short-report targets are special names (jumpy, often already falling) — we ask
    specifically whether the *report timing* adds extra underperformance over a random day on
    that name.
    """
    h = TRADING_DAYS[months]
    spy = prices["SPY"].dropna()
    pools = []
    obs = event_excess_drift(prices, table, months, lag=lag)
    for _, r in table.iterrows():
        tkr = r["ticker"]
        if tkr not in prices.columns:
            continue
        s = prices[tkr].dropna()
        i0 = _pos_on_or_after(s.index, r["report"])
        if i0 is None:
            continue
        entry0 = i0 + lag
        if entry0 + h >= len(s):
            continue
        common = s.index.intersection(spy.index)
        sc = s.reindex(common)
        spc = spy.reindex(common)
        n = len(common)
        if n <= h + 2:
            continue
        tgt_fwd = sc.values[h:] / sc.values[:-h] - 1.0
        spy_fwd = spc.values[h:] / spc.values[:-h] - 1.0
        pool = tgt_fwd - spy_fwd
        pool = pool[np.isfinite(pool)]
        if len(pool):
            pools.append(pool)
    k = len(obs)
    if k == 0 or not pools:
        return {"k": 0, "obs_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        means[i] = np.mean([p[rng.integers(0, len(p))] for p in pools])
    obs_mean = float(obs.mean())
    # one-sided toward the claim: random basket at least as NEGATIVE as observed
    p = float((means <= obs_mean).mean())
    return {"k": k, "obs_mean": obs_mean, "placebo_mean": float(means.mean()),
            "p_value": p}


def summarize(prices: pd.DataFrame, table: pd.DataFrame, months: int,
              lag: int = 1) -> dict:
    """Headline stats for one horizon: n, mean/median excess drift, hit-rate (excess<0) vs 50%,
    Welch t vs zero, and the placebo p-value."""
    ex = event_excess_drift(prices, table, months, lag=lag)
    pl = placebo_pvalue(prices, table, months, lag=lag)
    return {
        "months": months,
        "n": int(len(ex)),
        "mean_excess": float(ex.mean()) if len(ex) else float("nan"),
        "median_excess": float(np.median(ex)) if len(ex) else float("nan"),
        "hit_rate": float((ex < 0).mean()) if len(ex) else float("nan"),
        "t": welch_t_vs_zero(ex),
        "p_placebo": pl["p_value"],
    }


def short_net_of_costs(prices: pd.DataFrame, table: pd.DataFrame, months: int,
                       cost_bps: float = 10.0, borrow_ann_bps: float = 800.0,
                       lag: int = 1) -> dict:
    """Short-the-drift book: enter one day after the report, hold ``months``, pay costs + borrow.

    The short book's P&L is ``-excess`` (you profit when the target underperforms SPY). We charge
    one round-trip cost (enter + exit) plus a **punitive annual borrow** pro-rated over the hold:
    a freshly-crashed, allegation-hit name is exactly the hardest, most-expensive borrow (800
    bps/yr is a realistic-to-generous small-cap short fee — many of these names traded far higher).

    Returns average gross and net short P&L (positive = the short made money).
    """
    ex = event_excess_drift(prices, table, months, lag=lag)
    if not len(ex):
        return {"n_trades": 0, "gross_short": float("nan"), "net_short": float("nan"),
                "cost_bps": cost_bps, "borrow_ann_bps": borrow_ann_bps}
    gross_short = -ex                                  # short P&L before costs
    c = cost_bps / 1e4
    borrow = borrow_ann_bps / 1e4 * (months / 12.0)
    net_short = gross_short - c - borrow
    return {
        "n_trades": int(len(ex)),
        "gross_short": float(gross_short.mean()),
        "net_short": float(net_short.mean()),
        "cost_bps": cost_bps,
        "borrow_ann_bps": borrow_ann_bps,
    }
