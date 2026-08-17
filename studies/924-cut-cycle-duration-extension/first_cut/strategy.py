"""Event study + inference for Study 924 — First Cut.

The trade under test: the FOMC announces the **first cut of an easing cycle** on day
``t``; you buy long-duration Treasuries (TLT, or IEF for the belly) at the **close of day
t+1** — the study's single, documented execution lag — and hold for 1, 3, 6 or 12 months,
selling at the close of the last session on or before the horizon date. The scorecard is
the **excess of cash**: the duration total return minus BIL's total return over exactly
the same days, so a 12-month hold in 2024 is not flattered by a 5% bill yield it never
earned. Costs are one-way × NAV, charged on the entry and the exit (two legs per event).

Three things are measured, in ascending order of how much they can be trusted:

1. **The per-event table.** Five hardcoded first-cut dates, of which only those with both
   a live ETF tape and a complete forward window survive. With N of that size a
   one-sample *t* is a decoration, and the code says so by reporting N next to it.
2. **The daily conditional series.** A daily excess-of-cash return that is "long duration"
   only inside post-cut windows and flat otherwise. This turns a handful of events into
   thousands of daily observations, which supports a Newey-West *t* and a circular
   block-bootstrap CI — the honest inferential spine of the study.
3. **The controls.** (a) every cut, not just the first of a cycle — if the "first" label
   is doing no work, both tables look alike; (b) a placebo of random pseudo-event dates
   matched in count and horizon, which is what a small-N mean must actually beat.

A long-duration / short-front-end variant (long TLT, short SHY) is also priced, because
the folk version of this trade is "buy duration" and the pure-duration expression is a
curve trade. That leg has a short, so it pays **borrow**, and the borrow rate is swept.

Returns are **simple** (arithmetic) throughout so window returns compound correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
HORIZONS = (1, 3, 6, 12)


# --------------------------------------------------------------------------- #
# Window arithmetic — the one execution lag lives here
# --------------------------------------------------------------------------- #
def window_bounds(index: pd.DatetimeIndex, event: pd.Timestamp, horizon_months: int):
    """Entry / exit timestamps for one event, or ``None`` if the window is incomplete.

    Entry is the **first session strictly after** the announcement day (the one-day
    execution lag: the cut is known at the close of ``t``, the position is opened at the
    close of ``t+1``). Exit is the last session on or before ``event + horizon_months``.
    Returns ``None`` when the tape does not cover the full window — never a truncated one.
    """
    event = pd.Timestamp(event)
    target = event + pd.DateOffset(months=int(horizon_months))
    after = index[index > event]
    if len(after) == 0:
        return None
    entry = after[0]
    before = index[index <= target]
    if len(before) == 0:
        return None
    exit_ = before[-1]
    if exit_ <= entry or index[-1] < target:
        return None
    return entry, exit_


def event_table(
    duration: pd.Series,
    cash: pd.Series,
    events,
    horizon_months: int = 6,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Per-event holding-period returns for one horizon, gross and net, excess-of-cash.

    Columns: ``entry``/``exit`` dates, ``n_days`` held, ``dur_pct`` (the duration leg's
    total return), ``cash_pct`` (BIL's total return over the identical days),
    ``excess_pct`` (gross) and ``excess_net_pct`` (after ``2 × cost_bps`` — one-way on
    the way in and one-way on the way out, charged on NAV). Events whose window is not
    fully covered by the tape are dropped, and the caller is expected to say how many.
    """
    idx = duration.index.intersection(cash.index)
    d = duration.reindex(idx).dropna()
    c = cash.reindex(idx).dropna()
    idx = d.index.intersection(c.index)
    d, c = d.loc[idx], c.loc[idx]
    cost = 2.0 * cost_bps * 1e-4

    rows = []
    for ev in pd.DatetimeIndex(pd.to_datetime(list(events))):
        b = window_bounds(idx, ev, horizon_months)
        if b is None:
            continue
        entry, exit_ = b
        dur_r = float(d.loc[exit_] / d.loc[entry] - 1.0)
        cash_r = float(c.loc[exit_] / c.loc[entry] - 1.0)
        rows.append({
            "event": ev,
            "entry": entry,
            "exit": exit_,
            "n_days": int(idx.get_loc(exit_) - idx.get_loc(entry) + 1),
            "dur_pct": dur_r * 100.0,
            "cash_pct": cash_r * 100.0,
            "excess_pct": (dur_r - cash_r) * 100.0,
            "excess_net_pct": (dur_r - cash_r - cost) * 100.0,
        })
    if not rows:
        return pd.DataFrame(
            columns=["event", "entry", "exit", "n_days", "dur_pct", "cash_pct",
                     "excess_pct", "excess_net_pct"]
        ).set_index("event")
    return pd.DataFrame(rows).set_index("event")


def in_window_mask(index: pd.DatetimeIndex, events, horizon_months: int = 6) -> pd.Series:
    """{0,1} daily flag: 1 on days the post-cut position is open (entry+1 … exit).

    Built from the same :func:`window_bounds` arithmetic, so the daily series and the
    per-event table describe exactly the same trade. Overlapping windows (adjacent cuts
    at the short horizons) are unioned — the position is binary long, never doubled.
    """
    mask = pd.Series(0.0, index=index, name="in_window")
    for ev in pd.DatetimeIndex(pd.to_datetime(list(events))):
        b = window_bounds(index, ev, horizon_months)
        if b is None:
            continue
        entry, exit_ = b
        mask.loc[(index > entry) & (index <= exit_)] = 1.0
    return mask


# --------------------------------------------------------------------------- #
# The daily conditional series (the inferential spine)
# --------------------------------------------------------------------------- #
def conditional_daily(
    duration: pd.Series,
    cash: pd.Series,
    events,
    horizon_months: int = 6,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Daily excess-of-cash returns split into in-window and out-of-window days.

    ``e_dur`` is duration minus cash every day (the unconditional duration premium).
    ``e_event`` is that same series gated by the post-cut window, minus the one-way cost
    charged on each entry and exit day. ``in_window`` is the {0,1} gate. Everything is
    aligned on the common duration∩cash index and dropped of NaNs.
    """
    idx = duration.index.intersection(cash.index)
    d = duration.reindex(idx).sort_index()
    c = cash.reindex(idx).sort_index()
    r_dur = d.pct_change().rename("r_dur")
    r_cash = c.pct_change().rename("r_cash")
    e_dur = (r_dur - r_cash).rename("e_dur")

    mask = in_window_mask(idx, events, horizon_months)
    turns = mask.diff().abs().fillna(mask.iloc[0])
    e_event = (mask * e_dur - turns * cost_bps * 1e-4).rename("e_event")

    out = pd.concat([r_dur, r_cash, e_dur, mask, e_event], axis=1)
    return out.dropna()


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x) -> float:
    """Plain one-sample *t* of mean(x) vs 0 — reported with N, because N here is tiny."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def newey_west_t(x, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) *t* of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def _auto_lags(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def diff_tstat(a: pd.Series, b: pd.Series) -> float:
    """HAC *t* on the daily difference ``a − b`` (do post-cut days beat other days?)."""
    diff = (pd.Series(a) - pd.Series(b)).dropna()
    if len(diff) < 3:
        return float("nan")
    return newey_west_t(diff.to_numpy(), lags=_auto_lags(len(diff)))


def block_bootstrap_mean_ci(
    x,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 924,
    alpha: float = 0.05,
    scale: float = 1.0,
) -> dict:
    """Circular block-bootstrap CI for the mean of a daily series (blocks keep clustering).

    ``scale`` multiplies the point and the interval — pass ``TRADING_DAYS`` to read the
    result as an annualised return rather than a daily mean.
    """
    r = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}
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
        "mean": float(r.mean() * scale),
        "ci_low": float(lo * scale),
        "ci_high": float(hi * scale),
        "frac_negative": float((boots < 0).mean()),
        "n_obs": int(n),
        "n_boot": int(n_boot),
        "block": int(block),
    }


# --------------------------------------------------------------------------- #
# The headline comparison
# --------------------------------------------------------------------------- #
def compare(
    duration: pd.Series,
    cash: pd.Series,
    events,
    horizon_months: int = 6,
    cost_bps: float = 5.0,
) -> dict:
    """Race post-cut days against all other days, both excess-of-cash.

    Reports the per-event table, the in-window vs out-of-window daily means (annualised),
    the HAC *t* on the in-window series and on the in-minus-out difference, the small-N
    one-sample *t* across events, and the hit rate. Nothing here is a Sharpe of a
    tradable fund: the conditional leg is only invested a fraction of the time, so the
    annualised numbers are *conditional* rates, labelled as such.
    """
    tbl = event_table(duration, cash, events, horizon_months, cost_bps)
    daily = conditional_daily(duration, cash, events, horizon_months, cost_bps)

    inw = daily.loc[daily["in_window"] == 1.0, "e_dur"]
    outw = daily.loc[daily["in_window"] == 0.0, "e_dur"]
    n_ev = int(len(tbl))

    excess = tbl["excess_pct"].to_numpy() if n_ev else np.array([])
    excess_net = tbl["excess_net_pct"].to_numpy() if n_ev else np.array([])

    return {
        "table": tbl,
        "daily": daily,
        "n_events": n_ev,
        "horizon_months": int(horizon_months),
        "mean_excess_pct": float(np.mean(excess)) if n_ev else float("nan"),
        "median_excess_pct": float(np.median(excess)) if n_ev else float("nan"),
        "mean_excess_net_pct": float(np.mean(excess_net)) if n_ev else float("nan"),
        "t_events": one_sample_t(excess),
        "t_events_net": one_sample_t(excess_net),
        "hit_rate": float((excess > 0).mean()) if n_ev else float("nan"),
        "in_ann_pct": float(inw.mean() * TRADING_DAYS * 100.0) if len(inw) else float("nan"),
        "out_ann_pct": float(outw.mean() * TRADING_DAYS * 100.0) if len(outw) else float("nan"),
        "t_in_window": newey_west_t(inw.to_numpy(), lags=_auto_lags(max(len(inw), 3))) if len(inw) > 3 else float("nan"),
        "t_in_vs_out": _welch_like(inw, outw),
        "t_event_leg_hac": newey_west_t(daily["e_event"].to_numpy(),
                                        lags=_auto_lags(len(daily))),
        "t_uncond_hac": newey_west_t(daily["e_dur"].to_numpy(), lags=_auto_lags(len(daily))),
        "in_frac": float(daily["in_window"].mean()),
        "n_days": int(len(daily)),
    }


def _welch_like(a: pd.Series, b: pd.Series) -> float:
    """Welch *t* on two daily-return samples (post-cut days vs the rest).

    Deliberately *not* a paired test: the two samples are disjoint sets of days. It is
    also deliberately naive about serial correlation, which is why the HAC *t* on the
    daily conditional leg is the statistic the verdict leans on.
    """
    a = np.asarray(pd.Series(a).dropna(), dtype=float)
    b = np.asarray(pd.Series(b).dropna(), dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Controls — the placebo and the all-cuts calendar
# --------------------------------------------------------------------------- #
def placebo(
    duration: pd.Series,
    cash: pd.Series,
    n_events: int,
    horizon_months: int = 6,
    n_draws: int = 2000,
    cost_bps: float = 5.0,
    seed: int = 924,
    min_date: str | None = None,
) -> dict:
    """Random pseudo-event dates, matched in count and horizon — the small-N null.

    With four or five real events, "the mean was positive" is worth nothing until you
    know how often a random date does as well. Each draw picks ``n_events`` sessions
    uniformly from the eligible range (a full window must fit) and computes the same mean
    excess-of-cash window return. Returns the empirical distribution and a one-sided
    *p*-value slot for the caller to fill via :func:`placebo_pvalue`.
    """
    idx = duration.index.intersection(cash.index)
    d = duration.reindex(idx).dropna()
    c = cash.reindex(idx).dropna()
    idx = d.index.intersection(c.index)
    d, c = d.loc[idx], c.loc[idx]
    if min_date is not None:
        idx = idx[idx >= pd.Timestamp(min_date)]
    eligible = [t for t in idx if window_bounds(d.index, t, horizon_months) is not None]
    eligible = pd.DatetimeIndex(eligible)
    cost = 2.0 * cost_bps * 1e-4
    if len(eligible) < n_events:
        return {"means": np.array([]), "n_eligible": 0}

    # Pre-compute the window return for every eligible date once (vectorised lookup).
    rets = {}
    for t in eligible:
        entry, exit_ = window_bounds(d.index, t, horizon_months)
        rets[t] = float((d.loc[exit_] / d.loc[entry] - 1.0)
                        - (c.loc[exit_] / c.loc[entry] - 1.0) - cost) * 100.0
    vals = np.array([rets[t] for t in eligible])

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(vals), size=(n_draws, n_events))
    means = vals[draws].mean(axis=1)
    return {
        "means": means,
        "n_eligible": int(len(eligible)),
        "all_mean_pct": float(vals.mean()),
        "all_median_pct": float(np.median(vals)),
        "n_events": int(n_events),
        "horizon_months": int(horizon_months),
    }


def placebo_pvalue(observed: float, means: np.ndarray) -> float:
    """One-sided empirical *p*: the share of placebo draws at least as good as observed."""
    means = np.asarray(means, dtype=float)
    if means.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float((means >= observed).mean())


def horizon_sweep(
    duration: pd.Series,
    cash: pd.Series,
    events,
    horizons=HORIZONS,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """The 1 / 3 / 6 / 12-month grid: mean excess, small-N *t*, hit rate, HAC *t*."""
    rows = []
    for h in horizons:
        cmp = compare(duration, cash, events, horizon_months=h, cost_bps=cost_bps)
        rows.append({
            "horizon_m": h,
            "n_events": cmp["n_events"],
            "mean_excess_pct": cmp["mean_excess_pct"],
            "mean_excess_net_pct": cmp["mean_excess_net_pct"],
            "median_excess_pct": cmp["median_excess_pct"],
            "t_events": cmp["t_events"],
            "hit_rate": cmp["hit_rate"],
            "in_ann_pct": cmp["in_ann_pct"],
            "out_ann_pct": cmp["out_ann_pct"],
            "t_in_vs_out": cmp["t_in_vs_out"],
            "t_event_leg_hac": cmp["t_event_leg_hac"],
        })
    return pd.DataFrame(rows).set_index("horizon_m")


# --------------------------------------------------------------------------- #
# The curve expression — long duration, short the front end (this one pays borrow)
# --------------------------------------------------------------------------- #
def long_short_table(
    duration: pd.Series,
    front: pd.Series,
    events,
    horizon_months: int = 6,
    cost_bps: float = 5.0,
    borrow_bps_ann: float = 50.0,
) -> pd.DataFrame:
    """Long duration / short front-end per event, net of costs **and borrow**.

    The dollar-neutral curve trade: long TLT, short SHY, both legs sized 1× NAV. Costs
    are one-way × NAV on four legs (two in, two out) and the short leg accrues
    ``borrow_bps_ann`` annualised on the shorted notional for the days held. Borrow is an
    **ASSUMPTION** — it is not on any free tape — so it is swept in
    :func:`borrow_sweep`.
    """
    idx = duration.index.intersection(front.index)
    d = duration.reindex(idx).dropna()
    f = front.reindex(idx).dropna()
    idx = d.index.intersection(f.index)
    d, f = d.loc[idx], f.loc[idx]
    cost = 4.0 * cost_bps * 1e-4

    rows = []
    for ev in pd.DatetimeIndex(pd.to_datetime(list(events))):
        b = window_bounds(idx, ev, horizon_months)
        if b is None:
            continue
        entry, exit_ = b
        n_days = int(idx.get_loc(exit_) - idx.get_loc(entry) + 1)
        long_r = float(d.loc[exit_] / d.loc[entry] - 1.0)
        short_r = float(f.loc[exit_] / f.loc[entry] - 1.0)
        borrow = borrow_bps_ann * 1e-4 * n_days / TRADING_DAYS
        rows.append({
            "event": ev,
            "n_days": n_days,
            "long_pct": long_r * 100.0,
            "short_pct": short_r * 100.0,
            "ls_gross_pct": (long_r - short_r) * 100.0,
            "ls_net_pct": (long_r - short_r - cost - borrow) * 100.0,
        })
    if not rows:
        return pd.DataFrame(columns=["event", "n_days", "long_pct", "short_pct",
                                     "ls_gross_pct", "ls_net_pct"]).set_index("event")
    return pd.DataFrame(rows).set_index("event")


def borrow_sweep(
    duration: pd.Series,
    front: pd.Series,
    events,
    horizon_months: int = 6,
    cost_bps: float = 5.0,
    borrow_grid=(0.0, 25.0, 50.0, 100.0, 200.0),
) -> pd.DataFrame:
    """Sweep the assumed borrow on the short leg of the curve trade."""
    rows = []
    for b in borrow_grid:
        t = long_short_table(duration, front, events, horizon_months, cost_bps, b)
        rows.append({
            "borrow_bps_ann": b,
            "n_events": int(len(t)),
            "mean_ls_net_pct": float(t["ls_net_pct"].mean()) if len(t) else float("nan"),
            "t_events": one_sample_t(t["ls_net_pct"].to_numpy()) if len(t) else float("nan"),
        })
    return pd.DataFrame(rows).set_index("borrow_bps_ann")


def cost_sweep(
    duration: pd.Series,
    cash: pd.Series,
    events,
    horizon_months: int = 6,
    cost_grid=(0.0, 5.0, 10.0, 25.0),
) -> pd.DataFrame:
    """Sweep the one-way cost on the plain long-duration expression."""
    rows = []
    for c in cost_grid:
        cmp = compare(duration, cash, events, horizon_months=horizon_months, cost_bps=c)
        rows.append({
            "cost_bps": c,
            "mean_excess_net_pct": cmp["mean_excess_net_pct"],
            "t_events_net": cmp["t_events_net"],
            "t_event_leg_hac": cmp["t_event_leg_hac"],
        })
    return pd.DataFrame(rows).set_index("cost_bps")


def era_split(
    duration: pd.Series,
    cash: pd.Series,
    events,
    split: str = "2020-01-01",
    horizon_months: int = 6,
    cost_bps: float = 5.0,
) -> dict:
    """Per-event results either side of ``split`` — with N this small, a sanity read only."""
    tbl = event_table(duration, cash, events, horizon_months, cost_bps)
    out = {}
    for tag, sel in [("early", tbl.index < pd.Timestamp(split)),
                     ("late", tbl.index >= pd.Timestamp(split))]:
        sub = tbl[sel]
        out[tag] = {
            "n_events": int(len(sub)),
            "mean_excess_pct": float(sub["excess_pct"].mean()) if len(sub) else float("nan"),
            "mean_excess_net_pct": float(sub["excess_net_pct"].mean()) if len(sub) else float("nan"),
            "events": [str(t.date()) for t in sub.index],
        }
    return out


# --------------------------------------------------------------------------- #
# Synthetic-control detector (machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(
    prices: pd.DataFrame,
    events,
    horizon_months: int = 6,
    cost_bps: float = 5.0,
) -> dict:
    """Run the event study on a synthetic tape with known planted events.

    On the planted world the mean post-event excess must be clearly positive and the
    daily HAC *t* on the conditional leg large; on the null both must sit near zero.
    Proves the harness is unbiased — it never supports a real-tape stamp.
    """
    cmp = compare(prices["duration"], prices["cash"], events,
                  horizon_months=horizon_months, cost_bps=cost_bps)
    return {
        "n_events": cmp["n_events"],
        "mean_excess_pct": cmp["mean_excess_pct"],
        "t_events": cmp["t_events"],
        "in_ann_pct": cmp["in_ann_pct"],
        "out_ann_pct": cmp["out_ann_pct"],
        "t_in_vs_out": cmp["t_in_vs_out"],
        "t_event_leg_hac": cmp["t_event_leg_hac"],
        "hit_rate": cmp["hit_rate"],
    }
