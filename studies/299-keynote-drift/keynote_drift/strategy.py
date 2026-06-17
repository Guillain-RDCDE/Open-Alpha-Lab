"""Strategy and analysis layer for Study 299 (Keynote-Drift).

The "buy the rumor, sell the news" folklore says AAPL drifts UP into an Apple
keynote (anticipation) and sells off after the products are unveiled. We treat
this as a classic event study on AAPL daily returns and pin it against honest
nulls:

  - The **pre-event cumulative abnormal return (CAR)**: the sum of AAPL daily
    returns over the ``pre_window`` trading days before each keynote, in excess of
    AAPL's own unconditional daily mean (so a generally up-trending stock does not
    get credit for "drift into keynotes"). This is the killer control: AAPL went
    up a lot 2008-2025 regardless of keynotes.
  - The **post-event CAR**: same, over the ``post_window`` days after the event.
  - A **t-test** on the per-event pre-window CARs (H0: mean = 0).
  - A **permutation test**: re-anchor the event windows on random non-event dates
    and see where the real mean CAR lands.
  - The **tradable leg**: buy AAPL ``pre_window`` days before the event (one-day
    execution lag), exit at the event-eve close; report gross AND net of one-way
    costs x NAV. Price-only returns (no dividends).

The small-n reckoning: ~47 keynotes 2008-2025. With ~2-3%/event window vol, the
minimum detectable mean CAR at |t| = 2 is roughly 0.6-0.9%/event -- and the
abnormal (de-meaned) drift is far smaller than that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def daily_returns(prices: pd.DataFrame, col: str = "adj_close") -> pd.Series:
    """Simple daily returns from an adj_close frame."""
    return prices[col].pct_change().rename("ret")


def _event_anchors(ret_index: pd.DatetimeIndex, events: pd.DataFrame) -> list[int]:
    """Integer positions of the first trading day at/after each event date."""
    anchors = []
    arr = ret_index
    for d in pd.to_datetime(events["date"]):
        future = arr[arr >= d]
        if len(future) == 0:
            continue
        anchors.append(arr.get_loc(future[0]))
    return anchors


# ---------------------------------------------------------------------------
# Event study: per-event CARs
# ---------------------------------------------------------------------------
def event_cars(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    pre_window: int = 5,
    post_window: int = 5,
    col: str = "adj_close",
    demean: bool = True,
) -> pd.DataFrame:
    """Per-event cumulative (abnormal) returns in the pre- and post-event windows.

    For each keynote, find the first trading day at/after the event (the "anchor",
    treated as day 0). The pre-window CAR sums the ``pre_window`` daily returns
    strictly *before* day 0; the post-window CAR sums ``post_window`` daily returns
    starting *at* day 0 (the reveal day and following days).

    If ``demean=True`` (the honest default) each daily return is taken in excess of
    AAPL's own full-sample unconditional daily mean, so that a stock with a strong
    secular uptrend does not get spurious "keynote drift" credit. This is the
    abnormal return (a constant-mean market model with AAPL as its own benchmark).

    Returns a DataFrame indexed by event with columns ``pre_car``, ``post_car``,
    ``event_type``, ``label``.
    """
    ret = daily_returns(prices, col=col)
    ret = ret.dropna()
    mean_d = float(ret.mean()) if demean else 0.0
    ar = ret - mean_d  # abnormal daily return
    ar_vals = ar.values
    idx = ar.index
    n = len(ar_vals)

    ev = events.reset_index(drop=True).copy()
    ev["date"] = pd.to_datetime(ev["date"])

    rows = []
    for _, e in ev.iterrows():
        future = idx[idx >= e["date"]]
        if len(future) == 0:
            continue
        a = idx.get_loc(future[0])
        pre_lo, pre_hi = a - pre_window, a  # [a-pre, a)  strictly before reveal
        post_lo, post_hi = a, a + post_window  # [a, a+post)
        if pre_lo < 0 or post_hi > n:
            continue
        pre_car = float(ar_vals[pre_lo:pre_hi].sum())
        post_car = float(ar_vals[post_lo:post_hi].sum())
        rows.append({
            "date": e["date"],
            "event_type": e.get("event_type", "NA"),
            "label": e.get("label", ""),
            "pre_car": pre_car,
            "post_car": post_car,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inference on the per-event CARs
# ---------------------------------------------------------------------------
def car_stats(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    pre_window: int = 5,
    post_window: int = 5,
    col: str = "adj_close",
    demean: bool = True,
    n_permutations: int = 10_000,
    seed: int = 299,
) -> dict:
    """Mean CARs, t-tests, and a permutation test on the pre-event drift.

    Returns a flat dict:
      - ``n_events``: number of usable events
      - ``mean_pre_car``, ``mean_post_car``: average per-event CAR (fraction)
      - ``t_pre``, ``p_pre``: one-sample t-test of pre_car vs 0
      - ``t_post``, ``p_post``: one-sample t-test of post_car vs 0
      - ``perm_p``: two-sided permutation p-value for the mean pre_car, re-anchoring
        the windows on random non-event trading days
      - ``pre_cars``, ``post_cars``: raw arrays (for plotting)
    """
    cars = event_cars(prices, events, pre_window=pre_window,
                       post_window=post_window, col=col, demean=demean)
    pre = cars["pre_car"].values.astype(float)
    post = cars["post_car"].values.astype(float)
    n_ev = len(pre)

    if n_ev >= 2:
        t_pre, p_pre = scipy_stats.ttest_1samp(pre, 0.0)
        t_post, p_post = scipy_stats.ttest_1samp(post, 0.0)
    else:
        t_pre = p_pre = t_post = p_post = float("nan")

    mean_pre = float(np.mean(pre)) if n_ev else float("nan")
    mean_post = float(np.mean(post)) if n_ev else float("nan")

    # Permutation: re-anchor pre-windows on random trading days, recompute mean CAR.
    ret = daily_returns(prices, col=col).dropna()
    mean_d = float(ret.mean()) if demean else 0.0
    ar_vals = (ret - mean_d).values
    n = len(ar_vals)
    rng = np.random.default_rng(seed)
    lo = pre_window
    hi = n - post_window
    perm_means = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        anchors = rng.integers(lo, hi, size=n_ev)
        m = 0.0
        for a in anchors:
            m += ar_vals[a - pre_window:a].sum()
        perm_means[i] = m / n_ev
    # Two-sided: how often is |perm mean| >= |observed mean|.
    perm_p = float((np.abs(perm_means) >= abs(mean_pre)).mean()) if n_ev else float("nan")

    return {
        "n_events": int(n_ev),
        "mean_pre_car": round(mean_pre, 6),
        "mean_post_car": round(mean_post, 6),
        "t_pre": round(float(t_pre), 3),
        "p_pre": round(float(p_pre), 4),
        "t_post": round(float(t_post), 3),
        "p_post": round(float(p_post), 4),
        "perm_p": round(float(perm_p), 4),
        "pre_cars": pre,
        "post_cars": post,
        "perm_means": perm_means,
    }


# ---------------------------------------------------------------------------
# Tradable "buy the rumor" leg -- gross and net of costs
# ---------------------------------------------------------------------------
def buy_the_rumor_backtest(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    pre_window: int = 5,
    col: str = "adj_close",
    cost_bps: float = 5.0,
    exec_lag: int = 1,
) -> dict:
    """Backtest the 'buy AAPL into the keynote' leg, gross and net of costs.

    For each keynote, enter long AAPL ``pre_window`` trading days before the event,
    applying an ``exec_lag`` of one trading day (you can only act on the signal at
    the *next* close), and exit at the event-eve close. Each round-trip pays
    ``2 * cost_bps`` one-way x NAV (in and out). Price-only returns -- no dividends.

    Returns a dict with per-trade gross/net returns and aggregate stats:
      ``n_trades``, ``mean_gross``, ``mean_net``, ``hit_rate_gross``,
      ``ann_factor`` (events/yr), ``total_gross``, ``total_net``, and the raw
      ``gross``/``net`` arrays.
    """
    ret = daily_returns(prices, col=col).dropna()
    idx = ret.index
    n = len(ret)
    ev = events.reset_index(drop=True).copy()
    ev["date"] = pd.to_datetime(ev["date"])

    gross = []
    for _, e in ev.iterrows():
        future = idx[idx >= e["date"]]
        if len(future) == 0:
            continue
        a = idx.get_loc(future[0])
        entry = a - pre_window + exec_lag  # one-day lag on entry
        exit_ = a  # hold through the day before reveal (day a is the reveal day)
        if entry < 1 or exit_ > n:
            continue
        # Compound the daily returns from entry (inclusive) up to exit (exclusive).
        seg = ret.values[entry:exit_]
        if len(seg) == 0:
            continue
        g = float(np.prod(1.0 + seg) - 1.0)
        gross.append(g)

    gross = np.array(gross, dtype=float)
    rt_cost = 2.0 * cost_bps * 1e-4
    net = gross - rt_cost
    n_trades = len(gross)

    # Events per year for annualization context.
    span_years = max(1e-9, (ev["date"].max() - ev["date"].min()).days / 365.25)
    ann_factor = n_trades / span_years if span_years > 0 else float("nan")

    return {
        "n_trades": int(n_trades),
        "mean_gross": round(float(gross.mean()), 6) if n_trades else float("nan"),
        "mean_net": round(float(net.mean()), 6) if n_trades else float("nan"),
        "hit_rate_gross": round(float((gross > 0).mean()), 4) if n_trades else float("nan"),
        "hit_rate_net": round(float((net > 0).mean()), 4) if n_trades else float("nan"),
        "total_gross": round(float(np.prod(1.0 + gross) - 1.0), 4) if n_trades else float("nan"),
        "total_net": round(float(np.prod(1.0 + net) - 1.0), 4) if n_trades else float("nan"),
        "ann_factor": round(float(ann_factor), 2),
        "cost_bps": cost_bps,
        "gross": gross,
        "net": net,
    }


# ---------------------------------------------------------------------------
# Summarize -- compact R-dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    pre_window: int = 5,
    post_window: int = 5,
    n_permutations: int = 10_000,
    seed: int = 299,
) -> dict:
    """One-stop summary dict mirroring R in build_notebooks.py and docs/results.md."""
    cs = car_stats(prices, events, pre_window=pre_window, post_window=post_window,
                   demean=True, n_permutations=n_permutations, seed=seed)
    bt = buy_the_rumor_backtest(prices, events, pre_window=pre_window)
    return {
        "n_events": cs["n_events"],
        "mean_pre_car_pct": round(cs["mean_pre_car"] * 100, 3),
        "mean_post_car_pct": round(cs["mean_post_car"] * 100, 3),
        "t_pre": cs["t_pre"],
        "p_pre": cs["p_pre"],
        "t_post": cs["t_post"],
        "p_post": cs["p_post"],
        "perm_p": cs["perm_p"],
        "n_trades": bt["n_trades"],
        "mean_gross_pct": round(bt["mean_gross"] * 100, 3),
        "mean_net_pct": round(bt["mean_net"] * 100, 3),
        "hit_rate_gross_pct": round(bt["hit_rate_gross"] * 100, 1),
        "cost_bps": bt["cost_bps"],
    }
