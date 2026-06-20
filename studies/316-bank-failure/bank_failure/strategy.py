"""The event-study engine and its honest controls — Study 316 (Bank-Failure).

The question: when a headline bank fails (Lehman, SVB, Credit Suisse ...), is the
broad market a *buy* (panic overshoot, snap-back) or a *warning* (the failure is the
first domino, more decline to come)?

We answer it as a classic **event study**. For each event date `e` we line up the
log returns of SPY on the trading days around `e`, and average across events:

- **CAR (cumulative abnormal-ish return).** The mean cumulative log return from the
  event close out to +`h` trading days, averaged over events. Because we use the
  *adjusted* (total-return-ish) series and a short window, the "abnormal" benchmark
  is simply zero drift — we are not de-meaning against a market model, so we call it
  "cumulative *return*", not a Fama-style abnormal return, and lean on the placebo
  arm to supply the counterfactual.
- **The placebo / synthetic control.** The same window machinery run on `n` random
  calendar dates. This is the counterfactual: if the post-event CAR is just the
  market's ambient drift (or its volatility-clustering), the placebo reproduces it.
  The event signal is the *difference* event − placebo, with a permutation p-value.

Inference: a per-event mean carries a HAC/Newey-West-flavoured t-stat (few events →
we also report a simple t and a block/permutation test); the event-vs-placebo gap
gets a **permutation p-value** (draw many random-date placebo panels, see how often
their mean CAR beats the real events'). A circular block bootstrap CI on the daily
event-window returns guards the volatility clustering.

Tradability: a long-SPY "buy the blood" overlay entered on the event date and held
`h` days, charged a one-way cost × NAV on entry and exit, excess-of-cash where it
sits in cash between events, compared against simple buy-and-hold over the same span.

Execution lag: event dates are **calendar-known** (a failure announced on day `e` is
public on day `e`), so the entry uses the event-day close with **no extra shift** —
the documented convention for calendar-known rules (METHODOLOGY → execution lag).
A `lag=1` knob is exposed as a conservative robustness variant (enter next close).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Aligning one event window
# ---------------------------------------------------------------------------
def _nearest_loc(index: pd.DatetimeIndex, date: pd.Timestamp) -> int | None:
    """Position of the first trading day >= ``date`` (the actionable bar)."""
    pos = index.searchsorted(pd.Timestamp(date), side="left")
    if pos >= len(index):
        return None
    return int(pos)


def event_window_returns(
    close: pd.Series,
    dates,
    pre: int = 10,
    post: int = 20,
    lag: int = 0,
) -> pd.DataFrame:
    """A matrix of log returns aligned on each event's day-0.

    Row = one event; columns = relative trading day in ``[-pre, +post]``. Column 0
    is the event day's own log return (close_{0}/close_{-1}). The entry/anchor is the
    first trading day on or after the event date, shifted by ``lag`` extra days
    (``lag=0`` is the canonical calendar-known anchor).

    Events whose full window does not fit inside the sample are dropped.
    """
    logp = np.log(close.to_numpy(dtype=float))
    idx = close.index
    rows = []
    kept = []
    cols = list(range(-pre, post + 1))
    for d in pd.DatetimeIndex(pd.to_datetime(list(dates))):
        loc = _nearest_loc(idx, d)
        if loc is None:
            continue
        loc = loc + lag
        if loc - pre - 1 < 0 or loc + post >= len(logp):
            continue
        # log return on relative day k = logp[loc+k] - logp[loc+k-1]
        seg = logp[loc - pre : loc + post + 1] - logp[loc - pre - 1 : loc + post]
        rows.append(seg)
        kept.append(d)
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(np.vstack(rows), columns=cols, index=pd.DatetimeIndex(kept))
    return out


def car_curve(win: pd.DataFrame) -> pd.Series:
    """Mean cumulative log return across events, indexed by relative day.

    Cumulation starts at the event day (column 0); pre-event days carry the
    run-up into the event (cumulated backwards from 0 for display).
    """
    if win.empty:
        return pd.Series(dtype=float)
    mean_daily = win.mean(axis=0)
    # cumulate forward from day 0; pre-event part shown as cumulative run-up.
    cols = mean_daily.index
    cum = pd.Series(0.0, index=cols)
    # forward from 0
    post = [c for c in cols if c >= 0]
    cum.loc[post] = mean_daily.loc[post].cumsum()
    # backward for pre (negative of the cumulated future-to-0... simpler: cumulate all)
    pre = sorted([c for c in cols if c < 0], reverse=True)
    running = 0.0
    for c in pre:
        running += mean_daily.loc[c + 1] if (c + 1) in mean_daily.index else 0.0
        cum.loc[c] = -running
    return cum


def post_event_car(win: pd.DataFrame, horizon: int = 20) -> np.ndarray:
    """Per-event cumulative log return from day 0 (inclusive) to +``horizon``."""
    if win.empty:
        return np.array([])
    cols = [c for c in win.columns if 0 <= c <= horizon]
    return win[cols].sum(axis=1).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC t-stat of the mean of ``x`` (auto lag); plain t for tiny n."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return float("nan")
    mu = x.mean()
    if n < 8:  # too few events for HAC — plain t on the mean
        se = x.std(ddof=1) / np.sqrt(n)
        return float(mu / se) if se > 0 else float("nan")
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def permutation_pvalue(
    close: pd.Series,
    event_dates,
    n_placebo: int = 2000,
    pre: int = 10,
    post: int = 20,
    horizon: int = 20,
    lag: int = 0,
    seed: int = 316,
) -> dict:
    """Permutation test of the event CAR vs random-date placebos.

    Draws ``n_placebo`` panels, each of the same size as the event set but on random
    calendar dates, computes each panel's mean post-event CAR, and reports the
    two-sided p-value (fraction of placebos whose |mean CAR| >= the events').

    Returns event mean CAR, placebo mean/std of mean-CAR, z-score and p-value.
    """
    from . import data as _d

    win = event_window_returns(close, event_dates, pre=pre, post=post, lag=lag)
    car = post_event_car(win, horizon=horizon)
    if car.size == 0:
        return {"event_car": float("nan"), "p_value": float("nan"), "z": float("nan"),
                "placebo_mean": float("nan"), "placebo_std": float("nan"), "n_events": 0}
    event_mean = float(car.mean())
    k = len(win)

    rng = np.random.default_rng(seed)
    placebo_means = np.empty(n_placebo)
    for b in range(n_placebo):
        rdates = _d.random_dates(close, n=k, margin=max(pre, post) + 5,
                                 seed=int(rng.integers(0, 2**31 - 1)))
        pw = event_window_returns(close, rdates, pre=pre, post=post, lag=lag)
        pc = post_event_car(pw, horizon=horizon)
        placebo_means[b] = pc.mean() if pc.size else np.nan
    placebo_means = placebo_means[np.isfinite(placebo_means)]
    pmu, psd = float(placebo_means.mean()), float(placebo_means.std(ddof=1))
    centered = abs(event_mean - pmu)
    p = float((np.abs(placebo_means - pmu) >= centered).mean())
    z = float((event_mean - pmu) / psd) if psd > 0 else float("nan")
    return {
        "event_car": event_mean,
        "placebo_mean": pmu,
        "placebo_std": psd,
        "z": z,
        "p_value": p,
        "n_events": int(k),
        "n_placebo": int(placebo_means.size),
    }


def block_bootstrap_ci(
    car: np.ndarray,
    n_boot: int = 5000,
    block: int = 2,
    seed: int = 316,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean per-event CAR.

    Block resampling on the per-event CAR vector (preserves any clustering of events
    in time, e.g. the 2008 and 2023 clusters). Tiny n, so the CI is wide by design.
    """
    car = np.asarray(car, dtype=float)
    car = car[np.isfinite(car)]
    n = car.size
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n_blocks = int(np.ceil(n / block))
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(starts[:, None] + np.arange(block)) % n]).ravel()[:n]
        means[b] = car[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


# ---------------------------------------------------------------------------
# Tradability — the "buy the blood" overlay
# ---------------------------------------------------------------------------
def buy_the_blood(
    close: pd.Series,
    event_dates,
    horizon: int = 20,
    lag: int = 0,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Per-trade ledger for a long-SPY overlay entered on each event, held ``horizon`` days.

    Entry at the event-day close (calendar-known; ``lag`` extra days optional), exit
    ``horizon`` trading days later at the close. ``cost_bps`` is a **one-way** cost,
    charged on both entry and exit (so 2 × cost per round trip) against NAV.

    Columns: ``event, entry, exit, ret_gross, ret_net``. The overlay sits in cash
    between events, so the fair benchmark for it is excess-of-cash (handled by the
    caller); this ledger is the gross/net per-trade arm.
    """
    logp = np.log(close.to_numpy(dtype=float))
    idx = close.index
    rows = []
    for d in pd.DatetimeIndex(pd.to_datetime(list(event_dates))):
        loc = _nearest_loc(idx, d)
        if loc is None:
            continue
        loc = loc + lag
        if loc + horizon >= len(logp):
            continue
        ret_gross = float(np.exp(logp[loc + horizon] - logp[loc]) - 1.0)
        ret_net = ret_gross - 2.0 * cost_bps * 1e-4  # one-way × 2 legs
        rows.append({
            "event": idx[loc],
            "entry": float(close.iloc[loc]),
            "exit": float(close.iloc[loc + horizon]),
            "ret_gross": ret_gross,
            "ret_net": ret_net,
        })
    return pd.DataFrame(rows)


def summarize_trades(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-trade stats for the overlay ledger (win-rate, mean, t)."""
    if ledger.empty:
        return {"n": 0, "win_rate": float("nan"), "mean_pct": float("nan"),
                "tstat": float("nan")}
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    return {
        "n": int(r.size),
        "win_rate": float((r > 0).mean()),
        "mean_pct": float(r.mean() * 100),
        "median_pct": float(np.median(r) * 100),
        "tstat": hac_tstat(r),
    }
