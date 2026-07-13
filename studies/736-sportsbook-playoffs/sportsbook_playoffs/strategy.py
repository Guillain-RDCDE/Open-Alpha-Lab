"""The event-study engine and its honest controls — Study 736 (Sportsbook-Playoffs).

The claim under test, steelmanned: **betting stocks rally *into* the big betting
seasons.** In the weeks before NFL Wild-Card weekend (January) and the March-Madness
Round of 64 (mid-March), anticipation of a wall of betting handle — new-customer
promos, deposit surges, record parlay volume — is supposed to lift DraftKings and the
wider sportsbook/iGaming basket *before* the games are even played. The tradable
corollary: buy the basket a couple of weeks ahead of the first game and ride the
anticipation into the tip-off.

The machinery, one execution convention documented throughout:

* ``abnormal_returns`` — a constant-mean market model (Brown & Warner 1985): the
  "normal" return is the sample mean, so the abnormal return is the demeaned series.
  This strips out the stock's own (large, and largely upward-then-downward) drift so a
  "rally into the season" is measured against the stock's *own* baseline, not the raw
  bull/bear tape. ``market_adjusted_returns`` is the beta≈1 robustness variant (return
  minus SPY's return) — a rally that is just the whole market going up should vanish
  from it.
* ``run_up_window`` / ``stack_run_ups`` — the cumulative abnormal return over the
  ``run_up`` sessions *ending the session before* the first game. ``event_date`` is
  snapped to the first session on/after the (public, pre-scheduled) season-start date
  via ``searchsorted``; the run-up window is the ``run_up`` sessions strictly before
  that snap. Because the schedule is public months ahead, there is **zero look-ahead** —
  no execution lag is applied (a calendar-known rule, like a turn-of-month window).
* ``run_up_stats`` — the headline: mean run-up CAR + one-sample *t* across the events
  (events are independent, non-overlapping calendar dates — the correct unit, not a
  daily panel).
* ``car_path_stats`` — the mean cumulative abnormal-return path across the whole
  ``[-pre..+post]`` window, anchored at the start — the "rally then fade?" chart.
* ``post_event_stats`` — the [0..+post] "sell-the-news" leg: does whatever ran up give
  it back once the games start?
* ``placebo_distribution`` — the synthetic control: the same run-up statistic computed
  on thousands of random non-event dates, so the observed number's percentile is the
  falsification test.
* ``run_up_timer`` — the tradable overlay: buy at the close ``run_up`` sessions before
  the first game, sell at the close the session before it; one round trip of one-way
  costs charged twice against NAV; long-only (no borrow).

Costs are one-way × NAV per leg; the overlay is long-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Returns + abnormal returns
# --------------------------------------------------------------------------- #
def daily_returns(close: pd.Series) -> pd.Series:
    """Simple close-to-close daily returns."""
    return close.pct_change()


def abnormal_returns(ret: pd.Series) -> pd.Series:
    """Abnormal return = return minus its own full-sample mean (constant-mean model).

    Demeaning removes the stock's own average drift (betting stocks had a violent
    2020-21 melt-up and a 2022 crash) so a run-up CAR is not just "the stock was
    trending". The mean is over whatever the series covers (NaNs excluded).
    """
    return ret - ret.mean(skipna=True)


def market_adjusted_returns(ret: pd.Series, bench_ret: pd.Series) -> pd.Series:
    """Beta≈1 market-adjusted return: stock return minus the benchmark's return.

    Reindexed onto the stock's calendar. A "rally" that is really just the whole market
    rising over the run-up window should shrink toward zero here. A blunt beta=1 adjust
    (not a fitted beta) — deliberately conservative and free of look-ahead beta fitting.
    """
    b = bench_ret.reindex(ret.index)
    return ret - b


def basket_returns(closes: dict[str, pd.Series], tickers) -> tuple[pd.Series, pd.Series]:
    """Equal-weight daily return of whichever basket tickers trade that day.

    Returns ``(basket_return, coverage)`` where ``coverage`` counts the tickers with a
    valid return on the date. No survivorship backfill: a day with 0 coverage is NaN.
    """
    rets = pd.DataFrame({t: daily_returns(closes[t]) for t in tickers})
    basket = rets.mean(axis=1, skipna=True)
    basket[rets.notna().sum(axis=1) == 0] = np.nan
    coverage = rets.notna().sum(axis=1)
    return basket, coverage


# --------------------------------------------------------------------------- #
# Event windows — the run-up
# --------------------------------------------------------------------------- #
def run_up_window(ar: pd.Series, event_date: pd.Timestamp, run_up: int = 10
                   ) -> np.ndarray | None:
    """The ``run_up`` abnormal returns *ending the session before* the first game.

    ``event_date`` (the public, pre-scheduled season start) is snapped to the first
    session on/after it (position ``pos``); the run-up is positions ``pos-run_up ..
    pos-1`` — the sessions strictly before the games begin. Returns a 1-D array of length
    ``run_up`` (chronological), or ``None`` if the window runs off the edge of the tape.
    """
    idx = ar.index
    pos = idx.searchsorted(pd.Timestamp(event_date))
    if pos >= len(idx):
        return None
    lo, hi = pos - run_up, pos - 1
    if lo < 0 or hi >= len(idx):
        return None
    return ar.to_numpy()[lo: hi + 1]


def event_path_window(ar: pd.Series, event_date: pd.Timestamp, pre: int = 10, post: int = 5
                       ) -> np.ndarray | None:
    """Abnormal returns from ``-pre`` (run-up) through ``+post`` (post first-game) sessions.

    Position ``pre`` in the returned array is the first session on/after the season
    start (offset 0). Length ``pre + post + 1``, or ``None`` off the edge.
    """
    idx = ar.index
    pos = idx.searchsorted(pd.Timestamp(event_date))
    if pos >= len(idx):
        return None
    lo, hi = pos - pre, pos + post
    if lo < 0 or hi >= len(idx):
        return None
    return ar.to_numpy()[lo: hi + 1]


def stack_run_ups(ar: pd.Series, event_dates, run_up: int = 10,
                   require_finite: bool = True) -> tuple[np.ndarray, list]:
    """Stack every valid run-up window into a ``(n_events, run_up)`` matrix.

    ``require_finite`` drops windows carrying any NaN instead of propagating them.
    Returns ``(matrix, kept_dates)``.
    """
    rows, kept = [], []
    for d in pd.to_datetime(pd.Series(event_dates)):
        w = run_up_window(ar, d, run_up)
        if w is None:
            continue
        if require_finite and not np.all(np.isfinite(w)):
            continue
        rows.append(w)
        kept.append(d)
    if not rows:
        return np.empty((0, run_up)), kept
    return np.vstack(rows), kept


def stack_paths(ar: pd.Series, event_dates, pre: int = 10, post: int = 5,
                 require_finite: bool = True) -> tuple[np.ndarray, list]:
    """Stack every valid [-pre..+post] event path into a ``(n_events, pre+post+1)`` matrix."""
    rows, kept = [], []
    for d in pd.to_datetime(pd.Series(event_dates)):
        w = event_path_window(ar, d, pre, post)
        if w is None:
            continue
        if require_finite and not np.all(np.isfinite(w)):
            continue
        rows.append(w)
        kept.append(d)
    if not rows:
        return np.empty((0, pre + post + 1)), kept
    return np.vstack(rows), kept


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> tuple[float, float]:
    """Mean and one-sample t-stat of ``x`` (events treated as independent)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2 or x.std(ddof=1) == 0:
        return float(np.nan if n == 0 else x.mean()), float("nan")
    se = x.std(ddof=1) / np.sqrt(n)
    return float(x.mean()), float(x.mean() / se)


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson (1927) score interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# The headline — the run-up CAR
# --------------------------------------------------------------------------- #
def run_up_stats(ar: pd.Series, event_dates, run_up: int = 10) -> dict:
    """Run-up cumulative abnormal return: mean + one-sample t across events.

    Per event, the run-up CAR is the SUM of the ``run_up`` abnormal returns before the
    first game (approx. the run-up log-return). Also reports the hit rate (events with a
    positive run-up) as a count for a Wilson interval.
    """
    w, kept = stack_run_ups(ar, event_dates, run_up)
    if w.shape[0] == 0:
        return {"n": 0, "mean": float("nan"), "t": float("nan"),
                "hits": 0, "per_event": np.array([]), "kept_dates": []}
    per_event = w.sum(axis=1)
    mean, t = one_sample_t(per_event)
    return {"n": int(w.shape[0]), "mean": mean, "t": t,
            "hits": int((per_event > 0).sum()),
            "per_event": per_event, "kept_dates": kept}


def post_event_stats(ar: pd.Series, event_dates, pre: int = 10, post: int = 5) -> dict:
    """Post-first-game [0..+post] cumulative abnormal return: mean + one-sample t.

    A negative, significant number here is the "sell the news" leg — the run-up handed
    back once the games start. Offset 0 is the first session on/after the season start.
    """
    w, kept = stack_paths(ar, event_dates, pre, post)
    if w.shape[0] == 0:
        return {"n": 0, "mean": float("nan"), "t": float("nan")}
    per_event = w[:, pre: pre + post + 1].sum(axis=1)
    mean, t = one_sample_t(per_event)
    return {"n": int(w.shape[0]), "mean": mean, "t": t}


def car_path_stats(ar: pd.Series, event_dates, pre: int = 10, post: int = 5) -> pd.DataFrame:
    """Mean CAR by offset (``-pre..+post``), anchored so CAR(-pre) == first bar.

    CAR(k) = cumulative mean abnormal return from ``-pre`` through offset ``k`` — the
    "rally into the season, then what?" path. Each offset carries its own one-sample t.
    """
    w, kept = stack_paths(ar, event_dates, pre, post)
    offsets = list(range(-pre, post + 1))
    if w.shape[0] == 0:
        return pd.DataFrame(columns=["offset", "mean_ar", "car", "t"]).set_index("offset")
    mean_ar = w.mean(axis=0)
    car = np.cumsum(mean_ar)
    rows = []
    for i, k in enumerate(offsets):
        _, t = one_sample_t(w[:, i])
        rows.append({"offset": k, "mean_ar": float(mean_ar[i]),
                      "car": float(car[i]), "t": t})
    return pd.DataFrame(rows).set_index("offset")


# --------------------------------------------------------------------------- #
# Synthetic control / falsification — the random-date placebo
# --------------------------------------------------------------------------- #
def placebo_distribution(ar: pd.Series, n_events: int, run_up: int = 10,
                          n_draws: int = 2000, seed: int = 736) -> np.ndarray:
    """Random-date placebo: mean run-up CAR over ``n_draws`` sets of ``n_events`` random
    dates. A real "rally into the season" must sit in the right tail of this cloud; in
    the bulk means the run-up is just what a random ``run_up``-day window produces anyway.
    """
    rng = np.random.default_rng(seed)
    vals = ar.dropna().to_numpy()
    n = vals.size
    lo, hi = run_up, n - 1
    if hi <= lo or n_events <= 0:
        return np.array([])
    out = np.empty(n_draws)
    for d in range(n_draws):
        locs = rng.integers(lo, hi, size=n_events)
        per = np.array([vals[loc - run_up: loc].sum() for loc in locs])
        out[d] = per.mean()
    return out


def placebo_pvalue(observed: float, placebo: np.ndarray, tail: str = "right") -> float:
    """Empirical one-sided p-value of ``observed`` within the placebo draws.

    The claim predicts a POSITIVE run-up, so the falsification test is the RIGHT tail:
    how often does a random calendar of the same size produce a run-up this big or bigger.
    """
    if placebo.size == 0 or not np.isfinite(observed):
        return float("nan")
    if tail == "right":
        return float((placebo >= observed).mean())
    return float((placebo <= observed).mean())


def bootstrap_ci(per_event: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
                  seed: int = 736) -> tuple[float, float]:
    """Percentile CI on the mean per-event run-up (events resampled with replacement).

    Events are independent, far-apart calendar dates, so the resampling unit is the event.
    """
    x = per_event[np.isfinite(per_event)]
    n = x.size
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = x[rng.integers(0, n, size=n)].mean()
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


# --------------------------------------------------------------------------- #
# The tradable overlay — buy the run-up
# --------------------------------------------------------------------------- #
def run_up_timer(close: pd.Series, event_dates, run_up: int = 10,
                  cost_bps: float = 0.0) -> pd.DataFrame:
    """Long-only overlay: buy ``run_up`` sessions before the first game, sell the session
    before it.

    The season-start date is public months ahead, so both the entry and exit sessions are
    known in advance — zero look-ahead, no execution lag (a calendar-known rule). One
    round trip per event: one-way cost charged twice (entry + exit) against NAV.
    """
    idx = close.index
    rows = []
    for d in pd.to_datetime(pd.Series(event_dates)):
        pos = idx.searchsorted(pd.Timestamp(d))
        if pos >= len(idx):
            continue
        entry = pos - run_up
        exit_ = pos - 1
        if entry < 0 or exit_ >= len(idx) or exit_ <= entry:
            continue
        gross = close.iat[exit_] / close.iat[entry] - 1.0
        net = gross - 2.0 * cost_bps * 1e-4
        rows.append({"entry_date": idx[entry], "exit_date": idx[exit_],
                      "run_up": run_up, "ret_gross": float(gross), "ret_net": float(net)})
    return pd.DataFrame(rows)


def summarize_timer(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline stats for the run-up timer ledger: n, win-rate, mean bps, t-stat."""
    if ledger.empty:
        return {"n": 0, "win_rate": float("nan"), "mean_bps": float("nan"), "t": float("nan")}
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    mean, t = one_sample_t(r)
    return {"n": int(n), "win_rate": float((r > 0).mean()) if n else float("nan"),
            "mean_bps": mean * 1e4 if n else float("nan"), "t": t}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(close: pd.DataFrame, events: pd.DatetimeIndex, run_up: int = 10) -> dict:
    """Run the headline run-up test on a synthetic world."""
    ar = abnormal_returns(daily_returns(close["Close"]))
    return run_up_stats(ar, events, run_up)
