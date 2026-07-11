"""The event-study engine and its honest controls — Study 707 (Plane-Crash-Effect).

The claim under test, steelmanned (Kaplanski & Levy 2010, *Journal of Financial and
Quantitative Analysis*, "Sentiment and Stock Prices: The Case of Aviation Disasters"):
**a major commercial air crash is bad news for the market's mood, not just for
aviation** — investor fear/sentiment dips broadly on the news, then fades within a few
sessions as the shock is digested; and airline stocks, being the most directly exposed,
should drop *harder* than the market as a whole.

The machinery, one execution lag documented throughout:

* ``abnormal_returns`` — a constant-mean market model (Brown & Warner 1985): the
  "normal" return is the sample mean, so the abnormal return is the demeaned series.
* ``event_window`` / ``stack_windows`` — the ``[-1..+5]`` abnormal-return path around
  each crash. ``event_date`` is snapped forward to the first NYSE session on/after the
  crash's calendar date via ``searchsorted`` — the study's single documented execution
  lag: a crash's calendar date is public before that session's close (weekday crashes
  land same-day, weekend/overnight-overseas crashes roll to the next open), so there is
  zero look-ahead.
* ``car_path`` — the mean cumulative abnormal-return path, anchored at 0 the session
  before the crash — the headline "dip and (maybe) recovery" chart.
* ``day0_stats`` — the crash-day abnormal return itself: mean + one-sample *t* across
  events (events are independent, non-overlapping calendar dates).
* ``post_event_car`` / ``reversal_stats`` — the cumulative abnormal return over
  ``+1..+5``: does the shock reverse (a positive post-event drift offsetting the dip)
  or does it stick?
* ``airline_extra_drop`` — pairs each event's SPY day-0 abnormal return with the
  4-carrier airline-basket's day-0 abnormal return (same date, same snap) and one-sample
  *t*'s the **difference** — does the airline sector fall harder than the market on its
  own bad news?
* ``placebo_distribution`` — the synthetic control: the same statistic computed on
  thousands of random non-crash dates, so the observed number's percentile is the
  falsification test.
* ``buy_the_dip`` — the tradable overlay, stated with the obvious ethical caveat kept
  clinical (this measures a statistical pattern in public market data, not a
  recommendation to profit from tragedy): enter at the close of the crash session, ride
  ``hold`` sessions, one round trip of one-way costs charged twice against NAV.

Costs are one-way x NAV per leg; the overlay is long-only (no borrow, no shorting the
grief).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Returns + abnormal returns (constant-mean market model)
# --------------------------------------------------------------------------- #
def daily_returns(close: pd.Series) -> pd.Series:
    """Simple close-to-close daily returns."""
    return close.pct_change()


def abnormal_returns(ret: pd.Series) -> pd.Series:
    """Abnormal return = return minus its own full-sample mean (constant-mean model).

    Demeaning removes the trivial up-drift of equities (or the airline basket's own
    average) so a post-crash CAR is not just "stocks go up" / "airlines are volatile".
    The mean is taken over whatever the series actually covers (NaNs excluded), so an
    airline-basket series that only starts partway through the sample is demeaned over
    its own covered window, not artificially diluted by leading NaNs.
    """
    return ret - ret.mean(skipna=True)


def airline_basket_returns(airlines: dict[str, pd.Series]) -> tuple[pd.Series, pd.Series]:
    """Equal-weight daily return of whichever airline tickers are trading that day.

    Returns ``(basket_return, coverage)`` where ``coverage`` is the count of tickers
    with a valid return on that date (0..4). No survivorship trick is used to backfill
    tickers before their current-entity trading start (United 2006-02, Delta 2007-05,
    American 2013-12) — a day with 0 coverage is simply NaN, and events that land in a
    zero-coverage window are dropped from the airline test (named honestly, not
    silently zero-filled).
    """
    rets = pd.DataFrame({t: daily_returns(s) for t, s in airlines.items()})
    basket = rets.mean(axis=1, skipna=True)
    basket[rets.notna().sum(axis=1) == 0] = np.nan
    coverage = rets.notna().sum(axis=1)
    return basket, coverage


# --------------------------------------------------------------------------- #
# Event windows
# --------------------------------------------------------------------------- #
def event_window(ar: pd.Series, event_date: pd.Timestamp, pre: int = 1, post: int = 5
                  ) -> np.ndarray | None:
    """The abnormal-return path from ``-pre`` to ``+post`` sessions around a crash.

    ``event_date`` is the crash's calendar date; ``searchsorted`` snaps it to the first
    session on/after that date (session 0). Returns a 1-D array of length
    ``pre + post + 1`` (positions ``-pre..0..+post``), or ``None`` if the window runs
    off the edge of the tape.
    """
    idx = ar.index
    pos = idx.searchsorted(pd.Timestamp(event_date))
    if pos >= len(idx):
        return None
    lo, hi = pos - pre, pos + post
    if lo < 0 or hi >= len(idx):
        return None
    return ar.to_numpy()[lo: hi + 1]


def stack_windows(ar: pd.Series, event_dates, pre: int = 1, post: int = 5,
                   require_finite: bool = True) -> tuple[np.ndarray, list]:
    """Stack every valid event window into a ``(n_events, pre+post+1)`` matrix.

    ``require_finite`` drops windows carrying any NaN (e.g. an airline-basket window
    that predates every carrier's current-entity trading start) instead of quietly
    propagating them into the mean. Returns ``(matrix, kept_dates)``.
    """
    rows, kept = [], []
    for d in pd.to_datetime(pd.Series(event_dates)):
        w = event_window(ar, d, pre, post)
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


# --------------------------------------------------------------------------- #
# The headline split — the crash-day dip
# --------------------------------------------------------------------------- #
def day0_stats(ar: pd.Series, event_dates, pre: int = 1, post: int = 5) -> dict:
    """Crash-day abnormal return: mean + one-sample t across events."""
    w, kept = stack_windows(ar, event_dates, pre, post)
    if w.shape[0] == 0:
        return {"n": 0, "mean": float("nan"), "t": float("nan")}
    day0 = w[:, pre]
    mean, t = one_sample_t(day0)
    return {"n": w.shape[0], "mean": mean, "t": t, "kept_dates": kept}


def reversal_stats(ar: pd.Series, event_dates, pre: int = 1, post: int = 5) -> dict:
    """Post-crash [+1..+post] cumulative abnormal return: mean + one-sample t.

    A positive, significant number here says the dip *reverses* (the folklore's
    "temporary" half); a number near zero (or negative) says it sticks.
    """
    w, kept = stack_windows(ar, event_dates, pre, post)
    if w.shape[0] == 0:
        return {"n": 0, "mean": float("nan"), "t": float("nan")}
    per_event = w[:, pre + 1: pre + 1 + post].sum(axis=1)
    mean, t = one_sample_t(per_event)
    return {"n": w.shape[0], "mean": mean, "t": t}


def car_path_stats(ar: pd.Series, event_dates, pre: int = 1, post: int = 5) -> pd.DataFrame:
    """Mean CAR by offset (``-pre..+post``), with each offset's own one-sample t.

    CAR(k) = cumulative mean abnormal return from ``-pre`` through offset ``k``,
    re-anchored so CAR(``-pre``) == 0 (nothing has happened yet before the pre-event
    bar) — the standard event-study convention (matches 313-geopolitical-shock).
    """
    w, kept = stack_windows(ar, event_dates, pre, post)
    offsets = list(range(-pre, post + 1))
    if w.shape[0] == 0:
        return pd.DataFrame(columns=["offset", "mean_ar", "car", "t"]).set_index("offset")
    mean_ar = w.mean(axis=0)
    car_anchored = np.cumsum(mean_ar) - mean_ar[0]
    rows = []
    for i, k in enumerate(offsets):
        _, t = one_sample_t(w[:, i])
        rows.append({"offset": k, "mean_ar": float(mean_ar[i]),
                      "car": float(car_anchored[i]), "t": t})
    return pd.DataFrame(rows).set_index("offset")


# --------------------------------------------------------------------------- #
# Airline extra-drop — does the sector fall harder than the market?
# --------------------------------------------------------------------------- #
def airline_extra_drop(ar_spy: pd.Series, ar_airline: pd.Series, event_dates,
                        pre: int = 1, post: int = 5) -> dict:
    """Paired day-0 (airline - SPY) abnormal-return difference: mean + one-sample t.

    Both series are reindexed onto the same trading calendar before windowing, so a
    "kept" event has a same-day SPY AND airline-basket observation. Events that predate
    every carrier's current trading entity (0 airline coverage) are dropped, not
    zero-filled — the ``n`` reported here is typically smaller than the SPY-only n.
    """
    idx = ar_spy.index.union(ar_airline.index)
    a_spy = ar_spy.reindex(idx)
    a_air = ar_airline.reindex(idx)
    w_spy, kept_spy = stack_windows(a_spy, event_dates, pre, post)
    w_air, kept_air = stack_windows(a_air, event_dates, pre, post)
    kept_both = sorted(set(kept_spy) & set(kept_air))
    if not kept_both:
        return {"n": 0, "mean_diff": float("nan"), "t": float("nan")}
    w_spy2, _ = stack_windows(a_spy, kept_both, pre, post)
    w_air2, _ = stack_windows(a_air, kept_both, pre, post)
    diff = w_air2[:, pre] - w_spy2[:, pre]
    mean, t = one_sample_t(diff)
    return {"n": len(kept_both), "mean_diff": mean, "t": t,
            "air_mean": float(np.nanmean(w_air2[:, pre])),
            "spy_mean": float(np.nanmean(w_spy2[:, pre]))}


# --------------------------------------------------------------------------- #
# Synthetic control / falsification — the random-date placebo
# --------------------------------------------------------------------------- #
def placebo_distribution(ar: pd.Series, n_events: int, pre: int = 1, post: int = 5,
                          n_draws: int = 2000, seed: int = 707,
                          stat: str = "day0") -> np.ndarray:
    """Random-date placebo: mean statistic over ``n_draws`` sets of ``n_events`` random
    non-crash dates. ``stat`` is ``"day0"`` (the headline dip) or ``"post_car"`` (the
    reversal). A real effect must sit in the tail of this distribution; sitting in the
    bulk means the "effect" is what a random calendar of the same size produces anyway.
    """
    rng = np.random.default_rng(seed)
    vals = ar.dropna().to_numpy()
    n = vals.size
    lo, hi = pre, n - post - 1
    if hi <= lo or n_events <= 0:
        return np.array([])
    out = np.empty(n_draws)
    for d in range(n_draws):
        locs = rng.integers(lo, hi, size=n_events)
        if stat == "day0":
            per = vals[locs]
        else:
            per = np.array([vals[loc + 1: loc + 1 + post].sum() for loc in locs])
        out[d] = per.mean()
    return out


def placebo_pvalue(observed: float, placebo: np.ndarray, tail: str = "left") -> float:
    """Empirical one-sided p-value of ``observed`` within the placebo draws."""
    if placebo.size == 0 or not np.isfinite(observed):
        return float("nan")
    if tail == "left":
        return float((placebo <= observed).mean())
    return float((placebo >= observed).mean())


def block_bootstrap_ci(per_event: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
                        seed: int = 707) -> tuple[float, float]:
    """Percentile CI on the mean per-event statistic (events resampled with replacement).

    Events are independent, far-apart calendar dates (a within-event serial-correlation
    concern doesn't apply once each event is summarised to one number), so the
    resampling unit is the event.
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
# The tradable overlay — "buy the dip" after a crash (ethical caveat: this is a
# statistical test of public market data, not a recommendation to trade on tragedy)
# --------------------------------------------------------------------------- #
def buy_the_dip(close: pd.Series, event_dates, hold: int = 5,
                 cost_bps: float = 0.0) -> pd.DataFrame:
    """Long-only overlay: enter at the close of the crash session, hold ``hold`` sessions.

    The crash is known at the close of session 0 (see the module docstring's execution
    lag), so the position earns sessions ``+1..+hold`` — a single lag, applied once. One
    round trip per event: one-way cost charged twice (entry + exit) against NAV.
    """
    idx = close.index
    rows = []
    for d in pd.to_datetime(pd.Series(event_dates)):
        pos = idx.searchsorted(pd.Timestamp(d))
        if pos >= len(idx):
            continue
        entry = pos
        exit_ = entry + hold
        if exit_ >= len(idx):
            continue
        gross = close.iat[exit_] / close.iat[entry] - 1.0
        net = gross - 2.0 * cost_bps * 1e-4
        rows.append({"entry_date": idx[entry], "hold": hold,
                      "ret_gross": float(gross), "ret_net": float(net)})
    return pd.DataFrame(rows)


def summarize_dip(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline stats for the buy-the-dip ledger: n, win-rate, mean, t-stat."""
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
def synthetic_detect(close: pd.DataFrame, events: pd.DatetimeIndex,
                      pre: int = 1, post: int = 5) -> dict:
    """Run the headline day-0 dip test on a synthetic world."""
    ar = abnormal_returns(daily_returns(close["Close"]))
    return day0_stats(ar, events, pre, post)
