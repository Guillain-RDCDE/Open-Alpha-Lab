"""The event-study + seasonal engine and its honest controls — Study 739 (Wildfire-Season).

The claim under test, steelmanned: **California's fire season is a repeating, tradable
risk event for the state's utilities and property insurers.** Two testable halves:

* **The event half.** On the day a major California wildfire breaks out, the exposed
  basket — the two investor-owned utilities (EIX/PCG) whose lines might have sparked it,
  plus the property insurers on the hook for claims (ALL/TRV/MCY/CB) — should fall, and
  fall harder than the market.
* **The seasonal half.** The whole late-summer-to-early-winter *fire window*
  (Jul->Dec) should carry a systematically worse basket return than the rest of the
  calendar — a "sell in July" for California risk.

The machinery, one execution lag documented throughout:

* ``abnormal_returns`` — a constant-mean market model (Brown & Warner 1985): the
  "normal" return is the sample mean, so the abnormal return is the demeaned series.
* ``basket_returns`` — equal-weight daily return of a set of tickers (utility leg,
  insurer leg, or the combined basket).
* ``event_window`` / ``stack_windows`` — the ``[-1..+5]`` abnormal-return path around
  each fire. ``event_date`` is snapped forward to the first NYSE session on/after the
  ignition date via ``searchsorted`` — the study's single documented execution lag: an
  ignition date is public before that session's close (a weekday breakout lands
  same-day, a weekend breakout rolls to the next open), so there is zero look-ahead.
* ``day0_stats`` — the ignition-day abnormal return itself: mean + one-sample *t*
  across events (events are independent, non-overlapping calendar dates).
* ``reversal_stats`` — the cumulative abnormal return over ``+1..+5``: the crucial
  distinction for THIS study — does the drop actually arrive *on* the ignition day (a
  same-day sentiment/risk repricing), or only in the days after, as the *liability*
  (who-caused-it) news lands (a slower fundamental repricing)?
* ``leg_compare`` — utility leg vs insurer leg on the same dates: which half of the
  basket carries whatever drop exists?
* ``extra_drop`` — pairs each event's SPY abnormal return with the basket's, and
  one-sample *t*'s the **difference** — does the CA-exposed basket fall harder than the
  market on a fire day?
* ``seasonal_test`` / ``seasonal_placebo`` — the Jul->Dec fire-window mean vs the rest
  of the year, with a random-window null.
* ``placebo_distribution`` — the random-calendar control: the same statistic on
  thousands of random non-fire dates, so the observed number's percentile is the
  falsification test.
* ``fire_timer`` — the tradable overlay: **short** the basket at the ignition-session
  close (fires are bad news for the basket, so the folklore trade is a short), ride
  ``hold`` sessions; shorts pay borrow, one round trip of one-way costs charged twice
  against NAV.

Costs are one-way x NAV per leg; the short leg pays a borrow fee.
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

    Demeaning removes the trivial drift of the basket so a post-fire CAR is not just
    "utilities pay a dividend" / "insurers grind up". The mean is taken over whatever
    the series actually covers (NaNs excluded).
    """
    return ret - ret.mean(skipna=True)


def basket_nav(series: dict[str, pd.Series], tickers, base: float = 100.0) -> pd.Series:
    """Equal-weight, daily-rebalanced total-return NAV of ``tickers`` (a price the timer
    can enter/exit). Built by compounding the equal-weight daily basket return."""
    r = basket_returns(series, tickers).fillna(0.0)
    return base * (1.0 + r).cumprod()


def basket_returns(series: dict[str, pd.Series], tickers) -> pd.Series:
    """Equal-weight daily return across ``tickers`` (whichever trade that day).

    Every basket ticker in this study has continuous history over the sample, so
    coverage is full on every event date — no survivorship backfill trick is needed or
    used. A day with zero coverage would be NaN (never happens here).
    """
    rets = pd.DataFrame({t: daily_returns(series[t]) for t in tickers})
    basket = rets.mean(axis=1, skipna=True)
    basket[rets.notna().sum(axis=1) == 0] = np.nan
    return basket


# --------------------------------------------------------------------------- #
# Event windows
# --------------------------------------------------------------------------- #
def event_window(ar: pd.Series, event_date: pd.Timestamp, pre: int = 1, post: int = 5
                  ) -> np.ndarray | None:
    """The abnormal-return path from ``-pre`` to ``+post`` sessions around a fire.

    ``event_date`` is the fire's ignition date; ``searchsorted`` snaps it to the first
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

    ``require_finite`` drops windows carrying any NaN instead of propagating them into
    the mean. Returns ``(matrix, kept_dates)``.
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


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson (1927) score interval for a k/n hit rate."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# The headline split — the ignition-day move
# --------------------------------------------------------------------------- #
def day0_stats(ar: pd.Series, event_dates, pre: int = 1, post: int = 5) -> dict:
    """Ignition-day abnormal return: mean + one-sample t across events."""
    w, kept = stack_windows(ar, event_dates, pre, post)
    if w.shape[0] == 0:
        return {"n": 0, "mean": float("nan"), "t": float("nan"), "kept_dates": []}
    day0 = w[:, pre]
    mean, t = one_sample_t(day0)
    hit_down = int((day0 < 0).sum())
    return {"n": w.shape[0], "mean": mean, "t": t, "kept_dates": kept,
            "hit_down": hit_down}


def reversal_stats(ar: pd.Series, event_dates, pre: int = 1, post: int = 5) -> dict:
    """Post-ignition [+1..+post] cumulative abnormal return: mean + one-sample t.

    For THIS study the key contrast: a big negative number here (with a flat day 0)
    says the drop is the *liability* news landing in the days AFTER ignition — a slower
    fundamental repricing, not a same-day sentiment/risk flinch.
    """
    w, kept = stack_windows(ar, event_dates, pre, post)
    if w.shape[0] == 0:
        return {"n": 0, "mean": float("nan"), "t": float("nan")}
    per_event = w[:, pre + 1: pre + 1 + post].sum(axis=1)
    mean, t = one_sample_t(per_event)
    return {"n": w.shape[0], "mean": mean, "t": t, "per_event": per_event}


def car_path_stats(ar: pd.Series, event_dates, pre: int = 1, post: int = 5) -> pd.DataFrame:
    """Mean CAR by offset (``-pre..+post``), with each offset's own one-sample t.

    CAR(k) = cumulative mean abnormal return from ``-pre`` through offset ``k``,
    re-anchored so CAR(``-pre``) == 0 — the standard event-study convention.
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
# Utility leg vs insurer leg — which half carries the drop?
# --------------------------------------------------------------------------- #
def leg_compare(ar_util: pd.Series, ar_ins: pd.Series, event_dates,
                pre: int = 1, post: int = 5, window: str = "post") -> dict:
    """Compare the utility leg and insurer leg over the same event dates.

    ``window='post'`` uses each leg's [+1..+post] cumulative abnormal return (the
    liability-repricing window); ``'day0'`` uses the ignition-day abnormal return.
    Returns each leg's mean + one-sample t and the paired (util - ins) difference.
    """
    def per(ar):
        w, kept = stack_windows(ar, event_dates, pre, post)
        if w.shape[0] == 0:
            return np.array([]), []
        if window == "day0":
            return w[:, pre], kept
        return w[:, pre + 1: pre + 1 + post].sum(axis=1), kept
    u, ku = per(ar_util)
    i, ki = per(ar_ins)
    # align on common kept dates (all tickers cover all dates here, so ku == ki)
    n = min(len(u), len(i))
    u, i = u[:n], i[:n]
    mu, tu = one_sample_t(u)
    mi, ti = one_sample_t(i)
    diff = u - i
    md, td = one_sample_t(diff)
    return {"n": n, "util_mean": mu, "util_t": tu, "ins_mean": mi, "ins_t": ti,
            "diff_mean": md, "diff_t": td, "util_arr": u, "ins_arr": i}


# --------------------------------------------------------------------------- #
# Basket vs market — does the CA basket fall harder than SPY?
# --------------------------------------------------------------------------- #
def extra_drop(ar_spy: pd.Series, ar_basket: pd.Series, event_dates,
               pre: int = 1, post: int = 5, window: str = "post") -> dict:
    """Paired (basket - SPY) abnormal-return difference: mean + one-sample t.

    ``window='post'`` uses the [+1..+post] cumulative abnormal return, ``'day0'`` the
    ignition-day one. Pairing removes the common market-wide move on the day.
    """
    idx = ar_spy.index.union(ar_basket.index)
    a_spy = ar_spy.reindex(idx)
    a_bk = ar_basket.reindex(idx)
    w_spy, ks = stack_windows(a_spy, event_dates, pre, post)
    w_bk, kb = stack_windows(a_bk, event_dates, pre, post)
    kept = sorted(set(ks) & set(kb))
    if not kept:
        return {"n": 0, "mean_diff": float("nan"), "t": float("nan")}
    w_spy2, _ = stack_windows(a_spy, kept, pre, post)
    w_bk2, _ = stack_windows(a_bk, kept, pre, post)
    if window == "day0":
        spy_v, bk_v = w_spy2[:, pre], w_bk2[:, pre]
    else:
        spy_v = w_spy2[:, pre + 1: pre + 1 + post].sum(axis=1)
        bk_v = w_bk2[:, pre + 1: pre + 1 + post].sum(axis=1)
    diff = bk_v - spy_v
    mean, t = one_sample_t(diff)
    return {"n": len(kept), "mean_diff": mean, "t": t,
            "basket_mean": float(np.nanmean(bk_v)), "spy_mean": float(np.nanmean(spy_v))}


# --------------------------------------------------------------------------- #
# The seasonal half — the Jul->Dec fire window
# --------------------------------------------------------------------------- #
def seasonal_test(ar: pd.Series, fire_months=(7, 8, 9, 10, 11, 12)) -> dict:
    """Mean daily abnormal return inside the fire window vs the rest of the year.

    Returns each side's mean (in bps/day), the Welch t of (in - out), and the counts.
    A real "sell in July" seasonal for California risk would show a materially negative
    in-window mean and a Welch |t| >= 2.
    """
    a = ar.dropna()
    inw = a[a.index.month.isin(fire_months)].to_numpy()
    outw = a[~a.index.month.isin(fire_months)].to_numpy()
    mi, _ = one_sample_t(inw)
    mo, _ = one_sample_t(outw)
    return {"in_mean": mi, "out_mean": mo, "diff": mi - mo,
            "t": welch_t(inw, outw), "n_in": inw.size, "n_out": outw.size}


def seasonal_placebo(ar: pd.Series, k_months: int = 6, n_draws: int = 2000,
                     seed: int = 739) -> np.ndarray:
    """Random-window null: the (in - out) mean-return gap for a random set of
    ``k_months`` calendar months, ``n_draws`` times. A real fire-season gap must sit in
    the tail of this distribution; sitting in the bulk means any 6-month slice of the
    calendar produces a gap this size anyway.
    """
    rng = np.random.default_rng(seed)
    a = ar.dropna()
    months = a.index.month.to_numpy()
    vals = a.to_numpy()
    out = np.empty(n_draws)
    for d in range(n_draws):
        pick = rng.choice(np.arange(1, 13), size=k_months, replace=False)
        mask = np.isin(months, pick)
        out[d] = vals[mask].mean() - vals[~mask].mean()
    return out


# --------------------------------------------------------------------------- #
# Random-calendar placebo (the event half's falsification control)
# --------------------------------------------------------------------------- #
def placebo_distribution(ar: pd.Series, n_events: int, pre: int = 1, post: int = 5,
                          n_draws: int = 2000, seed: int = 739,
                          stat: str = "day0") -> np.ndarray:
    """Random-date placebo: mean statistic over ``n_draws`` sets of ``n_events`` random
    non-fire dates. ``stat`` is ``"day0"`` (the ignition-day move) or ``"post_car"``
    (the [+1..+post] window). A real effect must sit in the tail.
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
                        seed: int = 739) -> tuple[float, float]:
    """Percentile CI on the mean per-event statistic (events resampled with replacement).

    Events are independent, far-apart calendar dates, so the resampling unit is the
    event. Wide relative to the mean = the result hinges on a handful of events.
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


def jackknife_range(per_event: np.ndarray) -> dict:
    """Leave-one-out sensitivity of the one-sample t: min/max |t| across dropping each
    event once, and how many drops push |t| below 2. Exposes outlier dependence.
    """
    x = per_event[np.isfinite(per_event)]
    n = x.size
    if n < 3:
        return {"t_min": float("nan"), "t_max": float("nan"), "n_below2": 0, "n": n}
    ts = []
    for i in range(n):
        _, t = one_sample_t(np.delete(x, i))
        ts.append(t)
    ts = np.asarray(ts)
    return {"t_min": float(ts.min()), "t_max": float(ts.max()),
            "n_below2": int((np.abs(ts) < 2).sum()), "n": n}


# --------------------------------------------------------------------------- #
# The tradable overlay — SHORT the basket on the ignition headline
# --------------------------------------------------------------------------- #
def fire_timer(close_basket: pd.Series, event_dates, hold: int = 5,
               cost_bps: float = 0.0, borrow_bps_annual: float = 0.0) -> pd.DataFrame:
    """Short the basket at the ignition-session close, hold ``hold`` sessions.

    The fire is known at the close of session 0 (see the module docstring's execution
    lag), so the short earns sessions ``+1..+hold`` — a single lag, applied once. A
    short profits when the basket falls: ``ret_short = -(P_exit/P_entry - 1)``. One
    round trip per event: one-way cost charged twice (entry + exit) against NAV, plus a
    borrow fee accrued over the holding days (annual bps prorated by hold/252).
    """
    idx = close_basket.index
    rows = []
    for d in pd.to_datetime(pd.Series(event_dates)):
        pos = idx.searchsorted(pd.Timestamp(d))
        if pos >= len(idx):
            continue
        entry, exit_ = pos, pos + hold
        if exit_ >= len(idx):
            continue
        long_ret = close_basket.iat[exit_] / close_basket.iat[entry] - 1.0
        gross_short = -long_ret
        borrow = borrow_bps_annual * 1e-4 * (hold / 252.0)
        net = gross_short - 2.0 * cost_bps * 1e-4 - borrow
        rows.append({"entry_date": idx[entry], "hold": hold,
                      "ret_gross": float(gross_short), "ret_net": float(net)})
    return pd.DataFrame(rows)


def summarize_timer(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline stats for the short-the-fire ledger: n, win-rate, mean bps, t-stat."""
    if ledger.empty:
        return {"n": 0, "win_rate": float("nan"), "mean_bps": float("nan"), "t": float("nan")}
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    mean, t = one_sample_t(r)
    return {"n": int(n), "win_rate": float((r > 0).mean()) if n else float("nan"),
            "mean_bps": mean * 1e4 if n else float("nan"), "t": t,
            "median_bps": float(np.median(r) * 1e4) if n else float("nan")}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(close: pd.DataFrame, events: pd.DatetimeIndex,
                      pre: int = 1, post: int = 5, stat: str = "day0") -> dict:
    """Run the headline detector on a synthetic world (day0 or the post window)."""
    ar = abnormal_returns(daily_returns(close["Close"]))
    if stat == "day0":
        return day0_stats(ar, events, pre, post)
    return reversal_stats(ar, events, pre, post)
