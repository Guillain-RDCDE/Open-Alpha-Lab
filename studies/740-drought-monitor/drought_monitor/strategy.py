"""The event-study + regime engine and its honest controls — Study 740 (Drought-Monitor).

The claim under test, steelmanned: **a worsening US Drought Monitor print is tradable
supply-shock news for the ag complex.** When the weekly Drought Monitor shows severe
drought (D2+) rapidly expanding across the US crop belt, the harvest outlook worsens,
grain gets scarcer and pricier, and the ag names (Deere, Mosaic, ADM, the agribusiness
ETF) and grain itself (the DBA/CORN/WEAT basket) should rise — you could read the
Thursday print and buy the drought.

The machinery, one execution lag documented throughout. Two "abnormal return" ideas:
the basket's move *relative to SPY* (a beta-1 market model, ``abnormal_vs_bench``), so a
result is "ag outperformed the market on the print", not "stocks went up".

* ``basket_returns`` — equal-weight daily return of whichever tickers in a basket are
  trading that day, with a coverage count (no survivorship backfill before an ETF's
  inception; a zero-coverage day is NaN and its events are dropped, named honestly).
* ``event_window`` / ``stack_windows`` — the ``[-1..+POST]`` abnormal-return path around
  each drought print. ``event_date`` (the Thursday USDM release) is snapped forward to
  the first NYSE session on/after it via ``searchsorted`` — the study's single documented
  execution lag: the print is public that Thursday morning (~8:30 ET, before the close),
  so entering at session-0's close is zero look-ahead.
* ``car_path_stats`` — the mean cumulative abnormal-return path, anchored at 0 the
  session before the print — the headline "pop and fade?" chart, with each offset's own
  one-sample *t*.
* ``day0_stats`` / ``post_event_car`` — the print-day abnormal return and the cumulative
  ``+1..+POST`` reaction: one-sample *t* across events (independent, non-overlapping
  Thursday prints — the correct unit, not a daily panel).
* ``basket_extra_move`` — pairs each event's grain-basket day-0 abnormal return with the
  ag-equity basket's day-0 abnormal return (same date, same snap) and one-sample-*t*'s
  the **difference** — does grain react harder than the equities on the same drought news?
  (the study's third, myth-check axis).
* ``placebo_distribution`` — the random-calendar control: the same statistic on thousands
  of random non-drought dates, so the observed number's percentile is the falsification.
* ``trade_it`` — the costed overlay: long the basket at the print-session close, hold
  ``hold`` sessions, one round trip of one-way costs charged twice against NAV, long-only
  (no borrow), gross AND net reported, vs the unconditional same-horizon baseline.
* ``regime_stats`` — the drought-*regime* test on the labelled monthly proxy: split
  months by drought severity **known at the month's start** (one shift, no look-ahead)
  and compare that month's forward ag-basket abnormal return, high-drought vs the rest.

Costs are one-way × NAV per leg; the overlays are long-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEK_K = 5      # trading sessions ~ 1 week
POST = 5        # event-window post horizon
COST_BPS = 5.0  # one-way, per leg


# --------------------------------------------------------------------------- #
# Returns + baskets + abnormal (beta-1 market model vs SPY)
# --------------------------------------------------------------------------- #
def daily_returns(close: pd.Series) -> pd.Series:
    """Simple close-to-close daily returns."""
    return close.pct_change()


def basket_returns(names: dict[str, pd.Series], tickers) -> tuple[pd.Series, pd.Series]:
    """Equal-weight daily return of whichever ``tickers`` are trading that day.

    Returns ``(basket_return, coverage)`` where ``coverage`` is the count of tickers
    with a valid return on that date. No survivorship trick backfills a ticker before
    its inception (DBA 2007-01, MOO 2007-09, CORN 2010-06, WEAT 2011-09, MOS from its
    2004 IPO) — a day with 0 coverage is NaN, and events landing in a zero-coverage
    window are dropped from that basket's test, never silently zero-filled.
    """
    rets = pd.DataFrame({t: daily_returns(names[t]) for t in tickers})
    basket = rets.mean(axis=1, skipna=True)
    basket[rets.notna().sum(axis=1) == 0] = np.nan
    coverage = rets.notna().sum(axis=1)
    return basket, coverage


def abnormal_vs_bench(basket_ret: pd.Series, bench_ret: pd.Series) -> pd.Series:
    """Abnormal return = basket return minus the SPY benchmark return (a beta-1 market
    model). Both series are aligned on the union calendar first, so a "kept" event has a
    same-day basket AND SPY observation. Isolates the ag-specific move from the market's
    move that day — the claim predicts ag *outperformance* on drought news, not just "up".
    """
    idx = basket_ret.index.union(bench_ret.index)
    return basket_ret.reindex(idx) - bench_ret.reindex(idx)


# --------------------------------------------------------------------------- #
# Event windows
# --------------------------------------------------------------------------- #
def event_window(ar: pd.Series, event_date: pd.Timestamp, pre: int = 1, post: int = POST
                  ) -> np.ndarray | None:
    """The abnormal-return path from ``-pre`` to ``+post`` sessions around a print.

    ``event_date`` is the USDM Thursday release; ``searchsorted`` snaps it to the first
    session on/after that date (session 0). Returns a 1-D array of length
    ``pre + post + 1``, or ``None`` if the window runs off the edge of the tape.
    """
    idx = ar.index
    pos = idx.searchsorted(pd.Timestamp(event_date))
    if pos >= len(idx):
        return None
    lo, hi = pos - pre, pos + post
    if lo < 0 or hi >= len(idx):
        return None
    return ar.to_numpy()[lo: hi + 1]


def stack_windows(ar: pd.Series, event_dates, pre: int = 1, post: int = POST,
                   require_finite: bool = True) -> tuple[np.ndarray, list]:
    """Stack every valid event window into a ``(n_events, pre+post+1)`` matrix.

    ``require_finite`` drops windows carrying any NaN (e.g. a grain-basket window that
    predates every grain ETF's inception) instead of propagating them into the mean.
    Returns ``(matrix, kept_dates)``.
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
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# The headline split — the print-day abnormal move
# --------------------------------------------------------------------------- #
def day0_stats(ar: pd.Series, event_dates, pre: int = 1, post: int = POST) -> dict:
    """Print-day abnormal return: mean + one-sample t + down/up hit rate across events."""
    w, kept = stack_windows(ar, event_dates, pre, post)
    if w.shape[0] == 0:
        return {"n": 0, "mean": float("nan"), "t": float("nan"),
                "up": 0, "lo": float("nan"), "hi": float("nan"), "kept_dates": []}
    day0 = w[:, pre]
    mean, t = one_sample_t(day0)
    up = int((day0 > 0).sum())
    lo, hi = wilson_interval(up, w.shape[0])
    return {"n": w.shape[0], "mean": mean, "t": t, "up": up, "lo": lo, "hi": hi,
            "kept_dates": kept}


def post_event_car(ar: pd.Series, event_dates, pre: int = 1, post: int = POST) -> dict:
    """Post-print [+1..+post] cumulative abnormal return: mean + one-sample t.

    A positive, significant number says the ag complex drifts up in the days after a
    drought print (the tradable half of the claim); near zero says nothing sticks.
    """
    w, kept = stack_windows(ar, event_dates, pre, post)
    if w.shape[0] == 0:
        return {"n": 0, "mean": float("nan"), "t": float("nan")}
    per_event = w[:, pre + 1: pre + 1 + post].sum(axis=1)
    mean, t = one_sample_t(per_event)
    return {"n": w.shape[0], "mean": mean, "t": t}


def car_path_stats(ar: pd.Series, event_dates, pre: int = 1, post: int = POST) -> pd.DataFrame:
    """Mean CAR by offset (``-pre..+post``), each offset's own one-sample t.

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
# Third axis — does grain react harder than the ag-equities on the same print?
# --------------------------------------------------------------------------- #
def basket_extra_move(ar_equity: pd.Series, ar_grain: pd.Series, event_dates,
                       pre: int = 1, post: int = POST) -> dict:
    """Paired day-0 (grain - ag-equity) abnormal-return difference: mean + one-sample t.

    Both abnormal series are windowed on the same snapped print dates; a "kept" event
    has a same-day observation for BOTH baskets. Events predating the grain ETFs'
    inception (0 grain coverage) are dropped, not zero-filled — the ``n`` here is
    typically smaller than the equity-only n.
    """
    w_eq, kept_eq = stack_windows(ar_equity, event_dates, pre, post)
    w_gr, kept_gr = stack_windows(ar_grain, event_dates, pre, post)
    kept_both = sorted(set(kept_eq) & set(kept_gr))
    if not kept_both:
        return {"n": 0, "mean_diff": float("nan"), "t": float("nan"),
                "grain_mean": float("nan"), "equity_mean": float("nan")}
    w_eq2, _ = stack_windows(ar_equity, kept_both, pre, post)
    w_gr2, _ = stack_windows(ar_grain, kept_both, pre, post)
    diff = w_gr2[:, pre] - w_eq2[:, pre]
    mean, t = one_sample_t(diff)
    return {"n": len(kept_both), "mean_diff": mean, "t": t,
            "grain_mean": float(np.nanmean(w_gr2[:, pre])),
            "equity_mean": float(np.nanmean(w_eq2[:, pre]))}


# --------------------------------------------------------------------------- #
# Random-calendar placebo / falsification
# --------------------------------------------------------------------------- #
def placebo_distribution(ar: pd.Series, n_events: int, pre: int = 1, post: int = POST,
                          n_draws: int = 2000, seed: int = 740,
                          stat: str = "day0") -> np.ndarray:
    """Random-date placebo: mean statistic over ``n_draws`` sets of ``n_events`` random
    non-drought dates. ``stat`` is ``"day0"`` (the print-day move) or ``"post_car"`` (the
    +1..+post drift). A real effect must sit in the tail; sitting in the bulk means the
    "effect" is what a random calendar of the same size produces anyway.
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


def placebo_pvalue(observed: float, placebo: np.ndarray, tail: str = "right") -> float:
    """Empirical one-sided p-value of ``observed`` within the placebo draws.
    ``tail='right'`` for a claim predicting a POSITIVE move (share of null >= observed).
    """
    if placebo.size == 0 or not np.isfinite(observed):
        return float("nan")
    if tail == "right":
        return float((placebo >= observed).mean())
    return float((placebo <= observed).mean())


def block_bootstrap_ci(per_event: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
                        seed: int = 740) -> tuple[float, float]:
    """Percentile CI on the mean per-event statistic (events resampled with replacement).
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
# The costed overlay — "buy the drought" after a print
# --------------------------------------------------------------------------- #
def trade_it(basket_ret: pd.Series, bench_ret: pd.Series, event_dates, hold: int = 5,
             cost_bps: float = 0.0) -> pd.DataFrame:
    """Long-the-basket overlay (excess of SPY): enter at the print-session close, hold
    ``hold`` sessions. The print is known at session-0's close (see the module docstring's
    execution lag), so the position earns sessions ``+1..+hold`` of the basket's abnormal
    (basket - SPY) return — a single lag, applied once. One round trip per event: one-way
    cost charged twice against NAV.
    """
    ar = abnormal_vs_bench(basket_ret, bench_ret)
    idx = ar.index
    rows = []
    for d in pd.to_datetime(pd.Series(event_dates)):
        pos = idx.searchsorted(pd.Timestamp(d))
        if pos >= len(idx):
            continue
        entry, exit_ = pos, pos + hold
        if exit_ >= len(idx):
            continue
        seg = ar.to_numpy()[entry + 1: exit_ + 1]
        if not np.all(np.isfinite(seg)):
            continue
        gross = float(np.nansum(seg))
        net = gross - 2.0 * cost_bps * 1e-4
        rows.append({"entry_date": idx[entry], "hold": hold,
                      "ret_gross": gross, "ret_net": net})
    return pd.DataFrame(rows)


def summarize_trade(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline stats for a trade ledger: n, win-rate, mean (bps), t-stat."""
    if ledger.empty:
        return {"n": 0, "win_rate": float("nan"), "mean_bps": float("nan"), "t": float("nan")}
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    mean, t = one_sample_t(r)
    return {"n": int(n), "win_rate": float((r > 0).mean()) if n else float("nan"),
            "mean_bps": mean * 1e4 if n else float("nan"), "t": t}


# --------------------------------------------------------------------------- #
# The drought-regime test on the labelled monthly proxy
# --------------------------------------------------------------------------- #
def monthly_abnormal(basket_ret: pd.Series, bench_ret: pd.Series) -> pd.Series:
    """Month-end abnormal (basket - SPY) simple return, months with full basket coverage.

    Compounds daily abnormal returns within each calendar month. Used only by the
    regime test against the labelled monthly drought proxy.
    """
    ar = abnormal_vs_bench(basket_ret, bench_ret).dropna()
    grp = (1.0 + ar).groupby(ar.index.to_period("M")).prod() - 1.0
    grp.index = grp.index.to_timestamp("M")
    return grp


def regime_stats(proxy: pd.Series, monthly_ar: pd.Series, hi_pct: float = 66.0,
                 cost_bps: float = 5.0) -> dict:
    """High-drought vs low-drought monthly ag-abnormal-return split, NO look-ahead.

    The drought level that decides month M's regime is the proxy value **known at the
    start of M** (i.e. the prior month-end reading — one ``shift``), so the regime is set
    before the month's return is earned. ``hi_pct`` is the percentile of that lagged proxy
    above which a month is "high drought". Returns the two group means, a Welch *t* of the
    difference, and a costed long-only "hold ag only in high-drought months" timer
    (one round trip per entry month, 2× one-way cost × NAV) vs always-hold.
    """
    lagged = proxy.shift(1)                        # regime known at month start
    df = pd.DataFrame({"drought": lagged, "ar": monthly_ar}).dropna()
    if len(df) < 6:
        return {"n": 0}
    thr = np.percentile(df["drought"].to_numpy(), hi_pct)
    hi = df[df["drought"] >= thr]["ar"].to_numpy()
    lo = df[df["drought"] < thr]["ar"].to_numpy()
    t = welch_t(hi, lo)
    # costed timer: long ag (excess of SPY) only in high-drought months
    n_entries = len(hi)
    gross = float(np.mean(hi)) if len(hi) else float("nan")
    net = gross - 2.0 * cost_bps * 1e-4            # one round trip per high-drought month
    _, t_hi = one_sample_t(hi)
    return {"n": len(df), "thr": float(thr), "n_hi": len(hi), "n_lo": len(lo),
            "hi_mean": float(np.mean(hi)) if len(hi) else float("nan"),
            "lo_mean": float(np.mean(lo)) if len(lo) else float("nan"),
            "welch_t": t, "hi_t": t_hi,
            "hi_net_mean": net, "n_entries": n_entries}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(close: pd.DataFrame, events: pd.DatetimeIndex,
                      pre: int = 1, post: int = POST) -> dict:
    """Run the headline day-0 test on a synthetic world (abnormal = demeaned return, so
    the single-series synthetic tape needs no separate benchmark)."""
    ret = daily_returns(close["Close"])
    ar = ret - ret.mean(skipna=True)
    return day0_stats(ar, events, pre, post)
