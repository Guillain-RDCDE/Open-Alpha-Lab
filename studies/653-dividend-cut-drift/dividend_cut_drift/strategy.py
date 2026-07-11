"""Strategy + inference for Study 653 — Dividend-Cut-Drift.

The claim, in its "never catch a falling dividend" form: a stock that CUTS or OMITS its
dividend keeps underperforming afterward — management just told you, in the clearest way a
board can, that the outlook is worse than priced. The bounce-back counter-claim: dividend cuts
are backward-looking, bad news is already in the price by the ex-date, and a beaten-down
"cutter" is a contrarian buy.

**Event detection** (no look-ahead: an event is only flagged once the payment record shows it —
the same information a dividend-tracking service would have on the ex-dividend date):

* **Cut** — a scheduled dividend payment `<= cut_ratio` (default 0.70) times the prior payment.
* **Omission** — a gap between two consecutive payments that is `>= gap_mult` (default 1.8)
  times the ticker's own trailing typical inter-payment interval (computed from history strictly
  before the gap), restricted to tickers with a genuinely regular (sub-200-day) cadence so an
  annual/irregular payer doesn't manufacture false omissions. The event date is set to when the
  next payment was expected and didn't arrive.

Adjacent events on the same ticker within ``min_gap_days`` are de-duplicated (kept: the first)
so overlapping [-20, +120] windows don't double-count one underlying episode.

**Event study.** Abnormal return = ticker log return − SPY log return (both auto-adjusted /
total-return, so the ex-date's mechanical price drop never enters the math). CAR is cumulative
AR over trading-day offsets [-20, +120] relative to the event day (0). Pre-event drift is CAR at
offset −1 (cumulative [-20..-1]); post-event drift at a horizon h is CAR[h] − CAR[0] — i.e. it
excludes the event-day return itself, consistent with the study's ONE documented execution lag:
the cut is only knowable from the close of the event day, so a trade enters the NEXT session.

**Inference.** A cross-sectional one-sample t-test on each event's CAR at fixed horizons (the
planned primary — events are non-overlapping after de-dup, though the sample period is shared,
which is exactly why we cross-check with a calendar-time portfolio); a Newey-West (HAC) t on the
daily equal-weighted "cutters" portfolio abnormal return (handles the overlap the cross-sectional
test can't); a Wilson-bounded hit rate; and a random-date placebo.

**Two tradable expressions**, one execution lag (enter at the CLOSE of the trading session AFTER
the event day — the cut is only knowable from that close): short-the-cutter (borrow + costs both
ways) and buy-the-cutter (long, costs both ways, benchmarked excess-of-SPY over the identical
window).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inference primitives (shared shapes with the rest of the desk)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(y: np.ndarray, lags: int = 5) -> float:
    """HAC (Newey-West, Bartlett kernel) t that the mean of ``y`` is 0 (intercept-only OLS)."""
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)
    if n < 3:
        return float("nan")
    X = np.ones((n, 1))
    beta = y.mean()
    u = y - beta
    s = X * u[:, None]
    S = s.T @ s
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        G = s[l:].T @ s[:-l]
        S += w * (G + G.T)
    XtX_inv = np.array([[1.0 / n]])
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(V[0, 0])
    return float(beta / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Event detection
# --------------------------------------------------------------------------- #
def strip_special_dividends(div: pd.Series, special_mult: float = 1.8,
                             max_passes: int = 3) -> pd.Series:
    """Drop one-off special-dividend / stub-payment artifacts from the regular payment record.

    Two shapes, both seen live in this universe:

    * A **spike-then-revert** — a special dividend or (as Yahoo records it) a spinoff cash
      distribution folded into the same ``Dividends`` stream (Citigroup's Aug-2002 Travelers
      Property Casualty spinoff: a $24.15 entry between two $1.80 regular payments). Left in, it
      manufactures a false *cut* on the payment right after it.
    * A **dip-then-revert** — a stray stub/glitch payment far BELOW its neighbors that the very
      next regular payment reverts straight back up from (Exxon's one-off $0.01 between two
      $0.22 quarters; Medtronic's $0.001 stub between two $0.05 quarters; Philip Morris' split
      year-end stub pair $0.039/$0.085 between $0.85 quarters). Left in, this manufactures a
      false *cut* directly (the tiny payment itself reads as an 80%+ drop).

    A payment is flagged (and dropped from the series used for cut/omission detection) only when
    it sits at least ``special_mult`` away from BOTH neighbors, on the SAME side (both spike or
    both dip) — a real, sustained cut or raise (which does not revert) is never flagged. Applied
    over up to ``max_passes`` iterations so an adjacent PAIR of stub payments (the PM case above)
    gets fully cleaned once the first pass exposes the second one's true neighbors.
    """
    d = div[div > 0].sort_index()
    for _ in range(max_passes):
        if len(d) < 3:
            break
        vals = d.to_numpy()
        keep = np.ones(len(d), dtype=bool)
        for i in range(1, len(d) - 1):
            spike = vals[i] >= special_mult * vals[i - 1] and vals[i] >= special_mult * vals[i + 1]
            dip = vals[i] <= vals[i - 1] / special_mult and vals[i] <= vals[i + 1] / special_mult
            if spike or dip:
                keep[i] = False
        if keep.all():
            break
        d = d[keep]
    return d


def detect_cuts(div: pd.Series, cut_ratio: float = 0.70, min_prior: float = 0.01,
                 warmup: int = 4) -> pd.DataFrame:
    """Scheduled payments <= cut_ratio x the prior payment. No look-ahead: each event uses only
    the payment record up to and including the event's own ex-date. ``warmup`` requires at least
    that many payments of established history before the "prior" payment is trusted as a real
    baseline — without it, a freshly-initiated payer's first irregular special distributions
    (e.g. Wynn Resorts' $6/$4/$8 one-off specials before it started a genuine $0.25 regular
    quarterly in 2010) read as a string of 90%+ "cuts" that never happened."""
    d = strip_special_dividends(div)
    if len(d) < max(2, warmup + 1):
        return pd.DataFrame(columns=["event_date", "type", "prior_amt", "new_amt", "ratio"])
    amt = d.to_numpy()
    dates = d.index
    rows = []
    for i in range(warmup, len(d)):
        prior, new = float(amt[i - 1]), float(amt[i])
        if prior < min_prior:
            continue
        ratio = new / prior
        if ratio <= cut_ratio:
            rows.append({"event_date": dates[i], "type": "cut",
                         "prior_amt": prior, "new_amt": new, "ratio": ratio})
    return pd.DataFrame(rows)


def detect_omissions(div: pd.Series, gap_mult: float = 1.8, min_history: int = 6,
                      max_typical_days: float = 200.0) -> pd.DataFrame:
    """A gap between consecutive payments >= gap_mult x the ticker's own trailing typical
    interval (computed strictly before the gap, so no look-ahead), restricted to regular
    (sub-``max_typical_days``) payers. Event date = last payment + the typical interval — the
    date the market would have expected (and didn't get) the next payment."""
    d = strip_special_dividends(div)
    if len(d) < min_history + 1:
        return pd.DataFrame(columns=["event_date", "type", "prior_amt", "new_amt", "ratio"])
    dates = d.index
    amt = d.to_numpy()
    gaps = np.diff(dates.values).astype("timedelta64[D]").astype(float)
    rows = []
    for j in range(min_history, len(gaps)):
        typical = float(np.median(gaps[max(0, j - min_history):j]))
        if typical <= 0 or typical > max_typical_days:
            continue
        if gaps[j] >= gap_mult * typical:
            event_date = dates[j] + pd.Timedelta(days=typical)
            if event_date >= dates[j + 1]:
                event_date = dates[j + 1] - pd.Timedelta(days=1)
            rows.append({"event_date": pd.Timestamp(event_date), "type": "omission",
                         "prior_amt": float(amt[j]), "new_amt": 0.0, "ratio": 0.0})
    return pd.DataFrame(rows)


def _dedup(df: pd.DataFrame, min_gap_days: int = 90) -> pd.DataFrame:
    """Keep only the first event in any cluster of same-ticker events within ``min_gap_days``
    (overlapping [-20,+120] windows would otherwise double-count one underlying episode)."""
    if df.empty:
        return df
    keep = []
    for _, g in df.sort_values("event_date").groupby("ticker"):
        last_kept = None
        for _, row in g.iterrows():
            if last_kept is None or (row["event_date"] - last_kept).days >= min_gap_days:
                keep.append(row)
                last_kept = row["event_date"]
    return pd.DataFrame(keep).sort_values("event_date").reset_index(drop=True)


def build_event_table(divs: dict[str, pd.Series], cut_ratio: float = 0.70,
                       min_prior: float = 0.01, gap_mult: float = 1.8, min_history: int = 6,
                       min_gap_days: int = 90) -> pd.DataFrame:
    """All cut + omission events across the universe, ticker-tagged, de-duplicated."""
    rows = []
    for t, d in divs.items():
        c = detect_cuts(d, cut_ratio, min_prior)
        o = detect_omissions(d, gap_mult, min_history)
        for frame in (c, o):
            if len(frame):
                frame = frame.copy()
                frame["ticker"] = t
                rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["ticker", "event_date", "type", "prior_amt", "new_amt", "ratio"])
    out = pd.concat(rows, ignore_index=True)
    return _dedup(out, min_gap_days)


# --------------------------------------------------------------------------- #
# Event study — abnormal returns vs SPY
# --------------------------------------------------------------------------- #
def ar_series(px: pd.Series, spy: pd.Series) -> pd.Series:
    """Daily log abnormal return (ticker − SPY) on their shared trading-day index."""
    idx = px.index.intersection(spy.index)
    r_stock = np.log(px.reindex(idx)).diff()
    r_spy = np.log(spy.reindex(idx)).diff()
    return (r_stock - r_spy).dropna()


def event_car(ar: pd.Series, event_date: pd.Timestamp, lo: int = -20, hi: int = 120
              ) -> pd.Series | None:
    """Cumulative AR by offset in [lo, hi], offset 0 = the first trading day on/after
    ``event_date``. None if the tape doesn't have the full window (dropped, not padded)."""
    idx = ar.index
    after = idx[idx >= event_date]
    if len(after) == 0:
        return None
    ed = after[0]
    pos = idx.get_loc(ed)
    if pos + lo < 0 or pos + hi >= len(idx):
        return None
    window = idx[pos + lo: pos + hi + 1]
    sub = ar.reindex(window)
    car = sub.cumsum()
    car.index = np.arange(lo, hi + 1)
    return car


def build_cars(px_map: dict[str, pd.Series], spy: pd.Series, events: pd.DataFrame,
                lo: int = -20, hi: int = 120) -> tuple[pd.DataFrame, list[str]]:
    """CAR matrix (events x offsets) for every event whose full window is on the tape, plus the
    matching event_date used per row (for the timed-trade backtests)."""
    ar_cache: dict[str, pd.Series] = {}
    rows = []
    kept_dates = []
    for _, ev in events.iterrows():
        t = ev["ticker"]
        if t not in px_map:
            continue
        if t not in ar_cache:
            ar_cache[t] = ar_series(px_map[t], spy)
        car = event_car(ar_cache[t], ev["event_date"], lo, hi)
        if car is None:
            continue
        rows.append(car)
        kept_dates.append((t, ev["event_date"], ev["type"]))
    if not rows:
        return pd.DataFrame(columns=range(lo, hi + 1)), []
    mat = pd.DataFrame(rows)
    mat.columns = range(lo, hi + 1)
    mat.index = range(len(mat))
    return mat, kept_dates


def horizon_stats(car_mat: pd.DataFrame) -> dict:
    """Cross-sectional one-sample t at fixed horizons + Wilson-bounded hit rate on the
    [+1, +120] drift (post-event only, excludes the event-day return)."""
    if car_mat.empty:
        return {}
    pre = car_mat[-1].to_numpy()                              # CAR[-20..-1]
    post20 = (car_mat[20] - car_mat[0]).to_numpy()             # CAR[+1..+20]
    post60 = (car_mat[60] - car_mat[0]).to_numpy()             # CAR[+1..+60]
    post120 = (car_mat[120] - car_mat[0]).to_numpy()           # CAR[+1..+120]
    n = len(car_mat)
    k_neg = int((post120 < 0).sum())
    lo, hi = wilson_interval(k_neg, n)
    return {
        "n": n,
        "pre_mean": float(np.nanmean(pre)), "pre_t": one_sample_t(pre),
        "post20_mean": float(np.nanmean(post20)), "post20_t": one_sample_t(post20),
        "post60_mean": float(np.nanmean(post60)), "post60_t": one_sample_t(post60),
        "post120_mean": float(np.nanmean(post120)), "post120_t": one_sample_t(post120),
        "hit_neg": k_neg, "hit_rate": k_neg / n, "hit_lo": lo, "hit_hi": hi,
    }


def calendar_time_nw_t(px_map: dict[str, pd.Series], spy: pd.Series, kept: list[tuple],
                        hold: int = 120, lags: int = 5) -> dict:
    """Calendar-time equal-weight 'cutters' portfolio: on each trading day, average the AR of
    every event currently inside its [+1, +hold] post-event window; Newey-West t that the daily
    mean is 0. This is the overlap-robust cross-check to the cross-sectional t (events share
    calendar time, so their CARs are not independent draws)."""
    ar_cache: dict[str, pd.Series] = {}
    idx = spy.index
    daily: dict[pd.Timestamp, list[float]] = {}
    for t, ev_date, _ in kept:
        if t not in ar_cache:
            ar_cache[t] = ar_series(px_map[t], spy)
        ar = ar_cache[t]
        after = ar.index[ar.index >= ev_date]
        if len(after) == 0:
            continue
        pos = ar.index.get_loc(after[0])
        window = ar.index[pos + 1: pos + 1 + hold]
        for d in window:
            daily.setdefault(d, []).append(float(ar.loc[d]))
    if not daily:
        return {"n_days": 0, "mean_daily_bps": float("nan"), "nw_t": float("nan")}
    days = sorted(daily)
    port = np.array([np.mean(daily[d]) for d in days])
    return {
        "n_days": len(port),
        "mean_daily_bps": float(port.mean()) * 1e4,
        "nw_t": newey_west_t(port, lags=lags),
    }


def daily_welch(px_map: dict[str, pd.Series], spy: pd.Series, kept: list[tuple],
                 hold: int = 120) -> dict:
    """Welch t (the desk's other planned-primary shape): pool every ticker-day AR observation
    inside a [+1, +hold] post-event window ("treatment") against every OTHER ticker-day AR
    observation for the SAME tickers ("control" — their own non-event days), then a two-sample
    Welch t on the pooled daily-level distributions. A third, differently-shaped cut on the same
    question: unlike the cross-sectional test (one number per event) or the calendar-time NW test
    (one number per calendar day, averaged across simultaneous events), this pools every raw
    daily observation, treatment vs control, exactly like the desk's other Welch-based studies."""
    ar_cache: dict[str, pd.Series] = {}
    treat_days: dict[str, set] = {}
    for t, ev_date, _ in kept:
        if t not in ar_cache:
            ar_cache[t] = ar_series(px_map[t], spy)
        ar = ar_cache[t]
        after = ar.index[ar.index >= ev_date]
        if len(after) == 0:
            continue
        pos = ar.index.get_loc(after[0])
        window = ar.index[pos + 1: pos + 1 + hold]
        treat_days.setdefault(t, set()).update(window)

    treat_vals, control_vals = [], []
    for t, days in treat_days.items():
        ar = ar_cache[t]
        is_treat = ar.index.isin(days)
        treat_vals.append(ar.to_numpy()[is_treat])
        control_vals.append(ar.to_numpy()[~is_treat])
    a = np.concatenate(treat_vals) if treat_vals else np.array([])
    b = np.concatenate(control_vals) if control_vals else np.array([])
    return {"n_treat": len(a), "n_control": len(b),
            "mean_treat_bps": float(a.mean()) * 1e4 if len(a) else float("nan"),
            "mean_control_bps": float(b.mean()) * 1e4 if len(b) else float("nan"),
            "welch_t": welch_t(a, b)}


def random_date_placebo(px_map: dict[str, pd.Series], spy: pd.Series, tickers: list[str],
                         n_events: int, lo: int = -20, hi: int = 120,
                         n_draws_per_seed: int = 200, n_seeds: int = 20, base_seed: int = 653,
                         ) -> dict:
    """Placebo: draw ``n_events`` random (ticker, date) pairs (uniform over each ticker's tape,
    away from the edges so the window fits), compute mean post-``hi`` CAR (excludes the "event"
    day return, matching ``horizon_stats``), repeat n_seeds x n_draws_per_seed times. p = share
    of placebo means <= the observed mean (the claim is a NEGATIVE drift, left-tailed test).

    Prefix-sum vectorized: post_h(pos) = sum(ar[pos+1 : pos+h+1]) = prefix[pos+h+1] - prefix[pos+1]
    lets every draw be an O(1) lookup instead of re-summing a 141-day window from scratch."""
    prefix: dict[str, np.ndarray] = {}
    length: dict[str, int] = {}
    for t in tickers:
        if t not in px_map:
            continue
        ar = ar_series(px_map[t], spy).to_numpy()
        if len(ar) <= (hi - lo) + 50:
            continue
        prefix[t] = np.concatenate([[0.0], np.cumsum(ar)])
        length[t] = len(ar)
    names = np.array(list(prefix.keys()))
    means = np.empty(n_seeds * n_draws_per_seed)
    k = 0
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            picks = rng.choice(names, size=n_events, replace=True)
            vals = np.empty(n_events)
            for j, name in enumerate(picks):
                n = length[name]
                pos = int(rng.integers(-lo + 1, n - hi - 1))
                pre = prefix[name]
                vals[j] = pre[pos + hi + 1] - pre[pos + 1]
            means[k] = vals.mean()
            k += 1
    return {"placebo_mean": float(means.mean()), "placebo_sd": float(means.std(ddof=1)),
            "n_draws": len(means), "draws": means}


def placebo_pvalue(observed_mean: float, placebo: dict, draws: np.ndarray | None = None) -> float:
    if draws is not None:
        return float((draws <= observed_mean).mean())
    # Normal-approx fallback using the placebo mean/sd (used when raw draws weren't kept)
    from math import erf, sqrt
    z = (observed_mean - placebo["placebo_mean"]) / placebo["placebo_sd"]
    return float(0.5 * (1 + erf(z / sqrt(2))))


# --------------------------------------------------------------------------- #
# Tradable expressions — one documented execution lag (enter the close AFTER the event day)
# --------------------------------------------------------------------------- #
def timed_trade(px: pd.Series, spy: pd.Series, event_date: pd.Timestamp, side: str,
                 entry_lag: int = 1, hold: int = 120, cost_bps: float = 5.0,
                 borrow_bps_annual: float = 50.0) -> dict | None:
    """One event's net return for ``side`` in {'short', 'long'}. Entry = close of the session
    ``entry_lag`` trading days after the event day (the cut is only knowable from that close);
    exit = close ``hold`` trading days after entry. Costs = 2 x one-way x cost_bps (round trip);
    shorts additionally pay ``borrow_bps_annual`` accrued over the holding period."""
    idx = px.index
    after = idx[idx >= event_date]
    if len(after) == 0:
        return None
    pos = idx.get_loc(after[0])
    entry_pos, exit_pos = pos + entry_lag, pos + entry_lag + hold
    if entry_pos >= len(idx) or exit_pos >= len(idx):
        return None
    p_in, p_out = float(px.iloc[entry_pos]), float(px.iloc[exit_pos])
    spy_idx = spy.index
    spy_after = spy_idx[spy_idx >= idx[entry_pos]]
    spy_exit_after = spy_idx[spy_idx >= idx[exit_pos]]
    if len(spy_after) == 0 or len(spy_exit_after) == 0:
        return None
    s_in, s_out = float(spy.loc[spy_after[0]]), float(spy.loc[spy_exit_after[0]])
    spy_ret = s_out / s_in - 1.0

    if side == "short":
        gross = -(p_out / p_in - 1.0)
        borrow = borrow_bps_annual / 1e4 * (hold / TRADING_DAYS)
        net = gross - 2 * cost_bps / 1e4 - borrow
        # matched exposure for a directional short = shorting SPY over the identical window
        # (is the stock-specific call worth anything beyond generically being short the market?)
        matched = -spy_ret
    else:
        gross = p_out / p_in - 1.0
        net = gross - 2 * cost_bps / 1e4
        matched = spy_ret

    return {"entry_date": idx[entry_pos], "exit_date": idx[exit_pos],
            "gross": gross, "net": net, "spy_ret": spy_ret, "excess": net - matched}


def backtest(px_map: dict[str, pd.Series], spy: pd.Series, kept: list[tuple], side: str,
             entry_lag: int = 1, hold: int = 120, cost_bps: float = 5.0,
             borrow_bps_annual: float = 50.0) -> dict:
    """Aggregate ``timed_trade`` across every kept event; t on net (and on excess-vs-SPY for
    longs), hit rate, and an approximate annualized Sharpe using the sample's own event
    frequency (events/yr) since positions do not overlap 1:1 across the calendar."""
    trades = []
    for t, ev_date, _ in kept:
        if t not in px_map:
            continue
        tr = timed_trade(px_map[t], spy, ev_date, side, entry_lag, hold, cost_bps,
                          borrow_bps_annual)
        if tr is not None:
            trades.append(tr)
    if not trades:
        return {"n": 0}
    net = np.array([tr["net"] for tr in trades])
    excess = np.array([tr["excess"] for tr in trades])
    exits = pd.DatetimeIndex([tr["exit_date"] for tr in trades])
    entries = pd.DatetimeIndex([tr["entry_date"] for tr in trades])
    span_years = max((exits.max() - entries.min()).days / 365.25, 1.0)
    events_per_year = len(trades) / span_years
    sharpe_net = (net.mean() / net.std(ddof=1)) * np.sqrt(events_per_year) if net.std(ddof=1) > 0 else float("nan")
    sharpe_excess = (excess.mean() / excess.std(ddof=1)) * np.sqrt(events_per_year) if excess.std(ddof=1) > 0 else float("nan")
    k_pos = int((net > 0).sum())
    lo, hi = wilson_interval(k_pos, len(net))
    return {
        "n": len(trades), "events_per_year": events_per_year,
        "mean_net": float(net.mean()), "t_net": one_sample_t(net),
        "mean_excess": float(excess.mean()), "t_excess": one_sample_t(excess),
        "hit_rate": k_pos / len(net), "hit_lo": lo, "hit_hi": hi,
        "worst": float(net.min()), "best": float(net.max()),
        "sharpe_net": float(sharpe_net), "sharpe_excess": float(sharpe_excess),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(px_map: dict[str, pd.Series], bench: pd.Series,
                      events: dict[str, pd.Timestamp], lo: int = -20, hi: int = 120) -> dict:
    """Run the headline cross-sectional CAR test on a synthetic world."""
    rows = []
    for name, ev_date in events.items():
        ar = ar_series(px_map[name], bench)
        car = event_car(ar, ev_date, lo, hi)
        if car is not None:
            rows.append(car)
    if not rows:
        return {"n": 0, "post120_t": float("nan")}
    mat = pd.DataFrame(rows)
    mat.columns = range(lo, hi + 1)
    post120 = (mat[hi] - mat[0]).to_numpy()
    return {"n": len(mat), "post120_mean": float(np.nanmean(post120)), "post120_t": one_sample_t(post120)}
