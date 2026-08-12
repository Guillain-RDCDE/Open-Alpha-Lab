"""The event-study engine and its honest controls — Study 843 (Waffle House Index).

The claim under test, steelmanned: a **major US natural disaster** (a big
hurricane landfall) is a genuine cash-flow event for two obvious sectors —
**property & casualty insurers** face a payout shock (stock should *dip*), and
**home-improvement / rebuild** names face a reconstruction-demand tailwind (stock
should *rally*). FEMA reads storm severity off whether the always-open Waffle
House closes; we ask whether the market reads it off ALL/TRV/PGR and HD/LOW.

The machinery — a **market-adjusted** event study, one execution lag documented:

* ``daily_returns`` / ``market_adjusted`` — the "abnormal" return of a name is its
  simple daily return minus SPY's same-day return (a beta-1 market model, the
  standard cheap-and-robust abnormal-return proxy for a short event window). This
  strips the market-wide move so a post-storm CAR is a *sector-specific* signal,
  not "stocks went up that fortnight".
* ``basket_ar`` — the equal-weight market-adjusted return of a basket (insurers or
  rebuilders) on each date.
* ``event_window`` / ``stack_windows`` — the ``[-pre..+post]`` market-adjusted-return
  path around each landfall. ``event_date`` is snapped forward to the first NYSE
  session on/after the landfall calendar date via ``searchsorted`` — the study's
  single documented execution lag (a weekend landfall rolls to the next open).
* ``car_path_stats`` — the mean cumulative abnormal-return path by offset, anchored
  at 0 the ``pre`` sessions before landfall — the headline "dip / rally" chart. The
  window spans the pre-storm run-up because a hurricane is FORECAST ahead of
  landfall and can be priced in early.
* ``basket_car_stats`` — per-event CAR over a chosen horizon, mean + one-sample *t*
  across events (independent, far-apart calendar dates).
* ``long_short_stats`` — the paired (rebuilders − insurers) per-event CAR, the
  cleanest single test of the folklore's *directional* prediction; pairing removes
  any residual common shock.
* ``placebo_distribution`` — the synthetic control: the same statistic over
  thousands of random non-disaster dates; the observed number's percentile is the
  falsification test.
* ``timer`` — the tradable overlay (long rebuilders / short insurers, entered at the
  landfall-session close, held ``hold`` sessions), one-way cost × NAV per leg on
  both sides plus borrow on the short leg.

Honest up front: ~16 events is a **small sample → low power**; a null here means
"not detectable at this size", not "provably zero".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Returns + market-adjusted (abnormal) returns
# --------------------------------------------------------------------------- #
def daily_returns(close: pd.Series) -> pd.Series:
    """Simple close-to-close daily returns."""
    return close.pct_change()


def market_adjusted(ret: pd.Series, mkt_ret: pd.Series) -> pd.Series:
    """Market-adjusted (beta-1) abnormal return: ``r_i - r_mkt`` on aligned dates.

    A short-window abnormal-return proxy (Brown & Warner 1985's market-adjusted
    model): the "normal" return is the market's, so the abnormal return removes the
    common market-wide move and leaves the name-specific reaction.
    """
    idx = ret.index.union(mkt_ret.index)
    return (ret.reindex(idx) - mkt_ret.reindex(idx)).rename(ret.name)


def basket_ar(closes: dict[str, pd.Series], tickers, mkt: pd.Series) -> pd.Series:
    """Equal-weight market-adjusted return of ``tickers`` (whichever trade that day).

    Each leg is market-adjusted against ``mkt`` (SPY) first, then equal-weighted; a
    date with no covered leg is NaN (not zero-filled).
    """
    mkt_ret = daily_returns(mkt)
    ars = pd.DataFrame({t: market_adjusted(daily_returns(closes[t]), mkt_ret) for t in tickers})
    basket = ars.mean(axis=1, skipna=True)
    basket[ars.notna().sum(axis=1) == 0] = np.nan
    return basket.rename("+".join(tickers))


# --------------------------------------------------------------------------- #
# Event windows
# --------------------------------------------------------------------------- #
def event_window(ar: pd.Series, event_date, pre: int = 10, post: int = 20
                 ) -> np.ndarray | None:
    """The abnormal-return path from ``-pre`` to ``+post`` sessions around a landfall.

    ``event_date`` is the landfall calendar date; ``searchsorted`` snaps it to the
    first session on/after that date (session 0). Returns a 1-D array of length
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
    return ar.to_numpy()[lo:hi + 1]


def stack_windows(ar: pd.Series, event_dates, pre: int = 10, post: int = 20,
                  require_finite: bool = True) -> tuple[np.ndarray, list]:
    """Stack every valid event window into a ``(n_events, pre+post+1)`` matrix.

    ``require_finite`` drops any window carrying a NaN instead of quietly
    propagating it into the mean. Returns ``(matrix, kept_dates)``.
    """
    rows, kept = [], []
    for d in pd.to_datetime(pd.Series(list(event_dates))):
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


def newey_west_t(x: np.ndarray, lags: int = 4) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0; plain t for tiny n."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    if n < 8:  # too few events for a stable HAC — plain t on the mean
        se = x.std(ddof=1) / np.sqrt(n)
        return float(mu / se) if se > 0 else float("nan")
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson (1927) score interval for a hit rate ``k``/``n``."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# The headline — per-event CAR over a horizon + the CAR path
# --------------------------------------------------------------------------- #
def per_event_car(ar: pd.Series, event_dates, pre: int = 10, post: int = 20,
                  lo: int = 0, hi: int = 20) -> tuple[np.ndarray, list]:
    """Per-event cumulative abnormal return summed over offsets ``[lo..hi]``.

    ``lo``/``hi`` are relative sessions (0 = landfall session). Returns
    ``(per_event_car_array, kept_dates)``.
    """
    w, kept = stack_windows(ar, event_dates, pre, post)
    if w.shape[0] == 0:
        return np.array([]), kept
    cols = np.arange(-pre, post + 1)
    mask = (cols >= lo) & (cols <= hi)
    return w[:, mask].sum(axis=1), kept


def car_stats(ar: pd.Series, event_dates, pre: int = 10, post: int = 20,
              lo: int = 0, hi: int = 20) -> dict:
    """Mean per-event CAR over ``[lo..hi]``: mean + one-sample t + NW t + hit rate."""
    car, kept = per_event_car(ar, event_dates, pre, post, lo, hi)
    if car.size == 0:
        return {"n": 0, "mean": float("nan"), "t": float("nan"), "t_nw": float("nan"),
                "hits": 0, "kept_dates": kept}
    mean, t = one_sample_t(car)
    hits = int((car < 0).sum())    # "down" count — insurers expected to dip
    return {"n": int(car.size), "mean": mean, "t": t,
            "t_nw": newey_west_t(car), "hits": hits, "car": car, "kept_dates": kept}


def car_path_stats(ar: pd.Series, event_dates, pre: int = 10, post: int = 20
                   ) -> pd.DataFrame:
    """Mean CAR by offset (``-pre..+post``), each offset's own one-sample t.

    CAR(k) = cumulative mean abnormal return from ``-pre`` through offset ``k``,
    re-anchored so CAR(``-pre``) == 0 (nothing has happened before the first
    pre-event bar) — the standard event-study convention.
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
# The directional test — rebuilders minus insurers (paired per event)
# --------------------------------------------------------------------------- #
def long_short_stats(ar_ins: pd.Series, ar_reb: pd.Series, event_dates,
                     pre: int = 10, post: int = 20, lo: int = 0, hi: int = 20) -> dict:
    """Paired (rebuilders − insurers) per-event CAR over ``[lo..hi]``: mean + t.

    The folklore's directional prediction in one number: rebuilders should out-CAR
    insurers. Both baskets are windowed on the same kept dates, so the difference is
    a clean per-event paired statistic.
    """
    ins, kept_i = per_event_car(ar_ins, event_dates, pre, post, lo, hi)
    reb, kept_r = per_event_car(ar_reb, event_dates, pre, post, lo, hi)
    kept = sorted(set(kept_i) & set(kept_r))
    if not kept:
        return {"n": 0, "mean_diff": float("nan"), "t": float("nan"),
                "ins_mean": float("nan"), "reb_mean": float("nan")}
    ins2, _ = per_event_car(ar_ins, kept, pre, post, lo, hi)
    reb2, _ = per_event_car(ar_reb, kept, pre, post, lo, hi)
    diff = reb2 - ins2
    mean, t = one_sample_t(diff)
    return {"n": len(kept), "mean_diff": mean, "t": t, "t_nw": newey_west_t(diff),
            "ins_mean": float(np.nanmean(ins2)), "reb_mean": float(np.nanmean(reb2)),
            "diff": diff}


# --------------------------------------------------------------------------- #
# Synthetic control / falsification — the random-date placebo
# --------------------------------------------------------------------------- #
def placebo_distribution(ar: pd.Series, n_events: int, pre: int = 10, post: int = 20,
                         lo: int = 0, hi: int = 20, n_draws: int = 2000,
                         seed: int = 843) -> np.ndarray:
    """Random-date placebo: mean per-event CAR over ``n_draws`` sets of ``n_events``
    random non-disaster dates. A real effect must sit in the tail of this cloud;
    sitting in the bulk means the "effect" is what a random calendar of the same size
    produces anyway.
    """
    rng = np.random.default_rng(seed)
    vals = ar.dropna().to_numpy()
    n = vals.size
    span_lo, span_hi = pre, n - post - 1
    if span_hi <= span_lo or n_events <= 0:
        return np.array([])
    cols = np.arange(-pre, post + 1)
    mask = (cols >= lo) & (cols <= hi)
    off = np.where(mask)[0] - pre                 # relative offsets kept
    out = np.empty(n_draws)
    for d in range(n_draws):
        locs = rng.integers(span_lo, span_hi, size=n_events)
        per = np.array([vals[loc + off].sum() for loc in locs])
        out[d] = per.mean()
    return out


def placebo_pvalue(observed: float, placebo: np.ndarray, tail: str = "two") -> float:
    """Empirical p-value of ``observed`` within the placebo draws.

    ``tail='left'`` / ``'right'`` for one-sided; ``'two'`` centers on the placebo mean.
    """
    if placebo.size == 0 or not np.isfinite(observed):
        return float("nan")
    if tail == "left":
        return float((placebo <= observed).mean())
    if tail == "right":
        return float((placebo >= observed).mean())
    mu = placebo.mean()
    return float((np.abs(placebo - mu) >= abs(observed - mu)).mean())


def block_bootstrap_ci(per_event: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
                       seed: int = 843) -> tuple[float, float]:
    """Percentile CI on the mean per-event statistic (events resampled with replacement).

    Events are independent, far-apart calendar dates, so the resampling unit is the
    event. Tiny n → the CI is wide by design.
    """
    x = np.asarray(per_event, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = x[rng.integers(0, n, size=n)].mean()
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


# --------------------------------------------------------------------------- #
# The tradable overlay — long rebuilders / short insurers around a disaster
# --------------------------------------------------------------------------- #
def timer(closes: dict[str, pd.Series], event_dates, insurers, rebuilders,
          mkt: str = "SPY", hold: int = 20, lag: int = 0,
          cost_bps: float = 0.0, borrow_bps_yr: float = 50.0) -> pd.DataFrame:
    """Per-trade ledger for a market-neutral long-rebuilders / short-insurers overlay.

    Enter at the landfall-session close (``lag`` extra days optional — landfall is
    calendar-known and largely forecast, so ``lag=0`` is the canonical anchor), hold
    ``hold`` sessions. The leg return is the equal-weight basket's simple return over
    the hold; the trade return is ``reb_leg − ins_leg`` (dollar-neutral). ``cost_bps``
    is a one-way cost charged on all **four** legs (enter+exit × long+short), plus
    ``borrow_bps_yr`` annualised borrow on the short insurer leg over the hold.
    """
    def basket_close(tickers):
        df = pd.DataFrame({t: closes[t] for t in tickers}).sort_index()
        return df

    ins = basket_close(insurers)
    reb = basket_close(rebuilders)
    idx = closes[mkt].index
    rows = []
    borrow = (borrow_bps_yr / 1e4) / TRADING_DAYS
    for d in pd.to_datetime(pd.Series(list(event_dates))):
        pos = idx.searchsorted(pd.Timestamp(d)) + lag
        if pos < 0 or pos + hold >= len(idx):
            continue
        d0, d1 = idx[pos], idx[pos + hold]

        def leg_ret(df):
            a = df.reindex([d0, d1])
            r = (a.iloc[1] / a.iloc[0] - 1.0)
            return float(r.mean(skipna=True))

        ins_r = leg_ret(ins)
        reb_r = leg_ret(reb)
        gross = reb_r - ins_r
        cost = 4.0 * cost_bps * 1e-4 + borrow * hold
        rows.append({"event": d0, "hold": hold, "ins_ret": ins_r, "reb_ret": reb_r,
                     "ret_gross": gross, "ret_net": gross - cost})
    return pd.DataFrame(rows)


def summarize_trades(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-trade stats for the overlay ledger: n, win-rate, mean, t."""
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
def synthetic_detect(closes: dict[str, pd.Series], events, insurers, rebuilders,
                     mkt: str = "SPY", pre: int = 10, post: int = 20,
                     lo: int = 0, hi: int = 20) -> dict:
    """Run the headline directional (rebuilders − insurers) test on a synthetic world."""
    m = closes[mkt]
    ar_ins = basket_ar(closes, insurers, m)
    ar_reb = basket_ar(closes, rebuilders, m)
    ls = long_short_stats(ar_ins, ar_reb, events, pre, post, lo, hi)
    ins = car_stats(ar_ins, events, pre, post, lo, hi)
    reb = car_stats(ar_reb, events, pre, post, lo, hi)
    return {"ls_mean": ls["mean_diff"], "ls_t": ls["t"],
            "ins_mean": ins["mean"], "ins_t": ins["t"],
            "reb_mean": reb["mean"], "reb_t": reb["t"], "n": ls["n"]}
