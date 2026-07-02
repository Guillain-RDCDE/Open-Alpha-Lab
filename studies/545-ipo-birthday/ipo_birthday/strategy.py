"""The engine and its honest controls — Study 545 (IPO-Birthday).

The claim: a listed company shows a small positive **abnormal return** in a window around the
anniversary of its IPO each year — an attention/calendar 'birthday effect'. We measure it as a
classic **event study**:

1. **Abnormal returns.** For each name we work in *market-adjusted* returns: ``ar = r_i - r_SPY``
   (a beta-1 market model — simple, transparent, and the standard short-window CAR convention when
   an estimation-window beta would add noise on a 30-name basket).

2. **The event window.** For each firm-**year**, find the first trading day on/after the IPO
   anniversary; take the ``[-pre, +post]`` trading-day window around it and **cumulate** the
   abnormal returns → one CAR per firm-year event.

3. **The headline statistic.** The mean CAR across all firm-year events and a one-sample *t*-stat
   (H0: mean CAR = 0). The 'birthday effect' predicts a positive mean CAR clearing *t* ≥ 2.

4. **The random-anchor placebo.** Re-run the identical window machinery on **random calendar
   anchor dates** (not IPO anniversaries). If the anniversary CAR is just window noise, the placebo
   distribution of mean CARs brackets it; a real effect sits in the tail (a permutation p-value).

5. **Costs.** A birthday trade means entering before the window and exiting after — a round trip
   per event. We report gross and net (one-way bps × 2), and since a symmetric attention window is
   long-only there is no borrow, but a short-the-anniversary variant would pay borrow (noted).

6. **A synthetic positive control.** A deterministic world where a bump is planted
   (``bump_bps > 0``); the engine must recover a positive mean CAR / *t* and stay flat at the null,
   averaged over ≥ 20 seeds so no lucky seed manufactures significance.

Execution convention: the event window is defined purely by the historical IPO calendar date — no
future data enters it. The one documented convention is that day 0 is the first trading day on/after
the anniversary; shifting day 0 by a day changes the CAR trivially and no stamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BENCHMARK = "SPY"


# ---------------------------------------------------------------------------
# Abnormal (market-adjusted) returns
# ---------------------------------------------------------------------------
def abnormal_returns(prices: pd.DataFrame, benchmark: str = BENCHMARK) -> pd.DataFrame:
    """Daily market-adjusted abnormal returns: r_i - r_benchmark, per name.

    Uses a beta-1 market model (the standard short-window CAR convention). Returns a wide frame of
    abnormal returns (index=date), one column per non-benchmark ticker.
    """
    rets = prices.pct_change()
    if benchmark not in rets.columns:
        raise ValueError(f"benchmark {benchmark} missing from prices")
    bench = rets[benchmark]
    ar = rets.drop(columns=[benchmark]).subtract(bench, axis=0)
    return ar


# ---------------------------------------------------------------------------
# The event study — CAR per firm-year around each IPO anniversary
# ---------------------------------------------------------------------------
def _anchor_pos(index: pd.DatetimeIndex, anchor: pd.Timestamp) -> int | None:
    """Position of the first trading day on/after ``anchor`` in ``index`` (None if out of range)."""
    pos = int(np.searchsorted(index.values, np.datetime64(anchor)))
    if pos < 0 or pos >= len(index):
        return None
    return pos


def event_cars(
    ar: pd.DataFrame,
    ipo_dates: dict,
    pre: int = 5,
    post: int = 5,
    min_hist_days: int = 20,
) -> pd.DataFrame:
    """Cumulative abnormal return over the ``[-pre, +post]`` window around each IPO anniversary.

    For each ticker and each anniversary year that falls inside the abnormal-return sample (and at
    least ``min_hist_days`` after the IPO so we skip the immediate post-IPO period), cumulate the
    daily abnormal returns in the window into one CAR. Returns a long frame with columns
    ``ticker, year, anniv, car`` — one row per firm-year event.
    """
    idx = ar.index
    rows = []
    for tk, ipo in ipo_dates.items():
        if tk not in ar.columns:
            continue
        ipo = pd.Timestamp(ipo)
        col = ar[tk].dropna()
        if col.empty:
            continue
        first_valid = col.index[0]
        # iterate anniversaries from year 1 up to the last full year in the sample
        yr = 1
        while True:
            anniv = ipo + pd.DateOffset(years=yr)
            if anniv > idx[-1]:
                break
            yr += 1
            if anniv < first_valid + pd.Timedelta(days=min_hist_days):
                continue
            pos = _anchor_pos(idx, anniv)
            if pos is None:
                continue
            lo, hi = pos - pre, pos + post
            if lo < 0 or hi >= len(idx):
                continue
            window = ar[tk].iloc[lo : hi + 1]
            if window.isna().any():
                continue
            rows.append(
                {"ticker": tk, "year": anniv.year, "anniv": anniv, "car": float(window.sum())}
            )
    return pd.DataFrame(rows)


def car_tstat(cars: pd.DataFrame) -> dict:
    """One-sample *t* on the mean event CAR (H0: mean CAR = 0).

    Returns mean CAR, its *t*-stat, the standard error, and the number of events. The birthday
    effect predicts a positive mean clearing *t* ≥ 2.
    """
    if cars is None or cars.empty or len(cars) < 2:
        return {"mean_car": float("nan"), "t": float("nan"), "se": float("nan"), "n": 0}
    x = cars["car"].to_numpy(dtype=float)
    n = len(x)
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    t = m / se if se > 0 else float("nan")
    return {"mean_car": m, "t": float(t), "se": se, "n": int(n)}


# ---------------------------------------------------------------------------
# Random-anchor placebo null
# ---------------------------------------------------------------------------
def placebo_pvalue(
    ar: pd.DataFrame,
    ipo_dates: dict,
    pre: int = 5,
    post: int = 5,
    n_perm: int = 2000,
    seed: int = 545,
) -> dict:
    """Random-anchor placebo p-value for the mean event CAR.

    For each of ``n_perm`` iterations, replace every real anniversary anchor with a *random* trading
    day drawn from the same in-sample calendar (keeping the same number of events per name), cumulate
    the same ``[-pre, +post]`` window, and record the mean CAR. The two-sided p-value is the fraction
    of placebo mean-CARs whose magnitude ≥ the observed |mean CAR|. Real calendar structure sits in
    the tail; window noise does not.

    Returns ``{"obs": mean_car, "p": p, "placebo_mean": mean_of_null, "placebo_sd": sd_of_null}``.
    """
    obs_cars = event_cars(ar, ipo_dates, pre, post)
    if obs_cars.empty:
        return {"obs": float("nan"), "p": float("nan"), "placebo_mean": float("nan"),
                "placebo_sd": float("nan")}
    obs = abs(float(obs_cars["car"].mean()))
    idx = ar.index
    n_idx = len(idx)
    rng = np.random.default_rng(seed)

    # how many events per ticker (to keep the placebo the same shape as observed)
    counts = obs_cars.groupby("ticker").size().to_dict()
    cols = {tk: ar[tk].to_numpy(dtype=float) for tk in counts}

    lo_ok, hi_ok = pre, n_idx - post - 1
    if hi_ok <= lo_ok:
        return {"obs": obs, "p": float("nan"), "placebo_mean": float("nan"),
                "placebo_sd": float("nan")}

    means = np.empty(n_perm)
    for i in range(n_perm):
        vals = []
        for tk, k in counts.items():
            arr = cols[tk]
            anchors = rng.integers(lo_ok, hi_ok + 1, size=k)
            for pos in anchors:
                w = arr[pos - pre : pos + post + 1]
                if not np.isnan(w).any():
                    vals.append(float(w.sum()))
        means[i] = np.mean(vals) if vals else np.nan
    valid = means[~np.isnan(means)]
    p = (np.sum(np.abs(valid) >= obs - 1e-15) + 1) / (len(valid) + 1)
    return {"obs": float(obs_cars["car"].mean()), "p": float(p),
            "placebo_mean": float(np.nanmean(valid)), "placebo_sd": float(np.nanstd(valid))}


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------
def net_mean_car(mean_car: float, cost_bps: float = 5.0) -> float:
    """Mean event CAR net of a round-trip cost (enter before window, exit after).

    One event = one round trip = 2 one-way crossings of ``cost_bps``. Long-only (no borrow); a
    short-the-anniversary variant would additionally pay borrow (noted in the write-up).
    """
    return float(mean_car - 2.0 * cost_bps * 1e-4)


# ---------------------------------------------------------------------------
# Robustness — window sweep
# ---------------------------------------------------------------------------
def window_sweep(ar: pd.DataFrame, ipo_dates: dict, windows: list[tuple[int, int]]) -> pd.DataFrame:
    """Mean CAR and *t* across several ``[-pre, +post]`` event windows."""
    rows = []
    for pre, post in windows:
        c = event_cars(ar, ipo_dates, pre, post)
        st = car_tstat(c)
        rows.append({"window": f"[-{pre},+{post}]", "n": st["n"],
                     "mean_car_bps": st["mean_car"] * 1e4, "t": st["t"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic positive control — seed-robust (house rule)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, bump_bps: float, pre: int = 5, post: int = 5,
                     n_seeds: int = 25, base_seed: int = 545) -> dict:
    """Average the event-CAR *t* over ``n_seeds`` synthetic worlds for a planted ``bump_bps``.

    House rule: any synthetic-dependent claim averages the stat over ≥ 20 seeds so no single lucky
    RNG seed can manufacture significance. Returns the mean CAR (bps) and mean *t* across seeds.
    """
    ts, ms = [], []
    for s in range(base_seed, base_seed + n_seeds):
        prices, ipos, _ = data_mod.synthetic_panel(bump_bps=bump_bps, seed=s)
        ar = abnormal_returns(prices)
        cars = event_cars(ar, ipos, pre, post)
        st = car_tstat(cars)
        ts.append(st["t"])
        ms.append(st["mean_car"] * 1e4)
    return {"mean_car_bps": float(np.nanmean(ms)), "mean_t": float(np.nanmean(ts))}
