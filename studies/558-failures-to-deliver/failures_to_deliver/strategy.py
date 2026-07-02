"""The engine and its honest controls — Study 558 (Failures-To-Deliver).

The claim, at full strength (meme-stock-era microstructure folklore): a **spike in
failures-to-deliver** (settlement fails) means shorts are trapped and a **short squeeze** is
imminent, so the **forward return after an FTD spike is large and positive**. The tradable
expression is: when a name's FTD spikes, go long into the next few days and ride the pop.

This module measures that as an **event study**:

1. **Spike events.** For each name, flag the days its trailing FTD z-score exceeds ``spike_z``
   (computed from that name's own past FTD only — causal, no look-ahead).

2. **Forward returns.** For each spike event at day t, take the cumulative return over the forward
   window (t+1 .. t+h). The one-bar entry lag (enter at t's close, measure t+1 onward) is the
   single documented execution convention.

3. **The headline statistic.** A **one-sample t** on the cross-section of post-spike forward
   returns: is the mean post-spike forward return significantly positive? (The folklore says yes.)

4. **A placebo / label-shuffle null.** Shuffle the spike *labels* to random non-event days and
   recompute the mean forward return many times; the p-value is the fraction of shuffles whose
   mean is at least the observed one. A real squeeze signal sits in the tail; noise does not.

5. **Costs & borrow honesty.** The post-spike long is round-tripped (entry + exit one-way cost),
   and because you would typically pair it against a short (or the trapped-shorts story is *about*
   shorts), we also report the effect after a punitive borrow — FTD-spiking names are exactly the
   hard-to-borrow tail.

6. **A robustness sweep.** Re-run across several forward horizons (h in {1,3,5,10 days}) and a
   couple of spike thresholds; a real effect should be sign-stable, not a single lucky cut.

7. **A synthetic positive control.** A deterministic world where the squeeze is planted
   (``squeeze_alpha > 0``); the engine must recover a positive, significant post-spike return and
   stay flat at the null — averaged over >= 20 seeds so no single seed can manufacture it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as _data

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------
def forward_cum_return(ret: np.ndarray, t: int, horizon: int) -> float:
    """Cumulative simple return over the forward window (t+1 .. t+horizon].

    Enters at day t's close and measures t+1 onward — the single documented execution lag. Returns
    NaN if the forward window runs off the end of the series.
    """
    lo, hi = t + 1, t + 1 + horizon
    if hi > len(ret):
        return np.nan
    return float(np.prod(1.0 + ret[lo:hi]) - 1.0)


def _forward_cum_all(ret: np.ndarray, horizon: int) -> np.ndarray:
    """Vectorised forward cumulative return for every t: (t+1 .. t+horizon]. NaN where it overruns.

    fwd[t] = prod(1+ret[t+1 : t+1+horizon]) - 1. Uses a cumulative-log trick for O(n).
    """
    n = len(ret)
    logc = np.concatenate([[0.0], np.cumsum(np.log1p(ret))])  # logc[k] = sum log(1+ret[:k])
    fwd = np.full(n, np.nan)
    hi = np.arange(n) + 1 + horizon
    lo = np.arange(n) + 1
    valid = hi <= n
    fwd[valid] = np.expm1(logc[hi[valid]] - logc[lo[valid]])
    return fwd


def _market_forward(panel: pd.DataFrame, horizon: int) -> dict[int, float]:
    """Cross-name mean forward cumulative return by day t — the 'market' return event days are
    benchmarked against. Market-adjusting each event removes the common drift and any shared
    calendar-time effect, so the abnormal return isolates the FTD-spike's own information."""
    wide = panel.pivot(index="day", columns="name", values="ret").sort_index()
    days = wide.index.to_numpy()
    mkt_fwd: dict[int, float] = {}
    mats = {c: _forward_cum_all(wide[c].to_numpy(dtype=float), horizon) for c in wide.columns}
    fwd_mat = np.column_stack([mats[c] for c in wide.columns])  # (days, names)
    with np.errstate(invalid="ignore"):
        counts = np.sum(~np.isnan(fwd_mat), axis=1)
        row_mean = np.where(counts > 0, np.nansum(fwd_mat, axis=1) / np.maximum(counts, 1), np.nan)
    for i, d in enumerate(days):
        mkt_fwd[int(d)] = float(row_mean[i])
    return mkt_fwd


def spike_events(
    panel: pd.DataFrame, horizon: int = 5, spike_z: float = 2.5, lookback: int = 60
) -> pd.DataFrame:
    """Extract every FTD-spike event and its **market-adjusted** forward return, across all names.

    For each name: flag spike days from the trailing FTD z-score (causal), then attach the forward
    cumulative return over (t+1 .. t+horizon] and subtract the cross-name (market) forward return
    on the same day t. The resulting ``abn_ret`` is the *abnormal* post-spike return — the pop
    beyond the market's ordinary drift, which is what the squeeze folklore actually claims.
    Columns: ``[name, day, fwd_ret, abn_ret]``.

    **Non-overlapping events.** FTD is persistent (spikes cluster), so raw spike days sit inside
    each other's forward windows — overlapping event returns are *not* independent, which would
    inflate a one-sample t. We enforce a **refractory period**: once a spike is kept, the next
    spike within ``horizon`` days is skipped, so each kept event's forward window is disjoint from
    its neighbours and the event-return sample is approximately independent.
    """
    mkt = _market_forward(panel, horizon)
    rows = []
    for name, g in panel.groupby("name", sort=True):
        g = g.sort_values("day")
        ftd = g["ftd"].to_numpy(dtype=float)
        ret = g["ret"].to_numpy(dtype=float)
        days = g["day"].to_numpy(dtype=int)
        flags = _data.spike_flags(ftd, spike_z=spike_z, lookback=lookback)
        fwd = _forward_cum_all(ret, horizon)
        kept = _refractory(np.flatnonzero(flags), horizon)
        for t in kept:
            fr = fwd[t]
            if not np.isnan(fr):
                d = int(days[t])
                rows.append(
                    {"name": name, "day": d, "fwd_ret": float(fr),
                     "abn_ret": float(fr - mkt.get(d, np.nan))}
                )
    return pd.DataFrame(rows, columns=["name", "day", "fwd_ret", "abn_ret"])


def _refractory(spike_days: np.ndarray, horizon: int) -> list[int]:
    """Thin spike days so kept events are >= ``horizon`` apart (disjoint forward windows)."""
    kept: list[int] = []
    last = -(10**9)
    for t in spike_days:
        if t - last > horizon:
            kept.append(int(t))
            last = t
    return kept


def nonevent_forward_returns(
    panel: pd.DataFrame, horizon: int = 5, spike_z: float = 2.5, lookback: int = 60
) -> np.ndarray:
    """Forward returns on all *non*-spike days — the baseline the event mean is compared against."""
    out = []
    for name, g in panel.groupby("name", sort=True):
        g = g.sort_values("day")
        ftd = g["ftd"].to_numpy(dtype=float)
        ret = g["ret"].to_numpy(dtype=float)
        flags = _data.spike_flags(ftd, spike_z=spike_z, lookback=lookback)
        fwd = _forward_cum_all(ret, horizon)
        n = len(ret)
        idx = np.arange(lookback, n - horizon - 1)
        idx = idx[~flags[idx] & ~np.isnan(fwd[idx])]
        out.append(fwd[idx])
    return np.concatenate(out) if out else np.array([], dtype=float)


# ---------------------------------------------------------------------------
# The headline statistic
# ---------------------------------------------------------------------------
def event_study(
    panel: pd.DataFrame, horizon: int = 5, spike_z: float = 2.5, lookback: int = 60
) -> dict:
    """One-sample t on the **market-adjusted abnormal** post-spike forward returns.

    The folklore claims a pop **beyond** ordinary drift, so we market-adjust each event (subtract
    the cross-name forward return on the same day) and run a one-sample t on the abnormal returns
    (H0: no abnormal pop, mean = 0). Market-adjustment removes the common drift and shared
    calendar-time effects; the refractory period keeps the events non-overlapping, so the abnormal
    sample is approximately independent and the t is honest. Returns the event count, the raw and
    abnormal post-spike means, and the abnormal-return t (the headline statistic).
    """
    ev = spike_events(panel, horizon=horizon, spike_z=spike_z, lookback=lookback)
    if len(ev) < 3:
        return {
            "n_events": int(len(ev)),
            "mean_fwd": float("nan"),
            "t": float("nan"),
            "baseline_mean": float("nan"),
            "excess": float("nan"),
        }
    a = ev["abn_ret"].to_numpy(dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    se = a.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    t = a.mean() / se if se and se > 0 else float("nan")
    return {
        "n_events": int(n),
        "mean_fwd": float(ev["fwd_ret"].mean()),
        "t": float(t),
        "baseline_mean": float(ev["fwd_ret"].mean() - a.mean()),
        "excess": float(a.mean()),  # the abnormal (market-adjusted) post-spike return
    }


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(
    panel: pd.DataFrame,
    horizon: int = 5,
    spike_z: float = 2.5,
    lookback: int = 60,
    n_perm: int = 1000,
    seed: int = 558,
) -> float:
    """Label-shuffle placebo p-value for the mean **abnormal** post-spike return.

    Keep the same *number* of (refractory-thinned) events per name but move them to random *non*-
    event eligible days, recompute the mean **market-adjusted abnormal** return, and repeat
    ``n_perm`` times. The p-value is the fraction of shuffles whose mean abnormal return >= the
    observed event mean (one-sided; the folklore predicts a positive pop). A genuine post-spike
    signal sits in the right tail; pure noise does not.
    """
    obs = event_study(panel, horizon=horizon, spike_z=spike_z, lookback=lookback)
    if not np.isfinite(obs["excess"]) or obs["n_events"] < 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    mkt = _market_forward(panel, horizon)

    # Per-name: number of kept events, and the abnormal returns on eligible non-event days.
    per_name = []
    for name, g in panel.groupby("name", sort=True):
        g = g.sort_values("day")
        ftd = g["ftd"].to_numpy(dtype=float)
        ret = g["ret"].to_numpy(dtype=float)
        days = g["day"].to_numpy(dtype=int)
        flags = _data.spike_flags(ftd, spike_z=spike_z, lookback=lookback)
        n_kept = len(_refractory(np.flatnonzero(flags), horizon))
        fwd_all = _forward_cum_all(ret, horizon)
        n = len(ret)
        eligible = np.arange(lookback, n - horizon - 1)
        eligible = eligible[~flags[eligible]]
        fwd = fwd_all[eligible]
        abn = np.array([fwd[i] - mkt.get(int(days[eligible[i]]), np.nan)
                        for i in range(len(eligible))], dtype=float)
        abn = abn[~np.isnan(abn)]
        per_name.append((n_kept, abn))

    obs_mean = obs["excess"]
    count = 0
    for _ in range(n_perm):
        vals = []
        for n_kept, abn in per_name:
            if n_kept > 0 and len(abn) > 0:
                k = min(n_kept, len(abn))
                idx = rng.choice(len(abn), size=k, replace=False)
                vals.extend(abn[idx].tolist())
        if vals and np.mean(vals) >= obs_mean - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs + borrow
# ---------------------------------------------------------------------------
def net_event_return(
    event_mean: float,
    cost_bps: float = 5.0,
    borrow_ann_bps: float = 300.0,
    horizon_days: int = 5,
) -> float:
    """Post-spike long return net of round-trip cost and a punitive short-side borrow.

    The tradable long is entered and exited once (two one-way crossings of ``cost_bps``). The
    trapped-shorts story is *about* shorts, and FTD-spiking names are exactly the expensive,
    hard-to-borrow tail, so we charge a punitive annual borrow pro-rated over the holding window
    (as if the trade were financed against / paired with a short in the same name).
    """
    cost = 2.0 * cost_bps * 1e-4
    borrow = borrow_ann_bps * 1e-4 * (horizon_days / TRADING_DAYS)
    return float(event_mean - cost - borrow)


# ---------------------------------------------------------------------------
# Robustness sweep — horizons x thresholds
# ---------------------------------------------------------------------------
def robustness_sweep(
    panel: pd.DataFrame,
    horizons=(1, 3, 5, 10),
    thresholds=(2.0, 2.5, 3.0),
    lookback: int = 60,
) -> pd.DataFrame:
    """Event-study mean/t across forward horizons and spike thresholds.

    A real post-spike squeeze signal should be sign-stable across reasonable horizons and spike
    definitions; a fragile artifact will not be. Returns a tidy frame.
    """
    rows = []
    for h in horizons:
        for z in thresholds:
            es = event_study(panel, horizon=h, spike_z=z, lookback=lookback)
            rows.append(
                {
                    "horizon": h,
                    "spike_z": z,
                    "n_events": es["n_events"],
                    "mean_fwd_pct": es["mean_fwd"] * 100 if np.isfinite(es["mean_fwd"]) else np.nan,
                    "t": es["t"],
                    "excess_pct": es["excess"] * 100 if np.isfinite(es["excess"]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Seed-robust synthetic control (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(
    squeeze_alpha: float,
    n_seeds: int = 25,
    base_seed: int = 558,
    horizon: int = 5,
    spike_z: float = 2.5,
) -> float:
    """Average the event-study t over ``n_seeds`` synthetic worlds for a planted ``squeeze_alpha``.

    The house rule: any synthetic-dependent claim averages the test statistic over >= 20 seeds so
    no single lucky RNG seed can manufacture significance. Returns the mean event t across seeds.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = _data.synthetic_panel(squeeze_alpha=squeeze_alpha, seed=s)
        ts.append(event_study(panel, horizon=horizon, spike_z=spike_z)["t"])
    return float(np.nanmean(ts))
