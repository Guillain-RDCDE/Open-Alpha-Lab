"""Strategy + inference for Study 847 — Rotten-Tomatoes -> Studio.

The claim: **a film's critic reception (fresh vs. rotten) moves its distributing studio's
stock around the release.** We test it as a tier-conditioned event study.

Anchoring, one execution lag documented throughout. Each film's release date is snapped
to the first NYSE session on/after it (``searchsorted``) — the *anchor*, session 0. Critic
reviews lift their embargo a few days before a wide release, so the tier is public by the
anchor; opening-weekend box office is public by the Monday after. We measure two
market-adjusted abnormal-return windows on the DISTRIBUTING STUDIO, anchored there:

* **Opening-weekend CAR** — the studio's cumulative abnormal return over sessions
  ``[0..+1]`` (the release session and the next), the immediate reaction as the reviews
  and first weekend numbers land.
* **Following-week CAR** — sessions ``[+2..+6]`` (the trading week after), the "slow
  digest" window.

Abnormal return = studio daily return minus SPY daily return (a market-adjusted,
beta = 1 model), then demeaned by the studio's own full-sample mean of that market-adjusted
series (a constant-mean overlay) so a positive CAR is not just "this stock drifted up over
2022-2025". Events are independent, non-overlapping calendar dates, so the primary
statistic is a **one-sample t** per tier, and a **Welch t** of the fresh-minus-rotten gap
(the quantity the claim actually predicts to be positive: fresh studios out-return rotten
ones). Two nulls: a **tier-label permutation** placebo (break the tier -> CAR link,
preserving each event's CAR) and a **random-date** placebo (redraw pseudo-events). A costed
long-fresh / short-rotten timer asks whether any gap is tradable. N is small (~40 events,
~20 per tier) -> low power -> the honest prior is None.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

OW_LO, OW_HI = 0, 1       # opening-weekend window (sessions from the anchor, inclusive)
FW_LO, FW_HI = 2, 6       # following-week window
COST_BPS = 5.0            # one-way, per leg
BORROW_BPS_YR = 50.0      # annual borrow on the short leg


# --------------------------------------------------------------------------- #
# Market-adjusted abnormal returns
# --------------------------------------------------------------------------- #
def market_adjusted_ar(studio: pd.Series, spy: pd.Series) -> pd.Series:
    """Abnormal return = studio daily return minus SPY daily return, then demeaned.

    A beta = 1 market model (subtract the market's return) removes the common market move;
    the constant-mean overlay (subtract the series' own mean) removes the stock's average
    2022-2025 drift, so a nonzero event-window CAR reflects the film, not "this studio
    happened to rally that year". Both series are aligned on their common trading dates.
    """
    common = studio.index.intersection(spy.index).sort_values()
    rs = studio.loc[common].pct_change()
    rb = spy.loc[common].pct_change()
    ar = rs - rb
    return ar - ar.mean(skipna=True)


def studio_ars(prices: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Per-studio market-adjusted abnormal-return series (vs SPY)."""
    spy = prices[dt.BENCHMARK]
    return {s: market_adjusted_ar(prices[s], spy) for s in dt.STUDIOS}


# --------------------------------------------------------------------------- #
# Event resolution: curated table -> per-event window CARs (vectorised per studio)
# --------------------------------------------------------------------------- #
def _window_sum(ar_vals: np.ndarray, pos: int, lo: int, hi: int) -> float:
    """Sum of ``ar_vals[pos+lo .. pos+hi]`` (inclusive); NaN if the window runs off tape."""
    a, b = pos + lo, pos + hi
    if a < 0 or b >= ar_vals.size:
        return float("nan")
    seg = ar_vals[a:b + 1]
    if not np.all(np.isfinite(seg)):
        return float("nan")
    return float(seg.sum())


def build_event_table(prices: dict[str, pd.Series],
                      ars: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """One row per film: opening-weekend + following-week studio CAR + inclusion.

    The anchor is the first trading session on/after the film's release date. A row is
    INCLUDED only if the studio's AR series covers ``[anchor + OW_LO .. anchor + FW_HI]``.
    All searches are numpy ``searchsorted`` on the studio's own DatetimeIndex — no per-date
    pandas ``.loc`` scans.
    """
    if ars is None:
        ars = studio_ars(prices)
    films = dt.film_table()
    rows = []
    # Pre-extract numpy arrays + index positions per studio once.
    studio_idx = {s: ars[s].index for s in dt.STUDIOS}
    studio_val = {s: ars[s].to_numpy(dtype=float) for s in dt.STUDIOS}
    for r in films.itertuples(index=False):
        idx = studio_idx[r.studio]
        vals = studio_val[r.studio]
        pos = int(idx.searchsorted(pd.Timestamp(r.date)))
        row = dict(title=r.title, studio=r.studio, tier=r.tier, rt=r.rt,
                   date=str(pd.Timestamp(r.date).date()))
        if pos >= len(idx):
            row.update(included=False, reason="release past the tape")
            rows.append(row)
            continue
        ow = _window_sum(vals, pos, OW_LO, OW_HI)
        fw = _window_sum(vals, pos, FW_LO, FW_HI)
        full = _window_sum(vals, pos, OW_LO, FW_HI)
        if not (np.isfinite(ow) and np.isfinite(fw)):
            row.update(included=False, reason="window off the tape / ticker not yet trading")
            rows.append(row)
            continue
        row.update(included=True, reason="", anchor_date=str(idx[pos].date()),
                   ow_car=ow, fw_car=fw, full_car=full)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference primitives (canonical desk set)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> dict:
    """Mean + one-sample t of ``x`` vs 0 (independent, non-overlapping events)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else float("nan"),
                "sd": float("nan"), "t": float("nan")}
    sd = x.std(ddof=1)
    se = sd / np.sqrt(n)
    return {"n": n, "mean": float(x.mean()), "sd": float(sd),
            "t": float(x.mean() / se) if se > 0 else float("nan")}


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances) — the fresh-minus-rotten gap."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 5) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0 (included for the canonical set;
    events are independent so it tracks the one-sample t)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Headline — tier-conditioned stats
# --------------------------------------------------------------------------- #
def tier_stats(events: pd.DataFrame, col: str = "fw_car") -> dict:
    """Per-tier one-sample t on ``col`` plus the Welch fresh-minus-rotten gap.

    ``col`` is ``ow_car`` (opening weekend), ``fw_car`` (following week) or ``full_car``.
    The claim predicts fresh > rotten (a positive gap); the debunk prior is gap ~ 0.
    """
    inc = events[events["included"]] if "included" in events else events
    fresh = inc.loc[inc["tier"] == "fresh", col].to_numpy(dtype=float)
    rotten = inc.loc[inc["tier"] == "rotten", col].to_numpy(dtype=float)
    tf, tr = one_sample_t(fresh), one_sample_t(rotten)
    gap = float(np.nanmean(fresh) - np.nanmean(rotten))
    return {
        "col": col,
        "n_fresh": tf["n"], "fresh_bps": tf["mean"] * 1e4, "fresh_t": tf["t"],
        "n_rotten": tr["n"], "rotten_bps": tr["mean"] * 1e4, "rotten_t": tr["t"],
        "gap_bps": gap * 1e4, "gap_welch_t": welch_t(fresh, rotten),
    }


def hit_rate(events: pd.DataFrame, tier: str, col: str, positive: bool = True) -> dict:
    """Wilson-interval hit rate: share of ``tier`` events with CAR of the expected sign."""
    inc = events[events["included"]] if "included" in events else events
    x = inc.loc[inc["tier"] == tier, col].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    k = int((x > 0).sum()) if positive else int((x < 0).sum())
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


# --------------------------------------------------------------------------- #
# Placebos
# --------------------------------------------------------------------------- #
def permutation_placebo(events: pd.DataFrame, col: str = "fw_car",
                        n_seeds: int = 20, n_draws_per_seed: int = 1000,
                        base_seed: int = 847) -> dict:
    """Tier-label permutation null for the fresh-minus-rotten gap.

    Keep every event's CAR; SHUFFLE the fresh/rotten labels across events (breaking any
    tier -> CAR link, preserving the pooled CAR distribution). p = share of permuted worlds
    whose fresh-minus-rotten gap is >= the observed gap (right-tail: the claim says fresh
    beats rotten). A real tier effect sits in the right tail; sitting in the bulk means the
    observed gap is what random relabelling produces anyway.
    """
    inc = events[events["included"]] if "included" in events else events
    x = inc[col].to_numpy(dtype=float)
    tiers = inc["tier"].to_numpy()
    keep = np.isfinite(x)
    x, tiers = x[keep], tiers[keep]
    n_fresh = int((tiers == "fresh").sum())
    obs = float(x[tiers == "fresh"].mean() - x[tiers == "rotten"].mean())
    n = x.size
    gaps = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            perm = rng.permutation(n)
            f = x[perm[:n_fresh]]
            r = x[perm[n_fresh:]]
            gaps.append(f.mean() - r.mean())
    gaps = np.asarray(gaps)
    return {"obs_bps": obs * 1e4, "placebo_mean_bps": float(gaps.mean() * 1e4),
            "placebo_sd_bps": float(gaps.std(ddof=1) * 1e4),
            "p_value": float((gaps >= obs).mean()), "n_draws": len(gaps),
            "draws_bps": gaps * 1e4}


def random_date_placebo(events: pd.DataFrame, prices: dict[str, pd.Series],
                        col: str = "fw_car", lo: int = FW_LO, hi: int = FW_HI,
                        n_seeds: int = 20, n_draws_per_seed: int = 200,
                        base_seed: int = 847) -> dict:
    """Random-date null for the pooled window CAR magnitude.

    For each INCLUDED event, redraw a random (non-release) window of the same shape on that
    event's OWN studio AR series and recompute the CAR; average across the same n events;
    repeat many times. Two-sided p = share of |null means| >= |observed pooled mean|.
    """
    ars = studio_ars(prices)
    idxpos = {s: ars[s].to_numpy(dtype=float) for s in dt.STUDIOS}
    inc = events[events["included"]] if "included" in events else events
    inc = inc[np.isfinite(inc[col].to_numpy(dtype=float))]
    obs = float(inc[col].mean())
    studios = inc["studio"].to_numpy()
    means = []
    span = hi - lo
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            vals = []
            for stud in studios:
                arr = idxpos[stud]
                good = np.where(np.isfinite(arr))[0]
                good = good[(good >= span + 5) & (good < arr.size - span - 5)]
                if good.size == 0:
                    continue
                p = int(rng.choice(good))
                seg = arr[p:p + span + 1]
                vals.append(float(seg.sum()))
            if vals:
                means.append(np.mean(vals))
    means = np.asarray(means)
    p = float((np.abs(means) >= abs(obs)).mean()) if means.size else float("nan")
    return {"obs_bps": obs * 1e4, "placebo_mean_bps": float(means.mean() * 1e4),
            "placebo_sd_bps": float(means.std(ddof=1) * 1e4) if means.size > 1 else float("nan"),
            "p_value": p, "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# Event anatomy — mean CAR path by offset, by tier
# --------------------------------------------------------------------------- #
def car_path_by_tier(events: pd.DataFrame, prices: dict[str, pd.Series],
                     pre: int = 1, post: int = FW_HI) -> pd.DataFrame:
    """Mean cumulative AR by offset (``-pre..+post``), split fresh vs rotten.

    Each event's path is the cumulative studio AR from the anchor, re-anchored so
    offset ``-pre`` = 0. Returns a frame indexed by offset with ``fresh``/``rotten``
    mean-CAR columns (bps space is applied by the caller).
    """
    ars = studio_ars(prices)
    idxp = {s: ars[s].index for s in dt.STUDIOS}
    valp = {s: ars[s].to_numpy(dtype=float) for s in dt.STUDIOS}
    inc = events[events["included"]] if "included" in events else events
    offsets = list(range(-pre, post + 1))
    paths = {"fresh": [], "rotten": []}
    for r in inc.itertuples(index=False):
        idx = idxp[r.studio]; vals = valp[r.studio]
        pos = int(idx.searchsorted(pd.Timestamp(r.anchor_date)))
        lo, hi = pos - pre, pos + post
        if lo < 0 or hi >= vals.size:
            continue
        seg = vals[lo:hi + 1]
        if not np.all(np.isfinite(seg)):
            continue
        car = np.cumsum(seg) - seg[0]     # re-anchor at offset -pre
        paths[r.tier].append(car)
    out = {}
    for tier in ("fresh", "rotten"):
        if paths[tier]:
            out[tier] = np.vstack(paths[tier]).mean(axis=0)
        else:
            out[tier] = np.full(len(offsets), np.nan)
    return pd.DataFrame(out, index=offsets)


# --------------------------------------------------------------------------- #
# The costed timer — long fresh / short rotten the studio over the following week
# --------------------------------------------------------------------------- #
def timer_stats(events: pd.DataFrame, col: str = "fw_car",
                cost_bps: float = COST_BPS, borrow_bps_yr: float = BORROW_BPS_YR,
                hold_days: int = FW_HI - FW_LO + 1) -> dict:
    """Cost a long-fresh / short-rotten studio book on the ``col`` window.

    Per event the tradable return is ``+CAR`` (long) on fresh and ``-CAR`` (short) on
    rotten. Charge one round trip of one-way cost (x2) per leg against NAV, plus borrow on
    the short (rotten) legs over the hold. One-sample t across all event legs; a positive,
    cost-surviving t would be an edge.
    """
    inc = events[events["included"]] if "included" in events else events
    rt = 2.0 * cost_bps / 1e4
    borrow = (borrow_bps_yr / 1e4) / 365.0 * hold_days
    legs = []
    for r in inc.itertuples(index=False):
        c = getattr(r, col)
        if not np.isfinite(c):
            continue
        if r.tier == "fresh":
            legs.append(c - rt)                 # long the fresh studio
        else:
            legs.append(-c - rt - borrow)        # short the rotten studio (+ borrow)
    legs = np.asarray(legs, dtype=float)
    st = one_sample_t(legs)
    gross = float(np.nanmean([getattr(r, col) if r.tier == "fresh" else -getattr(r, col)
                              for r in inc.itertuples(index=False)
                              if np.isfinite(getattr(r, col))]))
    return {"n": st["n"], "gross_bps": gross * 1e4, "net_bps": st["mean"] * 1e4,
            "t_net": st["t"], "cost_bps_per_leg": (rt + borrow) * 1e4}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(edge: float, seed: int, post: int = FW_HI) -> dict:
    """Run the tier-conditioned fresh-minus-rotten detector on a synthetic world.

    Builds market-adjusted ARs from a planted-``edge`` world, sums each event's
    following-week window, and returns the fresh/rotten means + the Welch gap t. A larger
    planted ``edge`` must drive the gap (fresh - rotten) MORE POSITIVE, monotonically;
    ``edge = 0`` must not fire.
    """
    studio, bench, events = dt.synthetic_world(edge=edge, seed=seed)
    ar = (studio - bench)
    ar = (ar - ar.mean()).to_numpy(dtype=float)
    fresh, rotten = [], []
    for pos, tier in events:
        a, b = pos + 1, pos + post
        if b >= ar.size:
            continue
        c = float(ar[a:b + 1].sum())
        (fresh if tier == "fresh" else rotten).append(c)
    fresh, rotten = np.asarray(fresh), np.asarray(rotten)
    tf, tr = one_sample_t(fresh), one_sample_t(rotten)
    return {"fresh_bps": tf["mean"] * 1e4, "rotten_bps": tr["mean"] * 1e4,
            "gap_bps": (float(fresh.mean()) - float(rotten.mean())) * 1e4,
            "gap_welch_t": welch_t(fresh, rotten),
            "n_fresh": tf["n"], "n_rotten": tr["n"]}
