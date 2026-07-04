"""Strategy + inference for Study 602 — Macro-Announcement-Day Premium.

The claim (Savor & Wilson 2013, JFQA): a disproportionate share of the equity premium is
earned on **scheduled macro-announcement days** — CPI releases, Employment Situation
("NFP") releases and FOMC decision days. All three schedules are published in advance, so
the announcement-day tag is knowable ex-ante. We measure:

    * **Pooled announcement-day premium.** Mean SPY close-to-close return on announcement
      days ("A-days") minus the mean on all other days, with a **Welch t** (the desk's
      group-split statistic; daily close-to-close returns carry negligible serial
      correlation at this horizon, and the placebo below is the sharper calendar null).

    * **Per-type split.** Each announcement type (FOMC / CPI / NFP) against the same
      **pure non-announcement baseline** (days carrying no announcement of any type).
      Sessions carrying two announcements (e.g. CPI + FOMC on 2008-12-16) count for both
      types; the overlap count is reported.

    * **Placebo.** Random pseudo-announcement calendars of the same density — ``n_draws``
      same-size random day-sets; p = P[shuffled A-minus-rest >= observed]. Far beyond the
      desk's >= 20-seed floor for random baselines.

    * **Tradable overlay.** Long SPY **only** on announcement days: the schedule is known
      in advance, so the position is entered at the **prior session's close** (the one
      execution lag — the last price available before the news) and exited at the A-day
      close. One round trip per A-day; one-way costs x NAV, 2 legs. Long-only, no borrow.
      Returns are total-return (adjusted closes), raw (not excess): the overlay would
      additionally earn the T-bill rate on the ~87% of days it sits in cash, which we do
      not model — an omission that penalises the overlay, never flatters it.

The decisive number is the pooled A-day-vs-rest Welch t on the REAL tape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def daily_returns(px: pd.Series) -> pd.Series:
    """Simple close-to-close daily returns from an adjusted-close series."""
    return px.dropna().pct_change().dropna()


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of mean(sample) - mean(base) (unequal variance). NaN if either < 2."""
    sample = np.asarray(sample, dtype=float)
    base = np.asarray(base, dtype=float)
    if len(sample) < 2 or len(base) < 2:
        return float("nan")
    v1 = sample.var(ddof=1) / len(sample)
    v0 = base.var(ddof=1) / len(base)
    se = np.sqrt(v1 + v0)
    if se == 0:
        return float("nan")
    return float((sample.mean() - base.mean()) / se)


def ttest_vs_zero(sample: np.ndarray) -> float:
    """One-sample t of ``sample`` against 0."""
    sample = np.asarray(sample, dtype=float)
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float(sample.mean() / se)


# --------------------------------------------------------------------------- #
# The premium
# --------------------------------------------------------------------------- #
def event_vs_rest(ret: np.ndarray, ev: np.ndarray, base: np.ndarray | None = None) -> dict:
    """A-day-vs-baseline stats: means (bps/day), diff, Welch t, counts, share of return.

    ``ev`` tags the event days; ``base`` tags the baseline days (default: ``~ev``).
    ``share_of_total`` is the share of the full sample's cumulative log return earned on
    the event days (meaningful for the pooled mask; reported alongside the day share).
    """
    ret = np.asarray(ret, dtype=float)
    ev = np.asarray(ev, dtype=bool)
    base = ~ev if base is None else np.asarray(base, dtype=bool)
    r_ev, r_b = ret[ev], ret[base]
    lr = np.log1p(ret)
    tot = float(lr.sum())
    return {
        "n_ev": int(ev.sum()), "n_base": int(base.sum()),
        "frac_days": float(ev.mean()),
        "ev_mean_bps": float(r_ev.mean() * 1e4), "base_mean_bps": float(r_b.mean() * 1e4),
        "diff_bps": float((r_ev.mean() - r_b.mean()) * 1e4),
        "welch_t": welch_t(r_ev, r_b),
        "ev_t0": ttest_vs_zero(r_ev),
        "share_of_total": float(lr[ev].sum() / tot) if tot != 0 else float("nan"),
        "ann_contrib_pct": float(np.expm1(lr[ev].sum() / (len(ret) / TRADING_DAYS)) * 100),
    }


def subperiod_stats(ret: pd.Series, ev: np.ndarray,
                    splits: list[tuple[str, str, str]]) -> list[dict]:
    """`event_vs_rest` per sub-period. ``splits`` = [(label, start, end), ...]."""
    out = []
    for label, lo, hi in splits:
        sel = (ret.index >= lo) & (ret.index <= hi)
        s = event_vs_rest(ret.values[sel], ev[sel])
        s["label"] = label
        out.append(s)
    return out


def placebo_pvalue(ret: np.ndarray, ev: np.ndarray, n_draws: int = 20_000,
                   seed: int = 602) -> dict:
    """Random pseudo-announcement-calendar placebo.

    Keeps the *number* of announcement days fixed but relocates them uniformly at random
    across the sample; each draw recomputes the A-minus-rest mean difference.
    ``p`` = P[shuffled diff >= observed]. The honest answer to "could any random set of
    days of the same density have looked this special?".
    """
    rng = np.random.default_rng(seed)
    ret = np.asarray(ret, dtype=float)
    ev = np.asarray(ev, dtype=bool)
    n, k = len(ret), int(ev.sum())
    obs = float(ret[ev].mean() - ret[~ev].mean())
    tot = float(ret.sum())
    diffs = np.empty(n_draws)
    for d in range(n_draws):
        sel = rng.choice(n, size=k, replace=False)
        s = float(ret[sel].sum())
        diffs[d] = s / k - (tot - s) / (n - k)
    return {"obs": obs, "p_value": float((diffs >= obs).mean()),
            "null_mean": float(diffs.mean()), "draws": diffs}


# --------------------------------------------------------------------------- #
# Tradable overlay
# --------------------------------------------------------------------------- #
def overlay_stats(ret: pd.Series, ev: np.ndarray, cost_bps: float = 2.0) -> dict:
    """Long SPY only on announcement days vs always-invested buy & hold.

    The calendar is published in advance: enter at the close of the session **before**
    each A-day (the one execution lag), exit at the A-day close — the position earns
    exactly the A-day close-to-close return. One round trip per A-day: ``2 * cost_bps``
    one-way x NAV charged against that day's return. Long-only (no borrow). Raw
    total-return arithmetic (no T-bill credit on idle cash — a conservative omission).
    """
    r = np.asarray(ret.values, dtype=float)
    ev = np.asarray(ev, dtype=bool)
    years = len(r) / TRADING_DAYS
    c = 2.0 * cost_bps / 1e4
    gross = r[ev]
    net = gross - c
    net_t = ttest_vs_zero(net)
    return {
        "cost_bps": cost_bps, "n_rt_per_yr": float(ev.sum() / years),
        "gross_bps_per_aday": float(gross.mean() * 1e4),
        "net_bps_per_aday": float(net.mean() * 1e4),
        "net_t0": net_t,
        "gross_ann_pct": float(np.expm1(np.log1p(gross).sum() / years) * 100),
        "net_ann_pct": float(np.expm1(np.log1p(net).sum() / years) * 100),
        "bh_ann_pct": float(np.expm1(np.log1p(r).sum() / years) * 100),
        "time_in_market_pct": float(ev.mean() * 100),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control harness (seed-averaged, per the desk's >= 20-seed rule)
# --------------------------------------------------------------------------- #
def synthetic_sweep(edge: float, n_seeds: int = 20, n_days: int = 7000,
                    base_seed: int = 602) -> dict:
    """Run the A-vs-rest Welch t on ``n_seeds`` independent synthetic worlds.

    Returns the mean diff / mean t across seeds and the share of seeds clearing |t|>=2.
    With ``edge = 0`` the mean t must sit ~0 and the clearing share near the nominal
    false-positive rate; with a planted edge the mean t must light up.
    """
    from . import data as _data

    ts, diffs = [], []
    for s in range(n_seeds):
        ser, mask = _data.synthetic_world(n_days=n_days, edge=edge, seed=base_seed + s)
        st = event_vs_rest(ser.values, mask)
        ts.append(st["welch_t"])
        diffs.append(st["diff_bps"])
    ts = np.asarray(ts)
    return {"edge_bps": edge * 1e4, "n_seeds": n_seeds,
            "mean_diff_bps": float(np.mean(diffs)), "mean_t": float(ts.mean()),
            "share_t_ge2": float((np.abs(ts) >= 2).mean())}
