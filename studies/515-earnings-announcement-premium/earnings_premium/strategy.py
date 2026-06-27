"""Strategy + inference for Study 515 — Earnings-Announcement Premium.

The claim (Frazzini & Lamont 2007; Beaver 1968): a disproportionate share of a stock's total
return is earned in the few days *around* its scheduled earnings announcement. We measure it
two complementary ways:

    * **Per-day premium.** Pool every stock-day; compare the mean daily return on
      **announcement days** (inside the ``[D-1, D+1]`` window) against the mean on
      **non-announcement days**. The premium is announce-mean minus rest-mean (in basis points
      per day), with a Welch t.

    * **Tradable calendar overlay.** A long-only strategy that holds the basket equal-weighted
      ONLY when names are inside their announcement window (entered the close one day after the
      window is known — no look-ahead), versus always-invested buy & hold. We compare the
      mean daily return *while in an announcement window* to buy & hold, and charge per-entry
      turnover costs.

Inference, the desk's shared spirit:

  * a **one-sample t** of the per-event window-vs-baseline premium against 0 (the primary
    real-tape test) and a **Welch t** of pooled announce-days vs rest-days;
  * a **placebo / label-shuffle** null — randomly relocate the announcement flags within each
    name's calendar (preserving the count of "event" days) and ask how often a random tag set
    yields as large a premium (the honest test for a calendar effect);
  * a **share-of-return** decomposition: what fraction of total return is earned on the ~few
    percent of days that are announcement days;
  * **one-day execution lag** and **one-way costs × turnover** on the tradable overlay.

The decisive number is the per-event premium t on the REAL tape, and whether the overlay's
edge survives the per-entry turnover cost of trading a calendar that fires only ~12 days a year
per name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


# --------------------------------------------------------------------------- #
# Per-day premium (pooled announce vs rest)
# --------------------------------------------------------------------------- #
def pooled_returns(flags: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pool all stock-days into (announce-day returns, rest-day returns).

    ``flags`` is ``{ticker: (idx, ret, is_announce)}`` from ``data.build_announce_flags`` (or the
    synthetic equivalent). Drops the first NaN return of each name.
    """
    ann, rest = [], []
    for _tk, (_idx, ret, fl) in flags.items():
        r = ret[1:]
        f = fl[1:]
        ann.append(r[f])
        rest.append(r[~f])
    return np.concatenate(ann), np.concatenate(rest)


def per_event_premium(flags: dict) -> np.ndarray:
    """One window-vs-baseline premium per (ticker, announcement event).

    For each name, walk its tagged windows; each contiguous run of announce-days is one event.
    The event premium = mean daily return inside that window minus the name's mean daily return
    on its own non-announcement days (a within-name baseline, so cross-name level differences
    cancel). Returns one premium per event — the sample for the primary one-sample t.
    """
    out = []
    for _tk, (_idx, ret, fl) in flags.items():
        r = ret[1:]
        f = fl[1:]
        base = r[~f]
        if len(base) < 5:
            continue
        base_mu = float(base.mean())
        # contiguous runs of True == individual announcement windows
        i = 0
        n = len(f)
        while i < n:
            if f[i]:
                j = i
                while j < n and f[j]:
                    j += 1
                win = r[i:j]
                if len(win) > 0:
                    out.append(float(win.mean()) - base_mu)
                i = j
            else:
                i += 1
    return np.asarray(out, dtype=float)


def announce_day_fraction(flags: dict) -> float:
    """Share of all stock-days that fall inside an announcement window."""
    tot = 0
    ann = 0
    for _tk, (_idx, ret, fl) in flags.items():
        tot += len(fl) - 1
        ann += int(fl[1:].sum())
    return ann / tot if tot else float("nan")


def share_of_return(flags: dict) -> dict:
    """What fraction of cumulative log return is earned on announcement days.

    Sums log(1+r) over announce-days and over all days; the ratio is the 'disproportionate
    share' number (compared to the share of *days* that are announcement days).
    """
    ann_logret = 0.0
    all_logret = 0.0
    ann_days = 0
    all_days = 0
    for _tk, (_idx, ret, fl) in flags.items():
        r = ret[1:]
        f = fl[1:]
        lr = np.log1p(r)
        ann_logret += float(lr[f].sum())
        all_logret += float(lr.sum())
        ann_days += int(f.sum())
        all_days += len(r)
    return {
        "ann_logret": ann_logret, "all_logret": all_logret,
        "share_return": (ann_logret / all_logret) if all_logret else float("nan"),
        "share_days": (ann_days / all_days) if all_days else float("nan"),
        "ann_days": ann_days, "all_days": all_days,
    }


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def ttest_vs_zero(sample: np.ndarray) -> float:
    """One-sample t of ``sample`` against 0."""
    sample = np.asarray(sample, dtype=float)
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float(sample.mean() / se)


def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of mean(sample) - mean(base) (unequal variance). NaN if either < 2."""
    sample = np.asarray(sample, dtype=float)
    base = np.asarray(base, dtype=float)
    if len(sample) < 2 or len(base) < 2:
        return float("nan")
    m1, m0 = sample.mean(), base.mean()
    v1 = sample.var(ddof=1) / len(sample)
    v0 = base.var(ddof=1) / len(base)
    se = np.sqrt(v1 + v0)
    if se == 0:
        return float("nan")
    return float((m1 - m0) / se)


def placebo_pvalue(flags: dict, n_draws: int = 20_000, seed: int = 515) -> dict:
    """Random-tag placebo null for the per-day premium.

    For each name we keep the *number* of announcement days fixed but randomly relocate which
    days carry the tag (a random calendar of the same density). Each draw recomputes the pooled
    announce-minus-rest mean; ``p`` = P[shuffled premium >= observed]. The honest answer to
    "could a random set of calendar days of the same density have looked this special?".
    """
    rng = np.random.default_rng(seed)
    # precompute per-name returns and announce-day counts
    rets = []
    counts = []
    for _tk, (_idx, ret, fl) in flags.items():
        r = ret[1:]
        rets.append(r)
        counts.append(int(fl[1:].sum()))
    ann, rest = pooled_returns(flags)
    obs = float(ann.mean() - rest.mean())

    means = np.empty(n_draws)
    for d in range(n_draws):
        ann_sum = 0.0
        ann_n = 0
        rest_sum = 0.0
        rest_n = 0
        for r, k in zip(rets, counts):
            n = len(r)
            if k <= 0 or k >= n:
                rest_sum += r.sum()
                rest_n += n
                continue
            sel = rng.choice(n, size=k, replace=False)
            mask = np.zeros(n, dtype=bool)
            mask[sel] = True
            ann_sum += r[mask].sum()
            ann_n += k
            rest_sum += r[~mask].sum()
            rest_n += n - k
        means[d] = (ann_sum / ann_n) - (rest_sum / rest_n)
    p = float((means >= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()), "p_value": p, "draws": means}


def summarize_premium(flags: dict, n_draws: int = 20_000, placebo: bool = True) -> dict:
    """Headline per-day premium stats: pooled means, Welch t, per-event one-sample t, placebo."""
    ann, rest = pooled_returns(flags)
    ev = per_event_premium(flags)
    welch = welch_t(ann, rest)
    t_ev = ttest_vs_zero(ev)
    p = (placebo_pvalue(flags, n_draws=n_draws)["p_value"]
         if placebo else float("nan"))
    sr = share_of_return(flags)
    return {
        "n_ann": int(len(ann)), "n_rest": int(len(rest)), "n_events": int(len(ev)),
        "ann_mean_bps": float(ann.mean() * 1e4), "rest_mean_bps": float(rest.mean() * 1e4),
        "premium_bps": float((ann.mean() - rest.mean()) * 1e4),
        "welch_t": welch, "event_premium_bps": float(ev.mean() * 1e4), "t_event": t_ev,
        "p_placebo": p,
        "share_return": sr["share_return"], "share_days": sr["share_days"],
    }


# --------------------------------------------------------------------------- #
# Tradable calendar overlay
# --------------------------------------------------------------------------- #
def overlay_stats(flags: dict, cost_bps: float = 5.0, window: int = 1) -> dict:
    """Long-only announcement-window overlay vs always-invested buy & hold.

    The overlay holds a name equal-weighted ONLY during its announcement windows (entered the
    day the window opens, which is already known one day ahead from the scheduled date — but we
    still apply a 1-day execution lag implicitly: the window's first day return is realised by a
    position we set at the prior close). Each window is one round trip: enter + exit =
    ``2 * cost_bps`` one-way. We report the gross and net mean daily return *while in a window*,
    annualised, versus the basket's buy & hold mean daily return annualised.

    Turnover note: a name fires ~4 windows/yr × (2*window+1) days, so the overlay is in-market
    only a small fraction of the time but pays a round trip every window — the cost driver.
    """
    ann, rest = pooled_returns(flags)
    all_ret = np.concatenate([ann, rest])
    c = cost_bps / 1e4
    span_days = 2 * window + 1
    # one round trip (enter+exit) amortised over the in-window days
    per_day_cost = (2.0 * c) / span_days
    gross_daily = float(ann.mean())
    net_daily = gross_daily - per_day_cost
    bh_daily = float(all_ret.mean())
    return {
        "gross_daily_bps": gross_daily * 1e4, "net_daily_bps": net_daily * 1e4,
        "bh_daily_bps": bh_daily * 1e4,
        "gross_ann_pct": (np.expm1(gross_daily * TRADING_DAYS)) * 100,
        "net_ann_pct": (np.expm1(net_daily * TRADING_DAYS)) * 100,
        "bh_ann_pct": (np.expm1(bh_daily * TRADING_DAYS)) * 100,
        "cost_bps": cost_bps, "per_day_cost_bps": per_day_cost * 1e4,
        "n_window_days": int(len(ann)),
    }


def event_premium_by_seed_robust(flags: dict) -> dict:
    """The per-event premium t is deterministic on the real tape (no RNG); report it plainly.

    Kept as a named entry point so the verify script + notebooks can show the primary t is not
    a lucky-seed artefact: it is a fixed function of the cached data, with no random component.
    """
    ev = per_event_premium(flags)
    return {"n_events": int(len(ev)), "mean_bps": float(ev.mean() * 1e4),
            "t": ttest_vs_zero(ev)}
