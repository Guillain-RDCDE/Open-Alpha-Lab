"""Strategy + inference for Study 517 — Pre-FOMC-Drift.

The claim (Lucca & Moench 2015): a large fraction of the equity premium is earned in the
~24 hours BEFORE a scheduled FOMC announcement. We test it as a clean event/calendar study:

    * **Pre-FOMC day** = the trading session that ends just before each scheduled FOMC
      announcement (the meeting date is pre-known months ahead — no look-ahead).
    * **The signal** = the mean SPY return on pre-FOMC days vs all other days (a two-sample
      Welch t), plus the share of cumulative return those ~3% of sessions carry.
    * **The decay test** = the same, split pre/post-2012 (the post-publication era) — the
      McLean-Pontiff question: did the edge survive being written up?
    * **The overnight/intraday decomposition** (study-specific third axis) = is the pre-FOMC
      return earned in the overnight gap (prev close -> open) or the intraday session
      (open -> close)? The original paper is about the run-up *into* the 14:15 release.
    * **Tradability** = a calendar timing rule that holds SPY only on pre-FOMC days, net of
      one-way costs x turnover (two trades per meeting).

Inference, the desk's shared spirit:
  * a **Welch t** of pre-FOMC mean return vs the other-days mean;
  * a **placebo / label-shuffle** null — randomly relocate the same number of "event" days
    many times and ask how often a random calendar matches the observed pre-FOMC mean;
  * **one-day execution lag** is moot for a pre-known calendar, but the timing strategy still
    pays one-way costs x turnover on every entry/exit.

The decisive number is whether the pre-FOMC mean clears t >= 2 on the REAL tape AND survives
the placebo AND does not evaporate post-publication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Return helpers
# --------------------------------------------------------------------------- #
def daily_returns(prices: pd.Series) -> pd.Series:
    """Simple daily returns from an adjusted-close series."""
    return prices.dropna().pct_change().dropna()


def overnight_intraday(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Split SPY daily return into overnight (prev close -> open) and intraday (open -> close).

    Uses raw Open/Close. ``overnight`` = open / prev_close - 1; ``intraday`` = close/open - 1;
    ``daily`` (close-to-close, raw) = their compounded sum. Returns a frame indexed by date.
    """
    o = ohlc["Open"].astype(float)
    c = ohlc["Close"].astype(float)
    prev_c = c.shift(1)
    overnight = o / prev_c - 1.0
    intraday = c / o - 1.0
    daily = c / prev_c - 1.0
    out = pd.DataFrame({"overnight": overnight, "intraday": intraday, "daily": daily}).dropna()
    return out


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variance). NaN if either has < 2 points."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = a.var(ddof=1) / len(a)
    vb = b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    if se == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / se)


def one_sample_t(sample: np.ndarray) -> float:
    """One-sample t of ``sample`` against 0 (the pre-FOMC-day-mean edge test)."""
    sample = np.asarray(sample, dtype=float)
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float(sample.mean() / se)


def event_vs_rest(ret: pd.Series, mask: np.ndarray) -> dict:
    """Compare event-day returns (mask True) to all other days.

    Returns the two means, the count, the Welch t of (event - rest), the one-sample t of the
    event-day returns vs 0, and the share of cumulative (summed) return carried by the event
    days. ``ret`` and ``mask`` must be aligned (same length / order).
    """
    ret = np.asarray(ret, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    ev = ret[mask]
    rest = ret[~mask]
    total = ret.sum()
    share = float(ev.sum() / total) if total != 0 else float("nan")
    return {
        "n_event": int(mask.sum()), "n_rest": int((~mask).sum()),
        "ev_mean": float(ev.mean()) if len(ev) else float("nan"),
        "rest_mean": float(rest.mean()) if len(rest) else float("nan"),
        "ev_t0": one_sample_t(ev),
        "welch_t": welch_t(ev, rest),
        "ev_sum": float(ev.sum()), "total_sum": float(total),
        "share_of_total": share,
        "frac_days": float(mask.mean()),
    }


def placebo_pvalue(ret: pd.Series, mask: np.ndarray, n_draws: int = 20_000,
                   seed: int = 517) -> dict:
    """Label-shuffle placebo: relocate the same number of event days at random many times.

    Each draw picks ``n_event`` random days and computes their mean return; ``p`` = share of
    draws whose random mean >= the observed pre-FOMC mean. The honest "could a random calendar
    of the same size have looked this good?" null.
    """
    ret = np.asarray(ret, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    n = len(ret)
    k = int(mask.sum())
    obs = float(ret[mask].mean())
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        idx = rng.choice(n, size=k, replace=False)
        means[i] = ret[idx].mean()
    p = float((means >= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()), "p_value": p, "draws": means}


# --------------------------------------------------------------------------- #
# Tradability — the calendar timing rule
# --------------------------------------------------------------------------- #
def timing_strategy(ret: pd.Series, mask: np.ndarray, cost_bps: float = 1.0) -> dict:
    """Hold SPY only on pre-FOMC days; charge one-way costs x turnover.

    The rule is in cash except on pre-FOMC days. Each meeting is two trades (in the day before,
    out after): turnover = 2 per event. Net daily return on an event day = day return -
    2 * cost_bps (round trip). Reports gross/net mean per event-day, the annualised drag, and
    the net Welch t vs other days. Tiny ``cost_bps`` default (large, liquid SPY ETF).
    """
    ret = np.asarray(ret, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    c = cost_bps / 1e4
    round_trip = 2.0 * c
    ev = ret[mask]
    ev_net = ev - round_trip
    rest = ret[~mask]
    gross = float(ev.mean())
    net = float(ev_net.mean())
    return {
        "n_event": int(mask.sum()),
        "gross_mean": gross, "net_mean": net,
        "round_trip_cost": round_trip,
        "net_t0": one_sample_t(ev_net),
        "net_welch_t": welch_t(ev_net, rest),
        # if you sat in the pre-FOMC day every meeting, summed net return vs gross
        "gross_sum": float(ev.sum()), "net_sum": float(ev_net.sum()),
    }


def split_summary(ret: pd.Series, mask: np.ndarray, split: str = "2012-01-01") -> dict:
    """Pre/post-publication split of the event-vs-rest comparison (the decay test).

    ``ret`` must be a Series with a DatetimeIndex; ``mask`` aligned. Returns the full-sample,
    pre-split, and post-split ``event_vs_rest`` dicts.
    """
    ret = ret.dropna()
    mask = pd.Series(np.asarray(mask, dtype=bool), index=ret.index)
    sp = pd.Timestamp(split)
    pre = ret.index < sp
    post = ret.index >= sp
    return {
        "full": event_vs_rest(ret.values, mask.values),
        "pre": event_vs_rest(ret[pre].values, mask[pre].values),
        "post": event_vs_rest(ret[post].values, mask[post].values),
        "split": split,
    }
