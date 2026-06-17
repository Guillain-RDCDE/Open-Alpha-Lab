"""The analysis and its honest controls — Study 296 (Oscars-Effect).

The claim under test: the Academy Awards are a giant cultural event watched by
tens of millions, so "the market must react" — to who wins Best Picture, to the
implied box-office and streaming windfall for the winning studios, to the mood of
the nation. We test the cleanest tradeable version: does the broad market (^GSPC)
do anything unusual on the first trading session *after* the ceremony?

Four tests, all event-study flavoured:

  (A) **Abnormal return on event day 0** — mean return on the first post-ceremony
      session vs the full-tape baseline, with a Newey-West HAC t-stat. n = 31
      ceremonies, so the standard error is large by construction.
  (B) **Cumulative abnormal return (CAR)** over a small window (-1 .. +3) — does a
      drift build in the days bracketing the ceremony?
  (C) **A tradable rule** — buy the close before the ceremony reaction, sell a few
      days later, every year; net of one-way costs and an explicit execution lag.
      Compared against buy-and-hold over the same exposed days.
  (D) **A permutation / placebo control** — relabel 5,000 random sets of 31 sessions
      as "post-Oscar" and ask how often a random set looks as strong as the real one.

Honesty:
  - One execution lag: event day 0 is the first FULL session after the broadcast,
    never the ceremony evening itself.
  - One-way costs are charged on every entry and exit (round trips only on the days
    actually traded); the tape is price-only ^GSPC (no dividends), labelled as such.
  - Survivorship is not a concern on the index tape, but the n=31 small-sample
    caveat is the binding constraint and is stated on the Signal axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core statistics helper
# ---------------------------------------------------------------------------
def _hac_tstat(arr: np.ndarray) -> float:
    """Newey-West HAC t-statistic for the sample mean."""
    a = np.asarray(arr, dtype=float)
    n = a.size
    if n < 6:
        return float("nan")
    mu = a.mean()
    e = a - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def _plain_tstat(arr: np.ndarray) -> float:
    """Ordinary one-sample t-stat for the mean (i.i.d. assumption)."""
    a = np.asarray(arr, dtype=float)
    n = a.size
    if n < 2:
        return float("nan")
    sd = a.std(ddof=1)
    return float(a.mean() / (sd / np.sqrt(n))) if sd > 0 else float("nan")


# ---------------------------------------------------------------------------
# A: Abnormal return on event day 0
# ---------------------------------------------------------------------------
def event_day_test(ret: pd.Series, windows: pd.DataFrame) -> dict:
    """Mean return on the first post-ceremony session vs the full-tape baseline.

    The "abnormal return" is the event-day-0 mean minus the unconditional daily
    mean of the whole tape (a market-model with constant expected return). We
    report both a plain and a HAC t-stat for the event-day means; with only n = 31
    ceremonies these are the same order of magnitude.

    Returns a dict with ``n_events``, ``event_mean_bps``, ``baseline_mean_bps``,
    ``abnormal_bps``, ``event_t`` (HAC t on event-day returns), ``event_plain_t``,
    ``up_rate`` (fraction of event days that were positive), ``baseline_up_rate``.
    """
    r = pd.Series(ret).astype(float).dropna()
    base_mean = float(r.mean())
    base_up = float((r > 0).mean())

    ev = windows[windows["rel_day"] == 0]["ret"].to_numpy(dtype=float)
    n = ev.size
    ev_mean = float(ev.mean()) if n else float("nan")
    ev_up = float((ev > 0).mean()) if n else float("nan")

    return {
        "n_events": int(n),
        "event_mean_bps": float(ev_mean * 1e4),
        "baseline_mean_bps": float(base_mean * 1e4),
        "abnormal_bps": float((ev_mean - base_mean) * 1e4),
        "event_t": _hac_tstat(ev),
        "event_plain_t": _plain_tstat(ev),
        "up_rate": float(ev_up),
        "baseline_up_rate": float(base_up),
    }


# ---------------------------------------------------------------------------
# B: Cumulative abnormal return over the event window
# ---------------------------------------------------------------------------
def car_profile(ret: pd.Series, windows: pd.DataFrame) -> pd.DataFrame:
    """Average abnormal return by relative day, and the running CAR.

    For each relative day in the window we compute the cross-ceremony mean return,
    subtract the full-tape daily baseline to get the average abnormal return (AAR),
    and accumulate AAR into the cumulative abnormal return (CAR). A plain t-stat per
    relative day is included.

    Returns a DataFrame indexed by ``rel_day`` with columns ``aar_bps``, ``car_bps``,
    ``t``, ``n``.
    """
    r = pd.Series(ret).astype(float).dropna()
    base = float(r.mean())

    rows = []
    for rel, grp in windows.groupby("rel_day"):
        x = grp["ret"].to_numpy(dtype=float)
        aar = float(x.mean()) - base
        rows.append({
            "rel_day": int(rel),
            "aar_bps": aar * 1e4,
            "t": _plain_tstat(x - base),
            "n": int(x.size),
        })
    out = pd.DataFrame(rows).sort_values("rel_day").reset_index(drop=True)
    out["car_bps"] = out["aar_bps"].cumsum()
    return out[["rel_day", "aar_bps", "car_bps", "t", "n"]]


# ---------------------------------------------------------------------------
# C: A tradable rule — net of costs and execution lag
# ---------------------------------------------------------------------------
def tradable_rule(
    ret: pd.Series,
    windows: pd.DataFrame,
    hold_days: int = 1,
    cost_bps_one_way: float = 1.0,
) -> dict:
    """Buy the open of event day 0 and hold ``hold_days`` sessions, every year.

    The trade owns relative days ``0 .. hold_days-1`` (the first ``hold_days``
    sessions after the broadcast). One round trip per ceremony: a one-way cost is
    charged on entry and on exit. The tape is price-only ^GSPC.

    We compare the strategy's annualised return *per exposed day* against simple
    buy-and-hold over the same exposed sessions (to neutralise the obvious "you are
    out of the market most of the year" confound).

    Returns ``n_trades``, ``gross_mean_bps`` (per trade), ``net_mean_bps`` (per
    trade), ``net_t`` (HAC t on per-trade net returns), ``bah_mean_bps`` (buy-and-hold
    over the same exposed days), ``total_net_ret``, ``total_bah_ret``.
    """
    sel = windows[(windows["rel_day"] >= 0) & (windows["rel_day"] < hold_days)].copy()
    cost = 2.0 * cost_bps_one_way * 1e-4  # entry + exit, charged once per trade

    per_trade_gross = []
    per_trade_net = []
    for _, grp in sel.groupby("ceremony"):
        compounded = float((1.0 + grp["ret"]).prod() - 1.0)
        per_trade_gross.append(compounded)
        per_trade_net.append(compounded - cost)

    g = np.asarray(per_trade_gross, dtype=float)
    nnet = np.asarray(per_trade_net, dtype=float)
    n_trades = g.size

    # Buy-and-hold over exactly the same exposed sessions (no costs, no entry/exit).
    bah = sel["ret"].to_numpy(dtype=float)

    total_net = float(np.prod(1.0 + nnet) - 1.0) if n_trades else float("nan")
    total_bah = float(np.prod(1.0 + bah) - 1.0) if bah.size else float("nan")

    return {
        "n_trades": int(n_trades),
        "hold_days": int(hold_days),
        "cost_bps_one_way": float(cost_bps_one_way),
        "gross_mean_bps": float(g.mean() * 1e4) if n_trades else float("nan"),
        "net_mean_bps": float(nnet.mean() * 1e4) if n_trades else float("nan"),
        "net_t": _hac_tstat(nnet),
        "bah_mean_bps": float(bah.mean() * 1e4) if bah.size else float("nan"),
        "total_net_ret": total_net,
        "total_bah_ret": total_bah,
    }


# ---------------------------------------------------------------------------
# D: Permutation / placebo control
# ---------------------------------------------------------------------------
def permutation_null(
    ret: pd.Series,
    windows: pd.DataFrame,
    n_perm: int = 5000,
    seed: int = 296,
) -> dict:
    """How often does a *random* set of 31 sessions match the real event-day mean?

    We draw ``n_perm`` random samples of ``n_events`` trading days (matching the real
    count) and compute each sample's mean return. The two-sided placebo p-value is
    the fraction of random samples whose absolute mean is at least as large as the
    observed event-day absolute mean. If the Oscars carried real information, the
    real event-day mean should sit far in the tail of this placebo distribution.

    Returns ``event_mean_bps``, ``placebo_p`` (two-sided), ``pct_below``, ``n_perm``.
    """
    r = pd.Series(ret).astype(float).dropna()
    vals = r.to_numpy(dtype=float)
    n_total = vals.size

    ev = windows[windows["rel_day"] == 0]["ret"].to_numpy(dtype=float)
    n_events = ev.size
    obs_mean = float(ev.mean()) if n_events else float("nan")
    obs_abs = abs(obs_mean - float(r.mean()))
    base = float(r.mean())

    rng = np.random.default_rng(seed)
    ge = 0
    below = 0
    for _ in range(n_perm):
        idx = rng.choice(n_total, size=n_events, replace=False)
        m = float(vals[idx].mean()) - base
        if abs(m) >= obs_abs:
            ge += 1
        if m < (obs_mean - base):
            below += 1

    return {
        "event_mean_bps": float(obs_mean * 1e4),
        "abnormal_bps": float((obs_mean - base) * 1e4),
        "placebo_p": float(ge / n_perm),
        "pct_below": float(below / n_perm),
        "n_perm": int(n_perm),
        "n_events": int(n_events),
    }


# ---------------------------------------------------------------------------
# Summary: all tests in one call
# ---------------------------------------------------------------------------
def summarize(
    ret: pd.Series,
    windows: pd.DataFrame,
    hold_days: int = 1,
    cost_bps_one_way: float = 1.0,
    n_perm: int = 5000,
    seed: int = 296,
) -> dict:
    """Run all event-study tests and return a combined results dict.

    This is the study's spine — call it with the real tape's returns and the event
    windows to reproduce all headline numbers in ``docs/results.md``.
    """
    et = event_day_test(ret, windows)
    car = car_profile(ret, windows)
    trade = tradable_rule(ret, windows, hold_days=hold_days, cost_bps_one_way=cost_bps_one_way)
    perm = permutation_null(ret, windows, n_perm=n_perm, seed=seed)

    r = pd.Series(ret).astype(float).dropna()
    n = len(r)
    years = n / 252.0
    cagr = float((1 + r).prod() ** (1 / years) - 1) if years > 0 else float("nan")

    # CAR over the -1..+post window and over 0..+post window.
    car_full = float(car["car_bps"].iloc[-1])
    car_post = float(car.loc[car["rel_day"] >= 0, "aar_bps"].sum())

    return {
        "n_obs": int(n),
        "years": float(years),
        "cagr": float(cagr),
        # event-day-0
        "n_events": et["n_events"],
        "event_mean_bps": et["event_mean_bps"],
        "baseline_mean_bps": et["baseline_mean_bps"],
        "abnormal_bps": et["abnormal_bps"],
        "event_t": et["event_t"],
        "event_plain_t": et["event_plain_t"],
        "up_rate": et["up_rate"],
        "baseline_up_rate": et["baseline_up_rate"],
        # CAR
        "car_full_bps": car_full,
        "car_post_bps": car_post,
        # tradable rule
        "trade_gross_bps": trade["gross_mean_bps"],
        "trade_net_bps": trade["net_mean_bps"],
        "trade_net_t": trade["net_t"],
        "trade_bah_bps": trade["bah_mean_bps"],
        # permutation / placebo
        "placebo_p": perm["placebo_p"],
    }
