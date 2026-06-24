"""Strategy + inference for Study 441 — Camarilla Pivots.

The claim: the Camarilla **reversal** levels (L3 / H3, the headline lines) are support/
resistance the market *respects*. Touch L3 from above and price bounces back up toward the
central pivot; touch H3 from below and it rolls over. Day-traders fade L3/H3 toward P and run
breakouts beyond L4/H4.

We test the single falsifiable core — **does price respect the level more than a random line?**

    * **Ladder.** Each session's Camarilla levels come from the *prior* session's H/L/C
      (``data.camarilla_levels``) — known at the open, no look-ahead.
    * **Touch event.** Within a session, find the first 5-minute bar that touches L3 (a long
      candidate) or H3 (a short candidate). Enter the **next** bar's open (one-bar execution
      lag) and hold a fixed number of bars.
    * **Reversion return.** The signed forward return *in the direction of the central pivot*
      (long at L3, short at H3). If the level is real support/resistance, this should be
      positive on average.
    * **The placebo.** Re-run the identical touch→reversion rule on **random control levels**
      placed at the *same distance* from the day's open but at a shuffled offset. A real level
      must beat its own random twin; if a random line reverts just as well, "the line is
      special" is busted.

Inference (the desk's shared spirit):

  * a **one-sample t** of the per-event reversion return against 0, and a **HAC (Newey-West) t**
    on the daily-aggregated series (events cluster within a day);
  * a **random-control placebo** — the mean reversion at matched random levels, with the
    spread (real − random) and a bootstrap on the difference;
  * a **hit-rate** of "reverted toward the pivot" vs the 50% coin-flip;
  * **one-bar execution lag** and **round-trip intraday spread costs** charged loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import camarilla_levels

HOLD_BARS = 12          # ~1 hour on 5-minute bars
DEFAULT_LEVELS = ("L3", "H3")


# --------------------------------------------------------------------------- #
# Per-session ladder + touch detection
# --------------------------------------------------------------------------- #
def session_frames(bars: pd.DataFrame):
    """Yield ``(session_date, prev_hlc, day_bars)`` for each session after the first.

    ``prev_hlc`` = (high, low, close) of the *previous* session (the inputs to today's ladder).
    """
    groups = list(bars.groupby("session", sort=True))
    for k in range(1, len(groups)):
        prev_date, prev = groups[k - 1]
        date, day = groups[k]
        prev_hlc = (float(prev["high"].max()), float(prev["low"].min()),
                    float(prev["close"].iloc[-1]))
        yield date, prev_hlc, day


def _first_touch_idx(day: pd.DataFrame, level: float, side: str) -> int | None:
    """Index (integer position) of the first bar that touches ``level``.

    ``side='low'`` (support, L3): first bar whose low <= level. ``side='high'`` (resistance,
    H3): first bar whose high >= level. Returns None if never touched.
    """
    if side == "low":
        hit = np.where(day["low"].values <= level)[0]
    else:
        hit = np.where(day["high"].values >= level)[0]
    return int(hit[0]) if len(hit) else None


def _reversion_return(day: pd.DataFrame, touch_pos: int, hold: int,
                      direction: int, lag: int = 1) -> float | None:
    """Signed forward return from the bar after the touch, ``hold`` bars later.

    ``direction`` = +1 for a long (fade L3 up toward P), -1 for a short (fade H3 down). Enter at
    the open of ``touch_pos + lag``, exit at the close ``hold`` bars on. Returns None if the
    window overruns the session (no overnight carry).
    """
    entry = touch_pos + lag
    exit_ = entry + hold
    if entry >= len(day) or exit_ >= len(day):
        return None
    o = day["open"].values
    c = day["close"].values
    raw = c[exit_] / o[entry] - 1.0
    return float(direction * raw)


# --------------------------------------------------------------------------- #
# Event collection — real Camarilla levels and random controls
# --------------------------------------------------------------------------- #
def collect_events(bars: pd.DataFrame, levels=DEFAULT_LEVELS, hold: int = HOLD_BARS,
                   lag: int = 1) -> pd.DataFrame:
    """One row per (session, touched level) with the forward reversion return.

    Long the lower levels (L*) toward the pivot, short the upper levels (H*). The reversion
    return is signed so positive = "the level was respected" (price moved back toward P).
    """
    rows = []
    for date, (ph, pl, pc), day in session_frames(bars):
        lev = camarilla_levels(ph, pl, pc)
        day_open = float(day["open"].iloc[0])
        for name in levels:
            price = lev[name]
            is_support = name.startswith("L")
            side = "low" if is_support else "high"
            direction = +1 if is_support else -1
            tp = _first_touch_idx(day, price, side)
            if tp is None:
                continue
            rr = _reversion_return(day, tp, hold, direction, lag=lag)
            if rr is None:
                continue
            rows.append({"session": date, "level": name, "level_price": price,
                         "dist": abs(price - day_open) / day_open,
                         "touch_pos": tp, "rev": rr})
    return pd.DataFrame(rows)


def collect_random_controls(bars: pd.DataFrame, levels=DEFAULT_LEVELS, hold: int = HOLD_BARS,
                            lag: int = 1, n_draws: int = 50, seed: int = 441) -> pd.DataFrame:
    """The placebo: identical touch→reversion rule on RANDOM control levels.

    For each session we place control lines at the *same distance from the day open* as the real
    L3/H3 (so the touch geometry matches) but on a randomly chosen side/offset within the day's
    range, repeated ``n_draws`` times. Returns the pooled control reversion returns — the null
    "a line placed anywhere reverts this well." A real level must beat this cloud.
    """
    rng = np.random.default_rng(seed)
    revs = []
    sessions = list(session_frames(bars))
    for date, (ph, pl, pc), day in sessions:
        lev = camarilla_levels(ph, pl, pc)
        day_open = float(day["open"].iloc[0])
        rng_lo = float(day["low"].min())
        rng_hi = float(day["high"].max())
        span = rng_hi - rng_lo
        if span <= 0:
            continue
        for name in levels:
            dist = abs(lev[name] - day_open)
            is_support = name.startswith("L")
            for _ in range(n_draws):
                # random control at the matched distance, random side
                sign = -1 if (is_support if rng.random() < 0.5 else not is_support) else +1
                # place a random line within the day's range, matched distance from open
                base = day_open + rng.choice([-1, 1]) * dist * rng.uniform(0.6, 1.4)
                price = min(max(base, rng_lo + 1e-9), rng_hi - 1e-9)
                support_like = price < day_open
                side = "low" if support_like else "high"
                direction = +1 if support_like else -1
                tp = _first_touch_idx(day, price, side)
                if tp is None:
                    continue
                rr = _reversion_return(day, tp, hold, direction, lag=lag)
                if rr is not None:
                    revs.append(rr)
    return pd.DataFrame({"rev": revs})


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def ttest_vs_zero(sample: np.ndarray) -> float:
    """One-sample t of ``sample`` against 0."""
    sample = np.asarray(sample, dtype=float)
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    return float(sample.mean() / se) if se > 0 else float("nan")


def hac_t(daily: np.ndarray, lags: int = 5) -> float:
    """Newey-West (HAC) t of a daily-mean series against 0 — robust to within-window autocorr."""
    x = np.asarray(daily, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    gamma0 = (u @ u) / n
    var = gamma0
    for L in range(1, min(lags, n - 1) + 1):
        w = 1.0 - L / (lags + 1.0)
        cov = (u[L:] @ u[:-L]) / n
        var += 2.0 * w * cov
    se = np.sqrt(max(var, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def hit_rate(sample: np.ndarray) -> float:
    sample = np.asarray(sample, dtype=float)
    return float((sample > 0).mean()) if len(sample) else float("nan")


def bootstrap_diff_p(real: np.ndarray, control: np.ndarray,
                     n_boot: int = 5000, seed: int = 441) -> dict:
    """Bootstrap test of mean(real) - mean(control) > 0 (real level beats the random twin).

    Resamples both pools with replacement; ``p`` = P[bootstrapped diff <= 0] — the share of
    resamples where the real level fails to beat its random control.
    """
    real = np.asarray(real, dtype=float)
    control = np.asarray(control, dtype=float)
    obs = real.mean() - control.mean()
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    nr, nc = len(real), len(control)
    for i in range(n_boot):
        diffs[i] = (rng.choice(real, nr, replace=True).mean()
                    - rng.choice(control, nc, replace=True).mean())
    p = float((diffs <= 0).mean())
    return {"obs_diff": float(obs), "real_mean": float(real.mean()),
            "control_mean": float(control.mean()), "p_value": p}


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def net_of_costs(rev_mean: float, cost_bps: float = 2.0) -> dict:
    """Reversion mean net of a round-trip intraday spread (``cost_bps`` one-way × 2)."""
    c = 2.0 * cost_bps / 1e4
    return {"gross": float(rev_mean), "net": float(rev_mean - c),
            "cost_bps": cost_bps, "round_trip": float(c)}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(bars: pd.DataFrame, levels=DEFAULT_LEVELS, hold: int = HOLD_BARS,
                   lag: int = 1, n_control: int = 50, cost_bps: float = 2.0,
                   seed: int = 441) -> dict:
    """Full Camarilla teardown on one intraday tape (real or synthetic).

    Returns the event count, the mean reversion + one-sample t + HAC t + hit-rate, the
    random-control mean, the real−control spread with a bootstrap p, and the net-of-cost mean.
    """
    ev = collect_events(bars, levels=levels, hold=hold, lag=lag)
    ctrl = collect_random_controls(bars, levels=levels, hold=hold, lag=lag,
                                   n_draws=n_control, seed=seed)
    rev = ev["rev"].values if len(ev) else np.array([])
    daily = (ev.groupby("session")["rev"].mean().values if len(ev) else np.array([]))
    t = ttest_vs_zero(rev)
    th = hac_t(daily)
    diff = bootstrap_diff_p(rev, ctrl["rev"].values, seed=seed) if (len(rev) and len(ctrl)) else \
        {"obs_diff": float("nan"), "real_mean": float("nan"),
         "control_mean": float("nan"), "p_value": float("nan")}
    cost = net_of_costs(float(rev.mean()) if len(rev) else float("nan"), cost_bps=cost_bps)
    return {
        "n_events": int(len(ev)),
        "n_sessions": int(ev["session"].nunique()) if len(ev) else 0,
        "rev_mean": float(rev.mean()) if len(rev) else float("nan"),
        "t": t, "hac_t": th, "hit": hit_rate(rev),
        "control_mean": diff["control_mean"], "diff": diff["obs_diff"],
        "diff_p": diff["p_value"],
        "gross": cost["gross"], "net": cost["net"], "cost_bps": cost_bps,
    }
