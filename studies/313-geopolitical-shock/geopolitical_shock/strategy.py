"""The event-study engine and its honest controls — Study 313 (Geopolitical-Shock).

The claim under test, steelmanned: *major geopolitical shocks — wars, invasions, terror
attacks — knock the stock market down, but it shrugs them off within days.* The folk
corollary a trader actually cares about is the tradable one: **buy the geopolitical dip**
— go long SPY the session after a shock and ride the rebound.

The machinery:

- ``event_window`` cuts the ±k-session abnormal-return path around each event, where the
  "abnormal" return is the excess over the sample mean (a constant-mean market model — the
  simplest event-study benchmark; CAR = cumulative abnormal return).
- ``car_path`` averages those windows into the mean CAR path (Beat 4's headline chart).
- ``placebo_distribution`` is the **synthetic control / falsification**: re-run the exact
  same CAR computation on hundreds of *random non-event dates* to learn what a CAR of this
  size looks like by pure chance. A real shock effect must sit in the tail of that
  distribution; if it sits in the bulk, the "effect" is noise.
- ``block_bootstrap_car_ci`` puts a circular-block-bootstrap CI on the post-event CAR, so
  the volatility clustering the inference must respect is preserved (i.i.d. resampling
  would understate the error).
- ``buy_the_dip`` is the tradable overlay: enter at the close of the trade date (the
  shock is known), earn the next ``hold`` sessions of return — **one** execution lag,
  applied once — with one-way costs charged on entry and exit against NAV.

Costs are one-way × NAV; the strategy is long-only (no borrow). Returns are computed on
the cached adjusted close (total-return-ish if the cache was pulled with auto_adjust).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Returns + abnormal returns (constant-mean market model)
# ---------------------------------------------------------------------------
def daily_returns(bars: pd.DataFrame) -> pd.Series:
    """Simple close-to-close daily returns."""
    return bars["close"].pct_change()


def abnormal_returns(ret: pd.Series) -> pd.Series:
    """Abnormal return = return minus the full-sample mean (constant-mean model).

    The simplest event-study benchmark (Brown & Warner 1985): the "normal" return is a
    constant (the sample mean), so the abnormal return is the demeaned series. It removes
    the trivial up-drift of equities so a post-shock CAR is not just "stocks go up".
    """
    return ret - ret.mean()


# ---------------------------------------------------------------------------
# Event windows
# ---------------------------------------------------------------------------
def event_window(
    ar: pd.Series,
    event_date: pd.Timestamp,
    pre: int = 5,
    post: int = 10,
) -> np.ndarray | None:
    """The abnormal-return path from ``-pre`` to ``+post`` sessions around an event.

    ``event_date`` is the trade date (session 0). Returns a 1-D array of length
    ``pre + post + 1`` (positions -pre..0..+post) or ``None`` if the window runs off the
    edge of the tape or the event date is not in the index.
    """
    idx = ar.index
    pos = idx.searchsorted(event_date)
    if pos >= len(idx) or idx[pos] != event_date:
        # Event date not an exact session — snap to the next session on or after it.
        if pos >= len(idx):
            return None
    lo, hi = pos - pre, pos + post
    if lo < 0 or hi >= len(idx):
        return None
    return ar.to_numpy()[lo : hi + 1]


def stack_windows(
    ar: pd.Series,
    event_dates: pd.Series | list,
    pre: int = 5,
    post: int = 10,
) -> np.ndarray:
    """Stack every valid event window into a (n_events, pre+post+1) matrix."""
    rows = []
    for d in pd.to_datetime(pd.Series(event_dates)):
        w = event_window(ar, d, pre, post)
        if w is not None:
            rows.append(w)
    if not rows:
        return np.empty((0, pre + post + 1))
    return np.vstack(rows)


def car_path(windows: np.ndarray, pre: int = 5) -> np.ndarray:
    """Mean cumulative abnormal return (CAR) path, indexed from -pre.

    CAR at offset h = average across events of the *cumulative* abnormal return from
    session 0 (the event) through session h. Pre-event offsets accumulate backward to 0,
    so CAR(0) is the event-day abnormal return and CAR(post) is the headline drift.
    """
    if windows.size == 0:
        return np.array([])
    mean_ar = windows.mean(axis=0)            # avg abnormal return at each offset
    car = np.cumsum(mean_ar)                   # cumulative from the left edge
    # Re-anchor so CAR == 0 the session *before* the event (offset -1).
    anchor = car[pre - 1] if pre >= 1 else 0.0
    return car - anchor


def post_event_car(windows: np.ndarray, pre: int = 5, post: int = 10) -> np.ndarray:
    """Per-event post-event CAR (sum of abnormal returns over offsets +1..+post).

    One number per event — the cumulative abnormal return over the ``post`` sessions
    after the shock. This is the vector the t-test and bootstrap operate on.
    """
    if windows.size == 0:
        return np.array([])
    # columns: 0..pre-1 = pre, pre = event day (offset 0), pre+1.. = post.
    return windows[:, pre + 1 : pre + 1 + post].sum(axis=1)


# ---------------------------------------------------------------------------
# Inference — HAC-free here (events are far apart, treat as i.i.d. across events),
# but we report the cross-event t and a block bootstrap of the path.
# ---------------------------------------------------------------------------
def car_tstat(per_event_car: np.ndarray) -> float:
    """Cross-event t-stat of the mean post-event CAR (events ~independent in time)."""
    x = per_event_car[np.isfinite(per_event_car)]
    n = x.size
    if n < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(n)))


def placebo_distribution(
    ar: pd.Series,
    n_events: int,
    pre: int = 5,
    post: int = 10,
    n_draws: int = 2000,
    seed: int = 313,
) -> np.ndarray:
    """The synthetic control: mean post-event CAR on ``n_draws`` sets of random dates.

    Each draw picks ``n_events`` random sessions (away from the edges), computes the same
    mean post-event CAR, and records it. The real CAR's percentile in this distribution
    is the falsification test: a real shock effect sits in the tail. This is a *machinery
    /placebo* control — it can show the harness detects nothing where nothing is planted;
    it can never, by itself, certify a real-market signal.
    """
    rng = np.random.default_rng(seed)
    vals = ar.to_numpy()
    n = vals.size
    lo, hi = pre + 1, n - post - 1
    if hi <= lo or n_events <= 0:
        return np.array([])
    out = np.empty(n_draws)
    for d in range(n_draws):
        locs = rng.integers(lo, hi, size=n_events)
        # mean post-event CAR over the random "events"
        per = np.array([vals[loc + 1 : loc + 1 + post].sum() for loc in locs])
        out[d] = per.mean()
    return out


def placebo_percentile(real_car: float, placebo: np.ndarray) -> float:
    """Two-sided percentile of ``real_car`` within the placebo distribution (0..100)."""
    if placebo.size == 0 or not np.isfinite(real_car):
        return float("nan")
    return float((placebo <= real_car).mean() * 100.0)


def block_bootstrap_car_ci(
    per_event_car: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 313,
) -> tuple[float, float]:
    """Percentile CI on the mean post-event CAR by resampling events with replacement.

    Events are far apart in time, so the resampling unit is the *event* (an i.i.d.
    bootstrap over events; there is no within-event serial structure to preserve once we
    summarise each event to a single CAR). Returns ``(lo, hi)`` at level ``1 - alpha``.
    """
    x = per_event_car[np.isfinite(per_event_car)]
    n = x.size
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = x[rng.integers(0, n, size=n)].mean()
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


# ---------------------------------------------------------------------------
# The tradable overlay — "buy the geopolitical dip"
# ---------------------------------------------------------------------------
def buy_the_dip(
    bars: pd.DataFrame,
    event_dates: pd.Series | list,
    hold: int = 10,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Long-only overlay: enter at the close of the trade date, hold ``hold`` sessions.

    The shock is known at the **close** of the trade date (session 0), so the position
    earns sessions ``+1 .. +hold`` of return — a single execution lag, applied once. One
    round trip per event, so a one-way cost ``cost_bps`` is charged **twice** (entry +
    exit) against NAV. Long-only: no borrow.

    Returns a per-event ledger: ``entry_date, hold, ret_gross, ret_net``.
    """
    close = bars["close"]
    idx = close.index
    rows = []
    for d in pd.to_datetime(pd.Series(event_dates)):
        pos = idx.searchsorted(d)
        if pos >= len(idx) or idx[pos] != d:
            continue
        entry = pos               # known at close of session 0
        exit_ = entry + hold      # hold sessions later
        if exit_ >= len(idx):
            continue
        gross = close.iat[exit_] / close.iat[entry] - 1.0
        net = gross - 2.0 * cost_bps * 1e-4   # one-way x2 (entry + exit)
        rows.append(
            {
                "entry_date": idx[entry],
                "hold": hold,
                "ret_gross": float(gross),
                "ret_net": float(net),
            }
        )
    return pd.DataFrame(rows)


def summarize_dip(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline stats for the buy-the-dip ledger: n, win-rate, mean, t-stat."""
    if ledger.empty:
        return {"n": 0, "win_rate": float("nan"), "mean_bps": float("nan"),
                "tstat": float("nan")}
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out = {
        "n": int(n),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "tstat": float("nan"),
    }
    if n >= 2 and r.std(ddof=1) > 0:
        out["tstat"] = float(r.mean() / (r.std(ddof=1) / np.sqrt(n)))
    return out
