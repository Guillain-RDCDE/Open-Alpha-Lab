"""Strategy + honest controls — Study 440 (Floor-Trader Pivot Points).

The folk claim: the classic floor-trader pivots — pivot ``P`` and its supports/resistances
``S1/S2/S3`` and ``R1/R2/R3``, all computed from the **prior** session's high/low/close — act
as intraday support and resistance. Price is supposed to *respect* them: it touches the level
and bounces (reversion) or breaks and runs (continuation). So you fade the touch.

We turn that into two falsifiable measurements, both pinned against a **randomly-placed control
level** drawn on the *same* tape (a horizontal line at a random price inside the session range)
— the only honest baseline, because *any* line gets touched and price wanders away from *any*
line some of the time:

1. **Touch → bounce hit-rate.** When price touches a pivot level, how often does it reverse
   over the next ``k`` bars (move back away from the level)? Compare the pivot hit-rate to the
   control-line hit-rate. If pivots are special, pivot bounces beat control bounces.

2. **Fade-the-level trade.** On a touch of S1/S2/S3 go long (or R1/R2/R3 go short) for ``k``
   bars; measure the per-trade return, its HAC *t* against zero, and the same trade on control
   levels. Charge intraday round-trip costs (spread is the killer here).

Honest arbiters, the desk's shared spirit:
- a **one-sample / HAC t** of the fade-trade return against zero;
- a **permutation placebo** — re-draw the control levels many times and ask how often a random
  line matches the pivots' bounce-rate / trade edge (the small-effect null);
- **one-bar execution lag** (touch detected on bar *t* close, enter bar *t+1* open);
- **intraday round-trip costs** charged on the net return.

No look-ahead: pivots for session *D* use only sessions strictly before *D*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import _pivot_levels

# Which pivot lines are "support" (fade = go long on a touch) vs "resistance" (fade = short).
SUPPORTS = ("S1", "S2", "S3")
RESISTS = ("R1", "R2", "R3")
ALL_LEVELS = ("S3", "S2", "S1", "P", "R1", "R2", "R3")


# --------------------------------------------------------------------------- #
# Session helpers
# --------------------------------------------------------------------------- #
def drop_partial_sessions(bars: pd.DataFrame, min_bars: int = 75) -> pd.DataFrame:
    """Drop any session with fewer than ``min_bars`` bars (a partial/half day or the live day).

    A full 5-minute RTH session has ~78 bars; the in-progress day and exchange half-days are
    incomplete and would bias the touch counts, so a *stamped* run drops them (house rule:
    no partial bars in a stamped run).
    """
    n = bars.groupby("date").size()
    full = n[n >= min_bars].index
    return bars[bars["date"].isin(full)].copy()


def session_frames(bars: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    """Split an intraday tape into ``[(session_date, session_bars), ...]`` in time order."""
    out = []
    for d, sl in bars.groupby("date"):
        out.append((pd.Timestamp(d), sl.sort_index()))
    out.sort(key=lambda kv: kv[0])
    return out


def daily_hlc(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-session high/low/close (the inputs to the next day's pivots)."""
    g = bars.groupby("date")
    return pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last()}).sort_index()


# --------------------------------------------------------------------------- #
# Touch detection
# --------------------------------------------------------------------------- #
def _touches(session: pd.DataFrame, level: float, tol_bps: float) -> np.ndarray:
    """Integer bar positions where the session's bar range straddles ``level`` (a touch).

    A bar "touches" the level if ``low - tol <= level <= high + tol`` (tol = ``tol_bps`` of the
    level). Consecutive touch-bars collapse to the first (the entry bar of a touch episode).
    """
    tol = tol_bps / 1e4 * level
    hi = session["high"].to_numpy()
    lo = session["low"].to_numpy()
    hit = (lo - tol <= level) & (level <= hi + tol)
    if not hit.any():
        return np.empty(0, dtype=int)
    first = hit & ~np.concatenate([[False], hit[:-1]])
    return np.where(first)[0]


def bounce_after(session: pd.DataFrame, pos: int, level: float, side: str,
                 k: int = 6, lag: int = 1) -> float | None:
    """Did price *bounce away* from ``level`` over the next ``k`` bars after a touch at ``pos``?

    ``side='support'`` → a bounce is price moving UP (away from support);
    ``side='resistance'`` → a bounce is price moving DOWN.
    Enter at ``pos + lag`` close, measure to ``pos + lag + k`` close. Returns the signed
    *bounce return* (positive = price moved in the bounce direction), or None if the window
    overruns the session.
    """
    close = session["close"].to_numpy()
    entry = pos + lag
    exit_ = entry + k
    if entry < 0 or exit_ >= len(close):
        return None
    raw = close[exit_] / close[entry] - 1.0
    return raw if side == "support" else -raw


# --------------------------------------------------------------------------- #
# Build the event table: every pivot touch, with its forward bounce return
# --------------------------------------------------------------------------- #
def pivot_touch_events(bars: pd.DataFrame, levels: tuple[str, ...] = SUPPORTS + RESISTS,
                       tol_bps: float = 5.0, k: int = 6, lag: int = 1) -> pd.DataFrame:
    """One row per (session, pivot level, touch) with the forward bounce return.

    Pivots for session *D* come from session *D-1*'s H/L/C (no look-ahead). Columns:
    ``date, level_name, level, side, bounce`` (signed forward return in the bounce direction).
    """
    sess = session_frames(bars)
    rows = []
    for j in range(1, len(sess)):
        d, s = sess[j]
        _, prev = sess[j - 1]
        piv = _pivot_levels(float(prev["high"].max()), float(prev["low"].min()),
                            float(prev["close"].iloc[-1]))
        for name in levels:
            lv = piv[name]
            side = "support" if name in SUPPORTS or name == "P" else "resistance"
            if name == "P":          # P is treated as support for the fade convention
                side = "support"
            for pos in _touches(s, lv, tol_bps):
                b = bounce_after(s, pos, lv, side, k=k, lag=lag)
                if b is not None:
                    rows.append({"date": d, "level_name": name, "level": lv,
                                 "side": side, "bounce": b})
    return pd.DataFrame(rows)


def control_touch_events(bars: pd.DataFrame, tol_bps: float = 5.0, k: int = 6,
                         lag: int = 1, seed: int = 440,
                         n_per_session: int = 6) -> pd.DataFrame:
    """Same machinery on **randomly-placed** horizontal lines (the honest baseline).

    For each session we draw ``n_per_session`` random price levels uniformly inside that
    session's [low, high] range, randomly tag each as support/resistance, and run the identical
    touch → bounce measurement. This is what *any* arbitrary line yields on the same tape — the
    bar a real pivot must clear.
    """
    rng = np.random.default_rng(seed)
    sess = session_frames(bars)
    rows = []
    for j in range(1, len(sess)):
        d, s = sess[j]
        lo, hi = float(s["low"].min()), float(s["high"].max())
        if not np.isfinite(lo) or hi <= lo:
            continue
        for _ in range(n_per_session):
            lv = float(rng.uniform(lo, hi))
            side = "support" if rng.random() < 0.5 else "resistance"
            for pos in _touches(s, lv, tol_bps):
                b = bounce_after(s, pos, lv, side, k=k, lag=lag)
                if b is not None:
                    rows.append({"date": d, "level": lv, "side": side, "bounce": b})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_t(x: np.ndarray) -> float:
    """Newey-West HAC one-sample t of mean(x) against 0."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for kk in range(1, lags + 1):
        w = 1.0 - kk / (lags + 1.0)
        lrv += 2.0 * w * float(e[kk:] @ e[:-kk]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def bounce_rate(events: pd.DataFrame) -> float:
    """Share of touches that bounced (forward bounce return > 0)."""
    if events.empty:
        return float("nan")
    return float((events["bounce"].to_numpy() > 0).mean())


def trade_stats(events: pd.DataFrame, cost_bps: float = 0.0) -> dict:
    """Per-touch fade-trade stats: mean bounce return, HAC t, win-rate, net of round-trip cost.

    The fade trade goes *with* the bounce direction (long on support touch, short on resistance
    touch); ``bounce`` is already signed in that direction. ``cost_bps`` is the one-way cost;
    a round trip (in + out) costs ``2 * cost_bps``.
    """
    if events.empty:
        return {"n": 0, "mean_bps": float("nan"), "win": float("nan"),
                "t": float("nan"), "net_bps": float("nan")}
    r = events["bounce"].to_numpy(float)
    r = r[np.isfinite(r)]
    rt = 2.0 * cost_bps / 1e4
    net = r - rt
    return {
        "n": int(r.size),
        "mean_bps": float(r.mean() * 1e4),
        "win": float((r > 0).mean()),
        "t": hac_t(r),
        "net_bps": float(net.mean() * 1e4),
    }


def permutation_placebo(bars: pd.DataFrame, observed_rate: float, n_draws: int = 400,
                        tol_bps: float = 5.0, k: int = 6, lag: int = 1,
                        seed: int = 440, n_per_session: int = 6) -> dict:
    """How often does a *random* line match the pivots' bounce-rate?

    Re-draw the random control lines ``n_draws`` times (different seeds); each draw yields a
    control bounce-rate. ``p`` = P[control rate >= observed pivot rate] — the honest "could an
    arbitrary line have looked this good?" null.
    """
    rates = np.empty(n_draws)
    for i in range(n_draws):
        ev = control_touch_events(bars, tol_bps=tol_bps, k=k, lag=lag,
                                  seed=seed + i, n_per_session=n_per_session)
        rates[i] = bounce_rate(ev)
    rates = rates[np.isfinite(rates)]
    p = float((rates >= observed_rate).mean()) if rates.size else float("nan")
    return {"p_value": p, "control_mean": float(np.nanmean(rates)),
            "control_sd": float(np.nanstd(rates)), "draws": rates}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(bars: pd.DataFrame, tol_bps: float = 5.0, k: int = 6, lag: int = 1,
                   cost_bps: float = 1.0, n_placebo: int = 400, seed: int = 440,
                   n_per_session: int = 6) -> dict:
    """Full pivot-vs-control teardown on one intraday tape.

    Returns a dict with the pivot touch/bounce stats, the random-control stats, the difference,
    a permutation placebo p-value on the bounce-rate, and the net-of-cost fade-trade edge.
    """
    piv = pivot_touch_events(bars, tol_bps=tol_bps, k=k, lag=lag)
    ctl = control_touch_events(bars, tol_bps=tol_bps, k=k, lag=lag, seed=seed,
                               n_per_session=n_per_session)
    piv_rate = bounce_rate(piv)
    ctl_rate = bounce_rate(ctl)
    piv_tr = trade_stats(piv, cost_bps=cost_bps)
    ctl_tr = trade_stats(ctl, cost_bps=cost_bps)
    plac = permutation_placebo(bars, piv_rate, n_draws=n_placebo, tol_bps=tol_bps,
                               k=k, lag=lag, seed=seed + 1, n_per_session=n_per_session)
    return {
        "n_pivot_touches": piv_tr["n"], "n_control_touches": ctl_tr["n"],
        "pivot_bounce_rate": piv_rate, "control_bounce_rate": ctl_rate,
        "rate_diff": (piv_rate - ctl_rate) if np.isfinite(piv_rate) and np.isfinite(ctl_rate)
        else float("nan"),
        "pivot_mean_bps": piv_tr["mean_bps"], "pivot_t": piv_tr["t"],
        "pivot_win": piv_tr["win"], "pivot_net_bps": piv_tr["net_bps"],
        "control_mean_bps": ctl_tr["mean_bps"], "control_t": ctl_tr["t"],
        "placebo_p": plac["p_value"], "control_rate_mean": plac["control_mean"],
        "cost_bps": cost_bps, "tol_bps": tol_bps, "k": k,
    }
