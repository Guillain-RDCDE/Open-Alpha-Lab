"""Bump-and-Run Reversal (BARR) as a falsifiable mechanical rule — Study 467.

Thomas Bulkowski's *bump-and-run reversal* (from *Encyclopedia of Chart Patterns*) has three
ingredients, in order:

* **Lead-in trendline** — a gentle, low-slope trendline supporting a quiet up-move.
* **Bump** — a burst of speculation: price *steepens* and surges away from the lead-in line,
  rising to at least roughly **2×** the lead-in's height above the line (the "bump").
* **Break** — the speculation exhausts and price **breaks back below the lead-in trendline**.
  Bulkowski calls this the reversal signal; the folklore rule is to **short** the break.

The folklore (Bulkowski, repeated on every chart-pattern site): *the bump-then-break forecasts
a reversal* — once the speculative bump collapses through the lead-in line, price keeps falling,
so the trendline break is a high-probability **short**.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Lead-in trendline** — fit a least-squares line on a *trailing* window of ``lead`` closes
   (the lead-in). Its slope must be **gently positive** (a calm up-trend), not a steep ramp.
2. **Bump test** — over the next ``bump`` bars, the close must rise to at least ``bump_mult``×
   the lead-in's typical above-line distance: a genuine steepening surge, not a drift.
3. **Break** — a short entry fires the first bar whose close pierces **below** the (extended)
   lead-in trendline after a confirmed bump. Entry is at the **next** close (one documented
   lag); we then measure the forward H-day return of a **short** (negative of price change).
4. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold, same
   short side) that captures the tape's drift, and (b) a **shuffled-window placebo** that fits
   the lead-in line on a *permuted* slice of the trailing returns, destroying the bump-and-run
   geometry while keeping the marginal — the honest "is the BARR shape doing anything?" null.

No look-ahead: the lead-in line is fit only on bars ``< t``, the bump is measured on bars
``<= t``, the break is read on the close of *t*, the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Lead-in trendline geometry
# --------------------------------------------------------------------------- #
def _fit_line(x: np.ndarray, y: np.ndarray):
    """Least-squares line ``y = slope*x + intercept`` over (x, y); returns (slope, intercept)."""
    n = x.size
    sx = x.sum()
    sxx = float(x @ x)
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0, float(y.mean())
    sy = y.sum()
    sxy = float(x @ y)
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return float(slope), float(intercept)


# --------------------------------------------------------------------------- #
# Bump-and-run entries
# --------------------------------------------------------------------------- #
def barr_break_entries(
    close: pd.Series,
    lead: int = 60,
    bump: int = 30,
    bump_mult: float = 2.0,
    max_lead_slope_bps: float = 25.0,
) -> pd.DatetimeIndex:
    """Bars whose close breaks **below** the lead-in trendline after a confirmed speculative bump.

    For each candidate bar ``t`` we look back over a lead-in window of ``lead`` bars ending at
    ``t - bump`` (so the lead-in is entirely in the past and disjoint from the bump), fit the
    lead-in trendline, require its slope to be **gently positive** (0 < slope-per-bar below
    ``max_lead_slope_bps`` of price, in bps — a calm up-trend, not a ramp), then require a
    genuine **bump**: over the ``bump`` bars after the lead-in, the max close must exceed the
    extended lead-in line by at least ``bump_mult`` times the lead-in's own average above-line
    distance. The short fires the first bar whose close pierces **below** the extended lead-in
    line. Only the *first* bar of each consecutive run is kept (the break, not every day below).
    Entry is executed at the next close by :func:`forward_returns`.
    """
    p = close.to_numpy(dtype=float)
    n = p.size
    fire = np.zeros(n, dtype=bool)
    warm = lead + bump
    in_break = False
    for t in range(warm, n):
        # lead-in window: [t - bump - lead, t - bump)
        l0 = t - bump - lead
        l1 = t - bump
        xl = np.arange(l0, l1, dtype=float)
        yl = p[l0:l1]
        slope, intercept = _fit_line(xl, yl)
        # lead-in must be a gentle UP slope (calm trend), not flat or a steep ramp
        ref = float(yl.mean())
        slope_bps = slope / ref * 1e4 if ref > 0 else 0.0
        if not (0.0 < slope_bps <= max_lead_slope_bps):
            in_break = False
            continue
        # lead-in's own typical above-line distance (the unit of "bump height")
        line_lead = slope * xl + intercept
        above = yl - line_lead
        lead_height = float(np.maximum(above, 0.0).mean())
        if lead_height <= 0:
            in_break = False
            continue
        # bump window: [t - bump, t]; extended lead-in line over those bars
        xb = np.arange(l1, t + 1, dtype=float)
        line_bump = slope * xb + intercept
        bump_above = p[l1:t + 1] - line_bump
        bump_height = float(bump_above.max())
        if bump_height < bump_mult * lead_height:
            in_break = False
            continue
        # the bump must be RECENT — its peak (max above the line) sits in the second half of
        # the bump window, i.e. price has just rolled over. This stops stale geometry from a
        # long-past bump re-firing a "break" once price has gone quiet far below the projected
        # line. (Pure no-look-ahead: argmax is within the trailing bump window.)
        if int(np.argmax(bump_above)) < (bump_above.size // 2):
            in_break = False
            continue
        # break: the close *downcrosses* the extended lead-in line at bar t — i.e. the prior
        # close was at/above the line and this close is below it. A genuine rollover, fired once
        # (the latch blocks repeats until price climbs back above the line).
        line_t = slope * t + intercept
        line_tm1 = slope * (t - 1) + intercept
        below = p[t] < line_t
        above_prev = p[t - 1] >= line_tm1
        if below and above_prev and not in_break:
            fire[t] = True
            in_break = True
        elif not below:
            in_break = False
    return close.index[fire]


def random_entries(close: pd.Series, n: int, lead: int = 60, bump: int = 30,
                   seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched short-side baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[lead + bump:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine (SHORT side — the BARR rule is a short)
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0,
                    side: int = -1) -> np.ndarray:
    """Forward ``horizon``-day return of a ``side`` position, entered at the *next* close (one lag).

    ``side = -1`` (default) is a **short** — the bump-and-run break rule — so the trade profits
    when price falls. ``cost_bps`` is a one-way cost charged twice (in + out). Trades whose
    window overruns the tape are dropped.
    """
    pos = {d: i for i, d in enumerate(close.index)}
    p = close.to_numpy(dtype=float)
    n = p.size
    out = []
    for d in entries:
        i = pos.get(d)
        if i is None or i + 1 + horizon >= n:
            continue
        e = i + 1                      # enter at next close
        raw = p[e + horizon] / p[e] - 1.0
        out.append(side * raw - 2.0 * cost_bps * 1e-4)
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_t(x: np.ndarray) -> float:
    """Newey-West (HAC) one-sample t-stat of the mean against zero."""
    x = np.asarray(x, dtype=float)
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


def summarize(returns: np.ndarray) -> dict:
    """Headline per-trade stats: count, win-rate, mean (bps), per-trade Sharpe, HAC t."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    return {
        "n": int(n),
        "win": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "sharpe": float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan"),
        "t": hac_t(r),
    }


def shuffled_window_placebo(close: pd.Series, horizon: int, lead: int = 60, bump: int = 30,
                            bump_mult: float = 2.0, n_draws: int = 1000, seed: int = 467) -> dict:
    """Placebo: scramble the lead-in/bump *shape* while keeping the price marginal.

    We rebuild the entries on a tape whose **per-bar log-returns have been permuted** (a fixed
    permutation per draw): the price level still wanders over the same marginal return
    distribution, but the *ordering* — and therefore every lead-in trendline + bump geometry —
    is destroyed. If the real BARR short result survives the scramble, the bump-and-run shape
    was never load-bearing. Returns the share of placebo runs whose mean BARR short forward
    return **beats** the real one (the honest geometry p-value), plus the observed mean.
    """
    obs = float(np.mean(forward_returns(
        close, barr_break_entries(close, lead=lead, bump=bump, bump_mult=bump_mult), horizon)))
    rng = np.random.default_rng(seed)
    logret = np.diff(np.log(close.to_numpy(dtype=float)))
    p0 = float(close.iloc[0])
    idx = close.index
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = rng.permutation(logret)
        scr = np.empty(len(idx))
        scr[0] = p0
        scr[1:] = p0 * np.exp(np.cumsum(perm))
        s = pd.Series(scr, index=idx)
        ent = barr_break_entries(s, lead=lead, bump=bump, bump_mult=bump_mult)
        rr = forward_returns(s, ent, horizon)
        if rr.size == 0:
            continue
        valid += 1
        if rr.mean() >= obs:
            beats += 1
    p = (beats + 1) / (valid + 1) if valid else float("nan")
    return {"obs": obs, "p_value": float(p), "n_draws": valid}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(close: pd.Series, lead: int = 60, bump: int = 30, bump_mult: float = 2.0,
                   cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: BARR-break short vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the BARR-short summary (gross + net), the
    drift-matched random-entry baseline (same short side), and the break-minus-random delta.
    """
    ent = barr_break_entries(close, lead=lead, bump=bump, bump_mult=bump_mult)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), lead=lead, bump=bump,
                                   seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
