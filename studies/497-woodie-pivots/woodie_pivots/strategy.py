"""Woodie's Pivot Points as a falsifiable mechanical rule — Study 497.

Woodie's pivots are a close-weighted variant of the floor-trader pivot. Where the classic
pivot is P = (H + L + C) / 3, **Woodie double-weights the close**:

    P  = (H + L + 2C) / 4
    R1 = 2P - L          S1 = 2P - H
    R2 = P + (H - L)     S2 = P - (H - L)

where (H, L, C) are **yesterday's** high, low, close — the levels are fixed for the whole of
today (one documented lag, no look-ahead). The folklore (Woodie's CCI Club, every pivot-point
write-up): *yesterday's pivots act as intraday support/resistance* — price reaching down to
**S1** should find support and bounce.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Levels from the prior bar.** Today's Woodie pivot and S1/S2/R1/R2 are computed from
   *yesterday's* (H, L, C). They are knowable at today's open — no future data leaks in.
2. **S1 support touch.** A long entry fires when today's **low pierces (or undercuts) the
   prior-day S1** — the "support holds" setup. Entry is at the **next** close (one documented
   lag); we then measure the forward H-day return.
3. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold) that
   captures the tape's drift, and (b) a **random-level placebo** that replaces the real Woodie
   S1 with a level placed a *random* (re-sampled) distance below the prior close — destroying
   the close-weighting geometry while keeping the touch frequency and the price marginal — the
   honest "is the *specific* Woodie S1 doing anything?" null.

No look-ahead: Woodie's levels carry a one-bar lag, the touch is read on the bar of *t*, the
position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Woodie pivot geometry
# --------------------------------------------------------------------------- #
def woodie_levels(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-day Woodie pivot levels (P, R1, S1, R2, S2) from the **prior** day's H/L/C.

    P = (H + L + 2C)/4 (close double-weighted). Levels for day *t* use the bar at *t-1*, so they
    are knowable at *t*'s open — a one-bar lag, no look-ahead. The first row is NaN.
    """
    h = bars["high"].shift(1)
    l = bars["low"].shift(1)
    c = bars["close"].shift(1)
    rng = h - l
    P = (h + l + 2.0 * c) / 4.0
    out = pd.DataFrame(index=bars.index)
    out["P"] = P
    out["R1"] = 2.0 * P - l
    out["S1"] = 2.0 * P - h
    out["R2"] = P + rng
    out["S2"] = P - rng
    return out


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def s1_touch_entries(bars: pd.DataFrame) -> pd.DatetimeIndex:
    """Bars whose **low** reaches down to (or below) the prior-day Woodie **S1** — the
    'support holds, buy the bounce' rule.

    Only the *first* bar of each consecutive run is kept (the touch, not every day price keeps
    poking S1). Entry is executed at the next close by :func:`forward_returns`.
    """
    lev = woodie_levels(bars)
    s1 = lev["S1"]
    mask = (bars["low"] <= s1) & s1.notna()
    first = mask & ~mask.shift(1, fill_value=False)
    return bars.index[first.to_numpy()]


def random_entries(bars: pd.DataFrame, n: int, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after a short warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = bars.index[2:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(bars: pd.DataFrame, entries, horizon: int, cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    ``cost_bps`` is a one-way cost (charged twice: in + out) subtracted from each trade's
    return. Trades whose window overruns the tape are dropped.
    """
    close = bars["close"]
    pos = {d: i for i, d in enumerate(close.index)}
    p = close.to_numpy(dtype=float)
    n = p.size
    out = []
    for d in entries:
        i = pos.get(d)
        if i is None or i + 1 + horizon >= n:
            continue
        e = i + 1                      # enter at next close
        r = p[e + horizon] / p[e] - 1.0
        out.append(r - 2.0 * cost_bps * 1e-4)
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


def random_level_placebo(bars: pd.DataFrame, horizon: int,
                         n_draws: int = 1000, seed: int = 497) -> dict:
    """Placebo: replace the real Woodie S1 with a *random* support level, geometry destroyed.

    For each draw we resample (with replacement) the per-day distances ``(prior_close - S1)``
    and lay them under the prior close in scrambled order, producing a fake support line with
    the **same marginal depth distribution and touch frequency** but no close-weighting
    geometry. Returns the share of placebo runs whose mean S1-touch forward return **beats** the
    real one — the honest "is the specific Woodie S1 adding anything?" p-value, plus the
    observed mean.
    """
    real_ent = s1_touch_entries(bars)
    obs = float(np.mean(forward_returns(bars, real_ent, horizon))) if len(real_ent) else float("nan")
    lev = woodie_levels(bars)
    prev_close = bars["close"].shift(1)
    depth = (prev_close - lev["S1"])           # distance from prior close down to S1
    valid_mask = depth.notna()
    depths = depth[valid_mask].to_numpy(dtype=float)
    if depths.size < 10 or not np.isfinite(obs):
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    low = bars["low"]
    pc = prev_close
    idx = bars.index
    beats = 0
    valid = 0
    for _ in range(n_draws):
        fake_depth = pd.Series(np.nan, index=idx)
        fake_depth[valid_mask] = rng.choice(depths, size=depths.size, replace=True)
        fake_s1 = pc - fake_depth
        mask = (low <= fake_s1) & fake_s1.notna()
        first = mask & ~mask.shift(1, fill_value=False)
        ent = idx[first.to_numpy()]
        rr = forward_returns(bars, ent, horizon)
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
def run_experiment(bars: pd.DataFrame, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: S1-touch vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the S1-touch summary (gross + net), the drift-matched
    random-entry baseline, and the touch-minus-random delta.
    """
    ent = s1_touch_entries(bars)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(bars, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(bars, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            bars, random_entries(bars, max(len(ent), 50), seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
