"""The Hikkake pattern as a falsifiable mechanical rule — Study 459.

The **hikkake** (Japanese for "trap" / "to ensnare"), introduced by Daniel L. Chesler, is a
two-stage candlestick false-breakout pattern:

1. an **inside bar** — a bar whose entire range sits *inside* the prior bar's range
   (``high < prior_high`` and ``low > prior_low``);
2. a **false breakout** out of the inside-bar range one way (price pokes above the inside high,
   or below the inside low) within the next few bars;
3. a **snap-back** — price closes back *through* the inside-bar range in the **opposite**
   direction, confirming the breakout was a trap.

The folklore: the trap forecasts a move in the **reversal** direction. A false *upside* break
that fails is a **short**; a false *downside* break that fails is a **long**. We encode the
tightest mechanical version a proponent would accept and test it honestly:

1. **Inside bar.** Objective: ``high_i < high_{i-1}`` and ``low_i > low_{i-1}``.
2. **Confirmation window.** The false break must occur within ``window`` bars of the inside
   bar; the *snap-back close* completes the pattern and is read on its own close (bar *t*).
   No look-ahead: the entire trap is closed by *t*, we enter at the close of *t+1* (one lag).
3. **Direction.** Trade the reversal: long after a failed downside break, short after a failed
   upside break. Forward returns are signed by the trade direction.
4. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold, and the same
   long/short *mix* as the real signal) that captures the tape's drift, and (b) a **scrambled-
   direction placebo** that keeps every hikkake date but randomly flips the trade direction,
   destroying the trap's defining geometry (which way it forecasts) while keeping the entry
   marginal — the honest "is the trap's *direction* load-bearing?" null.

No look-ahead: the inside bar and the false break are confirmed on the close of *t*, the
position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Hikkake detection
# --------------------------------------------------------------------------- #
def hikkake_signals(bars: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Detect completed hikkake traps. Returns a frame indexed by the *confirmation* bar.

    Columns: ``dir`` (+1 long after a failed downside break, -1 short after a failed upside
    break). The pattern is: an **inside bar** at ``i`` (``high_i < high_{i-1}`` and
    ``low_i > low_{i-1}``); within ``window`` bars a **false breakout** beyond the inside bar's
    high or low; then a bar whose **close** snaps back *through* the inside-bar range in the
    opposite direction (the trap confirmed). The confirmation bar's close completes the
    pattern — everything is known by that bar, so :func:`forward_returns` (enter at *t+1*)
    carries no look-ahead.
    """
    o = bars["open"].to_numpy(dtype=float)  # noqa: F841 (kept for parity / future use)
    h = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    idx = bars.index
    n = len(idx)
    rows = []
    last_confirm = -1  # don't let overlapping windows double-fire on the same confirm bar

    for i in range(1, n):
        # is bar i an inside bar relative to i-1?
        if not (h[i] < h[i - 1] and lo[i] > lo[i - 1]):
            continue
        ih, il = h[i], lo[i]
        broke_up = False
        broke_dn = False
        # scan the confirmation window for false break + snap-back
        for j in range(i + 1, min(i + 1 + window, n)):
            if not broke_up and not broke_dn:
                # first leg: a breakout beyond the inside range
                if h[j] > ih:
                    broke_up = True
                    continue
                if lo[j] < il:
                    broke_dn = True
                    continue
            else:
                # second leg: a close back through the inside range, opposite the break
                if broke_up and c[j] < ih:
                    if j > last_confirm:
                        rows.append((idx[j], -1))   # failed upside break -> short
                        last_confirm = j
                    break
                if broke_dn and c[j] > il:
                    if j > last_confirm:
                        rows.append((idx[j], +1))   # failed downside break -> long
                        last_confirm = j
                    break
    if not rows:
        return pd.DataFrame({"dir": []}, index=pd.DatetimeIndex([], name="date"))
    dates = pd.DatetimeIndex([r[0] for r in rows], name="date")
    return pd.DataFrame({"dir": [r[1] for r in rows]}, index=dates)


def hikkake_entries(bars: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Convenience wrapper: the dated, signed hikkake entries (alias of :func:`hikkake_signals`)."""
    return hikkake_signals(bars, window=window)


def random_entries(bars: pd.DataFrame, dirs: np.ndarray, window: int = 3,
                   seed: int = 0) -> pd.DataFrame:
    """Random entry dates carrying the **same long/short mix** as the real signal.

    ``dirs`` is the array of real signal directions; we draw ``len(dirs)`` random dates (after
    the warm-up) and assign them the same multiset of directions, so the baseline matches the
    real signal's net exposure — the drift-matched, exposure-matched control.
    """
    rng = np.random.default_rng(seed)
    idx = bars.index
    valid = idx[2 * window + 2:]
    n = len(dirs)
    if len(valid) == 0 or n == 0:
        return pd.DataFrame({"dir": []}, index=pd.DatetimeIndex([], name="date"))
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    chosen = pd.DatetimeIndex(sorted(chosen), name="date")
    d = np.asarray(dirs, dtype=float).copy()
    rng.shuffle(d)
    d = d[: len(chosen)]
    return pd.DataFrame({"dir": d.astype(int)}, index=chosen)


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(bars: pd.DataFrame, sig: pd.DataFrame, horizon: int,
                    cost_bps: float = 0.0) -> np.ndarray:
    """Direction-signed forward ``horizon``-day return per entry, entered at the *next* close.

    ``sig`` is a frame indexed by confirmation date with a ``dir`` column (+1 long / -1 short).
    Entry is the close of *t+1* (one documented lag). Return is signed by ``dir``. ``cost_bps``
    is a one-way cost charged twice (in + out). Trades whose window overruns the tape are dropped.
    """
    close = bars["close"]
    pos = {d: i for i, d in enumerate(close.index)}
    p = close.to_numpy(dtype=float)
    n = p.size
    out = []
    for d, row in sig.iterrows():
        i = pos.get(d)
        if i is None or i + 1 + horizon >= n:
            continue
        e = i + 1
        raw = p[e + horizon] / p[e] - 1.0
        out.append(row["dir"] * raw - 2.0 * cost_bps * 1e-4)
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


def scrambled_direction_placebo(bars: pd.DataFrame, horizon: int, window: int = 3,
                                n_draws: int = 1000, seed: int = 459) -> dict:
    """Placebo: keep every hikkake *date* but randomly flip the trade *direction*.

    The hikkake's whole content is *which way* the trap forecasts. We keep the entry marginal
    (same dates, same number of trades) but scramble the directions, destroying that geometry.
    Returns the share of placebo runs whose mean signed forward return **beats** the real one —
    the honest "is the trap's direction load-bearing?" p-value, plus the observed mean.
    """
    sig = hikkake_signals(bars, window=window)
    obs = float(np.mean(forward_returns(bars, sig, horizon)))
    if len(sig) < 6 or not np.isfinite(obs):
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    dirs = sig["dir"].to_numpy(dtype=float)
    beats = 0
    for _ in range(n_draws):
        flip = rng.choice([-1.0, 1.0], size=len(dirs))
        scr = sig.copy()
        scr["dir"] = (dirs * flip).astype(int)
        rr = forward_returns(bars, scr, horizon)
        if rr.size and rr.mean() >= obs:
            beats += 1
    p = (beats + 1) / (n_draws + 1)
    return {"obs": obs, "p_value": float(p), "n_draws": n_draws}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(bars: pd.DataFrame, window: int = 3, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: hikkake vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the hikkake summary (gross + net), the
    drift/exposure-matched random-entry baseline, and the hikkake-minus-random delta.
    """
    sig = hikkake_signals(bars, window=window)
    dirs = sig["dir"].to_numpy()
    res = {"n_entries": int(len(sig)), "n_long": int((dirs > 0).sum()),
           "n_short": int((dirs < 0).sum()), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(bars, sig, h, cost_bps=0.0))
        net = summarize(forward_returns(bars, sig, h, cost_bps=cost_bps))
        rsig = random_entries(bars, dirs, window=window, seed=random_seed)
        rnd = summarize(forward_returns(bars, rsig, h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
