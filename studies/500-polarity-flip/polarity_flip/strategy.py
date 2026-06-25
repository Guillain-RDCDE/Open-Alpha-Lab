"""Polarity-Flip (role reversal) as a falsifiable mechanical rule — Study 500.

The chartists' **polarity principle** (a.k.a. *role reversal*, "support-becomes-resistance and
vice-versa"): once price decisively **breaks above** a prior swing-high resistance level, that
old ceiling is supposed to *flip role* and act as a **floor** — so the first pullback down to a
freshly broken resistance is a high-probability **buy** (it "should" hold as support and bounce).

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Resistance levels are confirmed fractals** — a swing high is a local maximum with ``k``
   strictly-lower bars on each side. A pivot at bar ``t`` is only *usable* from bar ``t+k``
   onward (the confirmation lag), so a level is never known with future bars.
2. **Break above** — the level becomes "broken resistance" the first time the close prints a
   clear margin (``break_buf``) above the confirmed swing-high price.
3. **First pullback retest** — a long fires when, after the break, the close pulls back *down
   into a band* (``band``) around the broken level for the **first time** (role reversal: the
   old ceiling tested from above as a floor). Entry is at the **next** close (one documented
   lag); we then measure the forward H-day return.
4. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold) that
   captures the tape's drift, and (b) a **scrambled-level placebo** that fires retests against
   a permutation of the level *prices*, destroying the "this specific broken level" geometry
   while keeping the marginal distribution — the honest "is the level doing anything?" null.

No look-ahead: levels carry a ``k``-bar confirmation lag, the break and the retest are read on
the close of *t*, the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Pivot detection (confirmed swing highs)
# --------------------------------------------------------------------------- #
def find_swing_highs(close: pd.Series, k: int = 10) -> pd.DataFrame:
    """Confirmed swing highs: local maxima with ``k`` strictly-lower bars on each side.

    Returns a DataFrame indexed by the pivot bar position with a ``price`` column. A pivot at
    position ``i`` is only *confirmed* (knowable) at bar ``i + k`` — the callers enforce that
    lag, so no future data leaks in.
    """
    p = close.to_numpy(dtype=float)
    n = p.size
    rows = []
    for i in range(k, n - k):
        win = p[i - k:i + k + 1]
        c = p[i]
        if c == win.max() and (win[:k] < c).all() and (win[k + 1:] < c).all():
            rows.append((i, c))
    if not rows:
        return pd.DataFrame(columns=["pos", "price"]).set_index("pos")
    return pd.DataFrame(rows, columns=["pos", "price"]).set_index("pos")


# --------------------------------------------------------------------------- #
# Entries — the polarity-flip (role-reversal) retest
# --------------------------------------------------------------------------- #
def polarity_entries(
    close: pd.Series,
    k: int = 10,
    band: float = 0.01,
    break_buf: float = 0.005,
    level_max_age: int = 250,
) -> pd.DatetimeIndex:
    """Bars that are the **first pullback retest** of a freshly broken resistance level.

    Mechanics, all at the close of bar ``t`` and using only information confirmed by ``t``:

    1. A confirmed swing high (fractal, lag ``k``) is a resistance level.
    2. The level becomes *broken* the first bar whose close exceeds ``level*(1+break_buf)``.
    3. After the break, the **first** bar whose close re-enters the band ``[level*(1-band),
       level*(1+band)]`` from above is the polarity-flip retest — a long entry. The level then
       retires (each broken level fires at most once); a level older than ``level_max_age`` bars
       since its break is dropped.

    Entry is executed at the next close by :func:`forward_returns`.
    """
    piv = find_swing_highs(close, k=k)
    idx = close.index
    cl = close.to_numpy(dtype=float)
    n = cl.size
    if piv.empty:
        return pd.DatetimeIndex([])

    # each level usable from confirm bar = pivot_pos + k
    levels = [(int(pos) + k, float(pr)) for pos, pr in zip(piv.index, piv["price"])]
    levels.sort()

    # state per active level: 'pending' (waiting to break) or 'broken@t' (waiting to retest)
    active = []   # list of dicts: {level, born, broke_at or None}
    li = 0
    entries = []
    for t in range(n):
        c = cl[t]
        # admit newly-confirmed levels
        while li < len(levels) and levels[li][0] <= t:
            active.append({"level": levels[li][1], "born": t, "broke": None})
            li += 1
        new_active = []
        for st_ in active:
            lvl = st_["level"]
            if st_["broke"] is None:
                # waiting for a break above
                if c > lvl * (1.0 + break_buf):
                    st_["broke"] = t
                new_active.append(st_)
            else:
                if t - st_["broke"] > level_max_age:
                    continue  # retire stale broken level
                # first pullback into the band from above => polarity-flip retest
                if lvl * (1.0 - band) <= c <= lvl * (1.0 + band):
                    entries.append(t)
                    # level fires once, then retires
                else:
                    new_active.append(st_)
        active = new_active

    if not entries:
        return pd.DatetimeIndex([])
    # collapse to first bar of each consecutive run (defensive; entries are already discrete)
    pos = sorted(set(entries))
    return pd.DatetimeIndex([idx[i] for i in pos])


def random_entries(close: pd.Series, n: int, k: int = 10, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[2 * k:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    ``cost_bps`` is a one-way cost (charged twice: in + out) subtracted from each trade's
    return. Trades whose window overruns the tape are dropped.
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


def scrambled_level_placebo(close: pd.Series, horizon: int, k: int = 10, band: float = 0.01,
                            break_buf: float = 0.005, n_draws: int = 1000,
                            seed: int = 500) -> dict:
    """Placebo: fire retests against a permutation of the resistance-level *prices*.

    Keeps the swing-high *positions* and the marginal distribution of level prices, but permutes
    which level-price sits at which pivot, so "this specific broken resistance" becomes a
    meaningless number. Returns the share of placebo runs whose mean polarity-retest forward
    return **beats** the real one — the honest "is the broken level load-bearing?" p-value,
    plus the observed mean.
    """
    obs = float(np.mean(forward_returns(close, polarity_entries(close, k=k, band=band,
                                                                break_buf=break_buf), horizon)))
    piv = find_swing_highs(close, k=k)
    if len(piv) < 3:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    prices = piv["price"].to_numpy(dtype=float)
    positions = [int(p) for p in piv.index]
    idx = close.index
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = rng.permutation(prices)
        ent = _entries_from_levels(close, list(zip(positions, perm)), k, band, break_buf)
        rr = forward_returns(close, ent, horizon)
        if rr.size == 0:
            continue
        valid += 1
        if rr.mean() >= obs:
            beats += 1
    p = (beats + 1) / (valid + 1) if valid else float("nan")
    return {"obs": obs, "p_value": float(p), "n_draws": valid}


def _entries_from_levels(close, level_pairs, k, band, break_buf, level_max_age=250):
    """Internal: run the polarity-retest entry logic over an arbitrary set of (pos, price) levels."""
    idx = close.index
    cl = close.to_numpy(dtype=float)
    n = cl.size
    levels = sorted([(int(pos) + k, float(pr)) for pos, pr in level_pairs])
    active = []
    li = 0
    entries = []
    for t in range(n):
        c = cl[t]
        while li < len(levels) and levels[li][0] <= t:
            active.append({"level": levels[li][1], "born": t, "broke": None})
            li += 1
        new_active = []
        for st_ in active:
            lvl = st_["level"]
            if st_["broke"] is None:
                if c > lvl * (1.0 + break_buf):
                    st_["broke"] = t
                new_active.append(st_)
            else:
                if t - st_["broke"] > level_max_age:
                    continue
                if lvl * (1.0 - band) <= c <= lvl * (1.0 + band):
                    entries.append(t)
                else:
                    new_active.append(st_)
        active = new_active
    if not entries:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex([idx[i] for i in sorted(set(entries))])


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(close: pd.Series, k: int = 10, band: float = 0.01, break_buf: float = 0.005,
                   cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: polarity-retest vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the polarity-retest summary (gross + net), the
    drift-matched random-entry baseline, and the retest-minus-random delta.
    """
    ent = polarity_entries(close, k=k, band=band, break_buf=break_buf)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), k=k, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
