"""Up-Down-Volume breadth as a falsifiable mechanical rule — Study 492.

The folklore (the *up/down volume ratio*, the volume side of the Arms index / TRIN, the
"selling climax" and "buying climax" of classic tape-reading): when *down-volume utterly
dominates up-volume* across the market, panic has exhausted itself — a **selling climax** — and
the index bounces. Symmetrically a **buying climax** (up-volume swamping down-volume, a blow-off
top) precedes weakness. The up/down-volume ratio is supposed to *forecast* the next move.

We build the breadth signal mechanically from a basket of liquid sector ETFs (a proxy for true
exchange up/down volume) and test it honestly:

1. **Daily up/down volume.** On each session, every basket member is "up" (close > prior close)
   or "down". ``up_vol`` = total volume of the up members, ``down_vol`` = total volume of the
   down members. The indicator is the **up-volume share** ``uvs = up_vol / (up_vol + down_vol)``
   in [0,1] (the bounded form of the up/down-volume ratio).
2. **Climax entries.** A long fires on a **selling climax**: a day whose ``uvs`` sits at or
   below its rolling lower quantile (down-volume dominating) — the "buy the panic" rule. Read on
   the close of *t*, entered at the **next** close (one documented lag); we then measure the
   forward H-day return on SPY.
3. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures the
   tape's drift; and (b) a **shuffled-volume placebo** that randomly re-pairs each day's up/down
   *volume magnitudes* with a different day's up/down *directions*, destroying the breadth
   geometry while preserving both marginal distributions — the honest "is the up/down structure
   load-bearing?" null.

No look-ahead: the rolling quantile uses only past bars, the climax is read on the close of *t*,
the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Breadth indicator: daily up/down volume across the basket
# --------------------------------------------------------------------------- #
def up_down_volume(basket: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate daily up-volume and down-volume across a basket of OHLCV frames.

    For each common session, a member contributes its volume to ``up_vol`` if it closed up vs
    its own prior close, else to ``down_vol`` (an unchanged close counts as down — the
    conservative, climax-friendly convention). Returns a frame indexed by date with columns
    ``up_vol``, ``down_vol``, and ``uvs`` = the **up-volume share** ``up_vol/(up_vol+down_vol)``
    centred and put on a comparable scale. We use the bounded share (in [0,1]) rather than a raw
    log-ratio because a small basket frequently prints an all-up or all-down day (``down_vol=0``),
    which would blow a log-ratio up to ±∞; the share degrades gracefully (0 or 1). A *selling
    climax* is a session whose up-volume share is in its rolling lower tail (down-volume
    dominating).
    """
    closes = pd.DataFrame({t: b["close"] for t, b in basket.items()}).sort_index()
    vols = pd.DataFrame({t: b["volume"] for t, b in basket.items()}).reindex(closes.index)
    up_mask = closes > closes.shift(1)                 # member up vs its own prior close
    up_vol = (vols.where(up_mask, 0.0)).sum(axis=1)
    down_vol = (vols.where(~up_mask, 0.0)).sum(axis=1)
    uvs = up_vol / (up_vol + down_vol + 1.0)         # up-volume share in [0,1]
    out = pd.DataFrame({"up_vol": up_vol, "down_vol": down_vol, "uvs": uvs})
    return out.iloc[1:]                                # first row has no prior close


def breadth_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Up/down-volume frame from a :func:`data.synthetic_panel` (already carries up/down vol).

    Uses the **same** up-volume-share indicator as the real basket (:func:`up_down_volume`) so
    the synthetic control exercises the identical detector geometry.
    """
    up_vol = panel["up_vol"].astype(float)
    down_vol = panel["down_vol"].astype(float)
    uvs = up_vol / (up_vol + down_vol + 1.0)            # up-volume share in [0,1]
    return pd.DataFrame({"up_vol": up_vol, "down_vol": down_vol, "uvs": uvs})


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def climax_entries(uvs: pd.Series, window: int = 60, q: float = 0.10,
                   side: str = "selling") -> pd.DatetimeIndex:
    """Dates of a volume **climax**: ``uvs`` at/below (selling) or at/above (buying) a rolling
    quantile of its own *past* ``window`` values.

    Only the *first* bar of each consecutive run is kept (the climax day, not every day the ratio
    stays extreme). The rolling quantile is computed on a window that **excludes** the current bar
    (``shift(1)``) so the threshold uses only past data — no look-ahead. Entry is executed at the
    next close by :func:`forward_returns`.
    """
    uvs = uvs.dropna()
    if side == "selling":
        thresh = uvs.shift(1).rolling(window, min_periods=window).quantile(q)
        mask = (uvs <= thresh) & thresh.notna()
    else:  # buying climax
        thresh = uvs.shift(1).rolling(window, min_periods=window).quantile(1.0 - q)
        mask = (uvs >= thresh) & thresh.notna()
    first = mask & ~mask.shift(1, fill_value=False)
    return uvs.index[first.to_numpy()]


def random_entries(close: pd.Series, n: int, warmup: int = 60, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[warmup:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    ``cost_bps`` is a one-way cost (charged twice: in + out) subtracted from each trade's return.
    Trades whose window overruns the tape are dropped.
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


def shuffled_volume_placebo(close: pd.Series, breadth: pd.DataFrame, horizon: int,
                            window: int = 60, q: float = 0.10, side: str = "selling",
                            n_draws: int = 500, seed: int = 492) -> dict:
    """Placebo: scramble the breadth **geometry** by permuting the ``uvs`` series in time.

    Keeps the exact marginal distribution of the up/down-volume ratio (it is the same multiset of
    values) but destroys its alignment with the index path — so a "climax" now falls on an
    unrelated day. If the real climax timing carried information, the scramble should collapse the
    result. Returns the share of placebo runs whose mean climax forward return **beats** the real
    one — the honest "is the up/down structure load-bearing?" p-value, plus the observed mean.
    """
    real_ent = climax_entries(breadth["uvs"], window=window, q=q, side=side)
    obs = float(np.mean(forward_returns(close, real_ent, horizon))) if len(real_ent) else float("nan")
    vals = breadth["uvs"].dropna()
    idx = vals.index
    rng = np.random.default_rng(seed)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = pd.Series(rng.permutation(vals.to_numpy()), index=idx)
        ent = climax_entries(perm, window=window, q=q, side=side)
        rr = forward_returns(close, ent, horizon)
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
def run_experiment(close: pd.Series, breadth: pd.DataFrame, window: int = 60, q: float = 0.10,
                   side: str = "selling", cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: climax entry vs random-entry baseline, all horizons.

    ``close`` is the forward-return instrument (SPY); ``breadth`` is the up/down-volume frame
    (its index need not match ``close`` exactly — entries are intersected with ``close`` inside
    :func:`forward_returns`). Returns a dict keyed by horizon with the climax summary (gross +
    net), the drift-matched random baseline, and the climax-minus-random delta.
    """
    ent = climax_entries(breadth["uvs"], window=window, q=q, side=side)
    # keep only entries that exist on the forward-return tape
    ent = ent[ent.isin(close.index)]
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), warmup=window, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
