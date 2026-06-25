"""The Arms Index (TRIN) as a falsifiable mechanical rule — Study 490.

Richard W. Arms Jr.'s *Trading Index* (1967), a.k.a. **TRIN** or the **Arms Index**:

    TRIN = (advancing issues / declining issues) / (advancing volume / declining volume)

A **high** TRIN (>~1.5–2) is read as a panic/washout — disproportionate volume crowding into the
few decliners — and is supposed to mark a short-term **bottom** that *bounces*; a **low** TRIN
(<~0.5–0.7) marks euphoria/froth. The folklore: fade the panic — *buy when TRIN spikes high*.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Breadth proxy.** True exchange breadth is unavailable offline, so we treat each basket ETF
   as one "issue": it *advances* if its daily return > 0, else *declines*; the *move magnitude*
   |return| stands in for that issue's volume. Then on each day

       TRIN_t = (A_t / D_t) / (Vu_t / Vd_t),

   where A,D are advancing/declining counts and Vu,Vd are the summed |returns| of advancers /
   decliners. (Equal counts and magnitudes -> TRIN = 1.)

2. **Panic entry.** A long fires when TRIN_t exceeds a high threshold (default the 90th
   percentile of the in-sample TRIN, the "panic" tail) — read on the close of *t*, entered at the
   **next** close (one documented lag). We measure the forward H-day return on the traded
   instrument (SPY).

3. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures the
   tape's drift, and (b) a **shuffled-TRIN placebo** that permutes the TRIN series in time,
   destroying its alignment with returns while keeping its marginal — the honest "is the *timing*
   of high-TRIN days doing anything?" null.

No look-ahead: TRIN is read on the close of *t*, the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Breadth proxy -> TRIN
# --------------------------------------------------------------------------- #
def compute_trin(panel: dict[str, pd.DataFrame]) -> pd.Series:
    """Daily breadth-proxy TRIN from a basket of OHLC frames (one "issue" per name).

    Each name advances if its daily close-to-close return > 0; |return| is its volume proxy.
    Returns a TRIN series on the common trading calendar.

    Regularisation. With only a handful of "issues" the raw ratio can blow up on tiny-move days
    (a near-zero declining-side volume sends TRIN to infinity even though *nothing happened*). We
    therefore floor each side's volume at a small fraction of the typical daily basket move
    (``vfloor``) and add a 1-issue Laplace prior to the advance/decline counts. This keeps high
    TRIN meaning *real* selling pressure rather than a quiet up-day artifact. (Equal counts and
    magnitudes -> TRIN = 1.) This regularisation is part of the documented breadth-proxy cap.
    """
    rets = {}
    for nm, bars in panel.items():
        rets[nm] = bars["close"].pct_change()
    R = pd.DataFrame(rets).dropna(how="all")
    R = R.dropna()  # common calendar (drop days where any member is missing)
    # typical per-issue daily move sets the volume floor (so tiny-move days do not explode)
    vfloor = float(R.abs().to_numpy().mean()) * 0.25
    adv = (R > 0)
    dec = (R < 0)
    A = adv.sum(axis=1).astype(float) + 1.0          # Laplace prior on counts
    D = dec.sum(axis=1).astype(float) + 1.0
    Vu = (R.where(adv, 0.0).abs()).sum(axis=1) + vfloor
    Vd = (R.where(dec, 0.0).abs()).sum(axis=1) + vfloor
    ad_ratio = A / D
    vol_ratio = Vu / Vd
    trin = ad_ratio / vol_ratio
    return trin.replace([np.inf, -np.inf], np.nan).dropna()


def panic_entries(trin: pd.Series, q: float = 0.90) -> pd.DatetimeIndex:
    """Bars whose TRIN exceeds the ``q`` quantile of the in-sample TRIN — the 'panic' tail.

    Only the *first* bar of each consecutive high-TRIN run is kept (the spike, not every day TRIN
    stays elevated). Entry is executed at the next close by :func:`forward_returns`.
    """
    thr = float(trin.quantile(q))
    mask = trin >= thr
    first = mask & ~mask.shift(1, fill_value=False)
    return trin.index[first.to_numpy()]


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


def shuffled_trin_placebo(trin: pd.Series, close: pd.Series, horizon: int, q: float = 0.90,
                          n_draws: int = 1000, seed: int = 490) -> dict:
    """Placebo: permute the TRIN series in time, destroying its alignment with future returns.

    Keeps the TRIN marginal (and thus the same number of 'panic' days at the same threshold) but
    scrambles *when* they fall, so any forecasting content from the timing is removed. Returns the
    share of placebo runs whose mean panic-day forward return **beats** the real one — the honest
    "is the timing of high-TRIN days load-bearing?" p-value, plus the observed mean.
    """
    obs = float(np.mean(forward_returns(close, panic_entries(trin, q=q), horizon)))
    rng = np.random.default_rng(seed)
    vals = trin.to_numpy(dtype=float)
    idx = trin.index
    thr = float(np.quantile(vals, q))
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = rng.permutation(vals)
        ser = pd.Series(perm, index=idx)
        mask = ser >= thr
        first = mask & ~mask.shift(1, fill_value=False)
        ent = idx[first.to_numpy()]
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
def run_experiment(panel: dict[str, pd.DataFrame], traded: str = "SPY", q: float = 0.90,
                   cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet: high-TRIN panic entry vs random-entry baseline, all horizons.

    ``panel`` is the breadth basket; ``traded`` is the instrument we buy (must be in the panel).
    Returns a dict keyed by horizon with the panic-entry summary (gross + net), the drift-matched
    random-entry baseline, and the panic-minus-random delta.
    """
    trin = compute_trin(panel)
    close = panel[traded]["close"]
    close = close[close.index.isin(trin.index)]
    trin = trin[trin.index.isin(close.index)]
    ent = panic_entries(trin, q=q)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
