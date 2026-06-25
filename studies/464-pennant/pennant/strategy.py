"""The Pennant as a falsifiable mechanical rule — Study 464.

A **pennant** is one of the textbook *continuation* patterns. Its anatomy:

* **The pole** — a steep, near-vertical thrust: a strong directional move over a handful of
  bars (the "flagpole"). It sets the pattern's *direction*.
* **The pennant body** — a brief consolidation that *converges*: a small symmetrical triangle
  whose swing highs tilt down and swing lows tilt up, so the intraday range shrinks toward a
  point. Volume dries up (we test on price only — daily total-return tapes have no clean
  volume — and state that limitation).
* **The breakout** — price escapes the converging body **in the pole's direction**.

The folklore (Edwards & Magee, Bulkowski, every chart-pattern site): *the pennant continues
the prior thrust* — after the breakout, price runs roughly another pole-length in the same
direction ("the flag flies at half-mast"). We encode the tightest mechanical version and test
it honestly:

1. **Pole.** Over a lookback ``pole_len``, the cumulative log-move must exceed ``pole_k`` ×
   (rolling daily sigma × sqrt(pole_len)) — a genuinely steep thrust. Its sign is the
   *pole direction*.
2. **Converging body.** Over the following ``pause_len`` bars the high-low range must
   **contract** (recent range < earlier range by a factor) AND the net move must be small
   (a pause, not a second leg) — a mechanical stand-in for the symmetrical triangle.
3. **Breakout.** The close on bar *t* escapes the body's recent range **in the pole
   direction** (above the body high if the pole was up; below the body low if down). Entry is
   at the **next** close (one documented lag); we hold long if the pole was up, short if down,
   and measure the forward H-day **pole-direction** return.
4. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold,
   same long/short mix) that captures the tape's drift, and (b) a **direction-scramble
   placebo** that keeps the exact same breakout dates but randomizes the traded direction,
   destroying the "continuation" structure while keeping the marginal — the honest "does
   trading *in the pole direction* matter?" null.

No look-ahead: the pole and converging body are read on bars up to and including *t*, the
breakout is detected on the close of *t*, the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Pennant detection
# --------------------------------------------------------------------------- #
def detect_pennants(
    bars: pd.DataFrame,
    pole_len: int = 8,
    pause_len: int = 12,
    pole_k: float = 1.0,
    converge: float = 0.85,
    vol_win: int = 40,
) -> pd.DataFrame:
    """Detect mechanical pennant breakouts; return a frame of (date, dir) signal rows.

    For each bar *t* we look back over a window [pole, then pause] entirely in the past
    (closed on *t*):

    * **pole** — bars ``[t-pause_len-pole_len+1 .. t-pause_len]``: the cumulative log-return
      over the pole must exceed ``pole_k`` × (rolling daily sigma × sqrt(pole_len)). Sign = dir.
    * **converging body** — bars ``[t-pause_len+1 .. t]``: the high-low range of the *recent*
      half must be ``< converge`` × the range of the *earlier* half (contraction toward a
      point), and the body's net move must be small relative to the pole (a genuine pause).
    * **breakout** — the close on *t* escapes the body's high (dir +1) or low (dir -1).

    Returns a DataFrame indexed by the breakout date with column ``dir`` (+1 long / -1 short).
    Only the *first* bar of a consecutive breakout run is kept. No look-ahead: every input bar
    is at or before *t*; the trade is entered at *t+1* by :func:`forward_returns`.
    """
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    idx = bars.index
    n = close.size
    logc = np.log(close)
    ret = np.diff(logc, prepend=logc[0])
    # rolling daily sigma (causal)
    sig = pd.Series(ret).rolling(vol_win, min_periods=vol_win // 2).std().to_numpy()

    body = pause_len
    span = pole_len + pause_len
    sigs = []
    last_fire = -10
    # geometry: pole [t-pause-pole .. t-pause]; converging body [t-pause .. t-1]; breakout bar t.
    for t in range(span + vol_win, n):
        s = sig[t - 1]
        if not np.isfinite(s) or s <= 0:
            continue
        # --- pole over [t-pause-pole_len .. t-pause] (all strictly before the body) ---
        p_end = t - body
        p_beg = p_end - pole_len
        pole_move = logc[p_end] - logc[p_beg]
        pole_thresh = pole_k * s * np.sqrt(pole_len)
        if abs(pole_move) < pole_thresh:
            continue
        pdir = 1 if pole_move > 0 else -1

        # --- converging body over [b0 .. t-1] (the triangle, *excluding* the breakout bar t) ---
        b0 = t - body
        half = body // 2
        early_hi = high[b0:b0 + half].max()
        early_lo = low[b0:b0 + half].min()
        late_hi = high[t - half:t].max()
        late_lo = low[t - half:t].min()
        early_rng = early_hi - early_lo
        late_rng = late_hi - late_lo
        if early_rng <= 0:
            continue
        if late_rng >= converge * early_rng:
            continue  # not contracting -> not a pennant body
        # body must be a *pause*: its net move small vs the pole
        body_move = abs(logc[t - 1] - logc[b0])
        if body_move > 0.6 * abs(pole_move):
            continue

        # --- breakout in pole direction on close of t (escapes the body range) ---
        body_hi = high[b0:t].max()      # body range (bars b0..t-1), excluding the breakout bar
        body_lo = low[b0:t].min()
        broke = (close[t] > body_hi) if pdir > 0 else (close[t] < body_lo)
        if not broke:
            continue
        if t - last_fire < span:        # de-dup: one fire per pattern span
            continue
        last_fire = t
        sigs.append((idx[t], pdir))

    if not sigs:
        return pd.DataFrame(columns=["dir"], index=pd.DatetimeIndex([], name="date"))
    out = pd.DataFrame(sigs, columns=["date", "dir"]).set_index("date")
    return out


def pennant_entries(bars: pd.DataFrame, **kw) -> pd.DataFrame:
    """Convenience: the (date, dir) breakout signals — alias of :func:`detect_pennants`."""
    return detect_pennants(bars, **kw)


def random_entries(close: pd.Series, sig_dirs, seed: int = 0) -> pd.DataFrame:
    """Random entry dates with a *matched* long/short direction mix (drift-matched baseline).

    ``sig_dirs`` is the real signal frame (for count + the +1/-1 mix). Returns a (date, dir)
    frame of the same size, dates drawn uniformly after the warm-up, directions a permutation
    of the real ones — so the baseline inherits the tape's drift *and* the same net exposure.
    """
    rng = np.random.default_rng(seed)
    n = len(sig_dirs)
    warm = 80
    valid = close.index[warm:]
    if len(valid) == 0 or n == 0:
        return pd.DataFrame(columns=["dir"], index=pd.DatetimeIndex([], name="date"))
    take = min(n, len(valid))
    chosen = rng.choice(valid, size=take, replace=False)
    dirs = rng.permutation(sig_dirs["dir"].to_numpy())[:take]
    out = pd.DataFrame({"dir": dirs}, index=pd.DatetimeIndex(sorted(chosen), name="date"))
    return out


# --------------------------------------------------------------------------- #
# Forward-return engine (pole-direction returns)
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, sigs: pd.DataFrame, horizon: int,
                    cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day **pole-direction** return per signal, entered the *next* close.

    Each signal carries a ``dir`` (+1 long / -1 short). The trade return is
    ``dir * (P[e+h]/P[e] - 1)`` where ``e = i+1`` (enter at the next close, one lag).
    ``cost_bps`` is a one-way cost charged twice (in + out). Windows that overrun are dropped.
    """
    pos = {d: i for i, d in enumerate(close.index)}
    p = close.to_numpy(dtype=float)
    n = p.size
    out = []
    for d, row in sigs.iterrows():
        i = pos.get(d)
        if i is None or i + 1 + horizon >= n:
            continue
        e = i + 1
        r = (p[e + horizon] / p[e] - 1.0) * float(row["dir"])
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


def direction_placebo(close: pd.Series, sigs: pd.DataFrame, horizon: int,
                      n_draws: int = 1000, seed: int = 464) -> dict:
    """Placebo: keep the breakout *dates* but randomize the traded direction.

    The pennant's whole claim is *continuation* — trade **in the pole direction**. The honest
    null is: do these exact breakout dates carry an edge once we forget which way the pole
    pointed? We keep the same dates and the same long/short *count* but reshuffle which sign is
    attached to which date. If the real (pole-direction) result sits far in the right tail, the
    direction is load-bearing; if it sits mid-pack, "continuation" is not doing the work.

    Returns ``{obs, p_value, n_draws}`` where ``obs`` is the real pole-direction mean (bps) and
    ``p_value`` the share of scrambled-direction draws whose mean **beats or ties** the real one.
    """
    obs = float(np.mean(forward_returns(close, sigs, horizon)))
    if len(sigs) < 6 or not np.isfinite(obs):
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    base_dirs = sigs["dir"].to_numpy(dtype=float)
    beats = 0
    for _ in range(n_draws):
        scr = sigs.copy()
        scr["dir"] = rng.permutation(base_dirs)
        m = float(np.mean(forward_returns(close, scr, horizon)))
        if np.isfinite(m) and m >= obs:
            beats += 1
    p = (beats + 1) / (n_draws + 1)
    return {"obs": obs, "p_value": float(p), "n_draws": n_draws}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(bars: pd.DataFrame, cost_bps: float = 1.0, random_seed: int = 7,
                   **detect_kw) -> dict:
    """Run the full gauntlet on one tape: breakout-continuation vs random-entry baseline.

    Returns a dict keyed by horizon with the pennant summary (gross + net), the drift-matched
    random-entry baseline (matched long/short mix), and the pennant-minus-random delta.
    """
    close = bars["close"]
    sigs = detect_pennants(bars, **detect_kw)
    res = {"n_entries": int(len(sigs)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, sigs, h, cost_bps=0.0))
        net = summarize(forward_returns(close, sigs, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close, sigs, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
