"""The Kicker candlestick pattern as a falsifiable mechanical rule — Study 457.

The **kicker** (a.k.a. "kicker signal") is a two-candle reversal pattern. Two *opposite*
marubozu candles separated by a **gap in the new direction**, ignoring the prior trend:

* **Bullish kicker** — a black (down) marubozu, then a white (up) marubozu whose **open gaps
  up above the first candle's open** (a gap up). The lore: a violent turn to the upside.
* **Bearish kicker** — the mirror: a white (up) marubozu, then a black (down) marubozu whose
  **open gaps down below the first candle's open**.

A *marubozu* is a candle with a big body and tiny/absent wicks (open ≈ extreme, close ≈ the
other extreme). The folklore (Bulkowski, Nison, every chart-pattern site): the kicker is "one
of the most reliable" reversal signals — when it prints you trade its direction, ignoring what
came before.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Marubozu test** — body / (high − low) ≥ ``body_frac`` (default 0.60): a strong candle
   whose wicks are at most ~40% of the range. A bullish marubozu closes above its open; a
   bearish one below.
2. **Kicker test** — bar ``t-1`` and bar ``t`` are opposite-colour marubozus, and bar ``t``
   gaps in its own direction past bar ``t-1``'s open: bullish needs ``open_t > open_{t-1}``,
   bearish needs ``open_t < open_{t-1}``. The pattern is *completed* on the close of bar ``t``.
3. **Trade the kicker direction** — long on a bullish kicker, short on a bearish kicker,
   entered at the **next** close (one documented lag); we measure the forward H-day return
   (signed by the kicker direction).
4. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold,
   same long/short mix) that captures the tape's drift, and (b) a **gap-scramble placebo**
   that keeps every bar's marubozu colours but permutes the day-over-day gap signs, destroying
   the gap-in-new-direction geometry while keeping the candle marginal — the honest "is the
   gap-reversal structure doing anything?" null.

No look-ahead: the pattern is read on the close of bar ``t`` (using only bars ``t-1`` and
``t``), the position is entered at the close of bar ``t+1``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Candle primitives
# --------------------------------------------------------------------------- #
def _candle_arrays(bars: pd.DataFrame):
    o = bars["open"].to_numpy(dtype=float)
    h = bars["high"].to_numpy(dtype=float)
    l = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    return o, h, l, c


def marubozu_flags(bars: pd.DataFrame, body_frac: float = 0.60):
    """Per-bar (is_marubozu, direction): direction +1 (white/up), -1 (black/down), 0 (none).

    A marubozu has body / range >= ``body_frac`` (tiny wicks). Direction is the sign of
    close − open. Returns two numpy arrays aligned to ``bars``.
    """
    o, h, l, c = _candle_arrays(bars)
    rng = h - l
    body = np.abs(c - o)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(rng > 0, body / rng, 0.0)
    is_maru = frac >= body_frac
    direction = np.sign(c - o).astype(int)
    direction = np.where(is_maru, direction, 0)
    return is_maru, direction


# --------------------------------------------------------------------------- #
# Kicker detection
# --------------------------------------------------------------------------- #
def kicker_signals(bars: pd.DataFrame, body_frac: float = 0.60) -> pd.Series:
    """Per-bar kicker signal completed on this bar: +1 bullish, -1 bearish, 0 none.

    A signal at bar ``t`` requires bar ``t-1`` and ``t`` to be opposite-colour marubozus and
    bar ``t`` to gap *in its own direction* past bar ``t-1``'s open. Uses only bars ``t-1`` and
    ``t`` — no look-ahead. The entry executes at ``t+1`` (see :func:`forward_returns`).
    """
    o, h, l, c = _candle_arrays(bars)
    is_maru, d = marubozu_flags(bars, body_frac=body_frac)
    n = len(bars)
    sig = np.zeros(n, dtype=int)
    for t in range(1, n):
        if not (is_maru[t] and is_maru[t - 1]):
            continue
        if d[t] == 0 or d[t - 1] == 0 or d[t] == d[t - 1]:
            continue  # need opposite colours
        if d[t] > 0 and o[t] > o[t - 1]:      # bullish kicker: up candle gaps up past prior open
            sig[t] = +1
        elif d[t] < 0 and o[t] < o[t - 1]:    # bearish kicker: down candle gaps down past prior open
            sig[t] = -1
    return pd.Series(sig, index=bars.index)


def kicker_entries(bars: pd.DataFrame, body_frac: float = 0.60):
    """Entry dates and their direction (+1 long / -1 short) — one row per kicker."""
    sig = kicker_signals(bars, body_frac=body_frac)
    nz = sig[sig != 0]
    return pd.DatetimeIndex(nz.index), nz.to_numpy()


def random_entries(bars: pd.DataFrame, n: int, dirs: np.ndarray | None = None,
                   seed: int = 0):
    """``n`` random entry dates (after a warm-up), the drift-matched baseline.

    If ``dirs`` is given, its long/short composition is reused (sampled with replacement) so
    the random baseline carries the *same* directional mix as the real kicker entries — the
    fair test on a tape that drifts up (a naive all-long baseline would beat an all-short rule
    for free).
    """
    rng = np.random.default_rng(seed)
    valid = bars.index[2:]
    if len(valid) == 0:
        return pd.DatetimeIndex([]), np.array([], dtype=int)
    k = min(n, len(valid))
    chosen = rng.choice(valid, size=k, replace=False)
    chosen = pd.DatetimeIndex(sorted(chosen))
    if dirs is not None and len(dirs):
        rdir = rng.choice(dirs, size=k, replace=True)
    else:
        rdir = np.ones(k, dtype=int)
    return chosen, rdir


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(bars: pd.DataFrame, entries, dirs, horizon: int,
                    cost_bps: float = 0.0) -> np.ndarray:
    """Signed forward ``horizon``-day return per entry, entered at the *next* close (one lag).

    The return is ``dir * (close[e+h]/close[e] - 1)`` where ``e = entry_index + 1``; a short
    kicker profits when price falls. ``cost_bps`` is a one-way cost charged twice (in + out).
    Trades whose window overruns the tape are dropped.
    """
    close = bars["close"].to_numpy(dtype=float)
    pos = {d: i for i, d in enumerate(bars.index)}
    n = close.size
    entries = pd.DatetimeIndex(entries)
    dirs = np.asarray(dirs, dtype=float)
    out = []
    for d, sgn in zip(entries, dirs):
        i = pos.get(d)
        if i is None or i + 1 + horizon >= n:
            continue
        e = i + 1                       # enter at next close
        r = sgn * (close[e + horizon] / close[e] - 1.0)
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


def gap_scramble_placebo(bars: pd.DataFrame, horizon: int, body_frac: float = 0.60,
                         n_draws: int = 1000, seed: int = 457) -> dict:
    """Placebo: keep marubozu colours, permute the day-over-day gap signs.

    The kicker's load-bearing geometry is *the gap in the new direction*. We keep every bar's
    marubozu flag and colour (the candle marginal) but build a synthetic "kicker" set where the
    gap direction is drawn from a **shuffled** permutation of the actual day-over-day open-gap
    signs across the tape. The opposite-colour marubozu condition is preserved; only the gap
    geometry is randomised. Returns the share of placebo runs whose mean kicker-direction
    forward return **beats** the real one — the honest "is the gap-reversal doing anything?"
    p-value, plus the observed mean.
    """
    ent, dirs = kicker_entries(bars, body_frac=body_frac)
    obs = float(np.mean(forward_returns(bars, ent, dirs, horizon))) if len(ent) else float("nan")

    o = bars["open"].to_numpy(dtype=float)
    is_maru, d = marubozu_flags(bars, body_frac=body_frac)
    n = len(bars)
    idx = bars.index
    # candidate bars: opposite-colour marubozu pairs (the kicker shell, ignoring the gap)
    cand = []
    cand_dir = []
    real_gapsign = []
    for t in range(1, n):
        if is_maru[t] and is_maru[t - 1] and d[t] != 0 and d[t - 1] != 0 and d[t] != d[t - 1]:
            cand.append(t)
            cand_dir.append(d[t])
            real_gapsign.append(np.sign(o[t] - o[t - 1]))
    cand = np.asarray(cand)
    cand_dir = np.asarray(cand_dir)
    real_gapsign = np.asarray(real_gapsign)
    if cand.size < 3 or not np.isfinite(obs):
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}

    rng = np.random.default_rng(seed)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = rng.permutation(real_gapsign)
        # placebo "kicker": opposite-colour marubozu pair whose (shuffled) gap sign agrees
        # with the candle direction — i.e. perm == cand_dir keeps the same definition shape
        keep = perm == cand_dir
        if keep.sum() < 3:
            continue
        ent_pos = cand[keep]
        ent_dates = idx[ent_pos]
        ent_dirs = cand_dir[keep]
        rr = forward_returns(bars, ent_dates, ent_dirs, horizon)
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
def run_experiment(bars: pd.DataFrame, body_frac: float = 0.60, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: kicker vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the kicker summary (gross + net), the drift-matched
    random-entry baseline (same direction mix), and the kicker-minus-random delta.
    """
    ent, dirs = kicker_entries(bars, body_frac=body_frac)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(bars, ent, dirs, h, cost_bps=0.0))
        net = summarize(forward_returns(bars, ent, dirs, h, cost_bps=cost_bps))
        re, rdir = random_entries(bars, max(len(ent), 50), dirs=dirs, seed=random_seed)
        rnd = summarize(forward_returns(bars, re, rdir, h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
