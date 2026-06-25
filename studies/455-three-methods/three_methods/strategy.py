"""Rising/Falling Three Methods as a falsifiable mechanical rule — Study 455.

The *three methods* (san-poh) is one of the classic Japanese candlestick **continuation**
patterns (Steve Nison, *Japanese Candlestick Charting Techniques*). It comes in two mirrored
forms:

* **Rising three methods** (bullish continuation):
    1. a long **white** (up) candle in an uptrend;
    2. three small candles that drift *against* the trend but stay **inside** the first
       candle's high–low range;
    3. a long white candle that closes **above** the first candle's close.
* **Falling three methods** is the bearish mirror (long black candle, three small inside
  candles, a long black candle closing below the first's close).

The folklore: the three small candles are a *pause* — profit-taking that does not reverse the
trend — and the fifth candle's break past the first **confirms continuation**. So you trade in
the trend's direction (long on rising, short on falling) and expect the move to resume.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Five closed candles.** The whole pattern ends at bar ``t``; every candle is fully closed,
   so the signal is read on the close of *t* and the trade entered at *t+1*'s close — no
   look-ahead at all (no pivot confirmation lag is even needed; the pattern is self-contained).
2. **Mechanical body/range tests.** The anchor candle's body must exceed a multiple of recent
   average body; the three middle candles must each be *small* (body < a fraction of the
   anchor) and held *inside* the anchor's high–low range; the fifth candle must be long and
   close past the anchor's close in the trend direction.
3. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold) that
   captures the tape's drift; and (b) a **shuffled-body placebo** that re-labels the same set
   of candidate dates onto random calendar dates' candle *bodies*, destroying the specific
   five-candle geometry while keeping the price marginal — the honest "is the pattern's shape
   doing anything?" null.

Direction matters: rising fires a **long**, falling fires a **short**, so the forward return is
signed by the pattern direction. The Signal axis is always signal-vs-random-baseline, never
signal-vs-zero (drift/beta would flatter a long-only rule on an upward tape).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Pattern detection (rising / falling three methods)
# --------------------------------------------------------------------------- #
def _bodies(bars: pd.DataFrame) -> np.ndarray:
    return np.abs(bars["close"].to_numpy() - bars["open"].to_numpy())


def three_methods_signals(
    bars: pd.DataFrame,
    avg_body_win: int = 20,
    long_mult: float = 1.0,
    small_frac: float = 0.7,
    tol: float = 0.10,  # noqa: ARG  charitable wick tolerance; 0 = strict containment
) -> pd.Series:
    """Signed three-methods signal on the close of each bar (+1 rising, -1 falling, 0 none).

    A pattern occupies bars ``t-4 .. t`` (all closed at ``t``). Conditions, for the *rising*
    form (the falling form is the exact mirror):

    * **anchor** (bar t-4): an up candle (close > open) whose body exceeds ``long_mult`` ×
      the trailing average body (a "long" white candle);
    * **three middles** (t-3, t-2, t-1): each held *inside* the anchor's high–low range — with a
      small ``tol`` tolerance (a fraction of the anchor range) since exact containment of every
      wick is rare on daily bars — and each *small* (body < ``small_frac`` × anchor body);
    * **confirm** (bar t): a long up candle (body exceeds ``long_mult`` × avg body) that closes
      **above** the anchor's close.

    The signal at bar ``t`` is read on the close of ``t``; :func:`forward_returns` enters at
    ``t+1``. No look-ahead. ``tol`` is the only "charitable" knob — set it to 0 for strict
    wick-containment.
    """
    o = bars["open"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    h = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    body = np.abs(c - o)
    n = len(bars)
    # trailing average body (causal, excludes the current bar)
    avg_body = pd.Series(body, index=bars.index).shift(1).rolling(
        avg_body_win, min_periods=avg_body_win).mean().to_numpy()

    sig = np.zeros(n, dtype=int)
    for t in range(4, n):
        a = t - 4  # anchor
        ab = avg_body[a]
        if not np.isfinite(ab) or ab <= 0:
            continue
        anchor_body = body[a]
        if anchor_body < long_mult * ab:
            continue
        ah, al, ac, ao = h[a], lo[a], c[a], o[a]
        rng_pad = tol * (ah - al)
        # three middles inside the anchor range (with a small tolerance) and small
        mids = (a + 1, a + 2, a + 3)
        inside = all(h[j] <= ah + rng_pad and lo[j] >= al - rng_pad for j in mids)
        smalls = all(body[j] < small_frac * anchor_body for j in mids)
        if not (inside and smalls):
            continue
        confirm_body = body[t]
        if confirm_body < long_mult * ab:
            continue
        up_anchor = ac > ao
        dn_anchor = ac < ao
        if up_anchor and c[t] > ac and c[t] > o[t]:
            sig[t] = +1            # rising three methods -> long
        elif dn_anchor and c[t] < ac and c[t] < o[t]:
            sig[t] = -1            # falling three methods -> short
    return pd.Series(sig, index=bars.index, name="tm_signal")


def pattern_entries(bars: pd.DataFrame, **kw) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Dates with a non-zero three-methods signal, plus the signed direction array."""
    sig = three_methods_signals(bars, **kw)
    mask = sig != 0
    dates = bars.index[mask.to_numpy()]
    dirs = sig[mask.to_numpy()].to_numpy()
    return dates, dirs


def random_entries(bars: pd.DataFrame, n: int, warmup: int = 25, seed: int = 0):
    """``n`` random (date, direction) entries after warm-up — the drift-matched baseline.

    Directions are drawn to **match the realised long/short mix** is the caller's job; here we
    just hand back random *dates* with a +1 (long) direction by default so the baseline inherits
    the same drift a long-only dip-buy would. The orchestrator passes the observed long-share
    so the baseline mirrors the pattern's actual directional balance.
    """
    rng = np.random.default_rng(seed)
    valid = bars.index[warmup:]
    if len(valid) == 0:
        return pd.DatetimeIndex([]), np.array([], dtype=int)
    m = min(n, len(valid))
    chosen = rng.choice(valid, size=m, replace=False)
    dates = pd.DatetimeIndex(sorted(chosen))
    dirs = np.ones(len(dates), dtype=int)
    return dates, dirs


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(bars: pd.DataFrame, entries, dirs, horizon: int,
                    cost_bps: float = 0.0) -> np.ndarray:
    """Signed forward ``horizon``-day return for each entry, entered at the *next* close.

    ``dirs[i]`` is +1 (long) or -1 (short); the realised return is multiplied by the sign, so a
    falling-three-methods short profits when price falls. ``cost_bps`` is a one-way cost charged
    on both legs. Trades whose window overruns the tape are dropped.
    """
    close = bars["close"]
    pos = {d: i for i, d in enumerate(close.index)}
    p = close.to_numpy(dtype=float)
    n = p.size
    out = []
    dirs = np.asarray(dirs, dtype=float)
    for d, s in zip(entries, dirs):
        i = pos.get(d)
        if i is None or i + 1 + horizon >= n:
            continue
        e = i + 1                      # enter at next close
        raw = p[e + horizon] / p[e] - 1.0
        out.append(s * raw - 2.0 * cost_bps * 1e-4)
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


def shuffled_body_placebo(bars: pd.DataFrame, horizon: int, n_draws: int = 1000,
                          seed: int = 455, **kw) -> dict:
    """Placebo: keep the *number* of signals and their directions, but fire them on random
    dates, destroying the five-candle geometry while keeping the price marginal.

    The real pattern selects specific dates by a precise shape; this null asks whether *that
    shape* matters by drawing the same count of (random-date, same-direction-mix) entries and
    measuring their mean signed forward return. Returns the share of placebo runs whose mean
    **beats** the real one — the honest "is the pattern's geometry adding anything?" p-value.
    """
    dates, dirs = pattern_entries(bars, **kw)
    obs_r = forward_returns(bars, dates, dirs, horizon)
    if obs_r.size == 0:
        return {"obs": float("nan"), "p_value": float("nan"), "n_draws": 0}
    obs = float(obs_r.mean())
    n_sig = len(dates)
    long_share = float((dirs > 0).mean())
    rng = np.random.default_rng(seed)
    valid = bars.index[25:-(horizon + 2)]
    if len(valid) < n_sig:
        valid = bars.index[25:]
    beats = 0
    valid_runs = 0
    for _ in range(n_draws):
        chosen = rng.choice(valid, size=min(n_sig, len(valid)), replace=False)
        d = pd.DatetimeIndex(sorted(chosen))
        rdirs = np.where(rng.random(len(d)) < long_share, 1, -1)
        rr = forward_returns(bars, d, rdirs, horizon)
        if rr.size == 0:
            continue
        valid_runs += 1
        if rr.mean() >= obs:
            beats += 1
    p = (beats + 1) / (valid_runs + 1) if valid_runs else float("nan")
    return {"obs": obs, "p_value": float(p), "n_draws": valid_runs}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(bars: pd.DataFrame, cost_bps: float = 1.0, random_seed: int = 7,
                   **kw) -> dict:
    """Run the full gauntlet on one tape: three-methods vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the pattern summary (gross + net), the drift-matched
    random-entry baseline (same count, same long/short mix), and the pattern-minus-random delta.
    """
    dates, dirs = pattern_entries(bars, **kw)
    long_share = float((dirs > 0).mean()) if len(dirs) else 1.0
    res = {"n_entries": int(len(dates)), "long_share": long_share, "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(bars, dates, dirs, h, cost_bps=0.0))
        net = summarize(forward_returns(bars, dates, dirs, h, cost_bps=cost_bps))
        rd, _ = random_entries(bars, max(len(dates), 50), seed=random_seed)
        rng = np.random.default_rng(random_seed + 1)
        rdirs = np.where(rng.random(len(rd)) < long_share, 1, -1)
        rnd = summarize(forward_returns(bars, rd, rdirs, h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
