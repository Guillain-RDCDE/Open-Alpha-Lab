"""The Bear-Flag as a falsifiable mechanical rule — Study 463.

The classic **bear flag** is a two-part continuation pattern:

* a **pole** — a sharp, fast drop over a short window (the impulsive first leg down);
* a **flag** — a small, *up-sloping* consolidation that drifts gently *against* the pole on
  a tighter range (a counter-trend pause, not a reversal).

The folklore (every chart-pattern site, *Encyclopedia of Chart Patterns*, and the technician
canon): *the flag forecasts continuation* — once price **breaks down** below the flag's lower
edge, the second leg down begins, often a "measured move" the size of the pole. So a short on
the flag breakdown is a high-probability continuation trade.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Pole** — over a lookback window the close falls by at least ``pole_min`` (a sharp drop),
   measured as the log-return from the window's start to the pole's bottom.
2. **Flag** — the next ``flag_len`` bars form a tight, *up-sloping* consolidation: a positive
   regression slope (retracing *up* against the pole) whose total retrace stays below the pole
   (a pause, not a full reversal), on a contained range.
3. **Breakdown** — a long-side *short* fires on the first close **below the flag's lower
   trendline** (the channel's lower edge), entered at the **next** close (one documented lag);
   we then measure the forward H-day return of the **short** (so a continued drop is a profit).
4. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold,
   short side) that captures the tape's drift, and (b) a **shuffled-flag placebo** that keeps
   the breakdown *count* and epoch but scrambles the flag's defining slope test, destroying the
   pattern's geometry while keeping the price marginal — the honest "is the flag's structure
   doing anything?" null.

No look-ahead: the pole/flag geometry uses only bars up to and including *t*, the breakdown is
read on the close of *t*, the short is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)

# Default pattern parameters (a proponent's tightest mechanical bear flag).
POLE_LOOKBACK = 10     # bars over which the sharp drop (pole) is measured
POLE_MIN = 0.06        # minimum pole drop, log units (~6%)
FLAG_LEN = 7           # bars of consolidation forming the flag
FLAG_MAX_RETRACE = 0.6  # the up-flag may retrace at most 60% of the pole (else it's a reversal)


# --------------------------------------------------------------------------- #
# Pattern detection (pole + up-sloping flag + breakdown)
# --------------------------------------------------------------------------- #
def _slope(y: np.ndarray) -> float:
    """OLS slope of ``y`` against 0..len-1 (per-bar)."""
    n = y.size
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xm = x.mean()
    denom = float(((x - xm) ** 2).sum())
    if denom == 0.0:
        return 0.0
    return float(((x - xm) * (y - y.mean())).sum() / denom)


def flag_lower_line(logc: np.ndarray, t: int, flag_len: int) -> float:
    """Value at bar ``t`` of the flag's *lower* trendline (parallel channel through flag lows).

    The flag spans bars ``t-flag_len .. t-1``; we fit an OLS line to those log-closes and shift
    it down to the most-negative residual (the lower channel edge), then extrapolate to bar
    ``t``. Returns the lower-line log-price at ``t`` (or NaN if the window is incomplete).
    """
    a = t - flag_len
    if a < 0:
        return np.nan
    y = logc[a:t]
    n = y.size
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    xm = x.mean()
    denom = float(((x - xm) ** 2).sum())
    if denom == 0.0:
        return np.nan
    slope = float(((x - xm) * (y - y.mean())).sum() / denom)
    intercept = y.mean() - slope * xm
    fit = slope * x + intercept
    resid = y - fit
    lower_off = float(resid.min())                 # shift line to lower channel edge
    # extrapolate to bar t (x = n in this local frame)
    return slope * n + intercept + lower_off


def detect_flags(close: pd.Series,
                 pole_lookback: int = POLE_LOOKBACK,
                 pole_min: float = POLE_MIN,
                 flag_len: int = FLAG_LEN,
                 flag_max_retrace: float = FLAG_MAX_RETRACE,
                 scramble_seed: int | None = None) -> pd.Series:
    """Boolean Series: is bar ``t`` a confirmed up-sloping bear-flag *consolidation end*?

    A bar ``t`` qualifies when, using only bars up to ``t``:
      * a **pole**: over the ``pole_lookback`` bars ending at the flag start, the log-close fell
        by at least ``pole_min`` (start-to-bottom);
      * an **up-sloping flag**: the last ``flag_len`` bars (ending at ``t``) have a *positive*
        OLS slope (drifting up against the pole) but their total up-retrace is at most
        ``flag_max_retrace`` of the pole (a pause, not a reversal).

    ``scramble_seed`` (placebo): when given, the flag's *slope sign* test is replaced by a
    deterministic coin keyed to the bar — destroying the "up-sloping" geometry while keeping the
    pole filter and the same epoch. The marginal flow of candidate bars is preserved.
    """
    logc = np.log(close.to_numpy(dtype=float))
    n = logc.size
    out = np.zeros(n, dtype=bool)
    rng = np.random.default_rng(scramble_seed) if scramble_seed is not None else None
    coin = rng.random(n) if rng is not None else None
    for t in range(pole_lookback + flag_len, n):
        flag_start = t - flag_len
        # pole: window ending where the flag begins
        win = logc[flag_start - pole_lookback:flag_start + 1]
        pole = float(win[0] - win.min())          # drop from window-start to its low (>=0)
        if pole < pole_min:
            continue
        flag = logc[flag_start:t + 1]
        slope = _slope(flag)
        retrace_up = float(flag[-1] - flag.min())  # how far the flag climbed back
        if coin is not None:
            up_ok = coin[t] > 0.5                   # placebo: random in place of slope>0
        else:
            up_ok = slope > 0.0                      # genuine up-sloping flag
        if up_ok and retrace_up <= flag_max_retrace * pole:
            out[t] = True
    return pd.Series(out, index=close.index)


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def breakdown_entries(close: pd.Series,
                      pole_lookback: int = POLE_LOOKBACK,
                      pole_min: float = POLE_MIN,
                      flag_len: int = FLAG_LEN,
                      flag_max_retrace: float = FLAG_MAX_RETRACE,
                      scramble_seed: int | None = None) -> pd.DatetimeIndex:
    """Bars whose close pierces *below* the flag's lower trendline after a confirmed bear flag.

    A short fires on the first bar where (a) a bear flag is confirmed at ``t-1`` (pole +
    up-sloping flag) and (b) the close at ``t`` closes **below** the flag's lower line. Only the
    *first* bar of each consecutive breakdown run is kept. Entry is executed at the next close
    by :func:`forward_returns` (short side).
    """
    logc = np.log(close.to_numpy(dtype=float))
    flags = detect_flags(close, pole_lookback, pole_min, flag_len, flag_max_retrace,
                         scramble_seed=scramble_seed).to_numpy()
    n = logc.size
    raw = np.zeros(n, dtype=bool)
    for t in range(1, n):
        if not flags[t - 1]:
            continue
        lower = flag_lower_line(logc, t, flag_len)
        if np.isfinite(lower) and logc[t] < lower:
            raw[t] = True
    mask = pd.Series(raw, index=close.index)
    first = mask & ~mask.shift(1, fill_value=False)
    return close.index[first.to_numpy()]


def random_entries(close: pd.Series, n: int, warmup: int = 20, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[warmup:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine (SHORT side — a continued drop is a profit)
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0,
                    side: int = -1) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    ``side = -1`` (default) is the **short** the bear-flag rule wants: the trade profits if
    price *continues down*, so the return is ``-(p[e+h]/p[e]-1)``. ``cost_bps`` is a one-way
    cost charged twice (in + out). Trades whose window overruns the tape are dropped.
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
        out.append(side * r - 2.0 * cost_bps * 1e-4)
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


def shuffled_flag_placebo(close: pd.Series, horizon: int,
                          pole_lookback: int = POLE_LOOKBACK, pole_min: float = POLE_MIN,
                          flag_len: int = FLAG_LEN, flag_max_retrace: float = FLAG_MAX_RETRACE,
                          n_draws: int = 1000, seed: int = 463) -> dict:
    """Placebo: replace the flag's *up-sloping* slope test with a coin, destroying the geometry.

    Keeps the **pole** filter, the same epoch and roughly the same flow of candidate bars, but
    scrambles which bars count as "up-sloping flags" (a per-bar coin instead of slope > 0), so
    the defining geometry of the flag is meaningless. Returns the share of placebo runs whose
    mean breakdown-short forward return **beats** the real one — the honest "is the flag's
    structure adding anything?" p-value, plus the observed mean.
    """
    obs = float(np.mean(forward_returns(
        close, breakdown_entries(close, pole_lookback, pole_min, flag_len, flag_max_retrace),
        horizon)))
    rng = np.random.default_rng(seed)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        s = int(rng.integers(1, 2**31 - 1))
        ent = breakdown_entries(close, pole_lookback, pole_min, flag_len, flag_max_retrace,
                                scramble_seed=s)
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
def run_experiment(close: pd.Series, cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: breakdown-short vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the breakdown-short summary (gross + net), the
    drift-matched random-entry baseline (also short side), and the breakdown-minus-random delta.
    """
    ent = breakdown_entries(close)
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
