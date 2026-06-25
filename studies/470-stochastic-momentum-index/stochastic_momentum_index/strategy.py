"""Stochastic Momentum Index as a falsifiable mechanical rule — Study 470.

William Blau's **Stochastic Momentum Index** (SMI, 1993) refines Lane's stochastic oscillator.
Where the classic %K measures the close's position relative to the *low* of the N-day range, the
SMI measures it relative to the **midpoint** of the range, then double-smooths both numerator and
denominator with two EMAs and scales to ±100:

    HH_t = max(high, N)        LL_t = min(low, N)
    M_t  = (HH_t + LL_t) / 2                          # range midpoint
    D_t  = close_t - M_t                              # distance from midpoint
    R_t  = HH_t - LL_t                                # full range
    SMI_t = 100 * EMA(EMA(D, s1), s2) / (EMA(EMA(R/2, s1), s2))

with the classic Blau parameters N=13, s1=25, s2=2 (a signal EMA is sometimes added; we use the
raw SMI for the turn). The SMI lives in [-100, +100]; conventional thresholds put **oversold** at
about -40 and overbought at +40.

The folklore (Blau's own teaching, echoed on every indicator site): *the SMI times turns* — when
it stops falling and **rises up out of oversold**, price is about to bottom, so that cross is a
high-probability **buy**.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **No look-ahead.** The SMI at bar ``t`` uses only highs/lows/closes through ``t``; EMAs are
   causal. The "rising out of oversold" cross is read on the close of *t*.
2. **Entry rule.** A long fires when the SMI was below the oversold threshold on the prior bar and
   **turns up** (SMI_t > SMI_{t-1}) while still in/just leaving the oversold zone. Entry is at the
   **next** close (one documented lag); we then measure the forward H-day return.
3. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures the
   tape's drift, and (b) a **parameter-scramble placebo** that recomputes the SMI with the smoothing
   lengths randomly permuted/perturbed, destroying the specific Blau geometry while keeping a
   "double-smoothed range oscillator" of the same family — the honest "is *this* SMI doing
   anything?" null.

No look-ahead: the SMI is causal, the turn is read on the close of *t*, the position is entered at
the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)

# Classic Blau SMI parameters and the conventional oversold gate.
SMI_N = 13          # range look-back
SMI_S1 = 25         # first (slow) smoothing EMA
SMI_S2 = 2          # second (fast) smoothing EMA
OVERSOLD = -40.0    # SMI oversold threshold


# --------------------------------------------------------------------------- #
# The indicator
# --------------------------------------------------------------------------- #
def _ema(x: pd.Series, span: int) -> pd.Series:
    """Causal exponential moving average (pandas adjust=False)."""
    return x.ewm(span=max(int(span), 1), adjust=False).mean()


def smi(high: pd.Series, low: pd.Series, close: pd.Series,
        n: int = SMI_N, s1: int = SMI_S1, s2: int = SMI_S2) -> pd.Series:
    """Blau's Stochastic Momentum Index, bounded in [-100, +100].

    SMI = 100 * EMA(EMA(close - midpoint, s1), s2) / EMA(EMA(range/2, s1), s2), where midpoint and
    range are over the rolling ``n``-day high/low band. Causal (no future bars). NaN during warm-up.
    """
    hh = high.rolling(n, min_periods=n).max()
    ll = low.rolling(n, min_periods=n).min()
    mid = (hh + ll) / 2.0
    dist = close - mid
    half_range = (hh - ll) / 2.0

    sm_d = _ema(_ema(dist, s1), s2)
    sm_r = _ema(_ema(half_range, s1), s2)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = 100.0 * sm_d / sm_r
    return val.replace([np.inf, -np.inf], np.nan).clip(-100.0, 100.0)


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def smi_turn_entries(high: pd.Series, low: pd.Series, close: pd.Series,
                     n: int = SMI_N, s1: int = SMI_S1, s2: int = SMI_S2,
                     oversold: float = OVERSOLD) -> pd.DatetimeIndex:
    """Bars where the SMI **rises up out of oversold** — Blau's 'time the turn' buy.

    The trigger: the SMI was below ``oversold`` on bar ``t-1`` and turns up on bar ``t``
    (SMI_t > SMI_{t-1}), while SMI_{t-1} is still in the oversold zone. Only the *first* bar of
    each consecutive run is kept (the turn, not every up-day thereafter). Entry is executed at the
    next close by :func:`forward_returns`.
    """
    s = smi(high, low, close, n=n, s1=s1, s2=s2)
    prev = s.shift(1)
    rising = s > prev
    was_oversold = prev < oversold
    mask = rising & was_oversold & s.notna() & prev.notna()
    first = mask & ~mask.shift(1, fill_value=False)
    return close.index[first.to_numpy()]


def random_entries(close: pd.Series, n: int, warmup: int = 40, seed: int = 0) -> pd.DatetimeIndex:
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


def scrambled_param_placebo(high: pd.Series, low: pd.Series, close: pd.Series, horizon: int,
                            n: int = SMI_N, s1: int = SMI_S1, s2: int = SMI_S2,
                            oversold: float = OVERSOLD,
                            n_draws: int = 500, seed: int = 470) -> dict:
    """Placebo: recompute the SMI with **scrambled smoothing/look-back parameters**.

    Keeps the same *family* of indicator (a double-smoothed range oscillator on the same price
    tape) but randomly perturbs the look-back ``n`` and the two EMA lengths ``s1, s2`` away from
    Blau's specific (13, 25, 2). If the *specific* SMI geometry forecasts turns, the real Blau SMI
    should beat the scrambled cousins. Returns the share of scrambled runs whose mean "turn-out-of-
    oversold" forward return **beats** the real one — the honest "is *this* parameterisation load-
    bearing?" p-value, plus the observed mean.
    """
    obs = float(np.mean(forward_returns(
        close, smi_turn_entries(high, low, close, n=n, s1=s1, s2=s2, oversold=oversold), horizon)))
    if not np.isfinite(obs):
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        nn = int(rng.integers(5, 30))      # range look-back 5..29
        ss1 = int(rng.integers(5, 40))     # first EMA 5..39
        ss2 = int(rng.integers(2, 15))     # second EMA 2..14
        ent = smi_turn_entries(high, low, close, n=nn, s1=ss1, s2=ss2, oversold=oversold)
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
def run_experiment(bars: pd.DataFrame, n: int = SMI_N, s1: int = SMI_S1, s2: int = SMI_S2,
                   oversold: float = OVERSOLD, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: SMI turn vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the turn summary (gross + net), the drift-matched
    random-entry baseline, and the turn-minus-random delta.
    """
    high, low, close = bars["high"], bars["low"], bars["close"]
    ent = smi_turn_entries(high, low, close, n=n, s1=s1, s2=s2, oversold=oversold)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        # large random draw => a stable drift-matched baseline (entries here are sparse)
        rnd = summarize(forward_returns(
            close, random_entries(close, 1000, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
