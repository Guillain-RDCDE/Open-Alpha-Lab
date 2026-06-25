"""Bill Williams' Accelerator Oscillator (AC) as a falsifiable mechanical rule — Study 474.

Bill Williams' **Accelerator Oscillator** measures the *acceleration* of momentum — the second
derivative of price. It is built on top of his **Awesome Oscillator (AO)**:

* **median price** ``MP = (high + low) / 2``;
* **Awesome Oscillator** ``AO = SMA5(MP) - SMA34(MP)`` (fast minus slow momentum);
* **Accelerator Oscillator** ``AC = AO - SMA5(AO)`` (momentum minus its own average = acceleration).

The folklore (Williams' *Trading Chaos* / *New Trading Dimensions*, echoed on every indicator
site): AC **leads** AO and price — it turns up *before* momentum does, so two consecutive rising
("green") AC bars are a high-probability **buy**, and the strongest signal is two green bars while
AC is **above zero** (the classic "don't buy when the bars are red" rule). We encode the tightest
mechanical version a proponent would accept and test it honestly:

1. **Trailing-only AC.** All SMAs are trailing (closed-bar) windows, so AC at bar ``t`` uses only
   bars ``≤ t`` — no look-ahead.
2. **Two-green-bars-above-zero entry.** A long fires when ``AC[t] > AC[t-1] > AC[t-2]`` (two
   consecutive rising bars) **and** ``AC[t] > 0``; only the *first* bar of each run is kept. Entry
   is at the **next** close (one documented lag); we then measure the forward H-day return.
3. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold) that
   captures the tape's drift, and (b) a **rotated-AC placebo** that circularly shifts the AC series
   relative to price by a random offset, destroying the AC-vs-price time alignment while keeping
   the AC value marginal *exactly* — the honest "is the AC-to-price timing doing anything?" null.

No look-ahead: the AC is read on the close of *t*, the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)

AO_FAST = 5
AO_SLOW = 34
AC_SMA = 5


# --------------------------------------------------------------------------- #
# The Accelerator Oscillator
# --------------------------------------------------------------------------- #
def awesome_oscillator(bars: pd.DataFrame, fast: int = AO_FAST, slow: int = AO_SLOW) -> pd.Series:
    """Bill Williams' Awesome Oscillator: SMA_fast(median) - SMA_slow(median). Trailing only."""
    mp = (bars["high"] + bars["low"]) / 2.0
    return mp.rolling(fast).mean() - mp.rolling(slow).mean()


def accelerator_oscillator(bars: pd.DataFrame, fast: int = AO_FAST, slow: int = AO_SLOW,
                           ac_sma: int = AC_SMA) -> pd.Series:
    """Bill Williams' Accelerator Oscillator: AO - SMA(AO). The second derivative of price.

    All windows are trailing (closed-bar), so ``AC[t]`` uses only bars ``<= t`` — no look-ahead.
    """
    ao = awesome_oscillator(bars, fast=fast, slow=slow)
    return ao - ao.rolling(ac_sma).mean()


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def ac_entries(bars: pd.DataFrame, above_zero: bool = True,
               fast: int = AO_FAST, slow: int = AO_SLOW, ac_sma: int = AC_SMA) -> pd.DatetimeIndex:
    """Bars where AC turns up — two consecutive rising bars (the classic 'two green bars').

    With ``above_zero`` (the canonical Williams rule) the entry also requires ``AC[t] > 0``.
    Only the *first* bar of each consecutive run is kept (the turn-up, not every day AC keeps
    rising). Entry is executed at the next close by :func:`forward_returns`.
    """
    ac = accelerator_oscillator(bars, fast=fast, slow=slow, ac_sma=ac_sma)
    rising = (ac > ac.shift(1)) & (ac.shift(1) > ac.shift(2))
    cond = rising & ac.notna() & ac.shift(2).notna()
    if above_zero:
        cond = cond & (ac > 0)
    first = cond & ~cond.shift(1, fill_value=False)
    return bars.index[first.to_numpy()]


def random_entries(close: pd.Series, n: int, warmup: int = AO_SLOW + AC_SMA,
                   seed: int = 0) -> pd.DatetimeIndex:
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


def rotated_ac_placebo(bars: pd.DataFrame, horizon: int, above_zero: bool = True,
                       fast: int = AO_FAST, slow: int = AO_SLOW, ac_sma: int = AC_SMA,
                       n_draws: int = 1000, seed: int = 474) -> dict:
    """Placebo: circularly rotate the AC series relative to price, destroying the AC->price timing.

    Keeps the AC value *marginal exactly* (same bars, same values, same two-green-bars run
    structure) but shifts AC by a random circular offset so its alignment with forward price is
    meaningless. Returns the share of placebo runs whose mean AC-up forward return **beats** the
    real one — the honest "is the AC-to-price timing adding anything?" p-value, plus the observed
    mean. This is the AC analogue of the template's shuffled-pivot geometry placebo.
    """
    close = bars["close"]
    obs = float(np.mean(forward_returns(close, ac_entries(
        bars, above_zero=above_zero, fast=fast, slow=slow, ac_sma=ac_sma), horizon)))
    ac = accelerator_oscillator(bars, fast=fast, slow=slow, ac_sma=ac_sma)
    valid = ac.dropna()
    if len(valid) < 2 * slow:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    ac_vals = ac.to_numpy()
    idx = bars.index
    finite = np.isfinite(ac_vals)
    fin_idx = np.where(finite)[0]
    lo, hi = fin_idx.min(), fin_idx.max()
    block = ac_vals[lo:hi + 1].copy()
    m = block.size
    beats = 0
    valid_draws = 0
    for _ in range(n_draws):
        shift = int(rng.integers(slow, m - slow))
        rot = np.roll(block, shift)
        rot_full = ac_vals.copy()
        rot_full[lo:hi + 1] = rot
        ser = pd.Series(rot_full, index=idx)
        rising = (ser > ser.shift(1)) & (ser.shift(1) > ser.shift(2))
        cond = rising & ser.notna() & ser.shift(2).notna()
        if above_zero:
            cond = cond & (ser > 0)
        first = cond & ~cond.shift(1, fill_value=False)
        ent = idx[first.to_numpy()]
        rr = forward_returns(close, ent, horizon)
        if rr.size == 0:
            continue
        valid_draws += 1
        if rr.mean() >= obs:
            beats += 1
    p = (beats + 1) / (valid_draws + 1) if valid_draws else float("nan")
    return {"obs": obs, "p_value": float(p), "n_draws": valid_draws}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(bars: pd.DataFrame, above_zero: bool = True, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: AC-up vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the AC-up summary (gross + net), the drift-matched
    random-entry baseline, and the AC-minus-random delta.
    """
    close = bars["close"]
    ent = ac_entries(bars, above_zero=above_zero)
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
