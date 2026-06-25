"""Relative Vigor Index as a falsifiable mechanical rule — Study 469.

John F. Ehlers' **Relative Vigor Index** (RVI) is a momentum oscillator built on a simple idea:
*in an up-trend the market closes above its open; in a down-trend it closes below.* So it reads
the bar **body** (close − open) relative to the bar **range** (high − low):

* a 4-bar symmetric-weighted smoother (weights 1,2,2,1) is applied to both (close − open) and
  (high − low) to damp the per-bar noise (Ehlers' "value" smoother);
* the **RVI** at bar *t* is the *N*-bar sum of the smoothed numerator divided by the *N*-bar sum
  of the smoothed denominator;
* a **signal line** is the same 4-bar symmetric-weighted smoother applied to the RVI itself.

The folklore (Ehlers' own teaching, echoed by every charting suite): the **RVI crossing above
its signal line is a buy** (vigor turning up), the cross below a sell. We encode the tightest
mechanical version a proponent would accept and test it honestly:

1. **Causal indicator** — every weighted average uses only the current and past *closed* bars,
   so the RVI and its signal line at bar *t* never see the future.
2. **Cross detection** — a long entry fires on the first bar where RVI crosses *from below to
   above* its signal line, read on the close of *t*; the trade is entered at the **next** close
   (one documented lag). We then measure the forward H-day return.
3. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold) that
   captures the tape's drift, and (b) a **phase-scramble placebo** that circularly rolls the RVI
   series relative to price, destroying the cross's timing while keeping its marginal — the honest
   "is the cross's *structure* doing anything?" null.

No look-ahead: the indicator is causal, the cross is read on the close of *t*, the position is
entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)

RVI_PERIOD = 10           # N-bar sum window (Ehlers' default)
_SWMA = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0   # Ehlers' 4-bar symmetric weighted MA


# --------------------------------------------------------------------------- #
# The indicator
# --------------------------------------------------------------------------- #
def _swma4(x: pd.Series) -> pd.Series:
    """Ehlers' causal 4-bar symmetric-weighted moving average (weights 1,2,2,1)/6.

    Output at bar t uses x[t], x[t-1], x[t-2], x[t-3] — strictly causal, NaN for the warm-up.
    """
    v = x.to_numpy(dtype=float)
    n = v.size
    out = np.full(n, np.nan)
    for t in range(3, n):
        out[t] = (_SWMA[0] * v[t] + _SWMA[1] * v[t - 1]
                  + _SWMA[2] * v[t - 2] + _SWMA[3] * v[t - 3])
    return pd.Series(out, index=x.index)


def rvi(bars: pd.DataFrame, period: int = RVI_PERIOD) -> pd.DataFrame:
    """The Relative Vigor Index and its signal line for an OHLC frame.

    RVI(t) = sum_{period}(swma4(close-open)) / sum_{period}(swma4(high-low)); the signal line is
    swma4(RVI). Both are causal. Returns a frame with columns ``rvi`` and ``signal`` aligned to
    ``bars.index`` (NaN over the warm-up).
    """
    body = _swma4(bars["close"] - bars["open"])
    rng = _swma4(bars["high"] - bars["low"])
    num = body.rolling(period).sum()
    den = rng.rolling(period).sum()
    rv = num / den.replace(0.0, np.nan)
    sig = _swma4(rv)
    return pd.DataFrame({"rvi": rv, "signal": sig}, index=bars.index)


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def cross_up_entries(bars: pd.DataFrame, period: int = RVI_PERIOD) -> pd.DatetimeIndex:
    """Bars where the RVI crosses *from below to above* its signal line — the buy rule.

    The cross is read on the close of t (RVI[t] > signal[t] and RVI[t-1] <= signal[t-1]); the
    trade is executed at the next close by :func:`forward_returns`. NaN warm-up bars never fire.
    """
    ind = rvi(bars, period=period)
    rv, sig = ind["rvi"], ind["signal"]
    above = rv > sig
    below_prev = (rv.shift(1) <= sig.shift(1))
    valid = rv.notna() & sig.notna() & rv.shift(1).notna() & sig.shift(1).notna()
    cross = above & below_prev & valid
    return bars.index[cross.to_numpy()]


def random_entries(close: pd.Series, n: int, warmup: int = 30, seed: int = 0) -> pd.DatetimeIndex:
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


def phase_scramble_placebo(bars: pd.DataFrame, horizon: int, period: int = RVI_PERIOD,
                           n_draws: int = 1000, seed: int = 469) -> dict:
    """Placebo: circularly roll the RVI/signal series vs price, destroying the cross's timing.

    Keeps the RVI and signal-line *marginal* (every value, every cross is still there) but breaks
    the alignment between the indicator's crosses and price by a random circular shift, so the
    up-crosses now fire on geometrically meaningless dates. Returns the share of placebo runs whose
    mean cross-up forward return **beats** the real one — the honest "is the cross's timing adding
    anything?" p-value, plus the observed mean.
    """
    close = bars["close"]
    obs = float(np.mean(forward_returns(close, cross_up_entries(bars, period=period), horizon)))
    ind = rvi(bars, period=period)
    rv, sig = ind["rvi"], ind["signal"]
    valid_mask = rv.notna() & sig.notna()
    idx = bars.index
    n = len(idx)
    first_valid = int(np.argmax(valid_mask.to_numpy())) if valid_mask.any() else n
    span = n - first_valid
    if span < 30:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    rv_v = rv.to_numpy()
    sig_v = sig.to_numpy()
    beats = 0
    valid = 0
    for _ in range(n_draws):
        shift = int(rng.integers(period + 5, span - 5))
        rv_r = rv_v.copy()
        sig_r = sig_v.copy()
        # roll only the populated tail circularly, leaving the warm-up NaNs in place
        rv_r[first_valid:] = np.roll(rv_v[first_valid:], shift)
        sig_r[first_valid:] = np.roll(sig_v[first_valid:], shift)
        rv_s = pd.Series(rv_r, index=idx)
        sig_s = pd.Series(sig_r, index=idx)
        above = rv_s > sig_s
        below_prev = rv_s.shift(1) <= sig_s.shift(1)
        v = rv_s.notna() & sig_s.notna() & rv_s.shift(1).notna() & sig_s.shift(1).notna()
        cross = above & below_prev & v
        ent = idx[cross.to_numpy()]
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
def run_experiment(bars: pd.DataFrame, period: int = RVI_PERIOD, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: RVI cross-up vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the cross-up summary (gross + net), the drift-matched
    random-entry baseline, and the cross-minus-random delta.
    """
    close = bars["close"]
    ent = cross_up_entries(bars, period=period)
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
