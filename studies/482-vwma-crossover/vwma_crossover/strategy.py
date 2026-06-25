"""VWMA-Crossover as a falsifiable mechanical rule — Study 482.

A **volume-weighted moving average** weights each bar's price by its volume:

    VWMA_N(t) = sum_{i=t-N+1..t} price_i * vol_i / sum_{i=t-N+1..t} vol_i

The folklore (every charting suite, countless YouTube setups): a VWMA crossover "front-runs"
a plain-SMA crossover because it leans toward the bars where *real money traded*, so a VWMA
golden cross (fast VWMA above slow VWMA) is a better long trigger than the equal-weighted SMA
golden cross.

We encode the tightest mechanical version and test it head-to-head:

1. **VWMA cross.** Long fires on the bar where ``VWMA_fast`` crosses *above* ``VWMA_slow``
   (a golden cross). Entry is the **next** close (one documented lag); we measure the forward
   H-day return.
2. **Plain-SMA cross (the incremental-value baseline).** The *same-length* equal-weighted SMA
   golden cross. The third-axis question — "does volume-weighting add edge?" — is answered by
   VWMA-minus-SMA, not by VWMA-vs-zero.
3. **Random-entry baseline.** Same instrument, epoch and hold, on random days — captures the
   tape's unconditional drift (the only honest Signal test on an up-drifting tape).
4. **Shuffled-volume placebo.** Recompute the VWMA with the volume series **permuted**
   (price path kept, volume marginal kept) so the weighting is destroyed while the marginals
   survive — the direct "is the volume term doing anything?" null.

No look-ahead: the moving averages are causal (trailing windows only); the cross is detected on
the close of *t*; the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)
FAST, SLOW = 10, 30


# --------------------------------------------------------------------------- #
# Moving averages
# --------------------------------------------------------------------------- #
def sma(price: pd.Series, n: int) -> pd.Series:
    """Causal simple (equal-weighted) moving average over a trailing window of ``n`` bars."""
    return price.rolling(n, min_periods=n).mean()


def vwma(price: pd.Series, volume: pd.Series, n: int) -> pd.Series:
    """Causal volume-weighted moving average: sum(price*vol)/sum(vol) over trailing ``n`` bars.

    Falls back to the equal-weighted mean on any window whose volume sums to zero (defensive;
    real tapes have positive volume). Trailing windows only — no look-ahead.
    """
    pv = (price * volume).rolling(n, min_periods=n).sum()
    v = volume.rolling(n, min_periods=n).sum()
    out = pv / v.replace(0.0, np.nan)
    # if a window had zero total volume, fall back to the plain SMA there
    return out.where(v > 0, sma(price, n))


# --------------------------------------------------------------------------- #
# Crossover entries
# --------------------------------------------------------------------------- #
def _golden_cross_dates(fast_ma: pd.Series, slow_ma: pd.Series) -> pd.DatetimeIndex:
    """Bars where ``fast_ma`` crosses strictly *above* ``slow_ma`` (a golden cross).

    Read on the close of the cross bar; :func:`forward_returns` enters at the next close.
    """
    above = fast_ma > slow_ma
    prev = above.shift(1, fill_value=False)
    cross = above & (~prev) & fast_ma.notna() & slow_ma.notna() & slow_ma.shift(1).notna()
    return fast_ma.index[cross.to_numpy()]


def vwma_cross_entries(price: pd.Series, volume: pd.Series,
                       fast: int = FAST, slow: int = SLOW) -> pd.DatetimeIndex:
    """Golden-cross dates of the **volume-weighted** MAs (fast VWMA crosses above slow VWMA)."""
    return _golden_cross_dates(vwma(price, volume, fast), vwma(price, volume, slow))


def sma_cross_entries(price: pd.Series, fast: int = FAST, slow: int = SLOW) -> pd.DatetimeIndex:
    """Golden-cross dates of the **plain SMAs** — the same-length incremental-value baseline."""
    return _golden_cross_dates(sma(price, fast), sma(price, slow))


def random_entries(price: pd.Series, n: int, warmup: int = SLOW, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = price.index[warmup:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(price: pd.Series, entries, horizon: int, cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    ``cost_bps`` is a one-way cost (charged twice: in + out) subtracted from each trade's
    return. Trades whose window overruns the tape are dropped.
    """
    pos = {d: i for i, d in enumerate(price.index)}
    p = price.to_numpy(dtype=float)
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


def shuffled_volume_placebo(price: pd.Series, volume: pd.Series, horizon: int,
                            fast: int = FAST, slow: int = SLOW,
                            n_draws: int = 1000, seed: int = 482) -> dict:
    """Placebo: recompute the VWMA cross with the **volume series permuted**.

    Keeps the price path and the volume *marginal* but scrambles which volume attaches to which
    bar, so the volume weighting becomes meaningless (it is no longer aligned with price).
    Returns the share of placebo runs whose mean VWMA-cross forward return **beats** the real
    one — the honest "is the volume term load-bearing?" p-value, plus the observed mean.
    """
    obs = float(np.mean(forward_returns(
        price, vwma_cross_entries(price, volume, fast, slow), horizon)))
    rng = np.random.default_rng(seed)
    v = volume.to_numpy(dtype=float)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        vp = pd.Series(rng.permutation(v), index=volume.index)
        ent = vwma_cross_entries(price, vp, fast, slow)
        rr = forward_returns(price, ent, horizon)
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
def run_experiment(bars: pd.DataFrame, fast: int = FAST, slow: int = SLOW,
                   cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one OHLCV tape: VWMA cross vs SMA cross vs random, all horizons.

    Returns a dict keyed by horizon with the VWMA-cross summary (gross + net), the same-length
    plain-SMA cross (the incremental-value baseline), the drift-matched random-entry baseline,
    and the VWMA-minus-SMA and VWMA-minus-random deltas.
    """
    price = bars["close"]
    volume = bars["volume"]
    ev = vwma_cross_entries(price, volume, fast, slow)
    es = sma_cross_entries(price, fast, slow)
    res = {"n_vwma": int(len(ev)), "n_sma": int(len(es)), "by_h": {}}
    for h in HORIZONS:
        gv = summarize(forward_returns(price, ev, h, cost_bps=0.0))
        nv = summarize(forward_returns(price, ev, h, cost_bps=cost_bps))
        gs = summarize(forward_returns(price, es, h, cost_bps=0.0))
        rnd = summarize(forward_returns(
            price, random_entries(price, max(len(ev), 50), warmup=slow, seed=random_seed), h))
        res["by_h"][h] = {
            "vwma": gv, "vwma_net": nv, "sma": gs, "random": rnd,
            "delta_vs_sma": (gv["mean_bps"] - gs["mean_bps"])
            if np.isfinite(gv["mean_bps"]) and np.isfinite(gs["mean_bps"]) else float("nan"),
            "delta_vs_random": (gv["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(gv["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
