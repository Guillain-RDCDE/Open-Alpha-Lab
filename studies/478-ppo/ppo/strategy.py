"""Percentage Price Oscillator (PPO) as a falsifiable mechanical rule — Study 478.

The **Percentage Price Oscillator** is the *normalized* cousin of MACD. Where MACD is the raw
spread between a fast and a slow EMA, the PPO divides that spread by the slow EMA and scales by
100, so it reads as a **percentage**:

    PPO_t   = 100 * (EMA_fast - EMA_slow) / EMA_slow         (fast=12, slow=26)
    signal_t = 9-period EMA of PPO

The folklore (the same lore that drives MACD, restated on every charting site): a **bullish
crossover** — the PPO line rising **above** its signal line — is a buy; the bearish crossover is
a sell. The normalization is *sold* as an upgrade: because PPO is a percentage it is comparable
across instruments and across time (a $400 SPY and a $40 GLD give comparable PPO readings, while
their raw MACDs are on wildly different scales).

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **EMAs** with the standard 12/26/9 parameters; the PPO and its signal are read on the close
   of bar *t* (a crossover needs *t* and *t-1*).
2. **Bullish crossover** — a long fires the first bar where ``PPO`` crosses strictly **above**
   ``signal``. Entry is at the **next** close (one documented lag); we then measure the forward
   H-day return.
3. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold) that
   captures the tape's drift, and (b) a **shuffled-sign placebo** that permutes the *direction*
   of the daily PPO-minus-signal differences (keeping their magnitudes/marginal), destroying the
   specific crossover structure while preserving how often crossings happen — the honest "is the
   normalized-crossover structure doing anything?" null.

The third-axis question is **does normalizing MACD add edge?** — so we also run the *raw* MACD
crossover (same 12/26/9, no division) on the same tape and compare deltas-vs-random.

No look-ahead: the oscillator is read on the close of *t*, the position is entered at the close
of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)

FAST, SLOW, SIGNAL = 12, 26, 9


# --------------------------------------------------------------------------- #
# Oscillator construction
# --------------------------------------------------------------------------- #
def ema(x: pd.Series, span: int) -> pd.Series:
    """Standard exponential moving average (adjust=False — the recursive textbook EMA)."""
    return x.ewm(span=span, adjust=False).mean()


def ppo_lines(close: pd.Series, fast: int = FAST, slow: int = SLOW,
              signal: int = SIGNAL) -> tuple[pd.Series, pd.Series]:
    """The PPO line and its signal line.

    ``PPO = 100 * (EMA_fast - EMA_slow) / EMA_slow`` and ``signal = EMA_signal(PPO)``.
    Returned aligned over ``close.index``.
    """
    ef = ema(close, fast)
    es = ema(close, slow)
    ppo = 100.0 * (ef - es) / es
    sig = ema(ppo, signal)
    return ppo, sig


def macd_lines(close: pd.Series, fast: int = FAST, slow: int = SLOW,
               signal: int = SIGNAL) -> tuple[pd.Series, pd.Series]:
    """The *raw* MACD line and its signal line (no normalization) — the thesis comparator.

    ``MACD = EMA_fast - EMA_slow`` and ``signal = EMA_signal(MACD)``.
    """
    ef = ema(close, fast)
    es = ema(close, slow)
    macd = ef - es
    sig = ema(macd, signal)
    return macd, sig


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def _cross_up_dates(line: pd.Series, sig: pd.Series, warmup: int) -> pd.DatetimeIndex:
    """Bars where ``line`` crosses strictly **above** ``sig`` (was <=, now >)."""
    diff = (line - sig)
    prev = diff.shift(1)
    cross = (prev <= 0.0) & (diff > 0.0) & diff.notna() & prev.notna()
    cross.iloc[:warmup] = False
    return line.index[cross.to_numpy()]


def ppo_cross_entries(close: pd.Series, fast: int = FAST, slow: int = SLOW,
                      signal: int = SIGNAL) -> pd.DatetimeIndex:
    """Bullish PPO crossovers — PPO rising above its signal line (the lore's buy).

    Entry is executed at the next close by :func:`forward_returns`.
    """
    ppo, sig = ppo_lines(close, fast, slow, signal)
    return _cross_up_dates(ppo, sig, warmup=slow + signal)


def macd_cross_entries(close: pd.Series, fast: int = FAST, slow: int = SLOW,
                       signal: int = SIGNAL) -> pd.DatetimeIndex:
    """Bullish *raw* MACD crossovers — the un-normalized comparator for the thesis axis."""
    macd, sig = macd_lines(close, fast, slow, signal)
    return _cross_up_dates(macd, sig, warmup=slow + signal)


def random_entries(close: pd.Series, n: int, warmup: int = SLOW + SIGNAL,
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


def shuffled_sign_placebo(close: pd.Series, horizon: int, fast: int = FAST, slow: int = SLOW,
                          signal: int = SIGNAL, n_draws: int = 1000, seed: int = 478) -> dict:
    """Placebo: permute the *sign* of the PPO-minus-signal differences, destroying the crossover.

    We keep the marginal distribution of |PPO - signal| (so crossovers happen about as often)
    but shuffle *which* bars sit above vs below the signal line, so the specific up-crossings
    become geometric nonsense. Returns the share of placebo runs whose mean crossover forward
    return **beats** the real one — the honest "is the normalized-crossover structure adding
    anything?" p-value, plus the observed mean.
    """
    obs = float(np.mean(forward_returns(close, ppo_cross_entries(close, fast, slow, signal),
                                        horizon)))
    ppo, sig = ppo_lines(close, fast, slow, signal)
    diff = (ppo - sig)
    warmup = slow + signal
    valid = diff.iloc[warmup:].dropna()
    if valid.empty:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    mags = valid.abs().to_numpy()
    idx = valid.index
    rng = np.random.default_rng(seed)
    beats = 0
    n_valid = 0
    for _ in range(n_draws):
        signs = rng.choice([-1.0, 1.0], size=mags.size)
        d = pd.Series(signs * mags, index=idx)
        prev = d.shift(1)
        cross = (prev <= 0.0) & (d > 0.0) & d.notna() & prev.notna()
        ent = idx[cross.to_numpy()]
        rr = forward_returns(close, ent, horizon)
        if rr.size == 0:
            continue
        n_valid += 1
        if rr.mean() >= obs:
            beats += 1
    p = (beats + 1) / (n_valid + 1) if n_valid else float("nan")
    return {"obs": obs, "p_value": float(p), "n_draws": n_valid}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(close: pd.Series, fast: int = FAST, slow: int = SLOW, signal: int = SIGNAL,
                   cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: PPO crossover vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the crossover summary (gross + net), the drift-matched
    random-entry baseline, the crossover-minus-random delta, and the *raw-MACD* comparator
    (for the 'does normalizing add edge?' axis).
    """
    ent = ppo_cross_entries(close, fast, slow, signal)
    macd_ent = macd_cross_entries(close, fast, slow, signal)
    res = {"n_entries": int(len(ent)), "n_macd": int(len(macd_ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), seed=random_seed), h))
        macd = summarize(forward_returns(close, macd_ent, h, cost_bps=0.0))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd, "macd": macd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
            "macd_delta_bps": (macd["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(macd["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
