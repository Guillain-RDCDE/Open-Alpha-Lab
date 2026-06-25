"""Bullish Percent Index as a falsifiable mechanical rule — Study 494.

The **Bullish Percent Index** (BPI), introduced by Abe Cohen / A.W. Cohen at Chartcraft in the
1950s, is a *breadth* oscillator: the percentage of a basket whose members sit on a Point &
Figure (P&F) **buy** signal. It ranges 0-100. The folklore (Cohen's own teaching, restated by
StockCharts' ChartSchool and every market-breadth write-up):

* BPI **above 70** = "overbought" — too many members already bullish, a market *top* warning;
* BPI **below 30** = "oversold" — breadth washed out, a market *bottom* and a high-probability
  **buy** of the index;
* the *reversal* up out of oversold (or the cross back below 70 from above) is the trade.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **BPI proxy.** True P&F-buy-signal counting needs box/reversal bookkeeping per member; the
   standard, transparent desk proxy is the **percentage of the basket trading above its
   moving average** (default 50-day SMA). This is a coarse stand-in for true exchange breadth
   and caps the test — stated plainly in the docs.
2. **Oversold-buy rule.** A long fires on the bar where BPI **crosses up through** the oversold
   threshold (default 30) after having been below it — Cohen's "bull alert / reversal into a
   column of X's". We also report the simpler "BPI < 30" level rule. Entry is at the **next**
   close (one documented lag); we measure the forward H-day return on SPY.
3. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures
   the tape's drift, and (b) a **scrambled-breadth placebo** that block-shuffles the BPI series
   in time, destroying its alignment with price while keeping its marginal — the honest "is the
   breadth timing doing anything?" null.

No look-ahead: every member's "above-MA" vote uses only data up to bar *t* (the SMA is causal),
the BPI cross is read on the close of *t*, and the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)
DEFAULT_MA = 50
OVERSOLD = 30.0
OVERBOUGHT = 70.0


# --------------------------------------------------------------------------- #
# Breadth oscillator (the BPI proxy)
# --------------------------------------------------------------------------- #
def bpi(members_close: pd.DataFrame, ma_win: int = DEFAULT_MA) -> pd.Series:
    """Bullish Percent Index proxy: % of basket members trading above their ``ma_win``-day SMA.

    Each member casts a causal vote (close_t > SMA_t, where the SMA uses only data through t).
    BPI_t is the mean vote in [0, 100]. NaN until the SMA window is full. No look-ahead: every
    vote uses only information available at bar t.
    """
    ma = members_close.rolling(ma_win, min_periods=ma_win).mean()
    above = (members_close > ma).astype(float)
    # only count rows where every member has a defined SMA vote
    valid = ma.notna().all(axis=1)
    pct = above.mean(axis=1) * 100.0
    pct[~valid] = np.nan
    return pct.rename("bpi")


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def oversold_cross_entries(bpi_series: pd.Series, oversold: float = OVERSOLD) -> pd.DatetimeIndex:
    """Bars where BPI **crosses up** through ``oversold`` (was below, now at/above).

    This is Cohen's "reversal up out of oversold" — the canonical BPI buy. Only the crossing
    bar is kept (read on the close of t); entry is executed at the next close by
    :func:`forward_returns`. No look-ahead (the cross uses BPI_{t-1} and BPI_t only).
    """
    b = bpi_series
    prev = b.shift(1)
    cross = (prev < oversold) & (b >= oversold)
    cross = cross & b.notna() & prev.notna()
    return b.index[cross.to_numpy()]


def oversold_level_entries(bpi_series: pd.Series, oversold: float = OVERSOLD) -> pd.DatetimeIndex:
    """Bars where BPI is below ``oversold`` for the *first* time of a run — the level rule.

    Reported alongside the cross rule as the simpler "buy while oversold" interpretation.
    """
    b = bpi_series
    mask = (b < oversold) & b.notna()
    first = mask & ~mask.shift(1, fill_value=False)
    return b.index[first.to_numpy()]


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


def scrambled_breadth_placebo(close: pd.Series, bpi_series: pd.Series, horizon: int,
                              oversold: float = OVERSOLD, block: int = 21,
                              n_draws: int = 1000, seed: int = 494) -> dict:
    """Placebo: block-shuffle the BPI series in time, destroying its alignment with price.

    Keeps the BPI *marginal* (the same distribution of breadth levels, the same number of
    oversold crosses on average) but breaks the *timing* link between breadth and the index, so
    the oversold cross fires on dates unrelated to the real breadth path. Returns the share of
    placebo runs whose mean oversold-cross forward return **beats** the real one — the honest
    "is the breadth timing adding anything?" p-value, plus the observed mean.
    """
    obs = float(np.mean(forward_returns(close, oversold_cross_entries(bpi_series, oversold), horizon)))
    b = bpi_series.dropna()
    if len(b) < block * 3:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    vals = b.to_numpy()
    idx = b.index
    n = len(vals)
    n_blocks = int(np.ceil(n / block))
    beats = 0
    valid = 0
    for _ in range(n_draws):
        order = rng.permutation(n_blocks)
        chunks = [vals[k * block:(k + 1) * block] for k in order]
        scr = np.concatenate(chunks)[:n]
        scr_ser = pd.Series(scr, index=idx)
        ent = oversold_cross_entries(scr_ser, oversold)
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
def run_experiment(close: pd.Series, bpi_series: pd.Series, oversold: float = OVERSOLD,
                   cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: oversold-cross vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the oversold-cross summary (gross + net), the
    drift-matched random-entry baseline, and the cross-minus-random delta.
    """
    ent = oversold_cross_entries(bpi_series, oversold)
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
