"""Zero-Lag EMA (ZLEMA) as a falsifiable mechanical rule — Study 483.

A plain exponential moving average lags price: with smoothing length ``L`` it sits roughly
``(L-1)/2`` bars behind. John Ehlers' **zero-lag EMA** removes that lag by feeding the EMA a
*de-lagged* input instead of the raw close:

    lag = (L - 1) / 2
    delagged_t = close_t + (close_t - close_{t-lag})        # extrapolate the recent move
    ZLEMA_t    = EMA_L( delagged_t )                         # then smooth as usual

The ``(close_t - close_{t-lag})`` term is the momentum over the lag window; adding it back
*pushes the input forward* so the smoothed line catches up to price. The folklore (Ehlers, and
every "zero-lag" / "instantaneous trendline" write-up) is that this earlier, less-laggy line
turns a sluggish trend filter into a timely one: **go long while price sits above the ZLEMA** —
you enter the uptrend sooner and exit sooner — so it should beat a plain EMA of the same length.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Causal ZLEMA** — uses only past closes and a fixed integer ``lag``; nothing future leaks.
2. **The folklore filter, as a discrete entry** — the rule is "long while ``price > ZLEMA``".
   We sample that long-state at a fixed ``step`` spacing (so trades don't fully overlap) and
   measure the forward H-day return from each sampled in-state bar. Entry is at the **next**
   close (one documented lag). [We also expose the bare ZLEMA *upcross* — and it is a useful
   foil: the de-lag overshoots, so the cross whipsaws and the upcross rule banks far less than
   the steady ``price > ZLEMA`` filter even on a planted trend.]
3. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures
   the tape's drift; (b) the **plain-EMA head-to-head** — the exact same ``price > EMA`` filter
   on a plain EMA of equal length, the thing ZLEMA claims to improve on; and (c) a **de-lag
   placebo** that replaces the de-lag offset with a *permuted* offset series (same marginal,
   destroyed alignment), the honest "is the zero-lag correction actually doing anything?" null.

No look-ahead: the ZLEMA is causal, the ``price > ZLEMA`` state is read on the close of *t*, the
position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)
DEFAULT_LENGTH = 20
DEFAULT_STEP = 5          # spacing of in-state samples (so trades don't fully overlap)


# --------------------------------------------------------------------------- #
# Moving averages
# --------------------------------------------------------------------------- #
def ema(series: pd.Series, length: int) -> pd.Series:
    """Plain exponential moving average, ``alpha = 2/(length+1)`` (causal)."""
    return series.ewm(span=length, adjust=False).mean()


def zlema(close: pd.Series, length: int = DEFAULT_LENGTH) -> pd.Series:
    """Ehlers' zero-lag EMA: EMA of the de-lagged input ``close + (close - close[lag])``.

    ``lag = (length - 1) // 2`` (the EMA's nominal lag). The de-lag term extrapolates the recent
    move forward so the smoothed line catches up to price. Causal — only past closes are used.
    """
    lag = (length - 1) // 2
    delagged = close + (close - close.shift(lag))
    return ema(delagged, length)


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def _state_entries(close: pd.Series, line: pd.Series, step: int = DEFAULT_STEP) -> pd.DatetimeIndex:
    """In-state samples of the ``price > line`` long filter, every ``step`` bars (non-overlap).

    The folklore rule is "stay long while price is above the line". We turn that stateful filter
    into discrete trades by sampling every ``step``-th bar on which price closes above the line.
    Entry is executed at the next close by :func:`forward_returns`.
    """
    above = (close > line) & line.notna()
    idx = close.index[above.to_numpy()]
    return idx[::step]


def _upcross_entries(close: pd.Series, line: pd.Series) -> pd.DatetimeIndex:
    """Bars whose close first crosses *above* ``line`` (the onset of a price>line regime).

    Only the *first* bar of each consecutive above-run is kept. A useful foil for the steady
    filter: the ZLEMA cross whipsaws.
    """
    above = (close > line) & line.notna()
    cross = above & ~above.shift(1, fill_value=False)
    return close.index[cross.to_numpy()]


def zlema_entries(close: pd.Series, length: int = DEFAULT_LENGTH,
                  step: int = DEFAULT_STEP) -> pd.DatetimeIndex:
    """Long entries: sampled ``price > ZLEMA`` long-state — the 'zero-lag trend filter' rule."""
    return _state_entries(close, zlema(close, length), step=step)


def ema_entries(close: pd.Series, length: int = DEFAULT_LENGTH,
                step: int = DEFAULT_STEP) -> pd.DatetimeIndex:
    """Head-to-head baseline: the same ``price > EMA`` filter on a plain EMA of equal length."""
    return _state_entries(close, ema(close, length), step=step)


def zlema_upcross_entries(close: pd.Series, length: int = DEFAULT_LENGTH) -> pd.DatetimeIndex:
    """The bare ZLEMA upcross (a foil to show the de-lag makes the cross whippy)."""
    return _upcross_entries(close, zlema(close, length))


def random_entries(close: pd.Series, n: int, length: int = DEFAULT_LENGTH,
                   seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[2 * length:]
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


def delag_placebo(close: pd.Series, horizon: int, length: int = DEFAULT_LENGTH,
                  n_draws: int = 1000, seed: int = 483) -> dict:
    """Placebo: replace the de-lag offset with a *permuted* offset, destroying its alignment.

    ZLEMA = EMA(close + offset), with ``offset_t = close_t - close_{t-lag}`` the de-lag term
    that is the whole point of the indicator. We keep the offset's marginal distribution but
    permute *when* each offset value is applied, so the "zero-lag" correction no longer lines up
    with the price it was supposed to extrapolate. If ZLEMA's edge really comes from the de-lag,
    the real entries should beat a scrambled-offset version far in the right tail. Returns the
    share of placebo runs whose mean entry return **beats** the real one (the honest p-value),
    plus the observed mean.
    """
    obs = float(np.mean(forward_returns(close, zlema_entries(close, length), horizon)))
    lag = (length - 1) // 2
    offset = (close - close.shift(lag)).to_numpy(dtype=float)
    valid_mask = np.isfinite(offset)
    base = close.to_numpy(dtype=float)
    idx = close.index
    rng = np.random.default_rng(seed)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm_off = offset.copy()
        v = np.where(valid_mask)[0]
        perm_off[v] = rng.permutation(offset[v])
        delagged = pd.Series(base + np.nan_to_num(perm_off, nan=0.0), index=idx)
        line = ema(delagged, length)
        ent = _state_entries(close, line)
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
def run_experiment(close: pd.Series, length: int = DEFAULT_LENGTH, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: ZLEMA upcross vs random + plain-EMA, all horizons.

    Returns a dict keyed by horizon with the ZLEMA-entry summary (gross + net), the
    drift-matched random-entry baseline, the plain-EMA head-to-head, and the ZLEMA-minus-random
    delta.
    """
    ent = zlema_entries(close, length=length)
    ent_ema = ema_entries(close, length=length)
    res = {"n_entries": int(len(ent)), "n_entries_ema": int(len(ent_ema)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        emah = summarize(forward_returns(close, ent_ema, h, cost_bps=0.0))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), length=length, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "ema": emah, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
            "delta_ema_bps": (g["mean_bps"] - emah["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(emah["mean_bps"]) else float("nan"),
        }
    return res
