"""FRAMA — the Fractal Adaptive Moving Average — as a falsifiable mechanical rule (Study 488).

John Ehlers' FRAMA (*Stocks & Commodities*, 2005) is a moving average whose smoothing constant
adapts to the **fractal dimension** of recent price. Over an ``N``-bar window (N even):

* split the window into two halves; let ``H1, L1`` be the high/low range of the older half,
  ``H2, L2`` the newer half, and ``H, L`` the range of the whole window;
* ``N1 = (H1-L1)/(N/2)``, ``N2 = (H2-L2)/(N/2)``, ``N3 = (H-L)/N`` are the per-bar amplitudes;
* the **fractal dimension** is ``D = (log(N1+N2) - log(N3)) / log(2)``, clipped to ``[1, 2]``;
* the adaptive smoothing constant is ``alpha = exp(-4.6 * (D - 1))``, clipped to ``[0.01, 1]``;
* ``FRAMA_t = alpha_t * price_t + (1 - alpha_t) * FRAMA_{t-1}`` (a strictly causal recursion).

When price moves in a clean trend, ``D -> 1`` so ``alpha -> 1`` and FRAMA hugs price (fast);
in choppy sideways action ``D -> 2`` so ``alpha -> exp(-4.6) ~ 0.01`` and FRAMA flattens (slow).
The folklore (Ehlers + every charting forum): this *fractal-adaptive smoothing* lets a
``price > FRAMA`` (or FRAMA-cross-up) long **catch trends sooner and dodge whipsaws**, beating a
plain fixed-length EMA. The third-axis question: **does fractal-adaptive smoothing buy edge?**

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Causal FRAMA.** The recursion above; each bar uses only bars up to ``t`` (no look-ahead).
2. **The rule.** A long is *on* whenever the close is above FRAMA. We trade the **cross-up**
   (the bar the close first crosses above FRAMA, the entry signal), read on the close of *t*
   and entered at the **next close** (one documented lag). We then measure the forward H-day
   return.
3. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) capturing the
   tape's drift; (b) a **fixed-EMA** comparator — the same cross-up rule on a plain EMA with the
   same average smoothing — the "does the *adaptive* part add anything over a static MA?" test;
   (c) a **shuffled-alpha placebo** that permutes the per-bar adaptive smoothing constants in
   time (destroying the fractal-dimension link) while keeping the alpha marginal — the honest
   "is the fractal adaptation load-bearing?" null.

No look-ahead: the FRAMA is strictly causal, the cross is read on the close of *t*, the position
is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# FRAMA + fixed EMA
# --------------------------------------------------------------------------- #
def fractal_dimension(high: pd.Series, low: pd.Series, n: int = 16) -> pd.Series:
    """Per-bar fractal dimension D in [1, 2] over a trailing N-bar window (Ehlers).

    Uses only bars up to ``t`` (rolling, right-aligned). NaN until the first full window.
    """
    if n % 2 != 0:
        n += 1
    half = n // 2
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    m = h.size
    D = np.full(m, np.nan)
    for t in range(n - 1, m):
        w_h = h[t - n + 1:t + 1]
        w_l = l[t - n + 1:t + 1]
        h1 = w_h[:half].max(); l1 = w_l[:half].min()
        h2 = w_h[half:].max(); l2 = w_l[half:].min()
        hh = w_h.max(); ll = w_l.min()
        n1 = (h1 - l1) / half
        n2 = (h2 - l2) / half
        n3 = (hh - ll) / n
        if n1 > 0 and n2 > 0 and n3 > 0:
            d = (np.log(n1 + n2) - np.log(n3)) / np.log(2.0)
            D[t] = min(2.0, max(1.0, d))
        else:
            D[t] = 1.0
    return pd.Series(D, index=high.index)


def frama(bars: pd.DataFrame, n: int = 16) -> pd.Series:
    """The Fractal Adaptive Moving Average of the close, strictly causal.

    ``bars`` needs high/low/close. Returns a Series aligned to the close; the recursion is seeded
    with the close at the first full window and runs forward with the adaptive alpha.
    """
    close = bars["close"].astype(float)
    D = fractal_dimension(bars["high"], bars["low"], n=n)
    alpha = np.exp(-4.6 * (D.to_numpy() - 1.0))
    alpha = np.clip(alpha, 0.01, 1.0)
    c = close.to_numpy(dtype=float)
    out = np.full(c.size, np.nan)
    prev = None
    for t in range(c.size):
        a = alpha[t]
        if not np.isfinite(a):
            continue
        if prev is None:
            prev = c[t]            # seed at the first usable bar
        else:
            prev = a * c[t] + (1.0 - a) * prev
        out[t] = prev
    return pd.Series(out, index=close.index)


def frama_alpha(bars: pd.DataFrame, n: int = 16) -> pd.Series:
    """The per-bar adaptive smoothing constant alpha_t used by :func:`frama` (for the placebo)."""
    D = fractal_dimension(bars["high"], bars["low"], n=n)
    alpha = np.exp(-4.6 * (D.to_numpy() - 1.0))
    alpha = np.clip(alpha, 0.01, 1.0)
    return pd.Series(alpha, index=bars.index)


def ema_from_alpha(close: pd.Series, alpha: np.ndarray) -> pd.Series:
    """A causal EMA of ``close`` driven by an arbitrary per-bar ``alpha`` array (NaN-skipping)."""
    c = close.to_numpy(dtype=float)
    out = np.full(c.size, np.nan)
    prev = None
    for t in range(c.size):
        a = alpha[t]
        if not np.isfinite(a):
            continue
        prev = c[t] if prev is None else a * c[t] + (1.0 - a) * prev
        out[t] = prev
    return pd.Series(out, index=close.index)


def fixed_ema(bars: pd.DataFrame, n: int = 16) -> pd.Series:
    """A plain fixed-length EMA comparator with the *same warm-up* as FRAMA.

    The smoothing constant is the FRAMA average alpha (so the two averages share an effective
    length); this isolates the *adaptive* part of FRAMA from a static MA of equal speed.
    """
    al = frama_alpha(bars, n=n).to_numpy()
    a_bar = np.nanmean(al)
    alpha = np.where(np.isfinite(al), a_bar, np.nan)  # same warm-up window, constant alpha
    return ema_from_alpha(bars["close"].astype(float), alpha)


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def _cross_up_entries(close: pd.Series, line: pd.Series) -> pd.DatetimeIndex:
    """Bars whose close first crosses *above* ``line`` (was <=, now >). The cross-up signal."""
    above = (close > line) & line.notna()
    first = above & ~above.shift(1, fill_value=False)
    return close.index[first.to_numpy()]


def frama_cross_entries(bars: pd.DataFrame, n: int = 16) -> pd.DatetimeIndex:
    """Long entries: the close crosses up through FRAMA (the Ehlers trend-following signal).

    Read on the close of *t*; executed at the next close by :func:`forward_returns`.
    """
    return _cross_up_entries(bars["close"].astype(float), frama(bars, n=n))


def ema_cross_entries(bars: pd.DataFrame, n: int = 16) -> pd.DatetimeIndex:
    """The same cross-up rule on the fixed-EMA comparator (the 'is adaptation worth it?' control)."""
    return _cross_up_entries(bars["close"].astype(float), fixed_ema(bars, n=n))


def random_entries(close: pd.Series, n_draws: int, n: int = 16, seed: int = 0) -> pd.DatetimeIndex:
    """``n_draws`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[2 * n:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n_draws, len(valid)), replace=False)
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


def shuffled_alpha_placebo(bars: pd.DataFrame, horizon: int, n: int = 16,
                           n_draws: int = 500, seed: int = 488) -> dict:
    """Placebo: permute the per-bar adaptive alpha in time, destroying the fractal-dimension link.

    Keeps the alpha *marginal* (the same set of smoothing constants is used) but shuffles which
    alpha lands on which bar, so the adaptation no longer tracks the local fractal dimension.
    Returns the share of placebo runs whose mean cross-up forward return **beats** the real
    FRAMA's — the honest "is the fractal adaptation load-bearing?" p-value, plus the observed
    mean. If shuffling the adaptation doesn't hurt, the fractal part isn't doing the work.
    """
    close = bars["close"].astype(float)
    obs = float(np.mean(forward_returns(close, frama_cross_entries(bars, n=n), horizon)))
    al = frama_alpha(bars, n=n).to_numpy()
    finite = np.isfinite(al)
    base_alpha = al[finite]
    if base_alpha.size < 3 or not np.isfinite(obs):
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = al.copy()
        perm[finite] = rng.permutation(base_alpha)   # scramble alpha in time, keep marginal
        line = ema_from_alpha(close, perm)
        ent = _cross_up_entries(close, line)
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
def run_experiment(bars: pd.DataFrame, n: int = 16, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: FRAMA cross-up vs random + fixed-EMA, all horizons.

    Returns a dict keyed by horizon with the FRAMA cross-up summary (gross + net), the
    drift-matched random-entry baseline, the fixed-EMA comparator, and the deltas.
    """
    close = bars["close"].astype(float)
    ent = frama_cross_entries(bars, n=n)
    ema_ent = ema_cross_entries(bars, n=n)
    res = {"n_entries": int(len(ent)), "n_ema_entries": int(len(ema_ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), n=n, seed=random_seed), h))
        ema = summarize(forward_returns(close, ema_ent, h, cost_bps=0.0))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd, "ema": ema,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
            "delta_ema_bps": (g["mean_bps"] - ema["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(ema["mean_bps"]) else float("nan"),
        }
    return res
