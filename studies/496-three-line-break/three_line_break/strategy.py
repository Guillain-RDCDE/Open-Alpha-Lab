"""Three-Line-Break as a falsifiable mechanical rule — Study 496.

The Three-Line-Break (TLB) chart (Japanese *Sakata* lineage, popularised in the West by
Steve Nison and Steve Achelis) ignores the calendar and draws a new **line** (a block) only
when the close pushes past the extreme of the prior line:

* in an **up** column, a close above the prior line's top draws a new up-line;
* the chart **reverses** to a down-line only when the close falls below the **lowest low of
  the 3 most-recent up-lines** (and symmetrically for a down→up reversal). The "3" is the
  break number — hence *Three*-Line-Break.

The folklore (Nison, *Beyond Candlesticks*; Achelis, *Technical Analysis A to Z*): a TLB
**reversal forecasts a new trend**. Trade it: go **long on an up-line**, **flat on a
reversal** (or short, but we test the long-flat market-timing version the claim describes),
and beat buy-and-hold.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Causal TLB construction** — lines are emitted bar-by-bar from *past* closes only; a
   reversal is a function of the prior lines' extremes, so nothing leaks from the future.
2. **Reversal entry** — a long fires on the bar whose close *flips the TLB from down to up*
   (the bullish 3-line break). Entry is at the **next** close (one documented lag); we then
   measure the forward H-day return.
3. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures
   the tape's drift, and (b) a **shuffled-returns placebo** that rebuilds TLB on a permutation
   of the daily returns, destroying the specific line-break *sequence* while keeping the price
   marginal — the honest "is the line-break geometry doing anything?" null.

No look-ahead: the reversal is read on the close of *t*, the position is entered at the close
of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Three-Line-Break construction (causal)
# --------------------------------------------------------------------------- #
def build_tlb(close: pd.Series, n_lines: int = 3) -> pd.DataFrame:
    """Causal Three-Line-Break state for each bar of ``close``.

    Walks the closes forward, emitting TLB lines. Returns a DataFrame aligned to ``close.index``
    with columns:

    * ``color`` — current TLB colour after this bar's close (+1 up, -1 down, 0 warm-up);
    * ``up_rev`` — True on the bar whose close flips the chart **down→up** (bullish reversal);
    * ``dn_rev`` — True on the bar whose close flips the chart **up→down** (bearish reversal).

    Reversals require breaking the extreme of the last ``n_lines`` opposite lines (the "3").
    Everything is a function of closes up to and including the current bar — no look-ahead.
    """
    c = close.to_numpy(dtype=float)
    n = c.size
    color = np.zeros(n, dtype=int)
    up_rev = np.zeros(n, dtype=bool)
    dn_rev = np.zeros(n, dtype=bool)

    # each line is (top, bottom); colour is the sign of the latest move
    lines: list[tuple[float, float]] = []
    col = 0
    for i in range(n):
        px = c[i]
        if not lines:
            lines.append((px, px))
            col = 0
        else:
            cur_top, cur_bot = lines[-1]
            if px > cur_top:                     # potential up extension / reversal
                if col < 0:                      # currently down: need a real reversal
                    ref = max(t for t, _ in lines[-n_lines:])
                    if px > ref:
                        up_rev[i] = True
                        col = +1
                        lines.append((px, cur_top))
                    # else: not enough to flip; chart unchanged
                else:
                    col = +1
                    lines.append((px, cur_top))
            elif px < cur_bot:                   # potential down extension / reversal
                if col > 0:                      # currently up: need a real reversal
                    ref = min(b for _, b in lines[-n_lines:])
                    if px < ref:
                        dn_rev[i] = True
                        col = -1
                        lines.append((cur_bot, px))
                else:
                    col = -1
                    lines.append((cur_bot, px))
            # else: inside the prior line -> no new line, state unchanged
        color[i] = col

    return pd.DataFrame(
        {"color": color, "up_rev": up_rev, "dn_rev": dn_rev}, index=close.index
    )


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def reversal_entries(close: pd.Series, n_lines: int = 3) -> pd.DatetimeIndex:
    """Bars whose close flips the TLB **down→up** — the bullish 3-line-break reversal.

    These are the "buy the reversal / go long the new up-trend" signals. Entry is executed at
    the next close by :func:`forward_returns`.
    """
    tlb = build_tlb(close, n_lines=n_lines)
    return close.index[tlb["up_rev"].to_numpy()]


def random_entries(close: pd.Series, n: int, n_lines: int = 3, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after a short warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    warm = 10
    valid = close.index[warm:]
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


def shuffled_returns_placebo(close: pd.Series, horizon: int, n_lines: int = 3,
                             n_draws: int = 1000, seed: int = 496) -> dict:
    """Placebo: rebuild TLB on a permutation of the daily returns, destroying the break sequence.

    Keeps the price **marginal** (the same set of daily log-returns, hence the same start/end
    drift and volatility) but permutes their *order*, so the specific line-break geometry — which
    runs get strung together into a reversal — becomes meaningless. Returns the share of placebo
    runs whose mean up-reversal forward return **beats** the real one (the honest "is the
    line-break geometry adding anything?" p-value), plus the observed mean.
    """
    obs = float(np.mean(forward_returns(close, reversal_entries(close, n_lines=n_lines), horizon)))
    logret = np.diff(np.log(close.to_numpy(dtype=float)))
    p0 = float(close.iloc[0])
    idx = close.index
    rng = np.random.default_rng(seed)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = rng.permutation(logret)
        path = p0 * np.exp(np.concatenate([[0.0], np.cumsum(perm)]))
        cser = pd.Series(path, index=idx)
        ent = reversal_entries(cser, n_lines=n_lines)
        rr = forward_returns(cser, ent, horizon)
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
def run_experiment(close: pd.Series, n_lines: int = 3, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: TLB reversal vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the reversal summary (gross + net), the drift-matched
    random-entry baseline, and the reversal-minus-random delta.
    """
    ent = reversal_entries(close, n_lines=n_lines)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), n_lines=n_lines, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
