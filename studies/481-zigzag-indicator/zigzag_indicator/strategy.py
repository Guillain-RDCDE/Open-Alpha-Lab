"""The ZigZag indicator as a falsifiable mechanical rule — Study 481.

The ZigZag is one of the oldest swing filters in technical analysis: starting from a pivot, it
walks forward and only marks a new swing when price **reverses by more than ``pct``** off the
running extreme, alternating swing lows and swing highs and drawing straight legs between them.
The folklore (every charting suite, from MetaTrader's standard ZigZag to TradingView) is that
the ZigZag *identifies turning points*: when the most recent leg turns **up** off a swing low,
that low was a tradable bottom — go long.

The fatal subtlety is **repaint**. The newest leg is *provisional*: the indicator draws it to
the latest extreme, but if price keeps going that line is erased and redrawn. A swing low at
bar ``t`` is only **confirmed** once price has rebounded ``pct`` above it — which happens at
some later bar ``c > t``. A naive "the last ZigZag pivot is a low, so buy" backtest reads the
*final, repainted* ZigZag and so peeks at the future. The honest test trades only **confirmed**
legs: we detect the confirmation at the bar where the rebound completes, read it on the close of
that bar ``t``, and enter at the **next** close ``t+1`` (one documented lag).

We encode the tightest mechanical version a proponent would accept:

1. **Threshold ZigZag** — swings are pivots separated by a ``pct``-percent reversal; consecutive
   same-kind candidate pivots are absorbed into the running extreme (standard ZigZag behaviour).
2. **Confirmation, not repaint** — a swing low is *confirmed* at the first bar whose close is
   ``pct`` above it. That bar is the up-leg-confirmed signal; nothing future is used.
3. **Confirmed up-leg long** — a long fires at the bar that confirms a swing low (the ZigZag
   just turned up), entered at the **next** close; we then measure the forward H-day return.
4. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures
   the tape's drift, and (b) a **shuffled-leg placebo** that keeps the confirmation *timing* but
   randomises whether each confirmed pivot is called a low or a high — destroying the ZigZag's
   "up-leg = buy" geometry while preserving the marginal — the honest "is the rule's structure
   load-bearing?" null.

No look-ahead: the confirmation carries the repaint lag, the up-leg is read on the close of
*t*, the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)
DEFAULT_PCT = 0.05  # 5% reversal threshold — the classic ZigZag default


# --------------------------------------------------------------------------- #
# ZigZag with explicit confirmation (no repaint)
# --------------------------------------------------------------------------- #
def zigzag_confirmations(close: pd.Series, pct: float = DEFAULT_PCT) -> pd.DataFrame:
    """Walk the close forward, emitting each swing pivot *at the bar it is confirmed*.

    Standard threshold ZigZag: we track a tentative extreme in the current direction and only
    declare a reversal (a confirmed pivot) when price retraces ``pct`` from that extreme. The
    pivot *price/date* is the extreme; the *confirm* date is the later bar where the retracement
    completes (the repaint lag). Returns a DataFrame with one row per confirmed pivot:

    - ``piv_pos`` : integer bar index of the extreme (the actual swing low/high),
    - ``confirm_pos`` : integer bar index at which it became known (>= piv_pos),
    - ``kind`` : +1 = swing high, -1 = swing low.

    The trading rule uses ``confirm_pos`` only — never ``piv_pos`` — so no future data leaks in.
    """
    p = close.to_numpy(dtype=float)
    n = p.size
    rows = []
    if n < 2:
        return pd.DataFrame(columns=["piv_pos", "confirm_pos", "kind"])

    # State machine. `direction` is the leg we are currently *in*:
    #   +1 = up-leg, tracking a running HIGH (a pct retrace down confirms that high);
    #   -1 = down-leg, tracking a running LOW  (a pct rebound up confirms that low);
    #    0 = undecided — track both extremes from the seed until the first pct move resolves it.
    direction = 0
    hi_pos, hi_val = 0, p[0]   # running high since the last confirmed low
    lo_pos, lo_val = 0, p[0]   # running low since the last confirmed high

    for i in range(1, n):
        c = p[i]
        if c > hi_val:
            hi_val, hi_pos = c, i
        if c < lo_val:
            lo_val, lo_pos = c, i

        if direction >= 0 and c <= hi_val * (1.0 - pct):
            # price retraced pct off the running high -> confirm a swing HIGH, flip to down-leg
            rows.append((hi_pos, i, +1))
            direction = -1
            lo_val, lo_pos = c, i      # start a fresh low-search from here
        elif direction <= 0 and c >= lo_val * (1.0 + pct):
            # price rebounded pct off the running low -> confirm a swing LOW, flip to up-leg
            rows.append((lo_pos, i, -1))
            direction = +1
            hi_val, hi_pos = c, i      # start a fresh high-search from here

    return pd.DataFrame(rows, columns=["piv_pos", "confirm_pos", "kind"])


def zigzag_line(close: pd.Series, pct: float = DEFAULT_PCT) -> pd.Series:
    """The *repainted* ZigZag line (extremes connected) — for plotting/illustration only.

    This is the line a charting package draws; it uses the final pivots and therefore is NOT
    safe to trade (it repaints). The trading rule lives in :func:`confirmed_uppleg_entries`,
    which keys off confirmation dates, not these extremes.
    """
    piv = zigzag_confirmations(close, pct=pct)
    idx = close.index
    out = pd.Series(np.nan, index=idx)
    if piv.empty:
        return out
    # anchor: first bar, then each pivot extreme, then last bar
    pts = [(0, float(close.iloc[0]))]
    for pos in piv["piv_pos"]:
        pts.append((int(pos), float(close.iloc[int(pos)])))
    pts.append((len(close) - 1, float(close.iloc[-1])))
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if x1 <= x0:
            continue
        xs = np.arange(x0, x1 + 1)
        ys = y0 + (y1 - y0) * (xs - x0) / (x1 - x0)
        out.iloc[x0:x1 + 1] = ys
    return out


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def confirmed_uppleg_entries(close: pd.Series, pct: float = DEFAULT_PCT) -> pd.DatetimeIndex:
    """Bars that **confirm a swing low** — the ZigZag has just turned up (the 'buy' signal).

    A confirmed swing low (``kind == -1``) means price has rebounded ``pct`` above a prior low:
    the new up-leg is now drawn and *will not repaint*. We read this on the close of the
    confirmation bar; :func:`forward_returns` enters at the next close (one lag).
    """
    piv = zigzag_confirmations(close, pct=pct)
    if piv.empty:
        return pd.DatetimeIndex([])
    lows = piv[piv["kind"] == -1]["confirm_pos"].to_numpy()
    lows = lows[lows < len(close)]
    return close.index[lows]


def random_entries(close: pd.Series, n: int, pct: float = DEFAULT_PCT,
                   seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after a short warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    warm = max(20, int(2 / max(pct, 1e-6)))
    valid = close.index[min(warm, len(close)):]
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


def shuffled_leg_placebo(close: pd.Series, horizon: int, pct: float = DEFAULT_PCT,
                         n_draws: int = 1000, seed: int = 481) -> dict:
    """Placebo: keep the confirmation *timing*, randomise which confirmations are 'lows'.

    The ZigZag's tradable claim is specifically that **up-legs** (confirmed lows) mark buys. We
    preserve the set of confirmation dates and the price marginal, but on each draw we relabel a
    random subset of confirmations as 'lows' (matching the real count) and treat *those* as the
    buy signals. This destroys the ZigZag's up/down geometry while keeping everything else —
    the honest "is the low-vs-high structure load-bearing, or is any confirmation date as good?"
    null. Returns the share of placebo runs whose mean entry return **beats** the real one.
    """
    piv = zigzag_confirmations(close, pct=pct)
    real = confirmed_uppleg_entries(close, pct=pct)
    obs = float(np.mean(forward_returns(close, real, horizon))) if len(real) else float("nan")
    if piv.empty or not np.isfinite(obs):
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    confirms = piv["confirm_pos"].to_numpy()
    confirms = confirms[confirms < len(close)]
    n_lows = int((piv["kind"] == -1).sum())
    n_lows = min(n_lows, len(confirms))
    idx = close.index
    rng = np.random.default_rng(seed)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        pick = rng.choice(len(confirms), size=n_lows, replace=False)
        ent = idx[np.sort(confirms[pick])]
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
def run_experiment(close: pd.Series, pct: float = DEFAULT_PCT, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: confirmed-up-leg vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the up-leg summary (gross + net), the drift-matched
    random-entry baseline, and the up-leg-minus-random delta.
    """
    ent = confirmed_uppleg_entries(close, pct=pct)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), pct=pct, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
