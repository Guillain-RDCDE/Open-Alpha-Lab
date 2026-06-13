"""The strategy and its honest controls — Study 91 (Death-Cross).

The folk claim: 'when the 50-day average crosses *below* the 200-day average — the
"death cross" — get out; re-enter on the "golden cross". You dodge the crashes and beat
buy-and-hold.' We implement the crossover timer and pin it against the two comparisons
that decide whether it is anything but lower exposure:

- **Buy-and-hold** — the thing it claims to beat, on a total-return tape.
- **Matched random-timing control** — a coin that is *out of the market on the same
  number of days, in runs of the same length*, but on **random** dates. If the crossover
  carries no forecasting information beyond "be in cash sometimes", the real timer should
  not beat this coin on a risk-adjusted basis.

The execution convention is the desk's: the cross is known at the close of day *t*, so
the position changes at *t+1* — one ``shift``, applied once. Cash earns **0%** (a stated,
conservative choice that biases *against* the timer; giving it the T-bill would only
flatter it). Switching costs are charged per round-trip in/out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Indicators & signal
# ---------------------------------------------------------------------------
def sma(close: pd.Series, n: int) -> pd.Series:
    """Simple moving average over ``n`` sessions (min_periods=n, so no partial windows)."""
    return close.rolling(n, min_periods=n).mean()


def crossover_position(
    close: pd.Series, fast: int = 50, slow: int = 200, lag: int = 1
) -> pd.Series:
    """Long-or-flat position from the SMA(fast)/SMA(slow) crossover.

    Position is **1** (long) while SMA(fast) >= SMA(slow), else **0** (cash). The raw
    signal is known at close *t*; the returned position is shifted forward by ``lag``
    days so it is acted on at *t+1*'s return — one lag, applied once. NaN warm-up days
    (before SMA(slow) is defined) are flat.
    """
    f = sma(close, fast)
    s = sma(close, slow)
    raw = (f >= s).astype(float)
    raw[f.isna() | s.isna()] = 0.0
    return raw.shift(lag).fillna(0.0)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def backtest(
    close: pd.Series,
    position: pd.Series,
    cost_bps: float = 5.0,
) -> dict:
    """Run a long-or-flat position over ``close`` and return net stats + the daily series.

    ``cost_bps`` is charged (one-way) on every change in position — entering or leaving
    the market each costs ``cost_bps``. Cash earns 0%. Returns a dict with the net daily
    return series and headline stats (CAGR, vol, Sharpe, max drawdown, switches,
    time-in-market).
    """
    ret = close.pct_change().fillna(0.0)
    pos = position.reindex(close.index).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())
    net = pos * ret - turn * cost_bps * 1e-4
    return _stats(net, pos, turn)


def buy_and_hold(close: pd.Series) -> dict:
    """Buy-and-hold the tape: position 1 every day, no costs after entry."""
    ret = close.pct_change().fillna(0.0)
    pos = pd.Series(1.0, index=close.index)
    return _stats(ret, pos, pd.Series(0.0, index=close.index))


def _stats(net: pd.Series, pos: pd.Series, turn: pd.Series) -> dict:
    net = net.astype(float)
    equity = (1.0 + net).cumprod()
    n = len(net)
    years = n / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(TRADING_DAYS)) if net.std() > 0 else float("nan")
    return {
        "net": net,
        "equity": equity,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": _max_drawdown(equity.to_numpy()),
        "switches": int((turn > 1e-9).sum()),
        "time_in_market": float((pos > 0).mean()),
        "final": float(equity.iloc[-1]),
    }


# ---------------------------------------------------------------------------
# Matched random-timing control
# ---------------------------------------------------------------------------
def _run_lengths(pos: np.ndarray) -> list[tuple[int, int]]:
    """Return (value, length) runs of a 0/1 position vector."""
    runs = []
    if len(pos) == 0:
        return runs
    cur, length = pos[0], 1
    for v in pos[1:]:
        if v == cur:
            length += 1
        else:
            runs.append((int(cur), length))
            cur, length = v, 1
    runs.append((int(cur), length))
    return runs


def matched_random_position(real_pos: pd.Series, seed: int = 0) -> pd.Series:
    """A random-timing coin that matches the real timer's *exposure profile*.

    It reproduces the exact multiset of in-market and out-of-market run *lengths* the
    real death-cross timer produced, but reshuffles their order — so it holds the same
    total time in cash, in similarly-sized blocks, at **random** points in history. If
    the crossover dates carry no information, the real timer should not beat this.
    """
    rng = np.random.default_rng(seed)
    p = real_pos.to_numpy()
    runs = _run_lengths(p)
    in_runs = [length for v, length in runs if v == 1]
    out_runs = [length for v, length in runs if v == 0]
    rng.shuffle(in_runs)
    rng.shuffle(out_runs)
    # Rebuild alternating, starting with whichever the real series started with.
    start_val = int(p[0])
    seq = []
    a = in_runs if start_val == 1 else out_runs
    b = out_runs if start_val == 1 else in_runs
    va, vb = start_val, 1 - start_val
    i = j = 0
    toggle = True
    while i < len(a) or j < len(b):
        if toggle and i < len(a):
            seq.extend([va] * a[i]); i += 1
        elif not toggle and j < len(b):
            seq.extend([vb] * b[j]); j += 1
        toggle = not toggle
    seq = (seq + [0] * len(p))[: len(p)]
    return pd.Series(seq, index=real_pos.index, dtype=float)


# ---------------------------------------------------------------------------
# Inference: HAC t-stat on a return-difference series
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat for the mean of ``x`` (local, no quantlab dep)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 5:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")
