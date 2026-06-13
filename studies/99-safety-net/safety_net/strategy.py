"""The trailing-stop engine and its honest controls — Study 99 (Safety-Net).

The folk claim: *"Always use a stop-loss. Cut your losses and let your winners run.
A trailing stop — exit when price falls X% from its peak — protects your capital,
limits drawdown, and improves returns."* We implement a trailing stop on a long
position and pin it against the two comparisons that decide whether the *timing*
of the exit carries any information beyond "be in cash sometimes":

- **Buy-and-hold** — the thing the stop claims to protect *and* beat, total-return tape.
- **Matched random-exit control** — a coin that is *out of the market on the same
  number of days, in runs of the same lengths*, but on **random** dates. If the
  stop's exit *dates* carry no information beyond the exposure profile, the real
  stop should not beat this coin on a risk-adjusted basis.

The trailing-stop rule
----------------------
While long, track the running peak of the close. If the close falls more than ``X``%
below that peak, the stop *fires*: we exit to cash (earning **0%**, a stated,
conservative choice). Re-entry rule: stay in cash for a fixed ``cooldown`` of trading
days, then re-enter and reset the trailing peak to the re-entry price. (A cooldown is
the simplest honest re-entry rule; "re-enter the instant the stop clears" would let a
single noisy day flip the position daily.) The drawdown is computed from close *t*,
so the exit is acted at *t+1* — one ``shift``, applied once. Switching costs are
charged one-way per change in position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Trailing-stop position
# ---------------------------------------------------------------------------
def trailing_stop_position(
    close: pd.Series,
    stop_pct: float = 10.0,
    cooldown: int = 21,
    lag: int = 1,
) -> pd.Series:
    """Long-or-flat position from a trailing stop with a fixed re-entry cooldown.

    While long, the running peak of the close is tracked. The stop fires the day the
    close is more than ``stop_pct`` percent below that peak; we go flat for
    ``cooldown`` trading days, then re-enter (resetting the peak to the re-entry
    price). The raw signal is known at close *t*; the returned position is shifted
    forward by ``lag`` days so it is acted on at *t+1*'s return — one lag, applied once.

    Returns a 0/1 :class:`pandas.Series` aligned to ``close``.
    """
    px = close.to_numpy(dtype=float)
    n = px.size
    raw = np.ones(n, dtype=float)  # 1 = long, 0 = cash (pre-lag, "decided at close t")
    peak = px[0]
    in_market = True
    cool = 0
    for t in range(n):
        if in_market:
            if px[t] > peak:
                peak = px[t]
            if px[t] <= peak * (1.0 - stop_pct / 100.0):
                # Stop fires at close t.
                in_market = False
                cool = cooldown
                raw[t] = 1.0  # still long *through* close t; exit happens next day via lag
            else:
                raw[t] = 1.0
        else:
            raw[t] = 0.0
            cool -= 1
            if cool <= 0:
                # Re-enter at close t; reset the trailing peak to here.
                in_market = True
                peak = px[t]
                # raw[t] stays 0 (flat through close t); the long is acted next day.
    pos = pd.Series(raw, index=close.index)
    return pos.shift(lag).fillna(0.0)


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
    """Run a long-or-flat position over ``close`` and return net stats + daily series.

    ``cost_bps`` is charged (one-way) on every change in position — entering or
    leaving the market each costs ``cost_bps``. Cash earns 0%. Returns a dict with the
    net daily return series and headline stats (CAGR, vol, Sharpe, max drawdown,
    switches, time-in-market).
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
    sharpe = (
        float(net.mean() / net.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if net.std() > 0
        else float("nan")
    )
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
# The X-sweep
# ---------------------------------------------------------------------------
def sweep(
    close: pd.Series,
    stops: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0),
    cooldown: int = 21,
    cost_bps: float = 5.0,
) -> dict[float, dict]:
    """Run the trailing stop for each threshold in ``stops``; return {stop_pct: stats}."""
    out = {}
    for x in stops:
        pos = trailing_stop_position(close, stop_pct=x, cooldown=cooldown)
        out[x] = backtest(close, pos, cost_bps=cost_bps)
    return out


# ---------------------------------------------------------------------------
# Matched random-exit control
# ---------------------------------------------------------------------------
def _run_lengths(pos: np.ndarray) -> list[tuple[int, int]]:
    """Return (value, length) runs of a 0/1 position vector."""
    runs: list[tuple[int, int]] = []
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
    """A random-exit coin that matches the real stop's *exposure profile*.

    It reproduces the exact multiset of in-market and out-of-market run *lengths* the
    real trailing stop produced, but reshuffles their order — so it holds the same
    total time in cash, in similarly-sized blocks, at **random** points in history. If
    the stop's *timing* carries no information, the real stop should not beat this on a
    risk-adjusted basis. (The desk's standard "beats a coin?" control, mirroring
    Study 91's matched random-timing coin.)
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
    seq: list[int] = []
    a = in_runs if start_val == 1 else out_runs
    b = out_runs if start_val == 1 else in_runs
    va, vb = start_val, 1 - start_val
    i = j = 0
    toggle = True
    while i < len(a) or j < len(b):
        if toggle and i < len(a):
            seq.extend([va] * a[i])
            i += 1
        elif not toggle and j < len(b):
            seq.extend([vb] * b[j])
            j += 1
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
