"""The trendline break as a falsifiable mechanical rule — Study 499.

The classic chartist signal: in an uptrend, connect the recent **swing lows** with a rising
**trendline** (support). While price holds above the line the trend is "intact"; the moment a
close drops **below** the line, support has *broken* — the textbook signal to exit longs (and,
for the bolder, to go short, because "the break forecasts a turn down").

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Swing lows are confirmed fractals** — a local minimum with ``k`` strictly-higher bars on
   each side. A pivot at bar ``t`` is only *usable* from bar ``t+k`` onward (the confirmation
   lag), so the trendline is never fit with future bars.
2. **Least-squares trendline** — at each bar we fit a line (ordinary least squares) through the
   ``n_lows`` most-recent *confirmed* swing lows, and require it to be **rising** (an uptrend
   support line, the only configuration the "break" lore applies to).
3. **The break** — a signal fires on the first close that pierces *below* the fitted trendline
   (a confirmed break of support). We measure the forward H-day return of the underlying.
   Because the lore is bearish ("support broke → it falls"), the natural tradable is a SHORT;
   we report the **break return sign-flipped** (so a positive number = the break correctly
   forecast a drop), and race it against random.
4. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures
   the tape's drift, and (b) a **shuffled-slope placebo** that re-fits the line from a
   permutation of the swing-low *prices*, destroying the geometry (the actual slope/level)
   while keeping the marginal — the honest "is the trendline's geometry doing anything?" null.

No look-ahead: pivots carry a ``k``-bar confirmation lag, the break is read on the close of
*t*, the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)
N_LOWS = 3  # swing lows fit by the trendline


# --------------------------------------------------------------------------- #
# Pivot detection (confirmed swing lows as fractals)
# --------------------------------------------------------------------------- #
def find_swing_lows(close: pd.Series, k: int = 10) -> pd.DataFrame:
    """Confirmed swing lows: local minima with ``k`` strictly-higher bars on each side.

    Returns a DataFrame indexed by the pivot bar position with column ``price``. A pivot at
    position ``i`` is only *confirmed* (knowable) at bar ``i + k`` — :func:`build_trendlines`
    enforces that lag, so no future data leaks in.
    """
    p = close.to_numpy(dtype=float)
    n = p.size
    rows = []
    for i in range(k, n - k):
        win = p[i - k:i + k + 1]
        c = p[i]
        if c == win.min() and (win[:k] > c).all() and (win[k + 1:] > c).all():
            rows.append((i, c))
    if not rows:
        return pd.DataFrame(columns=["pos", "price"]).set_index("pos")
    return pd.DataFrame(rows, columns=["pos", "price"]).set_index("pos")


# --------------------------------------------------------------------------- #
# Trendline geometry (least-squares through swing lows)
# --------------------------------------------------------------------------- #
def fit_line(points) -> tuple[float, float] | None:
    """Ordinary least-squares line ``price = slope*pos + intercept`` through ``points``.

    ``points`` is a list of ``(pos, price)``. Returns ``(slope, intercept)`` or ``None`` if the
    fit is degenerate (all positions equal).
    """
    if len(points) < 2:
        return None
    xs = np.array([float(x) for x, _ in points], dtype=float)
    ys = np.array([float(y) for _, y in points], dtype=float)
    if xs.std() == 0:
        return None
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def build_trendlines(close: pd.Series, k: int = 10, n_lows: int = N_LOWS):
    """For each bar, the *currently usable* rising trendline fit on the recent confirmed lows.

    Returns one aligned Series ``line`` over ``close.index`` (the trendline value at each bar);
    NaN until ``n_lows`` confirmed swing lows exist whose OLS fit is **rising** (slope > 0). A
    pivot at bar ``i`` becomes usable at ``i + k`` (confirmation lag), so the line at bar ``t``
    only uses lows confirmed by ``t``.
    """
    lows = find_swing_lows(close, k=k)
    n = len(close)
    line = np.full(n, np.nan)
    if len(lows) >= n_lows:
        pos_arr = np.array([int(p) for p in lows.index], dtype=float)
        price_arr = np.array([float(p) for p in lows["price"]], dtype=float)
        confirm = pos_arr + k                       # bar at which each low becomes known
        line = _trendline_loop(n, pos_arr, price_arr, confirm, n_lows)
    return pd.Series(line, index=close.index)


def _trendline_loop(n, pos_arr, price_arr, confirm, n_lows):
    """Incremental walk: at each bar use the ``n_lows`` most-recent *confirmed* swing lows.

    ``confirm`` is sorted ascending (pivots are found left-to-right), so a single pointer
    tracks how many lows are usable at bar ``t`` — O(n) overall, not O(n × pivots).
    """
    out = np.full(n, np.nan)
    j = 0                       # number of lows confirmed by bar t
    m = confirm.size
    for t in range(n):
        while j < m and confirm[j] <= t:
            j += 1
        if j < n_lows:
            continue
        xs = pos_arr[j - n_lows:j]
        ys = price_arr[j - n_lows:j]
        xm = xs.mean()
        dx = xs - xm
        denom = float(dx @ dx)
        if denom == 0.0:
            continue
        slope = float(dx @ (ys - ys.mean())) / denom
        if slope <= 0:          # only rising support lines (an uptrend) qualify
            continue
        intercept = float(ys.mean()) - slope * xm
        out[t] = slope * t + intercept
    return out


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def trendline_break_entries(close: pd.Series, k: int = 10, n_lows: int = N_LOWS) -> pd.DatetimeIndex:
    """Bars whose close pierces *below* the rising trendline — the 'support broke' signal.

    Only the *first* bar of each consecutive run is kept (the break, not every day price stays
    below the line). Entry is executed at the next close by :func:`forward_returns`.
    """
    line = build_trendlines(close, k=k, n_lows=n_lows)
    mask = (close < line) & line.notna()
    first = mask & ~mask.shift(1, fill_value=False)
    return close.index[first.to_numpy()]


def random_entries(close: pd.Series, n: int, k: int = 10, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[2 * k:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0,
                    short: bool = False) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    With ``short=True`` the sign is flipped — the break lore is bearish, so the tradable view
    is a short, and a positive number means "the break correctly forecast a drop". ``cost_bps``
    is a one-way round-trip cost (charged twice: in + out) subtracted from each trade's return.
    Trades whose window overruns the tape are dropped.
    """
    pos = {d: i for i, d in enumerate(close.index)}
    p = close.to_numpy(dtype=float)
    n = p.size
    sign = -1.0 if short else 1.0
    out = []
    for d in entries:
        i = pos.get(d)
        if i is None or i + 1 + horizon >= n:
            continue
        e = i + 1                      # enter at next close
        r = p[e + horizon] / p[e] - 1.0
        out.append(sign * r - 2.0 * cost_bps * 1e-4)
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


def shuffled_slope_placebo(close: pd.Series, horizon: int, k: int = 10, n_lows: int = N_LOWS,
                           short: bool = True, n_draws: int = 1000, seed: int = 499) -> dict:
    """Placebo: re-fit trendlines from permuted swing-low *prices*, destroying the geometry.

    Keeps the swing-low *positions* and the marginal price distribution but scrambles which
    price sits at which low, so the fitted line's slope/level become meaningless. Returns the
    share of placebo runs whose mean break forward return **beats** the real one — the honest
    "is the trendline's geometry adding anything?" p-value, plus the observed mean. (We compare
    on the same sign convention as the live rule via ``short``.)
    """
    obs = float(np.mean(forward_returns(
        close, trendline_break_entries(close, k=k, n_lows=n_lows), horizon, short=short)))
    lows = find_swing_lows(close, k=k)
    if len(lows) < n_lows:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    prices = lows["price"].to_numpy(dtype=float)
    pos_arr = np.array([int(p) for p in lows.index], dtype=float)
    confirm = pos_arr + k
    n = close.to_numpy().size
    idx = close.index
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = rng.permutation(prices)
        line = _trendline_loop(n, pos_arr, perm, confirm, n_lows)
        lineser = pd.Series(line, index=idx)
        mask = (close < lineser) & lineser.notna()
        first = mask & ~mask.shift(1, fill_value=False)
        ent = idx[first.to_numpy()]
        rr = forward_returns(close, ent, horizon, short=short)
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
def run_experiment(close: pd.Series, k: int = 10, n_lows: int = N_LOWS, cost_bps: float = 1.0,
                   short: bool = True, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: trendline-break vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the break summary (gross + net), the drift-matched
    random-entry baseline (same sign convention), and the break-minus-random delta. ``short``
    sign-flips the break return (the bearish tradable view) so a positive edge = the break
    correctly forecasts a drop.
    """
    ent = trendline_break_entries(close, k=k, n_lows=n_lows)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, short=short))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps, short=short))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), k=k, seed=random_seed), h, short=short))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
