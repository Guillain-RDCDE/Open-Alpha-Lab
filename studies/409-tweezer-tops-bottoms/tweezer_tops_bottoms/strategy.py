"""The detector, the timing rule, and the honest arbiters — Study 409 (Tweezers).

A **tweezer** is a two-candle reversal pattern:

- **Tweezer bottom** (bullish): in a short *downtrend*, two consecutive bars whose **lows
  are equal** within a tolerance.  The second bar tests the same floor and holds — the
  folklore says the down-move is exhausted, so *buy*.
- **Tweezer top** (bearish): in a short *uptrend*, two consecutive bars whose **highs are
  equal** within a tolerance.  The folklore says the up-move is exhausted, so *sell*.

Precise OHLC rules (signal evaluated at the close of the **second** bar *t*):

1. **Equal extreme.** ``|low_t - low_{t-1}| <= tol * close_t`` (bottom) or
   ``|high_t - high_{t-1}| <= tol * close_t`` (top).  ``tol`` defaults to 0.1% of price.
2. **Prior trend.** A bottom needs a short *down*-leg into it
   (``close_{t-1} < close_{t-1-trend_lookback}``); a top needs a short *up*-leg.  The
   trend filter is the optional myth-check knob (``require_trend``).
3. **Distinct bars.** The two bars must not be the identical doji collapse (a tiny guard).

Direction: a bottom is +1 (long the bounce), a top is −1 (short the fade).

No look-ahead: the signal is known at the close of bar *t*; the trade enters at the
**next** bar's open (*t+1*) and exits at the close of bar *t + hold_days* — one documented
execution lag, applied once.

Honest arbiters:
- ``base_rate`` — the unconditional forward return of the same tape (all bars), so the
  tweezer mean is read *against its own base rate*, not against zero in a vacuum.
- ``summarize`` — one-sample HAC (Newey-West) *t* on the per-event forward returns.
- ``label_shuffle_placebo`` — re-draw the entry dates at random (same count, same tape),
  thousands of times; the placebo *p* is the fraction of random baskets whose mean beats
  the real tweezer mean.  Destroys the "two aligned wicks" timing, keeps everything else.
- costs: ``cost_bps`` one-way × NAV per leg; shorts pay ``borrow_bps``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
def detect_tweezers(
    bars: pd.DataFrame,
    tol: float = 0.001,
    trend_lookback: int = 3,
    require_trend: bool = True,
) -> pd.DataFrame:
    """Detect tweezer tops & bottoms on a daily OHLCV tape.

    Returns a frame indexed by the **second-bar date** (the bar at whose close the signal
    is known) with columns ``dir`` (+1 bottom / −1 top) and ``kind`` ("bottom"/"top").

    - ``tol`` — equal-extreme tolerance as a fraction of close (0.001 = 10 bps).
    - ``trend_lookback`` — bars back used to define the short prior trend leg.
    - ``require_trend`` — if False, drop the prior-trend condition (the myth-check
      "raw two-aligned-wicks, no context" variant).
    """
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    n = len(bars)
    idx = bars.index

    rows = []
    start = max(1, trend_lookback + 1)
    for t in range(start, n):
        ctol = tol * c[t]
        body_t = abs(c[t] - o[t])
        body_p = abs(c[t - 1] - o[t - 1])
        # guard against the degenerate "two identical flat bars" collapse
        nondegen = (body_t + body_p) > (1e-6 * c[t])

        equal_low = abs(low[t] - low[t - 1]) <= ctol
        equal_high = abs(h[t] - h[t - 1]) <= ctol

        down_leg = c[t - 1] < c[t - 1 - trend_lookback]
        up_leg = c[t - 1] > c[t - 1 - trend_lookback]

        is_bottom = equal_low and nondegen and (down_leg or not require_trend)
        is_top = equal_high and nondegen and (up_leg or not require_trend)

        # if both fire (rare), prefer the one with the tighter matched extreme
        if is_bottom and is_top:
            if abs(low[t] - low[t - 1]) <= abs(h[t] - h[t - 1]):
                is_top = False
            else:
                is_bottom = False

        if is_bottom:
            rows.append((idx[t], 1, "bottom"))
        elif is_top:
            rows.append((idx[t], -1, "top"))

    out = pd.DataFrame(rows, columns=["date", "dir", "kind"]).set_index("date")
    return out


# ---------------------------------------------------------------------------
# Forward-return measurement (no look-ahead: enter t+1 open, hold N days)
# ---------------------------------------------------------------------------
def forward_returns(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    hold_days: int = 5,
    cost_bps: float = 0.0,
    borrow_bps: float = 0.0,
    directions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-event forward returns: enter at next bar's open, exit hold_days later.

    Direction is signed by ``events['dir']`` unless ``directions`` overrides it (the
    label/direction control passes its own ±1 vector aligned to ``events``).

    Costs: ``cost_bps`` one-way × NAV is charged on *each* of the two legs (entry+exit),
    and a short leg additionally pays ``borrow_bps`` × (hold_days/252) financing.

    Columns: ``entry_ts, dir, kind, ret_gross, ret_net``.
    """
    open_ = bars["open"].to_numpy(float)
    close_ = bars["close"].to_numpy(float)
    idx = bars.index
    n_bars = len(bars)
    pos = {ts: i for i, ts in enumerate(idx)}

    dirs = (np.asarray(directions, dtype=int) if directions is not None
            else events["dir"].to_numpy(int))
    kinds = events["kind"].to_numpy() if "kind" in events.columns else ["?"] * len(events)

    rows = []
    for sig_ts, d, kind in zip(events.index, dirs, kinds):
        i = pos.get(sig_ts)
        if i is None or i + 1 >= n_bars:
            continue
        e = i + 1
        exit_idx = min(e + hold_days - 1, n_bars - 1)
        entry_px = open_[e]
        exit_px = close_[exit_idx]
        ret_gross = d * (exit_px - entry_px) / entry_px
        cost = 2.0 * cost_bps * 1e-4  # two legs, one-way each
        if d < 0:
            cost += borrow_bps * 1e-4 * (hold_days / 252.0)
        rows.append({
            "entry_ts": idx[e], "dir": int(d), "kind": kind,
            "ret_gross": ret_gross, "ret_net": ret_gross - cost,
        })
    return pd.DataFrame(rows)


def base_rate(bars: pd.DataFrame, hold_days: int = 5, direction: int = 1) -> dict:
    """Unconditional forward return of the same tape (every bar, signed ``direction``).

    The honest baseline the tweezer mean is read against: if every bar earns the same
    forward drift, a pattern that simply selects bars carries no information.
    """
    open_ = bars["open"].to_numpy(float)
    close_ = bars["close"].to_numpy(float)
    n = len(bars)
    rets = []
    for e in range(1, n):
        exit_idx = min(e + hold_days - 1, n - 1)
        if exit_idx < e:
            continue
        rets.append(direction * (close_[exit_idx] - open_[e]) / open_[e])
    r = np.array(rets, float)
    return {"mean_bps": float(r.mean() * 1e4) if r.size else float("nan"),
            "n": int(r.size)}


# ---------------------------------------------------------------------------
# Inference & placebos
# ---------------------------------------------------------------------------
def _hac_t(r: np.ndarray) -> float:
    n = r.size
    if n <= 5:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-event statistics: count, win-rate, mean (bps), Sharpe, HAC *t*."""
    keys = ["n_trades", "win_rate", "mean_bps", "sharpe_per_trade", "tstat"]
    if len(ledger) == 0:
        return {k: float("nan") for k in keys}
    r = ledger[col].to_numpy(float)
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        return {k: float("nan") for k in keys}
    return {
        "n_trades": int(n),
        "win_rate": float((r > 0).mean()),
        "mean_bps": float(r.mean() * 1e4),
        "sharpe_per_trade": (float(r.mean() / r.std(ddof=1))
                             if n > 1 and r.std() > 0 else float("nan")),
        "tstat": _hac_t(r),
    }


def label_shuffle_placebo(
    bars: pd.DataFrame,
    n_events: int,
    direction_mix: np.ndarray,
    real_mean_bps: float,
    hold_days: int = 5,
    n_iter: int = 2000,
    seed: int = 409,
) -> dict:
    """Re-draw entry dates at random (same count, same direction mix), thousands of times.

    Destroys the "two aligned wicks" timing while preserving the tape, the event count,
    and the long/short mix.  Returns the placebo distribution mean/std and the one-sided
    *p* = fraction of random baskets whose mean ≥ the real tweezer mean.
    """
    rng = np.random.default_rng(seed)
    open_ = bars["open"].to_numpy(float)
    close_ = bars["close"].to_numpy(float)
    n = len(bars)
    # valid entry bars: need a t+1 open
    valid = np.arange(1, n - 1)
    means = np.empty(n_iter)
    k = min(n_events, valid.size)
    for it in range(n_iter):
        picks = rng.choice(valid, size=k, replace=False)
        dirs = rng.permutation(direction_mix)[:k] if direction_mix.size >= k \
            else rng.choice(direction_mix, size=k)
        rr = np.empty(k)
        for j, e in enumerate(picks):
            exit_idx = min(e + hold_days - 1, n - 1)
            rr[j] = dirs[j] * (close_[exit_idx] - open_[e]) / open_[e]
        means[it] = rr.mean()
    means_bps = means * 1e4
    p = float((means_bps >= real_mean_bps).mean())
    return {"placebo_mean_bps": float(means_bps.mean()),
            "placebo_std_bps": float(means_bps.std(ddof=1)),
            "p_value": p, "n_iter": n_iter}


# ---------------------------------------------------------------------------
# Pooled orchestrator over a basket
# ---------------------------------------------------------------------------
def run_experiment(
    panel: dict[str, pd.DataFrame],
    hold_days: int = 5,
    tol: float = 0.001,
    trend_lookback: int = 3,
    require_trend: bool = True,
    cost_bps: float = 0.0,
    borrow_bps: float = 0.0,
    kind: str | None = None,
    directions_seed: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Detect tweezers across the whole basket and pool the per-event forward returns.

    ``kind`` filters to "bottom" or "top" only (None = both, signed).  ``directions_seed``
    replaces every event's direction with a seeded ±1 coin (the direction control).
    Returns ``(pooled_ledger, summary_dict)`` where summary also carries the pooled
    unconditional base rate and the per-event count.
    """
    frames = []
    base_long = []
    base_short = []
    for tkr, bars in panel.items():
        ev = detect_tweezers(bars, tol=tol, trend_lookback=trend_lookback,
                             require_trend=require_trend)
        if kind is not None:
            ev = ev[ev["kind"] == kind]
        if len(ev) == 0:
            continue
        dirs = None
        if directions_seed is not None:
            rng = np.random.default_rng(directions_seed + hash(tkr) % 10_000)
            dirs = rng.choice([-1, 1], size=len(ev))
        led = forward_returns(bars, ev, hold_days=hold_days, cost_bps=cost_bps,
                              borrow_bps=borrow_bps, directions=dirs)
        led["ticker"] = tkr
        frames.append(led)
        base_long.append(base_rate(bars, hold_days=hold_days, direction=1)["mean_bps"])
        base_short.append(base_rate(bars, hold_days=hold_days, direction=-1)["mean_bps"])

    if not frames:
        return pd.DataFrame(), {"n_trades": 0}
    pooled = pd.concat(frames, ignore_index=True)
    summ = summarize(pooled, "ret_net" if (cost_bps or borrow_bps) else "ret_gross")
    summ["base_long_bps"] = float(np.nanmean(base_long)) if base_long else float("nan")
    summ["base_short_bps"] = float(np.nanmean(base_short)) if base_short else float("nan")
    summ["n_bottoms"] = int((pooled["kind"] == "bottom").sum())
    summ["n_tops"] = int((pooled["kind"] == "top").sum())
    return pooled, summ
