"""Strategy + honest inference for Study 447 — Gann Angles (the 1x1 line).

The folklore (W.D. Gann, *45 Years in Wall Street*, the "Gann fan"): a market moves in
geometric proportion to time, and the master angle is the **1x1** — a line rising **one
unit of price per one unit of time** from a significant pivot. Price *above* its 1x1
line is in a strong up-trend (the line is support); price *below* it has turned
(the line is resistance). The 1x1 is the spine of the fan (2x1, 1x2, etc. are the rest).

The method is usually drawn by eye, so we encode the **tightest mechanical rule a Gann
practitioner would accept** and make it fully falsifiable:

    * **The pivot.** A confirmed swing low — a bar whose low is the lowest of a
      ``pivot_win``-bar window centred on it. It is only *knowable* ``pivot_win//2`` bars
      later (we lag its use accordingly — no look-ahead).

    * **The 1x1 line.** From the most recent confirmed pivot low at price ``P0`` on day
      ``t0``, the up-line is the arithmetic ``P0 + slope*(t - t0)`` (a fixed *price*
      increment per bar — the literal 45° line a charting package draws). The slope is the
      only free choice and is *fixed* per instrument from its own scale: one "price unit
      per time unit" = the full price range divided by the number of bars (rise-over-run of
      the whole chart, so the angle is ~45° on the instrument's natural scale). We re-anchor
      whenever a newer pivot low prints. This is the literal 1x1 a charting package draws.

    * **The regime.** Close *above* the active 1x1 line → bullish (line is support).
      Close *below* → bearish. We test both the *un-latched* instantaneous flag and a
      *latched* version (stay long until a decisive break below) — Gann's "the trend
      holds until the angle is broken".

    * **The trade.** Be **long** the instrument while the regime is bullish, else **cash**.
      Regime known at close *t*, position earns the return of *t+1* — one execution lag.

Inference, the desk's shared spirit:

  * the strategy's daily return (long-when-above minus cash=0) vs **buy-and-hold**, both
    excess-of-cash, raced on Sharpe;
  * a **HAC (Newey-West) one-sample t** on the *active-minus-benchmark daily spread*
    — does the 1x1 angle add timing value over just holding the tape?
  * a **placebo** that replaces the real angle regime with a random regime of the *same
    on-fraction and same average run length* (is it the angle, or just being in cash part
    of the time?);
  * **one-way costs × turnover** charged on every regime flip.

The decisive number is the HAC t on the active-minus-benchmark spread: a real Gann edge
must clear **t >= 2** there. Most fixed-geometry chart folklore does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252.0
PIVOT_WIN = 21          # window (centred) defining a confirmed swing low (~1 month)


# --------------------------------------------------------------------------- #
# Pivots and the 1x1 line
# --------------------------------------------------------------------------- #
def confirmed_pivot_lows(low: pd.Series, win: int = PIVOT_WIN) -> pd.Series:
    """Boolean series: True on bars that are a centred-window swing low.

    A bar ``i`` is a pivot low if ``low[i]`` is the minimum of ``low[i-h : i+h+1]`` with
    ``h = win//2``. Such a pivot is only *confirmable* h bars after it prints; callers
    must lag its use by h bars (see :func:`gann_regime`). No look-ahead leaks here — this
    just marks where the pivots are.
    """
    h = win // 2
    lv = low.to_numpy(dtype=float)
    n = lv.size
    is_piv = np.zeros(n, dtype=bool)
    for i in range(h, n - h):
        seg = lv[i - h:i + h + 1]
        if lv[i] == seg.min():
            is_piv[i] = True
    return pd.Series(is_piv, index=low.index)


def daily_slope(bars: pd.DataFrame) -> float:
    """The fixed 1x1 slope: a **price increment per bar** (arithmetic, the chart 1x1).

    A Gann 1x1 line is the literal 45-degree line a charting package draws — it rises a
    fixed *price* amount each *bar*. We calibrate "one price unit per one time unit" to the
    instrument's own scale as the full price range divided by the number of bars
    (rise-over-run of the whole chart), a single structural constant fixed once over the
    sample (no per-day fitting). Returned in *price units per bar*.
    """
    c = bars["close"]
    rise = float(c.max() - c.min())
    run = max(len(c) - 1, 1)
    return rise / run


def gann_regime(bars: pd.DataFrame, win: int = PIVOT_WIN, slope: float | None = None,
                latched: bool = False) -> pd.DataFrame:
    """Build the 1x1 line and the bullish/bearish regime, lag-clean.

    For every bar we know the most recent **confirmed** pivot low (a pivot at bar ``j`` is
    only usable from bar ``j + win//2`` onward). From that anchor ``(t0, P0)`` we draw the
    arithmetic 1x1 up-line ``P0 + slope*(t-t0)`` (a fixed price increment per bar — the
    literal chart 1x1) and flag ``above`` = close >= line.

    ``latched=False`` → ``regime`` is the instantaneous ``above`` flag.
    ``latched=True``  → go bullish on a cross **above** the line, stay bullish until a
    cross **below** it (Gann's "trend holds until the angle breaks").

    Returns a frame with columns ``close, line, above, regime`` (regime in {0,1}).
    """
    if slope is None:
        slope = daily_slope(bars)
    h = win // 2
    close = bars["close"].to_numpy(dtype=float)
    n = close.size
    piv = confirmed_pivot_lows(bars["low"], win).to_numpy()

    line = np.full(n, np.nan)
    anchor_t = -1
    anchor_p = np.nan
    # a pivot at bar j becomes usable at bar j+h; scan forward keeping the latest usable one
    usable_at = {}  # bar index -> (t0, P0)
    for j in range(n):
        if piv[j]:
            uj = j + h
            if uj < n:
                usable_at.setdefault(uj, (j, close[j]))
    for t in range(n):
        if t in usable_at:
            anchor_t, anchor_p = usable_at[t]
        if anchor_t >= 0:
            line[t] = anchor_p + slope * (t - anchor_t)

    above = close >= line
    above = np.where(np.isnan(line), np.nan, above.astype(float))

    if not latched:
        regime = np.where(np.isnan(line), 0.0, above)
    else:
        regime = np.zeros(n)
        state = 0
        for t in range(n):
            if np.isnan(line[t]):
                regime[t] = 0
                continue
            if state == 0 and close[t] >= line[t]:
                state = 1
            elif state == 1 and close[t] < line[t]:
                state = 0
            regime[t] = state

    out = pd.DataFrame({"close": bars["close"].values, "line": line,
                        "above": above, "regime": regime}, index=bars.index)
    return out


# --------------------------------------------------------------------------- #
# Backtest: long-when-bullish vs buy-and-hold
# --------------------------------------------------------------------------- #
def backtest(bars: pd.DataFrame, win: int = PIVOT_WIN, slope: float | None = None,
             latched: bool = False, cost_bps: float = 2.0) -> dict:
    """Long-when-above-the-1x1 vs buy-and-hold, one execution lag, net of costs.

    Regime known at close ``t`` earns the asset return of ``t+1`` (one ``shift``). Costs
    are charged one-way x NAV on every change in the held position (turnover). Returns
    daily series + headline stats (excess-of-cash; cash earns 0, so excess == raw here,
    and both legs are raced excess-vs-excess consistently).
    """
    reg = gann_regime(bars, win=win, slope=slope, latched=latched)
    ret = bars["close"].pct_change().fillna(0.0)
    pos = reg["regime"].shift(1).fillna(0.0)              # one execution lag
    turn = pos.diff().abs().fillna(pos.abs())
    cost = turn * (cost_bps / 1e4)
    strat = pos * ret - cost
    bh = ret.copy()
    spread = strat - bh                                    # active minus benchmark

    valid = ~reg["line"].isna()
    strat_v = strat[valid]
    bh_v = bh[valid]
    spread_v = spread[valid]

    return {
        "ret": ret, "pos": pos, "strat": strat, "bh": bh, "spread": spread,
        "line": reg["line"], "regime": reg["regime"],
        "n": int(valid.sum()),
        "on_frac": float(pos[valid].mean()),
        "n_flips": int(turn[valid].gt(0).sum()),
        "sharpe_strat": _sharpe(strat_v),
        "sharpe_bh": _sharpe(bh_v),
        "ann_strat": float(strat_v.mean() * ANN),
        "ann_bh": float(bh_v.mean() * ANN),
        "spread_mean_ann": float(spread_v.mean() * ANN),
        "t_spread": hac_t(spread_v.to_numpy()),
        "cost_bps": cost_bps,
    }


def _sharpe(r: pd.Series) -> float:
    sd = r.std(ddof=1)
    if sd == 0 or len(r) < 2:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(ANN))


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_t(x: np.ndarray) -> float:
    """Newey-West (HAC) one-sample t of mean(x) vs 0, auto bandwidth."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def _run_lengths(pos: np.ndarray) -> tuple[float, float]:
    """Average on-run and off-run length of a 0/1 position (for a fair placebo)."""
    pos = np.asarray(pos, dtype=int)
    if pos.size == 0:
        return 1.0, 1.0
    on_runs, off_runs, cur, val = [], [], 1, pos[0]
    for p in pos[1:]:
        if p == val:
            cur += 1
        else:
            (on_runs if val == 1 else off_runs).append(cur)
            cur, val = 1, p
    (on_runs if val == 1 else off_runs).append(cur)
    return (float(np.mean(on_runs)) if on_runs else 1.0,
            float(np.mean(off_runs)) if off_runs else 1.0)


def placebo_pvalue(bars: pd.DataFrame, win: int = PIVOT_WIN, slope: float | None = None,
                   latched: bool = False, cost_bps: float = 2.0,
                   n_draws: int = 2000, seed: int = 447) -> dict:
    """Random-regime placebo: same on-fraction & run length, random timing.

    Replaces the real 1x1 regime with a Markov 0/1 regime matched to the strategy's
    on-fraction and average run length, re-runs the same one-lag, cost-charged backtest,
    and asks how often a *random* regime of the same shape produces an active-minus-
    benchmark spread at least as large. The honest "is it the angle, or just being in cash
    part-time?" null.
    """
    bt = backtest(bars, win=win, slope=slope, latched=latched, cost_bps=cost_bps)
    obs = bt["spread_mean_ann"]
    ret = bt["ret"].to_numpy()
    valid = ~bt["line"].isna().to_numpy()
    pos_real = bt["pos"].to_numpy()
    on_len, off_len = _run_lengths(pos_real[valid].astype(int))
    p_stay_on = max(0.0, 1.0 - 1.0 / max(on_len, 1.0))
    p_go_on = min(1.0, 1.0 / max(off_len, 1.0))
    c = cost_bps / 1e4
    n = ret.size

    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for d in range(n_draws):
        pos = np.zeros(n)
        state = 0
        u = rng.random(n)
        for t in range(n):
            if state == 1:
                state = 1 if u[t] < p_stay_on else 0
            else:
                state = 1 if u[t] < p_go_on else 0
            pos[t] = state
        pos_l = np.concatenate([[0.0], pos[:-1]])          # one lag
        turn = np.abs(np.diff(pos_l, prepend=0.0))
        strat = pos_l * ret - turn * c
        spread = (strat - ret)[valid]
        means[d] = spread.mean() * ANN
    p = float((means >= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()), "p_value": p,
            "draws": means, "on_frac": bt["on_frac"]}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(bars: pd.DataFrame, win: int = PIVOT_WIN, slope: float | None = None,
                   cost_bps: float = 2.0, n_draws: int = 2000,
                   placebo: bool = True) -> dict:
    """One-call teardown: un-latched + latched backtest, placebo, slope used."""
    if slope is None:
        slope = daily_slope(bars)
    unl = backtest(bars, win=win, slope=slope, latched=False, cost_bps=cost_bps)
    lat = backtest(bars, win=win, slope=slope, latched=True, cost_bps=cost_bps)
    out = {"slope": slope, "unlatched": unl, "latched": lat}
    if placebo:
        out["placebo"] = placebo_pvalue(bars, win=win, slope=slope, latched=False,
                                        cost_bps=cost_bps, n_draws=n_draws)
    return out
