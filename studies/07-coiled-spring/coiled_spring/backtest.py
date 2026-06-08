"""From signals to P&L — the trade simulator, the cost model, and the forward-return
measurement that powers the headline falsification.

Two things live here, kept straight:

  * :func:`forward_returns` — the **pure signal test**. For each breakout it just records
    the return of buying at the next open and selling ``h`` sessions later. No stop, no
    discretion, no costs. This is the apples-to-apples object the headline test compares
    against a same-stock random-entry baseline: *does the breakout day carry information
    beyond "this stock tends to drift"?*

  * :func:`run_trades` — the **tradable strategy**, exactly as the book prescribes: enter
    next open, initial stop under the breakout bar's low, then trail the stop under each
    prior bar's low once the trade is in profit; exit on a stop hit or a max hold. Returns a
    per-trade ledger with gross and net (cost-charged) returns, plus summary stats.

Costs are charged as a single round-trip in basis points (one buy + one sell), swept in
:mod:`coiled_spring.robustness`. A long-only cash position carries no overnight financing,
so unlike the overnight study there is no per-night carry term — just the round trip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import Params, find_signals

TRADING_DAYS = 252


@dataclass(frozen=True)
class ExitRules:
    """Deterministic exit rules — the book's stop-and-trail, with no eyeballed candles."""
    trail_after_profit: bool = True   # raise the stop under the prior bar's low once in profit
    max_hold: int = 30                # hard time stop (sessions) so a stalled trade can't dangle
    stop_eps: float = 0.001           # the book's "1 point below the breakout low" (0.1%)


# --------------------------------------------------------------------------- #
# Pure signal test — forward returns, no exit logic
# --------------------------------------------------------------------------- #

def forward_returns(frame: pd.DataFrame, signals: pd.DataFrame, horizons=(5, 10, 20)) -> pd.DataFrame:
    """Buy each breakout at the next open, mark out ``h`` sessions later. No stop, no costs.

    Returns one row per signal with a ``fwd_{h}`` column per horizon (close-to-entry return).
    A signal too close to the end of the series gets NaN for horizons it can't span.
    """
    f = frame.sort_index()
    close = f["Close"].to_numpy(float)
    pos = {d: i for i, d in enumerate(f.index)}
    rows = []
    for _, s in signals.iterrows():
        e = s["entry_date"]
        if pd.isna(e) or e not in pos or not (s["entry_price"] > 0):
            continue
        i0 = pos[e]
        entry = s["entry_price"]
        row = {"entry_date": e, "entry_price": entry}
        for h in horizons:
            j = i0 + h
            row[f"fwd_{h}"] = (close[j] / entry - 1.0) if j < len(close) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_forward_returns(
    frame: pd.DataFrame, n_draws: int, horizons=(5, 10, 20), seed: int = 0, warmup: int = 40
) -> pd.DataFrame:
    """Same-stock control: ``n_draws`` *random* entry dates, marked out over the same horizons.

    Buying at the open of a randomly chosen session and holding ``h`` days isolates the
    stock's ambient drift over that window — the thing a real breakout edge must *beat*.
    Deterministic given ``seed``.
    """
    f = frame.sort_index()
    close = f["Close"].to_numpy(float)
    open_ = f["Open"].to_numpy(float)
    n = len(f)
    hi = max(horizons)
    lo, top = warmup, n - hi - 1
    if top <= lo or n_draws <= 0:
        return pd.DataFrame(columns=["entry_date"] + [f"fwd_{h}" for h in horizons])
    rng = np.random.default_rng(seed)
    picks = rng.integers(lo, top, size=n_draws)
    rows = []
    for i0 in picks:
        entry = open_[i0]
        if not (entry > 0):
            continue
        row = {"entry_date": f.index[i0], "entry_price": entry}
        for h in horizons:
            row[f"fwd_{h}"] = close[i0 + h] / entry - 1.0
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Tradable strategy — enter, stop, trail, exit
# --------------------------------------------------------------------------- #

def _simulate_one(f: pd.DataFrame, sig: pd.Series, rules: ExitRules) -> dict | None:
    """Walk one trade forward bar by bar; return its ledger row (or None if unfillable)."""
    pos = {d: i for i, d in enumerate(f.index)}
    e = sig["entry_date"]
    if pd.isna(e) or e not in pos or not np.isfinite(sig["entry_price"]):
        return None
    i0 = pos[e]
    n = len(f)
    high = f["High"].to_numpy(float)
    low = f["Low"].to_numpy(float)
    close = f["Close"].to_numpy(float)

    entry = float(sig["entry_price"])
    stop = float(sig["stop"]) * (1.0 - rules.stop_eps)
    exit_price, exit_i, reason = None, None, None

    for i in range(i0, min(i0 + rules.max_hold, n - 1) + 1):
        # Stop check first (intrabar): a gap-down opens below the stop -> fill at the open.
        day_open = float(f["Open"].iloc[i])
        if low[i] <= stop:
            exit_price = min(day_open, stop) if day_open < stop else stop
            exit_i, reason = i, "stop"
            break
        if i == min(i0 + rules.max_hold, n - 1):
            exit_price, exit_i, reason = close[i], i, "time"
            break
        # Trail the stop under the prior bar's low once the trade is in profit.
        if rules.trail_after_profit and close[i] > entry:
            stop = max(stop, low[i])
    if exit_price is None:                       # ran off the end of the series
        exit_i = min(i0 + rules.max_hold, n - 1)
        exit_price, reason = close[exit_i], "time"

    return {
        "ticker": sig.get("ticker", ""),
        "entry_date": e,
        "exit_date": f.index[exit_i],
        "bars_held": int(exit_i - i0),
        "entry_price": entry,
        "exit_price": float(exit_price),
        "ret_gross": float(exit_price / entry - 1.0),
        "exit_reason": reason,
        "vol_ratio": float(sig.get("vol_ratio", np.nan)),
    }


def run_trades(
    frame: pd.DataFrame,
    params: Params = Params(),
    rules: ExitRules = ExitRules(),
    cost_bps: float = 15.0,
    signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Trade every signal in one name; return a per-trade ledger with gross & net returns.

    ``cost_bps`` is the **round-trip** cost (buy + sell) charged against each trade's return.
    """
    f = frame.sort_index()
    if signals is None:
        signals = find_signals(f, params)
    rows = []
    for _, s in signals.iterrows():
        row = _simulate_one(f, s, rules)
        if row is not None:
            row["ret_net"] = row["ret_gross"] - cost_bps / 1e4
            rows.append(row)
    return pd.DataFrame(
        rows,
        columns=["ticker", "entry_date", "exit_date", "bars_held", "entry_price",
                 "exit_price", "ret_gross", "ret_net", "exit_reason", "vol_ratio"],
    )


def signals_by_ticker(frames: dict[str, pd.DataFrame], params: Params = Params()) -> dict[str, pd.DataFrame]:
    """Detect signals once per name; reuse the dict everywhere so the scan runs a single pass."""
    return {tk: find_signals(f, params) for tk, f in frames.items()}


def run_universe(
    frames: dict[str, pd.DataFrame],
    params: Params = Params(),
    rules: ExitRules = ExitRules(),
    cost_bps: float = 15.0,
    sigs: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Run :func:`run_trades` across a universe; one stacked ledger with the ticker filled in.

    Pass ``sigs`` (from :func:`signals_by_ticker`) to skip re-detecting signals.
    """
    out = []
    for tk, f in frames.items():
        s = sigs.get(tk) if sigs is not None else None
        led = run_trades(f, params=params, rules=rules, cost_bps=cost_bps, signals=s)
        if len(led):
            led["ticker"] = tk
            out.append(led)
    if not out:
        return pd.DataFrame(
            columns=["ticker", "entry_date", "exit_date", "bars_held", "entry_price",
                     "exit_price", "ret_gross", "ret_net", "exit_reason", "vol_ratio"]
        )
    return pd.concat(out, ignore_index=True).sort_values("entry_date").reset_index(drop=True)


def trade_stats(ledger: pd.DataFrame) -> dict:
    """Per-trade summary: count, win rate, mean/median return, payoff, expectancy, tails."""
    if ledger.empty:
        return {"n_trades": 0}
    g = ledger["ret_gross"].to_numpy(float)
    net = ledger["ret_net"].to_numpy(float)
    wins = net[net > 0]
    losses = net[net <= 0]
    return {
        "n_trades": int(len(ledger)),
        "win_rate": float((net > 0).mean()),
        "mean_gross": float(g.mean()),
        "mean_net": float(net.mean()),
        "median_net": float(np.median(net)),
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
        "payoff": float(wins.mean() / abs(losses.mean())) if (wins.size and losses.size and losses.mean() != 0) else float("nan"),
        "best": float(net.max()),
        "worst": float(net.min()),
        "avg_bars_held": float(ledger["bars_held"].mean()),
        "pct_explosive": float((g >= 0.30).mean()),     # the book's "+30% in days" claim
    }
