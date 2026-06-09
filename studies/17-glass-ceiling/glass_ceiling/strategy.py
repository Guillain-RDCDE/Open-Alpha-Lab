"""The breakout trade itself, resolved trade-by-trade — and the arithmetic that decides its fate.

Koroush AK's execution, mechanized exactly: after ``confirm`` closes clear resistance, enter long,
put the stop at the swing low (floored at 1%), and take profit at **1R** — the same distance as the
stop, in the other direction. So every trade is a **symmetric ±1R bracket**, and that single fact
is the whole study:

    expectancy per trade (in R)  =  (2 · win_rate − 1)  −  cost_R

where ``cost_R`` is the round-trip transaction cost expressed in units of R. The break-even win
rate is therefore ``0.5 + cost_R / 2`` — strictly above a coin flip. On a driftless tape the win
rate *is* a coin flip (``data.synthetic_intraday`` with ``cont_drift=0`` makes this exact), so the
net expectancy is ``−cost_R`` and the strategy bleeds the spread. The job of the backtest is to
measure the real win rate honestly and put it next to that break-even line.

Intrabar resolution is **pessimistic by default**: if a single bar's range spans both the stop and
the target, we assume the stop filled first. This is the standard conservative assumption (you
cannot know the within-bar path) and it costs the strategy nothing it wouldn't really pay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .levels import breakout_triggers, stop_from_swing, swing_low


@dataclass(frozen=True)
class Trade:
    entry_idx: int
    entry_price: float
    stop: float
    target: float
    risk: float           # R in price units (entry − stop)
    exit_idx: int
    outcome: int          # +1 win (hit target), −1 loss (hit stop), 0 unresolved at tape end
    r_multiple: float     # gross PnL in R: +1, −1, or a fractional value if unresolved


def _resolve(
    bars: pd.DataFrame, entry_idx: int, entry: float, stop: float, target: float,
    pessimistic: bool = True, max_hold: int | None = None,
) -> tuple[int, int, float]:
    """First-touch resolution from ``entry_idx+1`` forward. Returns ``(exit_idx, outcome, r_mult)``."""
    n = len(bars)
    hi = bars["High"].to_numpy()
    lo = bars["Low"].to_numpy()
    risk = entry - stop
    end = n if max_hold is None else min(n, entry_idx + 1 + max_hold)
    for j in range(entry_idx + 1, end):
        hit_stop = lo[j] <= stop
        hit_tgt = hi[j] >= target
        if hit_stop and hit_tgt:
            return (j, -1, -1.0) if pessimistic else (j, +1, +1.0)
        if hit_stop:
            return j, -1, -1.0
        if hit_tgt:
            return j, +1, +1.0
    # Unresolved at the tape end (or max_hold): mark-to-market at the last close, in R.
    last = float(bars["Close"].iloc[end - 1])
    return end - 1, 0, float((last - entry) / risk) if risk > 0 else 0.0


def run(
    bars: pd.DataFrame,
    lookback: int = 30,
    confirm: int = 2,
    swing_window: int = 30,
    min_stop_frac: float = 0.01,
    pessimistic: bool = True,
    max_hold: int | None = None,
) -> pd.DataFrame:
    """Walk the tape, take every confirmed breakout (one position at a time), resolve each bracket.

    Entry is at the **confirmation bar's close** — a market entry the instant the breakout confirms.
    It always fills, so no winner is lost to a runaway that never pulls back; the test is generous on
    participation. (Koroush's "limit at the prior level on the pullback" would enter at a better price
    but miss the runaways — a realism variant for beat 7, not the clean symmetric-bracket core here.)
    Stop and target sit one R below and above this entry, so the bracket is exactly symmetric and the
    null tape is exactly a coin flip. Returns a tidy trades frame: one row per resolved trade with
    entry, stop, target, the realized ``r_multiple`` (+1/−1) and bars held.
    """
    trig = breakout_triggers(bars, lookback, confirm).to_numpy()
    trades: list[Trade] = []
    i = 0
    n = len(bars)
    while i < n:
        if not trig[i]:
            i += 1
            continue
        entry = float(bars["Close"].iloc[i])
        sw = swing_low(bars, i, swing_window)
        stop = stop_from_swing(entry, sw, min_stop_frac, bars, i, swing_window)
        risk = entry - stop
        if risk <= 0:                       # degenerate (stop above entry) — skip
            i += 1
            continue
        target = entry + risk               # 1R
        exit_idx, outcome, rmult = _resolve(bars, i, entry, stop, target, pessimistic, max_hold)
        trades.append(Trade(i, entry, stop, target, risk, exit_idx, outcome, rmult))
        i = exit_idx + 1                    # one position at a time: resume after the exit

    if not trades:
        return pd.DataFrame(columns=[
            "entry_idx", "entry_price", "stop", "target", "risk",
            "exit_idx", "outcome", "r_multiple", "risk_frac", "bars_held"])

    df = pd.DataFrame([t.__dict__ for t in trades])
    df["risk_frac"] = df["risk"] / df["entry_price"]      # R as a fraction of price
    df["bars_held"] = df["exit_idx"] - df["entry_idx"]
    return df


# --------------------------------------------------------------------------- #
# The arithmetic that decides it
# --------------------------------------------------------------------------- #

def cost_in_R(trades: pd.DataFrame, roundtrip_bps: float) -> float:
    """Round-trip cost expressed in units of R, averaged over trades.

    A trade risks ``risk_frac`` of notional to make 1R; a round-trip cost of ``roundtrip_bps`` bps of
    notional is therefore ``roundtrip_bps·1e-4 / risk_frac`` in R. Tight stops (small ``risk_frac``)
    make costs *bigger* in R — the reason a 1% stop on the 1-minute chart is so exposed to the spread.
    """
    if trades.empty:
        return float("nan")
    return float((roundtrip_bps * 1e-4 / trades["risk_frac"]).mean())


def summary(trades: pd.DataFrame, roundtrip_bps: float = 0.0) -> dict:
    """Headline: win rate, gross & net expectancy in R, and the break-even win rate costs demand."""
    resolved = trades[trades["outcome"] != 0]
    n = int(len(resolved))
    if n == 0:
        return {"n_trades": 0, "win_rate": float("nan"), "expectancy_R_gross": float("nan"),
                "expectancy_R_net": float("nan"), "breakeven_win_rate": float("nan"),
                "cost_R": float("nan")}
    win_rate = float((resolved["outcome"] == 1).mean())
    gross = 2.0 * win_rate - 1.0
    cR = cost_in_R(resolved, roundtrip_bps)
    return {
        "n_trades": n,
        "win_rate": win_rate,
        "expectancy_R_gross": float(gross),
        "expectancy_R_net": float(gross - cR),
        "breakeven_win_rate": float(0.5 + cR / 2.0),
        "cost_R": float(cR),
        "avg_risk_frac": float(resolved["risk_frac"].mean()),
        "avg_bars_held": float(resolved["bars_held"].mean()),
    }


def win_rate_ci(trades: pd.DataFrame, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for the win rate — the honest band around a proportion of ``n`` trades.

    Wilson (not normal-approx) because it stays inside [0,1] and is well-behaved at moderate ``n``;
    the question that matters is simply *does this interval contain 0.5?*.
    """
    resolved = trades[trades["outcome"] != 0]
    n = int(len(resolved))
    if n == 0:
        return (float("nan"), float("nan"))
    from scipy.stats import norm
    z = float(norm.ppf(1 - alpha / 2))
    p = float((resolved["outcome"] == 1).mean())
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(centre - half), float(centre + half))


def equity_curve(trades: pd.DataFrame, roundtrip_bps: float = 0.0) -> pd.Series:
    """Cumulative R, trade by trade, net of cost — the line a trader actually watches climb or bleed."""
    resolved = trades[trades["outcome"] != 0].reset_index(drop=True)
    if resolved.empty:
        return pd.Series(dtype="float64", name="cum_R")
    cR = roundtrip_bps * 1e-4 / resolved["risk_frac"]
    net = resolved["r_multiple"] - cR
    return net.cumsum().rename("cum_R")


def cost_sweep(trades: pd.DataFrame, roundtrip_bps=(0, 1, 2, 5, 10, 20)) -> pd.DataFrame:
    """Net expectancy (in R) as a function of round-trip cost — tradability made concrete.

    The break-even cost is where net expectancy crosses zero; below the strategy's true edge (often
    ≤ 0 already), above it the spread wins. On the 1-minute chart even a few bps matters because R is
    a ~1% stop, so cost-in-R is large.
    """
    rows = {}
    for c in roundtrip_bps:
        s = summary(trades, roundtrip_bps=c)
        rows[c] = {"win_rate": s["win_rate"], "cost_R": s["cost_R"],
                   "expectancy_R_net": s["expectancy_R_net"]}
    out = pd.DataFrame(rows).T
    out.index.name = "roundtrip_bps"
    return out
