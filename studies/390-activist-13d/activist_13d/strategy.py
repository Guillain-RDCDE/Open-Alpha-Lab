"""Strategy + inference for Study 390 — Activist-13D.

The folklore, made precise. Around each activist 13D announcement we measure two distinct
objects, because they have completely different tradability:

  * **The announcement pop** — the target's return on the announcement day (day 0) and the
    day-0→+1 move. This is the news being priced in; you cannot trade it unless you knew the
    filing was coming (you didn't), so it is *not* a strategy — it is the efficient-market
    re-pricing. We report it to show the news is real.

  * **The post-announcement drift** — what you get if you buy at the close *one day after*
    the announcement (a realistic, no-look-ahead entry) and hold 1 / 3 / 6 months, measured
    **in excess of SPY** over the same window (so we are racing the target against the market,
    not against cash). This is the only leg you can actually trade, and it is the one the
    "keeps drifting up" claim lives or dies on.

Inference on the drift leg (the Signal axis):

  * a **Welch t** of the mean excess drift against zero (no-drift null);
  * a **placebo / bootstrap** null — for each event, draw the *same* horizon excess return
    from a *random* date on the same target, repeat thousands of times, and ask how often a
    random-date basket matches/beats the real announcement basket (the honest small-sample
    test, controlling for the targets' own drift);
  * a **win-rate** (P[excess drift > 0]) vs the 50% coin-flip base rate;
  * **one-day execution lag** baked into the entry, and **one-way costs** applied to the
    buy-and-hold drift trade.

The decisive number is not the announcement pop (it is real and positive — the news matters)
but the *post-announcement excess drift you can buy*: across a basket selected because it is
famous, is the average drift bigger than what random dates on the same names deliver?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = {1: 21, 3: 63, 6: 126}        # months -> trading days


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pos_on_or_after(index: pd.DatetimeIndex, date: pd.Timestamp) -> int | None:
    """Integer position of the first index date >= ``date`` (None if past the tape)."""
    i = index.searchsorted(date, side="left")
    if i >= len(index):
        return None
    return int(i)


# --------------------------------------------------------------------------- #
# Announcement pop (day-0 effect — reported, not tradable)
# --------------------------------------------------------------------------- #
def announcement_pops(prices: pd.DataFrame, table: pd.DataFrame) -> np.ndarray:
    """Day-0 announcement return of each target (the news being priced in).

    Day 0 = the first trading day on/after the announcement date. The pop is that day's
    simple return (close vs prior close). Events whose target has no usable window dropped.
    """
    out = []
    for _, r in table.iterrows():
        tkr = r["ticker"]
        if tkr not in prices.columns:
            continue
        s = prices[tkr].dropna()
        i = _pos_on_or_after(s.index, r["announce"])
        if i is None or i < 1 or i >= len(s):
            continue
        out.append(s.iloc[i] / s.iloc[i - 1] - 1.0)
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Post-announcement drift (excess of SPY) — the tradable leg
# --------------------------------------------------------------------------- #
def event_excess_drift(prices: pd.DataFrame, table: pd.DataFrame, months: int,
                       lag: int = 1) -> np.ndarray:
    """Excess-of-SPY return of buying each target ``lag`` days after the announcement and
    holding ``months`` months (no look-ahead).

    For each event: entry = the close ``lag`` trading days after day 0; exit = ``horizon``
    days later. Excess = target return − SPY return over the identical calendar window.
    Events whose horizon overruns the target's or SPY's tape are dropped.
    """
    h = TRADING_DAYS[months]
    spy = prices["SPY"].dropna()
    out = []
    for _, r in table.iterrows():
        tkr = r["ticker"]
        if tkr not in prices.columns:
            continue
        s = prices[tkr].dropna()
        i0 = _pos_on_or_after(s.index, r["announce"])
        if i0 is None:
            continue
        entry = i0 + lag
        exit_ = entry + h
        if exit_ >= len(s):
            continue
        d_entry, d_exit = s.index[entry], s.index[exit_]
        je = _pos_on_or_after(spy.index, d_entry)
        jx = _pos_on_or_after(spy.index, d_exit)
        if je is None or jx is None or jx >= len(spy):
            continue
        tgt_ret = s.iloc[exit_] / s.iloc[entry] - 1.0
        spy_ret = spy.iloc[jx] / spy.iloc[je] - 1.0
        out.append(tgt_ret - spy_ret)
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t_vs_zero(sample: np.ndarray) -> float:
    """One-sample t of ``mean(sample)`` against 0 (the no-drift null). NaN if n < 2."""
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float(sample.mean() / se)


def placebo_pvalue(prices: pd.DataFrame, table: pd.DataFrame, months: int,
                   n_draws: int = 20_000, lag: int = 1, seed: int = 390) -> dict:
    """Small-sample placebo null controlling for the targets' own drift.

    For each event, the alternative-history return is the same ``months``-horizon excess
    (target − SPY) measured from a *random* valid entry date on the **same target**. We draw
    a full basket of such random-date excess returns ``n_draws`` times and ask how often the
    random basket's mean matches/beats the real announcement basket. This nulls out the fact
    that activist targets are special names (cheap, volatile, sometimes trending) — we ask
    specifically whether the *announcement timing* adds drift over a random day on that name.
    """
    h = TRADING_DAYS[months]
    spy = prices["SPY"].dropna()
    # Precompute, per real event, the pool of candidate excess returns on its own target.
    pools = []
    obs = event_excess_drift(prices, table, months, lag=lag)
    for _, r in table.iterrows():
        tkr = r["ticker"]
        if tkr not in prices.columns:
            continue
        s = prices[tkr].dropna()
        i0 = _pos_on_or_after(s.index, r["announce"])
        if i0 is None:
            continue
        entry0 = i0 + lag
        if entry0 + h >= len(s):
            continue
        # candidate entries: any day on this target with a full horizon ahead, aligned to SPY
        common = s.index.intersection(spy.index)
        sc = s.reindex(common)
        spc = spy.reindex(common)
        n = len(common)
        if n <= h + 2:
            continue
        tgt_fwd = sc.values[h:] / sc.values[:-h] - 1.0
        spy_fwd = spc.values[h:] / spc.values[:-h] - 1.0
        pool = tgt_fwd - spy_fwd
        pool = pool[np.isfinite(pool)]
        if len(pool):
            pools.append(pool)
    k = len(obs)
    if k == 0 or not pools:
        return {"k": 0, "obs_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        means[i] = np.mean([p[rng.integers(0, len(p))] for p in pools])
    obs_mean = float(obs.mean())
    p = float((means >= obs_mean).mean())
    return {"k": k, "obs_mean": obs_mean, "placebo_mean": float(means.mean()),
            "p_value": p}


def summarize(prices: pd.DataFrame, table: pd.DataFrame, months: int,
              lag: int = 1) -> dict:
    """Headline stats for one horizon: n, mean/median excess drift, win-rate vs 50%,
    Welch t vs zero, and the placebo p-value."""
    ex = event_excess_drift(prices, table, months, lag=lag)
    pl = placebo_pvalue(prices, table, months, lag=lag)
    return {
        "months": months,
        "n": int(len(ex)),
        "mean_excess": float(ex.mean()) if len(ex) else float("nan"),
        "median_excess": float(np.median(ex)) if len(ex) else float("nan"),
        "win_rate": float((ex > 0).mean()) if len(ex) else float("nan"),
        "t": welch_t_vs_zero(ex),
        "p_placebo": pl["p_value"],
    }


def net_of_costs(prices: pd.DataFrame, table: pd.DataFrame, months: int,
                 cost_bps: float = 10.0, lag: int = 1) -> dict:
    """Buy-1-day-after-announcement / hold-``months`` rule, one round-trip cost in bps.

    Returns the average gross and net excess-of-SPY trade return. Costs are one round-trip
    per trade (enter + exit). The point of this leg is that costs are small relative to a
    multi-month hold — so if the drift were real, cost would not kill it; the binding
    question is whether the gross drift exists at all."""
    gross = event_excess_drift(prices, table, months, lag=lag)
    c = cost_bps / 1e4
    net = gross - c
    return {
        "n_trades": int(len(gross)),
        "gross_mean": float(gross.mean()) if len(gross) else float("nan"),
        "net_mean": float(net.mean()) if len(net) else float("nan"),
        "cost_bps": cost_bps,
    }
