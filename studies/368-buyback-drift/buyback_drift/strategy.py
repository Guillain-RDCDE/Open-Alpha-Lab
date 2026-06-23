"""Strategy + inference for Study 368 — Buyback-Drift.

The claim (believers' framing): after a company announces a big share-buyback **authorization**,
the stock **drifts up for months**. We test it as a clean event study on **abnormal** returns —
the event stock's forward return *minus SPY's* over the same window, so we are asking about drift
*beyond the market*, not "did the stock go up while the whole market rallied."

For each announcement we:

  * enter the close **one day after** the announcement (no look-ahead — the press release is
    public at the close of day 0, you trade day +1);
  * hold a fixed horizon (1 / 3 / 6 months) and record the **abnormal** return
    ``(stock_T/stock_0) - (SPY_T/SPY_0)``;
  * test the mean abnormal drift against zero with a **Welch t**, and — because the sample is a
    few dozen — against a **same-names placebo null**: re-draw each event's entry on a random
    date *for the same ticker*, thousands of times, and ask how often chance matches the real
    announcement set (this controls for each name's own idiosyncratic drift/vol, not just SPY);
  * report the **win-rate** (P[abnormal return > 0]) against the 50% coin-flip base rate;
  * apply **one-way costs** to a "buy on the announcement, hold H months" rule.

The decisive object is not the sign of the point estimate but its **standard error**: with ~30
noisy single-name events, a few-percent abnormal drift is swamped by single-stock variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = {1: 21, 3: 63, 6: 126, 12: 252}      # months -> trading days


# --------------------------------------------------------------------------- #
# Abnormal forward returns
# --------------------------------------------------------------------------- #
def _pos_on_or_after(index: pd.DatetimeIndex, date: pd.Timestamp) -> int | None:
    """Integer position of the first index date on/after ``date`` (None if past the tape)."""
    i = index.searchsorted(date)
    if i >= len(index):
        return None
    return int(i)


def abnormal_returns(prices: pd.DataFrame, events: pd.DataFrame, months: int,
                     lag: int = 1) -> np.ndarray:
    """Abnormal (stock − SPY) forward ``months``-month returns after each announcement.

    Entry is the close ``lag`` trading days after the first available bar on/after the
    announcement date (no look-ahead); exit ``horizon`` days later. The abnormal return is
    ``(stock_exit/stock_entry - 1) - (SPY_exit/SPY_entry - 1)``. Events whose horizon runs past
    that ticker's tape (or whose ticker/SPY data is missing in the window) are dropped.
    """
    h = TRADING_DAYS[months]
    spy = prices["SPY"]
    out = []
    for _, row in events.iterrows():
        tk = row["ticker"]
        if tk not in prices.columns:
            continue
        s = prices[tk].dropna()
        idx = s.index
        a = _pos_on_or_after(idx, row["date"])
        if a is None:
            continue
        entry = a + lag
        exit_ = entry + h
        if exit_ >= len(idx):
            continue
        d_entry, d_exit = idx[entry], idx[exit_]
        try:
            sp_e, sp_x = spy.loc[d_entry], spy.loc[d_exit]
        except KeyError:
            continue
        if not np.isfinite([s.iloc[entry], s.iloc[exit_], sp_e, sp_x]).all():
            continue
        stock_ret = s.iloc[exit_] / s.iloc[entry] - 1.0
        mkt_ret = sp_x / sp_e - 1.0
        out.append(stock_ret - mkt_ret)
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, mu0: float = 0.0) -> float:
    """One-sample Welch/Student t of ``mean(sample) - mu0``. NaN if sample < 2."""
    if len(sample) < 2:
        return float("nan")
    m = sample.mean()
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float((m - mu0) / se)


def placebo_pvalue(prices: pd.DataFrame, events: pd.DataFrame, months: int,
                   n_draws: int = 20_000, lag: int = 1, seed: int = 368) -> dict:
    """Same-names placebo null: for each event re-draw the entry on a RANDOM valid date for the
    **same ticker**, many times, and ask how often a random draw's mean abnormal return
    matches/beats the real announcement set.

    This is stricter than a SPY-only baseline: it asks "could the same stocks, entered on
    random dates, have drifted this much beyond the market by luck?" — controlling for each
    name's own idiosyncratic drift and volatility. Returns the announcement mean, the placebo
    mean, and the empirical p-value P[placebo mean >= announcement mean].
    """
    h = TRADING_DAYS[months]
    spy = prices["SPY"]
    obs = abnormal_returns(prices, events, months, lag=lag)
    k = len(obs)
    if k == 0:
        return {"k": 0, "ann_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}

    # Precompute, per event ticker, the full vector of valid abnormal forward returns
    # (every date on which that name *could* have been entered) to sample from.
    pools: list[np.ndarray] = []
    for _, row in events.iterrows():
        tk = row["ticker"]
        if tk not in prices.columns:
            continue
        s = prices[tk].dropna()
        idx = s.index
        # align SPY to this name's dates
        sp = spy.reindex(idx)
        sv, pv = s.values, sp.values
        hi = len(idx) - h - lag - 1
        if hi <= 0:
            continue
        e = np.arange(lag, lag + hi)            # candidate entries
        x = e + h
        stock = sv[x] / sv[e] - 1.0
        mkt = pv[x] / pv[e] - 1.0
        ab = stock - mkt
        ab = ab[np.isfinite(ab)]
        if len(ab):
            pools.append(ab)
    if not pools:
        return {"k": k, "ann_mean": float(obs.mean()), "placebo_mean": float("nan"),
                "p_value": float("nan")}

    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        draw = [pool[rng.integers(0, len(pool))] for pool in pools]
        means[i] = np.mean(draw)
    obs_mean = float(obs.mean())
    p = float((means >= obs_mean).mean())
    return {"k": k, "ann_mean": obs_mean, "placebo_mean": float(means.mean()),
            "p_value": p}


def summarize(prices: pd.DataFrame, events: pd.DataFrame, months: int,
              lag: int = 1) -> dict:
    """Headline stats for one horizon: n, abnormal mean/median, win-rate (vs 50% base rate),
    one-sample Welch t (vs 0), and the same-names placebo p-value."""
    ab = abnormal_returns(prices, events, months, lag=lag)
    pl = placebo_pvalue(prices, events, months, lag=lag)
    return {
        "months": months,
        "n": int(len(ab)),
        "ab_mean": float(ab.mean()) if len(ab) else float("nan"),
        "ab_median": float(np.median(ab)) if len(ab) else float("nan"),
        "win": float((ab > 0).mean()) if len(ab) else float("nan"),
        "t": welch_t(ab, 0.0),
        "p_placebo": pl["p_value"],
        "placebo_mean": pl["placebo_mean"],
    }


def net_of_costs(prices: pd.DataFrame, events: pd.DataFrame, months: int,
                 cost_bps: float = 10.0, lag: int = 1) -> dict:
    """Buy-on-announcement / hold-``months`` rule, one round-trip cost in bps per trade.

    Single-name round-trips (buy the stock, sell H months later) pay more spread than a
    broad-ETF trade, so we charge 10 bps round-trip by default. Returns average gross and net
    abnormal trade return — but the point is not the cost, it is the variance: a per-trade
    abnormal drift inside its own error bar is not a strategy at any cost."""
    gross = abnormal_returns(prices, events, months, lag=lag)
    c = cost_bps / 1e4
    net = gross - c
    return {
        "n_trades": int(len(gross)),
        "gross_mean": float(gross.mean()) if len(gross) else float("nan"),
        "net_mean": float(net.mean()) if len(net) else float("nan"),
        "cost_bps": cost_bps,
    }
