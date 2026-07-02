"""Strategy + inference for Study 563 — Secondary-Offering-Drift.

The claim (dilution + negative-signalling framing): after a company sells a fresh slug of shares
in a **seasoned / secondary equity offering**, the stock **drifts down for months**. We test it as
a clean event study on **abnormal** returns — the event stock's forward return *minus SPY's* over
the same window, so we ask about drift *beyond the market*, not "did the stock fall while the whole
market fell." A *negative* abnormal drift is the claim's prediction.

For each offering we:

  * enter the close **one day after** the pricing/announcement (no look-ahead — the deal is
    public at the close of day 0, you act day +1);
  * hold a fixed horizon (1 / 3 / 6 / 12 months) and record the **abnormal** return
    ``(stock_T/stock_0) - (SPY_T/SPY_0)``;
  * test the mean abnormal drift against zero with a **Welch t**, and — because the sample is a
    few dozen — against a **same-names placebo null**: re-draw each event's entry on a random
    date *for the same ticker*, thousands of times, and ask how often chance produces a drift as
    *negative* as the real offering set (this controls for each name's own idiosyncratic
    drift/vol, not just SPY);
  * report the **loss-rate** (P[abnormal return < 0]) against the 50% coin-flip base rate;
  * cost the tradable expression — **short the issuer** — with a one-way cost AND a short borrow
    (the offering names skew hard-to-borrow high-fliers), because the honest trade here is a short.

The decisive object is not the sign of the point estimate but its **standard error**: with a few
dozen noisy single-name events, a few-percent abnormal drift is swamped by single-stock variance.
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
    """Abnormal (stock − SPY) forward ``months``-month returns after each offering.

    Entry is the close ``lag`` trading days after the first available bar on/after the offering
    date (no look-ahead); exit ``horizon`` days later. The abnormal return is
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
                   n_draws: int = 20_000, lag: int = 1, seed: int = 563) -> dict:
    """Same-names placebo null: for each event re-draw the entry on a RANDOM valid date for the
    **same ticker**, many times, and ask how often a random draw's mean abnormal return is as
    **negative as** the real offering set.

    This is stricter than a SPY-only baseline: it asks "could the same stocks, entered on random
    dates, have drifted down this much beyond the market by luck?" — controlling for each name's
    own idiosyncratic drift and volatility. Because the claim is a *negative* drift, the p-value
    is the left-tail probability ``P[placebo mean <= offering mean]``. Returns the offering mean,
    the placebo mean, and that p-value.
    """
    h = TRADING_DAYS[months]
    spy = prices["SPY"]
    obs = abnormal_returns(prices, events, months, lag=lag)
    k = len(obs)
    if k == 0:
        return {"k": 0, "off_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}

    pools: list[np.ndarray] = []
    for _, row in events.iterrows():
        tk = row["ticker"]
        if tk not in prices.columns:
            continue
        s = prices[tk].dropna()
        idx = s.index
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
        return {"k": k, "off_mean": float(obs.mean()), "placebo_mean": float("nan"),
                "p_value": float("nan")}

    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        draw = [pool[rng.integers(0, len(pool))] for pool in pools]
        means[i] = np.mean(draw)
    obs_mean = float(obs.mean())
    p = float((means <= obs_mean).mean())       # left tail: as negative as observed
    return {"k": k, "off_mean": obs_mean, "placebo_mean": float(means.mean()),
            "p_value": p}


def summarize(prices: pd.DataFrame, events: pd.DataFrame, months: int,
              lag: int = 1) -> dict:
    """Headline stats for one horizon: n, abnormal mean/median, loss-rate (vs 50% base rate),
    one-sample Welch t (vs 0), and the same-names (left-tail) placebo p-value."""
    ab = abnormal_returns(prices, events, months, lag=lag)
    pl = placebo_pvalue(prices, events, months, lag=lag)
    return {
        "months": months,
        "n": int(len(ab)),
        "ab_mean": float(ab.mean()) if len(ab) else float("nan"),
        "ab_median": float(np.median(ab)) if len(ab) else float("nan"),
        "loss": float((ab < 0).mean()) if len(ab) else float("nan"),
        "t": welch_t(ab, 0.0),
        "p_placebo": pl["p_value"],
        "placebo_mean": pl["placebo_mean"],
    }


# --------------------------------------------------------------------------- #
# Costs + borrow — the honest trade is SHORT the issuer
# --------------------------------------------------------------------------- #
def net_of_costs(prices: pd.DataFrame, events: pd.DataFrame, months: int,
                 cost_bps: float = 10.0, borrow_ann_bps: float = 300.0,
                 lag: int = 1) -> dict:
    """Short-the-issuer / hold-``months`` rule: the P&L of the SHORT is ``-abnormal_return``,
    charged one round-trip cost + a pro-rated short borrow.

    The tradable expression of "the stock slides after an offering" is to **short the issuer**,
    so the short's abnormal P&L is ``-ab``. Offering names skew toward hard-to-borrow high-fliers
    (Tesla, crypto miners, pandemic-era names), so we charge a punitive **300 bps/yr borrow**
    plus a 10 bps single-name round-trip. Returns the average SHORT abnormal P&L gross and net —
    but the point is not the cost, it is the variance: a per-trade drift inside its own error bar
    is not a strategy at any cost.
    """
    ab = abnormal_returns(prices, events, months, lag=lag)
    if len(ab) == 0:
        return {"n_trades": 0, "short_gross": float("nan"), "short_net": float("nan"),
                "cost_bps": cost_bps, "borrow_bps": borrow_ann_bps}
    short_gross = -ab                                  # short P&L = negative of the drift
    c = cost_bps / 1e4                                 # one round-trip
    borrow = borrow_ann_bps / 1e4 * (months / 12.0)    # pro-rated annual borrow
    short_net = short_gross - c - borrow
    return {
        "n_trades": int(len(ab)),
        "short_gross": float(short_gross.mean()),
        "short_net": float(short_net.mean()),
        "cost_bps": cost_bps, "borrow_bps": borrow_ann_bps,
    }


# --------------------------------------------------------------------------- #
# Robustness — split by deal size (does "bigger offering => bigger slide" hold?)
# --------------------------------------------------------------------------- #
def size_split(prices: pd.DataFrame, events: pd.DataFrame, months: int = 6,
               lag: int = 1) -> list[dict]:
    """Split the events at the median deal size and re-run the drift for each half.

    The dilution story predicts a *bigger* offering (more shares sold) => a *bigger* slide. We
    test that by median-splitting on ``size_bn`` and reporting each bucket's abnormal mean, Welch
    t and placebo p. A single tempting sub-bucket p is flagged as the median-split multiple-test
    it is.
    """
    ev = events.copy()
    med = ev["size_bn"].median()
    buckets = [("BIG (>=median)", ev[ev["size_bn"] >= med]),
               ("SMALL (<median)", ev[ev["size_bn"] < med])]
    rows = []
    for label, sub in buckets:
        s = summarize(prices, sub, months, lag=lag)
        rows.append({"bucket": label, "n": s["n"], "ab_mean": s["ab_mean"],
                     "t": s["t"], "p_placebo": s["p_placebo"]})
    return rows


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_control(synthetic_events, edges=(0.0, -0.20), n_events: int = 34,
                      months: int = 6, n_seeds: int = 25) -> list[dict]:
    """Faithful-engine / power control averaged over ``n_seeds`` synthetic worlds.

    For each planted (negative) ``edge`` cumulative abnormal drift, build ``n_seeds`` independent
    synthetic event panels (seed = 563+k), measure the ``months``-horizon abnormal drift, and
    average the Welch t and abnormal mean across seeds (the house rule: any synthetic-dependent
    claim is averaged over many seeds, never a single lucky draw). With ``edge=0`` the averaged
    |t| must stay well under 2 (no false positive); with a large planted negative edge it must
    clear −2 (a detected downside drift).
    """
    rows = []
    for edge in edges:
        ts, means = [], []
        for k in range(n_seeds):
            px, ev = synthetic_events(n_events=n_events, edge=edge, seed=563 + k)
            ab = abnormal_returns(px, ev, months, lag=1)
            ts.append(welch_t(ab, 0.0))
            means.append(float(ab.mean()) if len(ab) else float("nan"))
        rows.append({"edge": edge, "n_seeds": n_seeds,
                     "mean_t": float(np.nanmean(ts)),
                     "ab_mean": float(np.nanmean(means))})
    return rows
