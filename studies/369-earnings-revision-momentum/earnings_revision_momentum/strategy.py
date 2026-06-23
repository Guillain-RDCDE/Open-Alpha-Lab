"""Strategy + inference for Study 369 — Earnings-Revision-Momentum.

The claim (believers' framing): *stocks whose earnings estimates are being revised up keep
beating the market.* Paid analyst-revision feeds (IBES) are not free, so we test a
transparent **proxy** built from yfinance: a stock's realized **earnings surprise** (reported
EPS beats the consensus estimate) and the **q/q change in the consensus estimate** — a name
that keeps beating and whose estimate keeps rising is the public stand-in for "up-revision."
This is the post-earnings-announcement-drift / standardized-unexpected-earnings (SUE) family.

Each earnings quarter we:

  * score every stock by its revision proxy (``rev_score``);
  * rank cross-sectionally and form a **long top-tercile / short bottom-tercile** book;
  * enter the close **1 day after** the earnings date (no look-ahead) and hold ``horizon``
    trading days, measuring each leg's return **in excess of SPY** (does it beat the market?);
  * test the long-short spread with a **Welch t** vs zero and a **placebo** null that
    shuffles the score labels (so any spread is destroyed); apply **one-way costs × turnover**
    (+ a borrow charge on the short leg);
  * report the **hit-rate** of the long leg beating SPY vs the 50% coin-flip base rate.

The decisive object is whether the long-short spread survives execution lag, costs, borrow and
the placebo null on the **real** tape — not whether the raw surprise looks predictive in-sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_HORIZON = 63          # ~one quarter, the post-earnings-drift window
COST_BPS = 10.0               # one-way transaction cost per leg per rebalance
BORROW_BPS_ANN = 50.0         # annualized borrow on the short leg


# --------------------------------------------------------------------------- #
# Forward excess returns around each earnings event
# --------------------------------------------------------------------------- #
def _pos_map(index: pd.DatetimeIndex) -> dict:
    return {d: i for i, d in enumerate(index)}


def event_excess_returns(prices: pd.DataFrame, earnings: pd.DataFrame,
                         horizon: int = DEFAULT_HORIZON, lag: int = 1) -> pd.DataFrame:
    """Attach to each earnings event the stock's forward ``horizon``-day return in EXCESS of SPY.

    Entry is the close ``lag`` days after the earnings date (no look-ahead), exit ``horizon``
    days later. Events whose ticker is missing from the price cache, or whose horizon runs
    past the tape, are dropped. Returns the earnings frame with added columns
    ``entry_date, stk_ret, spy_ret, excess``.
    """
    spy = prices["SPY"]
    pos = _pos_map(prices.index)
    idx = prices.index
    out = []
    for row in earnings.itertuples(index=False):
        tk = row.ticker
        if tk not in prices.columns:
            continue
        d = pd.Timestamp(row.date).normalize()
        # first trading day on/after the earnings date
        loc = idx.searchsorted(d)
        if loc >= len(idx):
            continue
        i = loc + lag
        j = i + horizon
        if j >= len(idx):
            continue
        s = prices[tk]
        p0, p1 = s.iloc[i], s.iloc[j]
        sp0, sp1 = spy.iloc[i], spy.iloc[j]
        if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0 and np.isfinite(sp0) and sp0 > 0):
            continue
        stk = p1 / p0 - 1.0
        spr = sp1 / sp0 - 1.0
        out.append((tk, d, idx[i], float(stk), float(spr), float(stk - spr),
                    float(row.rev_score) if np.isfinite(row.rev_score) else np.nan))
    cols = ["ticker", "date", "entry_date", "stk_ret", "spy_ret", "excess", "rev_score"]
    return pd.DataFrame(out, columns=cols).dropna(subset=["rev_score"])


# --------------------------------------------------------------------------- #
# Cross-sectional revision score
# --------------------------------------------------------------------------- #
def build_rev_score(earnings: pd.DataFrame) -> pd.DataFrame:
    """Combine the surprise + estimate-revision proxies into one cross-sectional z-score.

    Within each earnings *quarter* (grouped on calendar quarter) we z-score the realized
    surprise and the q/q estimate revision and average them. Only past/realized inputs enter
    — the score is known on the earnings date, the trade enters the next day.
    """
    e = earnings.copy()
    e["q"] = e["date"].dt.to_period("Q")

    def _z(s):
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd and np.isfinite(sd) else s * 0.0

    parts = []
    for _, g in e.groupby("q"):
        g = g.copy()
        zs = _z(g["surprise"].fillna(0.0))
        zr = _z(g["est_revision"].fillna(0.0))
        g["rev_score"] = (zs + zr) / 2.0
        parts.append(g)
    out = pd.concat(parts).sort_values(["date", "ticker"])
    return out


# --------------------------------------------------------------------------- #
# Long-short tercile book
# --------------------------------------------------------------------------- #
def quantile_spread(ev: pd.DataFrame, q: float = 1 / 3) -> dict:
    """Per-quarter long top / short bottom tercile by ``rev_score``; spread = long − short.

    For each earnings quarter, rank events by ``rev_score``; the top ``q`` go long, the
    bottom ``q`` short. The quarter's spread is the equal-weight mean *excess* return of the
    long leg minus that of the short leg. Returns per-quarter arrays + the long-leg hit-rate.
    """
    ev = ev.copy()
    ev["q"] = ev["date"].dt.to_period("Q")
    long_ex, short_ex, spreads, long_hits, n_q = [], [], [], [], 0
    long_all_ex = []
    for _, g in ev.groupby("q"):
        if len(g) < 6:
            continue
        g = g.sort_values("rev_score")
        k = max(1, int(round(len(g) * q)))
        lo = g.iloc[:k]            # bottom tercile -> short
        hi = g.iloc[-k:]           # top tercile -> long
        l = hi["excess"].mean()
        s = lo["excess"].mean()
        long_ex.append(l); short_ex.append(s); spreads.append(l - s)
        long_all_ex.extend(hi["excess"].tolist())
        n_q += 1
    long_hits = np.asarray([x > 0 for x in long_all_ex], dtype=float)
    return {
        "n_quarters": n_q,
        "long_ex": np.asarray(long_ex),
        "short_ex": np.asarray(short_ex),
        "spread": np.asarray(spreads),
        "long_hit_rate": float(long_hits.mean()) if len(long_hits) else float("nan"),
        "long_n": int(len(long_hits)),
    }


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, mu: float = 0.0) -> float:
    """One-sample t of ``mean(sample) - mu`` (HAC-free; quarters ~ independent draws)."""
    sample = np.asarray(sample, dtype=float)
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float((sample.mean() - mu) / se)


def placebo_pvalue(ev: pd.DataFrame, q: float = 1 / 3, n_draws: int = 10_000,
                   seed: int = 369) -> dict:
    """Label-shuffle placebo: within each quarter permute ``rev_score`` across stocks, rebuild
    the long-short spread, and ask how often a shuffled book's mean spread matches/beats the
    real one. Destroys any real link between the score and forward returns.

    Vectorized over draws per quarter: a random score permutation is equivalent to a random
    partition into a top-k (long) and bottom-k (short) bucket, so we draw random key matrices
    and argsort them once per quarter.
    """
    real = quantile_spread(ev, q=q)
    obs = float(real["spread"].mean()) if len(real["spread"]) else float("nan")
    if not np.isfinite(obs):
        return {"obs": float("nan"), "placebo_mean": float("nan"), "p_value": float("nan")}
    ev = ev.copy()
    ev["q"] = ev["date"].dt.to_period("Q")
    groups = [g["excess"].values for _, g in ev.groupby("q") if len(g) >= 6]
    rng = np.random.default_rng(seed)
    nq = len(groups)
    acc = np.zeros(n_draws)
    for ex in groups:
        m = len(ex)
        k = max(1, int(round(m * q)))
        keys = rng.random((n_draws, m))          # random order per draw
        order = np.argsort(keys, axis=1)
        hi = ex[order[:, -k:]].mean(axis=1)
        lo = ex[order[:, :k]].mean(axis=1)
        acc += (hi - lo)
    means = acc / nq
    p = float((means >= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()), "p_value": p,
            "n_quarters": real["n_quarters"]}


def summarize(prices: pd.DataFrame, earnings: pd.DataFrame, horizon: int = DEFAULT_HORIZON,
              q: float = 1 / 3, lag: int = 1) -> dict:
    """Headline stats: long-short spread, its t, the placebo p, the long-leg hit-rate."""
    scored = build_rev_score(earnings)
    ev = event_excess_returns(prices, scored, horizon=horizon, lag=lag)
    qs = quantile_spread(ev, q=q)
    pl = placebo_pvalue(ev, q=q)
    spread = qs["spread"]
    return {
        "horizon": horizon,
        "n_events": int(len(ev)),
        "n_quarters": qs["n_quarters"],
        "long_ex_mean": float(qs["long_ex"].mean()) if len(qs["long_ex"]) else float("nan"),
        "short_ex_mean": float(qs["short_ex"].mean()) if len(qs["short_ex"]) else float("nan"),
        "spread_mean": float(spread.mean()) if len(spread) else float("nan"),
        "spread_t": welch_t(spread),
        "p_placebo": pl["p_value"],
        "long_hit_rate": qs["long_hit_rate"],
        "long_n": qs["long_n"],
    }


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def net_of_costs(prices: pd.DataFrame, earnings: pd.DataFrame, horizon: int = DEFAULT_HORIZON,
                 q: float = 1 / 3, cost_bps: float = COST_BPS,
                 borrow_bps_ann: float = BORROW_BPS_ANN, lag: int = 1) -> dict:
    """Gross vs net per-quarter long-short spread.

    Each quarter rebalances both legs: a one-way cost ``cost_bps`` per leg on entry and exit
    (4 crossings: buy/sell long, sell/buy short) plus a borrow charge on the short leg pro-rated
    to the holding period. Reported per-quarter (the natural rebalance frequency)."""
    scored = build_rev_score(earnings)
    ev = event_excess_returns(prices, scored, horizon=horizon, lag=lag)
    qs = quantile_spread(ev, q=q)
    gross = qs["spread"]
    c = cost_bps / 1e4
    turn_cost = 4 * c                                   # long+short, in and out
    borrow = (borrow_bps_ann / 1e4) * (horizon / 252.0)  # short-leg borrow over the hold
    net = gross - turn_cost - borrow
    return {
        "n_quarters": qs["n_quarters"],
        "gross_mean": float(gross.mean()) if len(gross) else float("nan"),
        "net_mean": float(net.mean()) if len(net) else float("nan"),
        "gross_t": welch_t(gross),
        "net_t": welch_t(net),
        "cost_bps": cost_bps,
        "borrow_bps_ann": borrow_bps_ann,
    }


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_spread(panel: pd.DataFrame, q: float = 1 / 3) -> dict:
    """Long-short tercile spread + inference on the synthetic panel (decorative int quarter).

    Mirrors ``quantile_spread`` but on the synthetic ``quarter/stock/rev_score/fwd_ret`` frame.
    With the planted ``edge=0`` the spread must be insignificant; with a large planted edge it
    must light up — the faithful-engine / power control.
    """
    spreads, long_hits = [], []
    for _, g in panel.groupby("quarter"):
        if len(g) < 6:
            continue
        g = g.sort_values("rev_score")
        k = max(1, int(round(len(g) * q)))
        lo = g.iloc[:k]["fwd_ret"].mean()
        hi = g.iloc[-k:]["fwd_ret"].mean()
        spreads.append(hi - lo)
        long_hits.extend((g.iloc[-k:]["fwd_ret"] > g["fwd_ret"].median()).tolist())
    spreads = np.asarray(spreads)
    return {
        "n_quarters": int(len(spreads)),
        "spread_mean": float(spreads.mean()) if len(spreads) else float("nan"),
        "spread_t": welch_t(spreads),
    }


def placebo_pvalue_synth(panel: pd.DataFrame, q: float = 1 / 3, n_draws: int = 5_000,
                         seed: int = 369) -> float:
    """Label-shuffle placebo p-value on the synthetic panel (vectorized over draws)."""
    obs = synthetic_spread(panel, q=q)["spread_mean"]
    if not np.isfinite(obs):
        return float("nan")
    groups = [g["fwd_ret"].values for _, g in panel.groupby("quarter") if len(g) >= 6]
    rng = np.random.default_rng(seed)
    nq = len(groups)
    acc = np.zeros(n_draws)
    for ex in groups:
        m = len(ex)
        k = max(1, int(round(m * q)))
        keys = rng.random((n_draws, m))
        order = np.argsort(keys, axis=1)
        acc += ex[order[:, -k:]].mean(axis=1) - ex[order[:, :k]].mean(axis=1)
    return float(((acc / nq) >= obs).mean())
