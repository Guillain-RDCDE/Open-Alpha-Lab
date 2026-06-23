"""Strategy + inference for Study 378 — ETF NAV premium/discount.

The claim under test: *buy an ETF when it trades below its NAV and you earn the discount
back.* We operationalise it on the NAV-proxy **basis** (see :mod:`etf_nav_premium.data`):

    A **discount** fires at date ``t`` when the basis z-score (rolling ``Z_WINDOW``) drops
    below ``z_thr`` (default -1). You enter the ETF the *next* close (1-day lag, no
    look-ahead) and hold ``h`` trading days. The thing you actually harvest is the
    **hedged residual** — the ETF's return net of its hedge-basket beta — summed over the
    hold. That isolates "did the gap close?" from "did the whole bond market rally?".

Inference, the desk's standard battery:

  * **HAC (Newey-West) t** of the mean hedged harvest (overlapping holds ⇒ autocorrelation,
    so a plain t lies — we use a Bartlett-kernel HAC SE);
  * a **placebo / randomization null** — draw the same number of *random* entry dates many
    times and ask how often chance harvests as much (the honest small-edge test);
  * a **win-rate** vs the 50% coin-flip base rate;
  * **one-way costs** (a realistic bond-ETF half-spread, charged on the ETF *and* the hedge),
    because the gross harvest is only a few basis points and costs decide everything.

The decisive number is not the sign (the basis does partially revert) but the magnitude vs
cost: a ~5 bp gross harvest cannot survive a multi-bp round-trip on two legs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Z_WINDOW


# --------------------------------------------------------------------------- #
# Discount detection
# --------------------------------------------------------------------------- #
def basis_zscore(basis: pd.Series, window: int = Z_WINDOW) -> pd.Series:
    """Rolling z-score of the prem/disc basis (the discount-depth gauge)."""
    mu = basis.rolling(window, min_periods=60).mean()
    sd = basis.rolling(window, min_periods=60).std()
    return (basis - mu) / sd


def detect_discounts(frame: pd.DataFrame, z_thr: float = -1.0,
                     window: int = Z_WINDOW) -> pd.DatetimeIndex:
    """Dates on which the ETF trades at a **discount** (basis z below ``z_thr``)."""
    z = basis_zscore(frame["basis"], window=window)
    sig = (z < z_thr) & z.notna()
    return frame.index[sig.values]


# --------------------------------------------------------------------------- #
# Hedged harvest (the "earn the discount back" payoff)
# --------------------------------------------------------------------------- #
def hedged_harvest(frame: pd.DataFrame, events: pd.DatetimeIndex, h: int,
                   lag: int = 1) -> np.ndarray:
    """Sum of the hedged residual return over the ``h`` days starting ``lag`` after each
    discount (no look-ahead). This is the discount actually *earned back*, net of the move
    in the underlying bond market. Events whose hold overruns the tape are dropped."""
    resid = frame["resid"].values
    pos = {d: i for i, d in enumerate(frame.index)}
    n = len(frame)
    out = []
    for d in events:
        i = pos.get(d)
        if i is None:
            continue
        a = i + lag
        b = a + h
        if b > n:
            continue
        out.append(resid[a:b].sum())
    return np.asarray(out, dtype=float)


def discount_depth(frame: pd.DataFrame, events: pd.DatetimeIndex) -> np.ndarray:
    """The basis level (the discount itself, negative) on each signal date."""
    return frame["basis"].reindex(events).dropna().values


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_t(x: np.ndarray, lags: int = 10) -> float:
    """Newey-West (Bartlett) HAC t-stat of mean(x) vs 0. NaN if < 3 obs."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    m = x.mean()
    xd = x - m
    g0 = (xd @ xd) / n
    s = g0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1)
        s += 2.0 * w * (xd[k:] @ xd[:-k]) / n
    if s <= 0:
        return float("nan")
    return float(m / np.sqrt(s / n))


def placebo_pvalue(frame: pd.DataFrame, events: pd.DatetimeIndex, h: int,
                   n_draws: int = 20_000, lag: int = 1, seed: int = 378) -> dict:
    """Randomization null: draw ``k`` random valid entry dates ``n_draws`` times and ask how
    often a random draw's mean hedged harvest matches/beats the discount set."""
    obs = hedged_harvest(frame, events, h, lag=lag)
    k = len(obs)
    resid = frame["resid"].values
    n = len(resid)
    hi = n - h - lag
    if k == 0 or hi <= k:
        return {"k": k, "obs_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    # precompute every overlapping h-day forward residual sum from a valid entry
    csum = np.concatenate([[0.0], np.cumsum(resid)])
    fwd = csum[lag + h: lag + h + hi] - csum[lag: lag + hi]
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        means[i] = fwd[rng.integers(0, hi, size=k)].mean()
    obs_mean = float(obs.mean())
    return {"k": k, "obs_mean": obs_mean, "placebo_mean": float(means.mean()),
            "p_value": float((means >= obs_mean).mean())}


def summarize(frame: pd.DataFrame, events: pd.DatetimeIndex, h: int,
              lag: int = 1) -> dict:
    """Headline stats for one ETF / horizon: n, mean & median discount depth, mean hedged
    harvest, win-rate, HAC t, placebo p."""
    hv = hedged_harvest(frame, events, h, lag=lag)
    depth = discount_depth(frame, events)
    pl = placebo_pvalue(frame, events, h, lag=lag)
    return {
        "h": h,
        "n": int(len(hv)),
        "depth_mean": float(np.mean(depth)) if len(depth) else float("nan"),
        "harvest_mean": float(hv.mean()) if len(hv) else float("nan"),
        "win": float((hv > 0).mean()) if len(hv) else float("nan"),
        "t": hac_t(hv),
        "p_placebo": pl["p_value"],
    }


def pooled_harvest(frames: dict, h: int, z_thr: float = -1.0, lag: int = 1) -> dict:
    """Pool the hedged harvest across every target ETF (one cross-ETF discount factor).

    Returns the pooled n, mean harvest, win-rate and HAC t — the cleanest single number for
    'does buying any discount earn it back?'."""
    xs = []
    for f in frames.values():
        ev = detect_discounts(f, z_thr=z_thr)
        xs.append(hedged_harvest(f, ev, h, lag=lag))
    x = np.concatenate(xs) if xs else np.array([])
    return {
        "h": h,
        "n": int(len(x)),
        "harvest_mean": float(x.mean()) if len(x) else float("nan"),
        "win": float((x > 0).mean()) if len(x) else float("nan"),
        "t": hac_t(x),
    }


def net_of_costs(harvest_mean: float, cost_bps_per_leg: float = 2.0,
                 n_legs: int = 2) -> dict:
    """Net the gross harvest against a realistic bond-ETF round-trip.

    A discount trade is round-tripped on the ETF (in then out) *and* — if you actually hedge
    the basket as the harvest measurement assumes — on the hedge leg(s). We charge one
    one-way half-spread per fill: ``2 * n_legs`` half-spreads of ``cost_bps_per_leg`` each
    (buy+sell on each of ``n_legs`` instruments). Bond-ETF half-spreads of ~1-3 bp are
    realistic for HYG/EMB/MUB in calm markets and blow out in stress."""
    cost = 2 * n_legs * cost_bps_per_leg / 1e4
    return {
        "gross": harvest_mean,
        "cost": cost,
        "net": harvest_mean - cost,
        "cost_bps_per_leg": cost_bps_per_leg,
        "n_legs": n_legs,
    }
