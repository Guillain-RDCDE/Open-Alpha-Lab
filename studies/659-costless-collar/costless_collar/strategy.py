"""Strategy + inference for Study 659 — Costless Collar.

The claim: *own SPY, buy a 5%-out-of-the-money put, sell a call whose premium exactly pays
for the put ("costless"), and you get equity upside with crash protection — for free.*

There is no live option chain anywhere in this study (yfinance has no historical SPY option
prices). Instead we build a **stylized monthly collar** with a Black-Scholes approximation:

1. Each month's put is struck ``PUT_OTM`` (5%) below spot.
2. Each month's call strike is chosen so its Black-Scholes premium exactly matches the put's
   premium — the model's definition of "costless" (zero net premium at inception).
3. Both legs are priced off the **trailing realized volatility** (``data.trailing_realized_vol``,
   known before the month begins — the study's one documented execution lag), because we have
   no implied-vol surface. Realized vol *understates* true implied vol on average (the variance
   risk premium — see ``docs/references.md``), which biases our modeled strikes SLIGHTLY closer
   to the money than a real market maker would quote. Stated honestly, not hidden.
4. At month end the position pays off exactly like a real collar: floored at the put strike,
   capped at the call strike, full participation in between.

Measurements:

* **The floor bites in crashes.** Restrict to months where SPY fell more than 5% — the put's
  own strike — and test whether the collar's return is significantly higher than SPY's (paired
  one-sample t, HAC cross-check). This is close to true "by construction", but it is exactly
  the mechanism the claim is selling, so we report the size and the significance of it anyway.
* **The cap bites in bull months.** Restrict to months where SPY's return exceeded that month's
  modeled cap — test whether the collar's return is significantly LOWER than SPY's there. This
  is the "no free lunch" half of the claim that the marketing omits.
* **The net drag, full sample.** Mean(collar − SPY) across every month, Newey-West HAC t — is
  the average cost of running this "free" structure statistically distinguishable from zero?
* **Terminal wealth & Sharpe.** Cumulative growth of $1, and Sharpe of MONTHLY EXCESS returns
  (over the same constant cash rate) for both legs — smoother ride vs. slower compounding.
* **Named crash windows.** Max drawdown, collar vs. SPY, inside the 2008 GFC and 2020 COVID
  windows specifically — the two events the claim is actually selling protection against.

The decisive numbers are the paired t-stats on the REAL SPY tape; the synthetic control only
proves the clip-and-drag machinery is unbiased.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Black-Scholes pricing, pure stdlib (no scipy dependency)
# --------------------------------------------------------------------------- #
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if sigma <= 0 or T <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if sigma <= 0 or T <= 0:
        return max(K * math.exp(-r * T) - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def solve_costless_call_strike(S: float, K_put: float, T: float, r: float, sigma: float,
                                lo_mult: float = 1.0001, hi_mult: float = 4.0,
                                tol: float = 1e-8, max_iter: int = 200) -> float:
    """Bisect for the call strike whose BS premium equals the put's BS premium.

    ``bs_call`` is strictly decreasing in K, so a target premium between
    ``bs_call(hi)`` and ``bs_call(lo)`` has a unique root. Falls back to the widest
    bracket's endpoint if the target sits outside it (near-zero-vol edge cases).
    """
    target = bs_put(S, K_put, T, r, sigma)
    lo, hi = S * lo_mult, S * hi_mult
    f_lo, f_hi = bs_call(S, lo, T, r, sigma) - target, bs_call(S, hi, T, r, sigma) - target
    if f_lo <= 0:      # even the tightest cap can't afford the put -> cap at spot*lo_mult
        return lo
    if f_hi >= 0:       # even a very wide cap still overpays -> cap at the wide bound
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = bs_call(S, mid, T, r, sigma) - target
        if abs(f_mid) < tol:
            return mid
        if f_mid > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def collar_caps(vol_in: pd.Series, put_otm: float = 0.05, r: float = 0.03,
                 T: float = 1.0 / 12.0) -> pd.Series:
    """Monthly cap (as a return, e.g. +0.058 = +5.8%) that makes each month's collar
    "costless" at that month's trailing realized vol. S normalized to 1."""
    K_put = 1.0 - put_otm
    caps = {dt: solve_costless_call_strike(1.0, K_put, T, r, float(v)) - 1.0
            for dt, v in vol_in.items()}
    return pd.Series(caps, name="cap_pct")


# --------------------------------------------------------------------------- #
# The collar payoff
# --------------------------------------------------------------------------- #
def collar_returns(spy_ret: pd.Series, caps: pd.Series, put_otm: float = 0.05,
                    cost_bps: float = 5.0, legs: int = 2) -> pd.Series:
    """Clip SPY's realized monthly return to [-put_otm, cap], then charge one round of
    option-roll costs (``legs`` legs, one-way ``cost_bps`` x notional each, charged once per
    month — the put and call are both rolled monthly). No cost on the equity leg (buy & hold,
    never traded)."""
    floor = -put_otm
    clipped = spy_ret.clip(lower=floor, upper=caps)
    drag = legs * cost_bps / 1e4
    return (clipped - drag).rename("collar_ret")


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 3) -> float:
    """HAC (Newey-West, Bartlett kernel) t of the sample mean of ``x`` (intercept-only)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    xc = x - x.mean()
    s0 = np.sum(xc ** 2)
    S = s0
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        g = np.sum(xc[l:] * xc[:-l])
        S += 2.0 * w * g
    var_mean = S / (n ** 2)
    se = np.sqrt(var_mean) if var_mean > 0 else float("nan")
    return float(x.mean() / se) if se and se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


# --------------------------------------------------------------------------- #
# The headline splits
# --------------------------------------------------------------------------- #
def full_sample_drag(df: pd.DataFrame) -> dict:
    """Mean(collar - SPY) full sample: one-sample t and Newey-West t (monthly overlap)."""
    d = (df["collar_ret"] - df["spy_ret"]).values
    return {
        "n": len(d), "mean_diff_bps": float(np.nanmean(d) * 1e4),
        "t_plain": one_sample_t(d), "t_nw": newey_west_t(d, lags=3),
    }


def crash_floor_effect(df: pd.DataFrame, thresh: float = -0.05) -> dict:
    """Months where SPY fell more than the put strike: does the collar demonstrably cushion?"""
    hit = df["spy_ret"] < thresh
    d = (df.loc[hit, "collar_ret"] - df.loc[hit, "spy_ret"]).values
    return {
        "n": int(hit.sum()), "mean_cushion_pts": float(np.nanmean(d) * 100) if len(d) else float("nan"),
        "t_plain": one_sample_t(d), "worst_spy_pct": float(df.loc[hit, "spy_ret"].min() * 100) if hit.any() else float("nan"),
        "worst_collar_pct": float(df.loc[hit, "collar_ret"].min() * 100) if hit.any() else float("nan"),
    }


def cap_cost_effect(df: pd.DataFrame) -> dict:
    """Months where SPY beat that month's modeled cap: does the collar demonstrably cost you?"""
    hit = df["spy_ret"] > df["cap_pct"]
    d = (df.loc[hit, "collar_ret"] - df.loc[hit, "spy_ret"]).values
    return {
        "n": int(hit.sum()), "mean_cost_pts": float(np.nanmean(d) * 100) if len(d) else float("nan"),
        "t_plain": one_sample_t(d), "share_of_months": float(hit.mean() * 100),
        "best_spy_pct": float(df.loc[hit, "spy_ret"].max() * 100) if hit.any() else float("nan"),
    }


def sharpe_excess(ret: pd.Series, rf_annual: float = 0.03, ann: int = 12) -> float:
    rf_m = rf_annual / ann
    ex = ret - rf_m
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(ann)) if sd > 0 else float("nan")


def terminal_wealth(ret: pd.Series) -> float:
    return float((1.0 + ret).prod())


def max_drawdown(ret: pd.Series) -> float:
    wealth = (1.0 + ret).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def exclude_windows(df: pd.DataFrame, windows: list[tuple[str, str]]) -> pd.DataFrame:
    """Drop every row whose index falls inside any [start, end] window (inclusive)."""
    keep = pd.Series(True, index=df.index)
    for lo, hi in windows:
        keep &= ~((df.index >= lo) & (df.index <= hi))
    return df.loc[keep]


def breakeven_cost_bps(spy_ret: pd.Series, caps: pd.Series, put_otm: float = 0.05,
                        legs: int = 2, lo: float = 0.0, hi: float = 100.0,
                        tol: float = 1e-4, max_iter: int = 60) -> float:
    """Per-leg one-way cost (bps) at which mean(collar - SPY) crosses zero, by bisection.
    Below this cost the modeled collar shows a positive full-sample edge; above it, a drag."""
    def diff_at(cb: float) -> float:
        coll = collar_returns(spy_ret, caps, put_otm=put_otm, cost_bps=cb, legs=legs)
        return float((coll - spy_ret).mean())
    f_lo = diff_at(lo)
    if f_lo <= 0:
        return lo
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if diff_at(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def window_drawdown(ret_full: pd.Series, start: str, end: str) -> float:
    """Max drawdown measured on the CONTINUOUS wealth path but reported only over
    [start, end] (a genuine peak-to-trough inside the window, not re-based to 1.0 at the
    window's first bar)."""
    wealth = (1.0 + ret_full).cumprod()
    peak_running = wealth.cummax()
    dd = wealth / peak_running - 1.0
    sl = dd.loc[(dd.index >= start) & (dd.index <= end)]
    return float(sl.min()) if len(sl) else float("nan")


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(spy_ret: pd.Series, floor: float, cap: float, cost_bps: float = 0.0
                      ) -> dict:
    """Run the full-sample drag test on a synthetic GBM world with a FIXED floor/cap
    (no vol-timing needed here — the point is purely to prove the clip-and-drag mechanics
    react correctly to a wide (null) vs tight (planted) band). ``cost_bps=0`` by default so
    the test isolates the CLIP effect itself — a flat cost would bias every world's mean the
    same deterministic amount and is not what this control is checking."""
    collar = (spy_ret.clip(lower=floor, upper=cap) - 2 * cost_bps / 1e4).rename("collar_ret")
    d = (collar - spy_ret).values
    return {"mean_diff_bps": float(np.nanmean(d) * 1e4), "t_plain": one_sample_t(d),
            "t_nw": newey_west_t(d, lags=3)}
