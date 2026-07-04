"""Tested rules + statistics for Study 634 — US-Leads-the-World.

The claim: today's US close predicts tomorrow's Tokyo / Frankfurt / London / Sydney
sessions — the time-zone spillover that never quite arbitrages away. Everything here runs
on the aligned pair frames from ``data.build_pairs`` (one row per US day: the SPY day-t
close-to-close return, then the NEXT foreign session's close-to-close ``cc``, overnight
``gap`` and intraday ``oc`` legs).

Statistics are autocorrelation-robust throughout: OLS slopes carry Newey-West (HAC)
standard errors with the Bartlett kernel (predictive daily regressions on overlapping
world calendars are exactly where naive OLS t's overstate); strategy means carry HAC t's
too. The shuffle placebo (a *random* baseline) is averaged over ``PLACEBO_SEEDS = 50``
seeds — single-seed baselines are banned by house rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PLACEBO_SEEDS = 50


# --------------------------------------------------------------------------- #
# Robust inference primitives
# --------------------------------------------------------------------------- #
def _nw_lags(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_mean(returns, lags: int | None = None) -> dict:
    """Newey-West (HAC) t for a sample mean (Bartlett kernel)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    mu = r.mean()
    e = r - mu
    if lags is None:
        lags = _nw_lags(n)
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {"mean_bps": mu * 1e4, "t": mu / se if se > 0 else np.nan, "n": n}


def ols_hac(x, y, lags: int | None = None) -> dict:
    """OLS slope of y on x with a Newey-West (HAC) standard error.

    slope b = Sxy/Sxx; Var(b) = n * LRV(u) / Sxx^2 with u_t = (x_t - x̄) e_t and the
    Bartlett-weighted long-run variance LRV. Returns slope, HAC se, t, n, R².
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    xd = x - x.mean()
    sxx = float(xd @ xd)
    b = float(xd @ (y - y.mean())) / sxx
    a = y.mean() - b * x.mean()
    e = y - a - b * x
    if lags is None:
        lags = _nw_lags(n)
    u = xd * e
    lrv = float(u @ u) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(u[k:] @ u[:-k]) / n
    se = np.sqrt(n * lrv) / sxx
    tss = float((y - y.mean()) @ (y - y.mean()))
    r2 = 1.0 - float(e @ e) / tss if tss > 0 else np.nan
    return {"slope": b, "se": se, "t": b / se if se > 0 else np.nan,
            "n": n, "r2": r2, "lags": lags}


# --------------------------------------------------------------------------- #
# The signal — per-market slope + the gap/intraday decomposition
# --------------------------------------------------------------------------- #
def market_summary(p: pd.DataFrame) -> dict:
    """cc / gap / oc predictive regressions on US day-t for one market.

    ``cc`` uses every aligned day. The open-based legs (``gap``, ``oc``) are measured on
    the **live-open** subsample only (days where Yahoo's index open is not a stale
    carry-forward of the previous close) — the stale share is returned so it is always
    reported next to the numbers it conditions.
    """
    lp = p[p["live_open"]]
    return {
        "cc": ols_hac(p["us"], p["cc"]),
        "gap": ols_hac(lp["us"], lp["gap"]),
        "oc": ols_hac(lp["us"], lp["oc"]),
        "live_share": len(lp) / len(p) if len(p) else np.nan,
        "n": len(p),
        "start": str(p["us_date"].min().date()),
        "end": str(p["f_date"].max().date()),
    }


def shuffle_placebo(bk: pd.DataFrame, n_seeds: int = PLACEBO_SEEDS) -> dict:
    """Random baseline (averaged over >= 20 seeds, house rule): shuffle the US predictor.

    Destroys the time link while keeping both marginals; the |t| of the slope must
    collapse to noise level (~1). Returns the mean |t| and mean |slope| across seeds.
    """
    ts, bs = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(634 + s)
        xs = rng.permutation(bk["us"].values)
        r = ols_hac(xs, bk["basket"].values)
        ts.append(abs(r["t"]))
        bs.append(abs(r["slope"]))
    return {"mean_abs_t": float(np.mean(ts)), "mean_abs_slope": float(np.mean(bs)),
            "n_seeds": n_seeds}


# --------------------------------------------------------------------------- #
# Decay / conditioning
# --------------------------------------------------------------------------- #
def era_slopes(p: pd.DataFrame, eras: list[tuple[str, str, str]],
               ycol: str = "cc") -> list[dict]:
    """Slope + HAC t per era; eras = [(label, start, end_inclusive_year), ...]."""
    out = []
    for lab, y0, y1 in eras:
        q = p[(p["us_date"] >= y0) & (p["us_date"] <= f"{y1}-12-31")]
        r = ols_hac(q["us"], q[ycol])
        r["label"] = lab
        out.append(r)
    return out


def size_conditional(bk: pd.DataFrame, n_q: int = 5) -> list[dict]:
    """Slope + HAC t by |US move| quintile (does the spillover need a big sneeze?)."""
    q = pd.qcut(bk["us"].abs(), n_q, labels=False)
    out = []
    for g in range(n_q):
        sub = bk[q == g]
        r = ols_hac(sub["us"], sub["basket"])
        r["label"] = f"Q{g + 1}"
        r["mean_abs_us_pct"] = float(sub["us"].abs().mean() * 100)
        out.append(r)
    return out


def prepost(p: pd.DataFrame, split: str = "2010-01-01", ycol: str = "cc") -> dict:
    """Pre/post-split slopes + a t on the DIFFERENCE (HAC se's combined in quadrature)."""
    pre = p[p["us_date"] < split]
    post = p[p["us_date"] >= split]
    r1 = ols_hac(pre["us"], pre[ycol])
    r2 = ols_hac(post["us"], post[ycol])
    t_diff = (r1["slope"] - r2["slope"]) / np.sqrt(r1["se"] ** 2 + r2["se"] ** 2)
    return {"pre": r1, "post": r2, "t_diff": float(t_diff)}


# --------------------------------------------------------------------------- #
# Tradability — the one feasible trade, and the phantom one
# --------------------------------------------------------------------------- #
def feasible_trade(p: pd.DataFrame, cost_bps_oneway: float = 5.0) -> dict:
    """The only executable version: follow the US sign at the NEXT foreign OPEN.

    You know the SPY close (16:00 ET) hours before Tokyo/Sydney open and the night
    before Frankfurt/London open — so you can be long/short from the foreign **open**
    to its close. The overnight gap is forfeit by construction (it happens before the
    first tradable print). One position per day, one round trip per day: net = gross −
    2 × one-way cost. Measured on live-open days only (a stale index open would credit
    the trade with repricing that happened before any order could fill).
    """
    lp = p[p["live_open"] & (p["us"] != 0)]
    strat = np.sign(lp["us"].values) * lp["oc"].values
    g = hac_mean(strat)
    rt = 2.0 * cost_bps_oneway
    return {"gross_bps": g["mean_bps"], "t": g["t"], "n": g["n"],
            "net_bps": g["mean_bps"] - rt, "cost_oneway_bps": cost_bps_oneway}


def phantom_trade(p: pd.DataFrame) -> dict:
    """The INFEASIBLE close-to-close version every naive backtest runs.

    Holding the foreign market close-to-close on the US sign requires entering at the
    foreign close of day t — which prints BEFORE the US day-t close exists (Tokyo closes
    ~01:00 ET, Europe ~11:30 ET). It is quoted only to show how much of the paper edge
    is a time-machine artefact; it is never a strategy.
    """
    strat = np.sign(p["us"].values) * p["cc"].values
    g = hac_mean(strat)
    return {"gross_bps": g["mean_bps"], "t": g["t"], "n": g["n"]}


def gap_share(summary: dict) -> float:
    """Fraction of the close-to-close spillover slope delivered by the overnight gap."""
    return summary["gap"]["slope"] / summary["cc"]["slope"]
