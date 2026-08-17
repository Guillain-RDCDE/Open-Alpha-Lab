"""Strategy + inference for Study 936 — Tolerance Bands.

One book, four rebalancing policies. The book is a fixed-weight sleeve mix (the
headline: 60% SPY / 40% IEF; the second book: 50/30/20 SPY/IEF/GLD). The weights
drift with the sleeves' daily total returns; a *policy* decides when to trade them
back to target:

- ``drift``     — never rebalance (the do-nothing floor, for reference only);
- ``annual``    — on the first session of each calendar year;
- ``quarterly`` — on the first session of each calendar quarter;
- ``bands``     — whenever any sleeve is off target by more than
  ``min(5 percentage points absolute, 25% relative)`` — the Daryanani "5/25" rule.

**Execution lag (exactly one).** Weights are inspected at the close of day ``t``
*after* that day's return has been earned. If the calendar or a band says "trade",
the trade prints at the close of day ``t+1``. Nothing is ever executed at the price
that triggered it. There is no other lag anywhere in this module.

**Costs.** ``cost_bps`` is a **one-way** cost charged on the traded notional as a
fraction of NAV: a rebalance that sells 3% of NAV and buys 3% of NAV trades 6% of
NAV and pays ``6% x cost_bps``. No sleeve is ever short, so there is no borrow leg;
the honest missing cost is **tax**, which is not modelled (see the README).

**The race.** Every policy holds the *same* four sleeves in the *same* target mix,
so this is not a return-forecasting contest: it is a friction-and-timing contest.
Sharpes are reported **excess-of-cash** (minus BIL's total return) on both sides, and
the headline test is an HAC (Newey-West) *t* on the daily return **difference**
between two policies plus a circular block-bootstrap CI on the Sharpe difference.
Because both arms subtract the same cash leg, the cash cancels **exactly** in the
daily return difference — which is why that difference, not the Sharpe, is the clean
test statistic here.

**Where the cash leg does NOT cancel.** A Sharpe *ratio* is a nonlinear function of
the series, so the difference of two Sharpes is *not* invariant to the cash leg even
when both arms subtract the same one: subtracting cash lowers both numerators and
changes both denominators. On the headline 60/40 window the bands-minus-annual Sharpe
difference is −0.0261 excess-of-cash and −0.0340 gross-of-cash — same sign, same
(non-)significance, but a 30% different magnitude. Any comparison of Sharpe
differences across windows must therefore hold the cash treatment fixed; the 2003-2026
cross-check (which predates BIL) is gross-of-cash and is compared only against the
gross-of-cash number, never against the excess-of-cash headline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

POLICIES = ("drift", "annual", "quarterly", "bands")

# The "5/25" tolerance band, stated as an ASSUMPTION (it is folklore, not a
# calibrated parameter): trade when a sleeve is off target by 5 percentage points
# absolute OR 25% of its own target weight, whichever binds first. Swept in
# `band_sweep`.
BAND_ABS = 0.05
BAND_REL = 0.25


# --------------------------------------------------------------------------- #
# Policy triggers
# --------------------------------------------------------------------------- #
def band_thresholds(target: np.ndarray, band_abs: float = BAND_ABS,
                    band_rel: float = BAND_REL) -> np.ndarray:
    """Per-sleeve trigger width: ``min(band_abs, band_rel * target_i)``.

    For a 60% sleeve the absolute 5-point band binds (5pp < 15pp); for a 10% sleeve
    the 25% relative band binds (2.5pp < 5pp). That asymmetry is the whole point of
    the 5/25 formulation: small sleeves get proportionally tighter bands.
    """
    t = np.asarray(target, dtype=float)
    return np.minimum(band_abs, band_rel * t)


def band_breached(weights: np.ndarray, target: np.ndarray,
                  band_abs: float = BAND_ABS, band_rel: float = BAND_REL) -> bool:
    """True iff any sleeve's weight sits outside its 5/25 tolerance band."""
    w = np.asarray(weights, dtype=float)
    thr = band_thresholds(target, band_abs=band_abs, band_rel=band_rel)
    # 1e-12 slack so an exactly-on-the-edge weight is not decided by float noise
    # (0.60 - 0.55 evaluates to 0.049999999999999996 in binary floating point).
    return bool(np.any(np.abs(w - np.asarray(target, dtype=float)) >= thr - 1e-12))


def calendar_flags(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """Boolean array marking the LAST session of each calendar period in ``index``.

    The flag is read at the close of that session and traded at the next session's
    close, so a ``freq='Y'`` policy prints its trade on the first session of the new
    year — which is what a real annual rebalance looks like.

    Note on information: identifying "the last session of December" uses the *date*
    index one step ahead, i.e. the published exchange calendar, and **no price
    whatsoever**. That is the same information a real rebalancer has on 31 December;
    it is not price look-ahead. The band rule uses no forward information at all.
    """
    idx = pd.DatetimeIndex(index)
    if freq == "Y":
        key = idx.year.astype(np.int64)
    elif freq == "Q":
        key = idx.year.astype(np.int64) * 4 + (idx.quarter.astype(np.int64) - 1)
    elif freq == "M":
        key = idx.year.astype(np.int64) * 12 + (idx.month.astype(np.int64) - 1)
    else:
        raise ValueError(f"unknown calendar freq {freq!r}")
    flags = np.zeros(len(idx), dtype=bool)
    if len(idx) > 1:
        flags[:-1] = key[1:] != key[:-1]
    return flags


# --------------------------------------------------------------------------- #
# The backtest engine (one execution lag, one-way cost on traded notional)
# --------------------------------------------------------------------------- #
def run_policy(
    rets: pd.DataFrame,
    target,
    policy: str = "bands",
    cost_bps: float = 5.0,
    band_abs: float = BAND_ABS,
    band_rel: float = BAND_REL,
) -> dict:
    """Run one rebalancing policy on a frame of daily simple sleeve returns.

    Parameters
    ----------
    rets:
        Daily **simple** total returns, one column per sleeve, no NaNs.
    target:
        Target weights, one per column of ``rets`` (must sum to 1).
    policy:
        One of :data:`POLICIES`.
    cost_bps:
        **One-way** cost in bps charged on traded notional / NAV at each rebalance.
    band_abs, band_rel:
        The tolerance band (only used by ``policy='bands'``).

    Returns a dict with the daily portfolio return series ``r`` (net of cost), the
    gross series ``r_gross``, the per-day traded notional, the weight path, and the
    rebalance count.

    The loop is deliberately explicit rather than vectorised: the one-day execution
    lag between *observing* a breach and *trading* it is the single most abused
    detail in rebalancing back-tests, and it should be readable.
    """
    cols = list(rets.columns)
    tgt = np.asarray(target, dtype=float)
    if tgt.size != len(cols):
        raise ValueError("target must have one weight per column of rets")
    if not np.isclose(tgt.sum(), 1.0):
        raise ValueError("target weights must sum to 1")
    if rets.isna().any().any():
        raise ValueError("rets must not contain NaNs")

    R = rets.to_numpy(dtype=float)
    n, k = R.shape
    idx = pd.DatetimeIndex(rets.index)

    if policy == "annual":
        cal = calendar_flags(idx, "Y")
    elif policy == "quarterly":
        cal = calendar_flags(idx, "Q")
    elif policy == "monthly":
        cal = calendar_flags(idx, "M")
    elif policy in ("drift", "bands"):
        cal = np.zeros(n, dtype=bool)
    else:
        raise ValueError(f"unknown policy {policy!r}")

    cost = cost_bps * 1e-4
    r_net = np.zeros(n)
    r_gross = np.zeros(n)
    traded = np.zeros(n)
    wpath = np.zeros((n, k))

    w = tgt.copy()          # weights applied to today's returns (start-of-day)
    pending = False         # a rebalance was DECIDED yesterday -> execute at today's close
    n_rebals = 0

    for t in range(n):
        wpath[t] = w
        gross = float(w @ R[t])
        r_gross[t] = gross

        # Weights after the day's move, before any trade.
        grown = w * (1.0 + R[t])
        w_close = grown / grown.sum()

        cost_today = 0.0
        if pending:
            notional = float(np.abs(tgt - w_close).sum())
            traded[t] = notional
            cost_today = cost * notional
            w_close = tgt.copy()
            pending = False
            n_rebals += 1

        r_net[t] = gross - cost_today

        # Decide (using information available at the close of day t) whether to
        # trade at the close of day t+1. Exactly one execution lag.
        if policy == "bands":
            pending = band_breached(w_close, tgt, band_abs=band_abs, band_rel=band_rel)
        elif policy != "drift":
            pending = bool(cal[t])

        w = w_close

    out_r = pd.Series(r_net, index=idx, name=f"r_{policy}")
    return {
        "policy": policy,
        "r": out_r,
        "r_gross": pd.Series(r_gross, index=idx, name=f"rg_{policy}"),
        "traded": pd.Series(traded, index=idx, name="traded_notional"),
        "weights": pd.DataFrame(wpath, index=idx, columns=cols),
        "n_rebals": int(n_rebals),
        "traded_total": float(traded.sum()),
        "traded_per_year": float(traded.sum() / (n / TRADING_DAYS)),
        "cost_bps": cost_bps,
    }


# --------------------------------------------------------------------------- #
# Inference primitives (desk-standard)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float(u[lag:] @ u[:-lag]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def auto_lags(n: int) -> int:
    """Newey-West bandwidth rule of thumb, ``floor(4 (n/100)^(2/9))``."""
    return int(np.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0)))


def diff_tstat(r1: pd.Series, r2: pd.Series) -> float:
    """HAC *t* on the daily return **difference** ``r1 - r2``.

    Both arms hold the same sleeves in the same target mix and are measured
    excess-of-cash, so the market and the cash leg cancel in the difference: what is
    left is purely the effect of *when* each policy trades. |*t*| >= 2 is the desk's
    bar for calling a difference real.
    """
    d = (r1 - r2).dropna()
    if len(d) < 3:
        return float("nan")
    return newey_west_t(d.to_numpy(), lags=auto_lags(len(d)))


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised stats for a daily simple-return series (pass an *excess* series
    for the excess-of-cash Sharpe)."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu = r.mean()
    sd = r.std(ddof=1)
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    cagr = (
        float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
        if years > 0 and n and wealth.iloc[-1] > 0 else float("nan")
    )
    return {
        "n_days": int(n),
        "sharpe": float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "vol_ann": float(sd * np.sqrt(periods_per_year)),
        "cagr": cagr,
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()) if n else float("nan"),
        "mean_daily_bps": float(mu * 1e4),
        "tstat": newey_west_t(r.to_numpy(), lags=auto_lags(n)),
    }


def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).dropna()
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def bootstrap_diff_ci(
    r1: pd.Series,
    r2: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 936,
    periods_per_year: int = TRADING_DAYS,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the **Sharpe difference** ``S(r1) - S(r2)``.

    The two series are resampled *jointly* (the same block indices for both), so the
    bootstrap preserves the day-by-day pairing as well as the volatility clustering —
    the paired structure is where nearly all the sampling noise cancels.
    """
    a = pd.Series(r1).astype(float)
    b = pd.Series(r2).astype(float)
    common = a.dropna().index.intersection(b.dropna().index)
    x = a.loc[common].to_numpy(dtype=float)
    y = b.loc[common].to_numpy(dtype=float)
    n = x.size
    ann = np.sqrt(periods_per_year)

    def _sh(v):
        sd = v.std(ddof=1)
        return v.mean() / sd * ann if sd > 0 else np.nan

    if n < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}

    point = float(_sh(x) - _sh(y))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offs = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for i in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offs[None, :]) % n).ravel()[:n]
        boots[i] = _sh(x[idx]) - _sh(y[idx])
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": point, "ci_low": float(lo), "ci_high": float(hi),
        "frac_negative": float((valid < 0).mean()),
        "n_obs": int(n), "n_boot": int(valid.size), "block": int(block),
    }


# --------------------------------------------------------------------------- #
# The race
# --------------------------------------------------------------------------- #
def race(
    rets: pd.DataFrame,
    target,
    cash: pd.Series | None = None,
    cost_bps: float = 5.0,
    band_abs: float = BAND_ABS,
    band_rel: float = BAND_REL,
    policies=POLICIES,
) -> dict:
    """Run every policy on the same book and report the excess-of-cash race.

    ``cash`` is the daily simple return of the cash leg (BIL). Every policy's Sharpe
    is computed on ``r_policy - r_cash``; passing ``cash=None`` runs the race
    **gross-of-cash** — legitimate for the *return-difference* test (the cash leg
    cancels there exactly) but NOT comparable, as a Sharpe difference, to an
    excess-of-cash run (see the module docstring).

    A cash series that is missing on more than 1% of the sample is rejected rather
    than zero-filled: silently treating missing cash as a 0% day would inflate every
    excess Sharpe in the table and would be invisible in the output.

    Returns per-policy summaries (on excess series), the absolute drawdown and
    turnover, and the pairwise difference tests against the calendar policies.
    """
    if cash is None:
        c = pd.Series(0.0, index=rets.index)
    else:
        c = pd.Series(cash).reindex(rets.index).astype(float)
        missing = int(c.isna().sum())
        if missing > max(1, int(0.01 * len(c))):
            raise ValueError(
                f"cash leg missing on {missing}/{len(c)} days of the sample; refusing to "
                f"zero-fill (that would silently inflate every excess Sharpe). Align the "
                f"cash series to `rets` first, or pass cash=None for a gross-of-cash race."
            )
        c = c.fillna(0.0)

    runs = {p: run_policy(rets, target, policy=p, cost_bps=cost_bps,
                          band_abs=band_abs, band_rel=band_rel)
            for p in policies}

    excess = {p: (runs[p]["r"] - c).rename(f"e_{p}") for p in policies}
    stats = {p: summary(excess[p]) for p in policies}

    out = {
        "runs": runs,
        "excess": excess,
        "stats": stats,
        "dd_abs": {p: max_drawdown(runs[p]["r"]) for p in policies},
        "traded_per_year": {p: runs[p]["traded_per_year"] for p in policies},
        "n_rebals": {p: runs[p]["n_rebals"] for p in policies},
        "cost_drag_bps": {
            p: float((runs[p]["r_gross"] - runs[p]["r"]).mean() * 1e4 * TRADING_DAYS)
            for p in policies
        },
        "cost_bps": cost_bps,
        "band_abs": band_abs, "band_rel": band_rel,
    }
    # The headline pairwise comparisons: bands vs each calendar rule.
    pairs = [("bands", "annual"), ("bands", "quarterly"), ("quarterly", "annual")]
    out["pairs"] = {}
    for a, b in pairs:
        if a in policies and b in policies:
            out["pairs"][f"{a}_vs_{b}"] = {
                "sharpe_diff": stats[a]["sharpe"] - stats[b]["sharpe"],
                "t_diff": diff_tstat(excess[a], excess[b]),
                "cagr_diff_pp": (stats[a]["cagr"] - stats[b]["cagr"]) * 100.0,
                "mean_diff_bps": stats[a]["mean_daily_bps"] - stats[b]["mean_daily_bps"],
            }
    return out


# --------------------------------------------------------------------------- #
# Sweeps and cuts
# --------------------------------------------------------------------------- #
def cost_sweep(rets: pd.DataFrame, target, cash: pd.Series | None = None,
               grid=(0.0, 2.0, 5.0, 10.0, 25.0, 50.0)) -> list[dict]:
    """Sharpe of each policy, and the band-minus-calendar gaps, across one-way costs.

    Turnover is the whole mechanism here, so the cost level is not a footnote: it is
    the axis on which any band-vs-calendar claim lives or dies.
    """
    rows = []
    for c in grid:
        r = race(rets, target, cash=cash, cost_bps=c)
        rows.append({
            "cost_bps": c,
            **{f"sharpe_{p}": r["stats"][p]["sharpe"] for p in r["stats"]},
            "diff_bands_annual": r["pairs"]["bands_vs_annual"]["sharpe_diff"],
            "t_bands_annual": r["pairs"]["bands_vs_annual"]["t_diff"],
            "diff_bands_quarterly": r["pairs"]["bands_vs_quarterly"]["sharpe_diff"],
            "t_bands_quarterly": r["pairs"]["bands_vs_quarterly"]["t_diff"],
        })
    return rows


def band_sweep(rets: pd.DataFrame, target, cash: pd.Series | None = None,
               cost_bps: float = 5.0,
               grid=((0.02, 0.10), (0.03, 0.15), (0.05, 0.25), (0.08, 0.40), (0.10, 0.50))
               ) -> list[dict]:
    """Sweep the ASSUMED band width. 5/25 is folklore — check it is not a lucky pick."""
    rows = []
    for ba, br in grid:
        r = race(rets, target, cash=cash, cost_bps=cost_bps, band_abs=ba, band_rel=br)
        rows.append({
            "band_abs": ba, "band_rel": br,
            "sharpe_bands": r["stats"]["bands"]["sharpe"],
            "n_rebals": r["n_rebals"]["bands"],
            "traded_per_year": r["traded_per_year"]["bands"],
            "diff_vs_annual": r["pairs"]["bands_vs_annual"]["sharpe_diff"],
            "t_vs_annual": r["pairs"]["bands_vs_annual"]["t_diff"],
        })
    return rows


def era_cut(rets: pd.DataFrame, target, cash: pd.Series | None = None,
            split: str = "2016-01-01", cost_bps: float = 5.0) -> dict:
    """Split the sample and re-run the race on each half.

    A real policy edge should show the same sign in both halves; the 2007-2015 and
    2016-2026 halves differ enormously in the stock/bond correlation, which is the
    single biggest driver of how far a 60/40 book drifts.
    """
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        sub = rets.loc[sl]
        if len(sub) < 300:
            out[tag] = None
            continue
        c = None if cash is None else pd.Series(cash).reindex(sub.index)
        r = race(sub, target, cash=c, cost_bps=cost_bps)
        out[tag] = {
            "n_days": len(sub),
            "start": str(sub.index[0].date()), "end": str(sub.index[-1].date()),
            **{f"sharpe_{p}": r["stats"][p]["sharpe"] for p in r["stats"]},
            "diff_bands_annual": r["pairs"]["bands_vs_annual"]["sharpe_diff"],
            "t_bands_annual": r["pairs"]["bands_vs_annual"]["t_diff"],
            "traded_bands": r["traded_per_year"]["bands"],
            "traded_annual": r["traded_per_year"]["annual"],
        }
    return out


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — it never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, target=(0.5, 0.5), cost_bps: float = 5.0) -> dict:
    """Race the policies on a synthetic panel from :func:`data.synthetic_panel`.

    On the *planted* mean-reverting world the band rule should beat the annual rule
    (it acts when dispersion is actually large); on the random-walk null the gap
    should be ~0 gross and slightly negative net (pure friction). This proves the
    harness reads the sign correctly — it never supports the real-tape verdict.
    """
    asset_cols = [c for c in prices.columns if c != "cash"]
    px = prices[asset_cols]
    rets = px.pct_change().dropna()
    cash = prices["cash"].pct_change().reindex(rets.index) if "cash" in prices else None
    r = race(rets, target, cash=cash, cost_bps=cost_bps)
    return {
        "sharpe_bands": r["stats"]["bands"]["sharpe"],
        "sharpe_annual": r["stats"]["annual"]["sharpe"],
        "sharpe_quarterly": r["stats"]["quarterly"]["sharpe"],
        "sharpe_drift": r["stats"]["drift"]["sharpe"],
        "diff_bands_annual": r["pairs"]["bands_vs_annual"]["sharpe_diff"],
        "t_bands_annual": r["pairs"]["bands_vs_annual"]["t_diff"],
        "traded_bands": r["traded_per_year"]["bands"],
        "traded_annual": r["traded_per_year"]["annual"],
        "n_rebals_bands": r["n_rebals"]["bands"],
        "n_days": r["stats"]["bands"]["n_days"],
    }
