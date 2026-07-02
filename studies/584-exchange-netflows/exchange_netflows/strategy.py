"""The engine and its honest controls — Study 584 (Exchange-Netflows).

The claim, at full strength (crypto on-chain desks — CryptoQuant, Glassnode, Nansen): a rising
exchange **net-inflow** (coins moving TO exchanges, deposits minus withdrawals) is **bearish** for
BTC's forward return, because deposited coins are coins about to be *sold*; a net-*outflow* (coins
to cold storage) is bullish accumulation. The tradable version: reduce/short exposure on
net-inflow spikes, add on net-outflow spikes.

This module measures that, honestly, on the deterministic synthetic world in ``data.py`` (a real
exchange-netflow tape is a paid, address-clustering product — see ``data.py`` for why this study
is synthetic-only and capped at WEAK/NONE):

1. **The signal.** Standardise the net-inflow (``z_flow``). The folklore predicts a *negative*
   relation between ``z_flow[t]`` and the forward BTC return over ``(t, t+lag]``.

2. **The headline statistic.** Two complementary reads:
   - a **cross-sectional OLS** slope of forward return on ``z_flow`` (the *sign* is the folklore:
     negative = bearish inflows), with its *t*-stat;
   - a **high-vs-low sort**: split days into a *high-inflow* tail and a *low-inflow* (net-outflow)
     tail; the folklore predicts high-inflow days earn *less*. A Welch two-sample *t* on the two
     tails' forward returns.

3. **The label-shuffle placebo null.** Shuffle the ``z_flow`` labels against forward returns; a
   real relation sits in the tail, noise does not.

4. **The tradable overlay + costs.** A signal-timed long/flat (or long/short) BTC book: go
   long when net-outflow (bullish), flat/short when net-inflow (bearish). Report gross AND net of
   a one-way cost per turn; the short leg pays a crypto borrow/funding charge.

5. **Robustness sweep.** Re-run the slope-*t* across execution lags and tail thresholds.

6. **A seed-robust synthetic positive control (>= 20 seeds).** Plant the folklore
   (``bear_beta < 0``) and confirm the engine recovers a negative slope-*t* past the bar, and
   stays flat (~0) at the null.

Execution convention: the signal on day *t* is the netflow **known at that day's close** (a public
on-chain quantity); it is tested against the return over ``(t, t+lag]``. The one documented lag is
``lag`` days (default 1: see today's netflow, trade tomorrow). No future data enters the signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as _data


# ---------------------------------------------------------------------------
# Align the signal with its forward target
# ---------------------------------------------------------------------------
def aligned(frame: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Return a clean frame with ``z_flow`` (signal at t) and ``fwd`` (return over (t, t+lag])."""
    fwd = _data.forward_return(frame, lag=lag)
    out = pd.DataFrame({"z_flow": frame["z_flow"], "fwd": fwd}).dropna()
    return out


# ---------------------------------------------------------------------------
# Headline statistic 1 — the cross-sectional slope (sign IS the folklore)
# ---------------------------------------------------------------------------
def flow_regression(frame: pd.DataFrame, lag: int = 1) -> dict:
    """OLS of forward BTC return on standardised net-inflow.

    Slope < 0 = the bearish folklore (more inflow -> lower forward return). Returns the slope, its
    *t*-stat, the Pearson correlation and n.
    """
    a = aligned(frame, lag=lag)
    if len(a) < 6:
        return {"slope": float("nan"), "slope_t": float("nan"), "corr": float("nan"), "n": len(a)}
    x = a["z_flow"].to_numpy(dtype=float)
    y = a["fwd"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, kk = X.shape
    dof = max(n - kk, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_slope = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return {
        "slope": float(coef[1]),
        "slope_t": float(coef[1] / se_slope) if se_slope > 0 else float("nan"),
        "corr": corr,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Headline statistic 2 — the high-inflow vs low-inflow sort
# ---------------------------------------------------------------------------
def sort_spread(frame: pd.DataFrame, lag: int = 1, frac: float = 0.2) -> dict:
    """Split days by net-inflow tail; compare forward returns of high-inflow vs low-inflow days.

    The folklore predicts high-inflow (bearish) days earn *less* than low-inflow (net-outflow,
    bullish) days, so ``spread = low_inflow_mean - high_inflow_mean`` should be **positive**.

    Returns bucket means, the spread and a Welch two-sample *t* (plus the per-day forward returns
    of each tail).
    """
    a = aligned(frame, lag=lag).sort_values("z_flow")
    if len(a) < 10:
        return {"spread": float("nan"), "t": float("nan"), "n": len(a)}
    k = max(2, int(round(len(a) * frac)))
    low = a.head(k)["fwd"].to_numpy(dtype=float)    # lowest inflow (net-outflow, bullish)
    high = a.tail(k)["fwd"].to_numpy(dtype=float)   # highest inflow (bearish)
    va, vb = low.var(ddof=1), high.var(ddof=1)
    se = np.sqrt(va / len(low) + vb / len(high))
    spread = low.mean() - high.mean()
    t = spread / se if se > 0 else float("nan")
    return {
        "low_inflow_mean": float(low.mean()),
        "high_inflow_mean": float(high.mean()),
        "spread": float(spread),
        "t": float(t),
        "n": int(len(a)),
        "k": int(k),
        "low_ret": low,
        "high_ret": high,
    }


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null (on the sort spread)
# ---------------------------------------------------------------------------
def placebo_pvalue(frame: pd.DataFrame, lag: int = 1, frac: float = 0.2,
                   n_perm: int = 2000, seed: int = _data.STUDY_SEED) -> float:
    """Label-shuffle placebo p-value for the high-vs-low sort spread.

    Shuffle ``z_flow`` against forward returns ``n_perm`` times; the p-value is the fraction of
    shuffles whose |spread| >= the observed |spread|.
    """
    a = aligned(frame, lag=lag)
    if len(a) < 12:
        return float("nan")
    obs = abs(sort_spread(frame, lag=lag, frac=frac)["spread"])
    z = a["z_flow"].to_numpy(dtype=float)
    y = a["fwd"].to_numpy(dtype=float)
    n = len(y)
    k = max(2, int(round(n * frac)))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        order = np.argsort(z[perm])          # sort forward returns by shuffled netflow
        ys = y[order]
        spread = ys[:k].mean() - ys[-k:].mean()
        if abs(spread) >= obs - 1e-18:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Tradable overlay + costs
# ---------------------------------------------------------------------------
def strategy_returns(frame: pd.DataFrame, lag: int = 1, band: float = 0.5,
                     allow_short: bool = True) -> pd.Series:
    """Signal-timed BTC book driven by the netflow sign.

    Position on day *t* (applied to the return over ``(t, t+lag]``): go **long** (+1) when the
    net-inflow is strongly negative (net-outflow, bullish), **flat/short** when strongly positive
    (bearish). ``band`` is the ``|z_flow|`` deadband; inside it the book is flat. If
    ``allow_short`` is False the short leg is clipped to flat (long/flat only).

    Returns the (gross) per-period strategy return series aligned to the forward window.
    """
    a = aligned(frame, lag=lag)
    z = a["z_flow"]
    pos = pd.Series(0.0, index=a.index)
    pos[z <= -band] = 1.0                 # bullish outflow -> long
    pos[z >= band] = -1.0 if allow_short else 0.0   # bearish inflow -> short/flat
    return (pos * a["fwd"]).rename("strat_ret")


def strategy_stats(frame: pd.DataFrame, lag: int = 1, band: float = 0.5,
                   allow_short: bool = True, cost_bps: float = 10.0,
                   borrow_ann_bps: float = 300.0) -> dict:
    """Gross and net stats for the netflow-timed book.

    - **Gross** mean per-period return and an annualised Sharpe (per-period Sharpe * sqrt(252/lag)).
    - **Net** charges a one-way ``cost_bps`` on every change of position (turnover) and, on days
      the book is short, an annual crypto borrow/funding of ``borrow_ann_bps`` pro-rated over the
      ``lag``-day holding period.

    Returns a dict with gross/net mean, turnover, and gross/net annualised Sharpe.
    """
    a = aligned(frame, lag=lag)
    z = a["z_flow"]
    pos = pd.Series(0.0, index=a.index)
    pos[z <= -band] = 1.0
    pos[z >= band] = -1.0 if allow_short else 0.0
    gross = (pos * a["fwd"])
    dpos = pos.diff().abs().fillna(pos.abs())
    turn_cost = dpos * (cost_bps * 1e-4)
    short_days = (pos < 0).astype(float)
    borrow = short_days * (borrow_ann_bps * 1e-4) * (lag / 252.0)
    net = gross - turn_cost - borrow

    periods_per_year = 252.0 / lag

    def _sharpe(s: pd.Series) -> float:
        sd = s.std(ddof=1)
        return float(s.mean() / sd * np.sqrt(periods_per_year)) if sd and sd > 0 else float("nan")

    return {
        "gross_mean": float(gross.mean()),
        "net_mean": float(net.mean()),
        "gross_sharpe": _sharpe(gross),
        "net_sharpe": _sharpe(net),
        "turnover": float(dpos.mean()),
        "n_active": int((pos != 0).sum()),
        "n": int(len(a)),
    }


# ---------------------------------------------------------------------------
# Robustness sweep
# ---------------------------------------------------------------------------
def robustness_sweep(frame: pd.DataFrame, lags=(1, 2, 3, 5), fracs=(0.1, 0.2, 0.3)) -> pd.DataFrame:
    """Slope-*t* and sort-*t* across execution lags and tail thresholds.

    A real signal keeps its sign (and stays significant) across reasonable choices; a fragile one
    depends on the exact lag/threshold.
    """
    rows = []
    for lag in lags:
        reg = flow_regression(frame, lag=lag)
        for fr in fracs:
            ss = sort_spread(frame, lag=lag, frac=fr)
            rows.append({
                "lag": lag,
                "frac": fr,
                "slope_t": reg["slope_t"],
                "sort_spread_bps": ss["spread"] * 1e4,
                "sort_t": ss["t"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Seed-robust synthetic positive control (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(bear_beta: float, n_seeds: int = 25, base_seed: int = _data.STUDY_SEED,
                     lag: int = 1, n_days: int = 1500) -> float:
    """Average the flow-regression slope-*t* over ``n_seeds`` synthetic worlds.

    The house rule: any synthetic-dependent claim averages the stat over >= 20 seeds so no single
    lucky RNG seed can manufacture significance. Returns the mean slope-*t* across seeds for a
    planted ``bear_beta`` (negative = the folklore).
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        frame, _ = _data.synthetic_series(n_days=n_days, bear_beta=bear_beta, seed=s)
        ts.append(flow_regression(frame, lag=lag)["slope_t"])
    return float(np.nanmean(ts))
